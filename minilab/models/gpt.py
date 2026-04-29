import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.checks import require
from minilab.config import BaseConfig
from minilab.nn.architecture import (
    GQA_ATTENTIONS,
    MOE_FFNS,
    QK_CLIP_ATTENTIONS,
    TOP_ONE_MOE_FFNS,
    resolve_deepseek_v4_attention,
)
from minilab.nn.connections import expand_residual_stream, reduce_residual_stream
from minilab.registry import get_attention, get_connection, get_ffn, get_norm, get_position, register_model


_ROPE_POSITIONS = {"rope", "gemma3_rope", "gemma4_rope", "yarn_rope", "qwen3_next_rope"}
_BIAS_POSITIONS = {"alibi", "t5_relative", "kerple_log", "kerple_power"}


@dataclass
class GPTConfig(BaseConfig):
    vocab_size: int = 50257
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int | None = None
    max_seq_len: int = 1024
    dropout: float = 0.0
    ffn_mult: float = 4.0
    attention: str = "mha"
    position: str = "rope"
    norm: str = "rmsnorm"
    ffn: str = "swiglu"
    connection: str = "residual"
    connection_expansion: int = 4
    num_experts: int = 8
    top_k_experts: int = 2
    post_norm: bool = False
    rope_base: float = 10000.0
    rope_local_base: float = 10000.0
    rope_global_base: float = 1000000.0
    rope_scaling_factor: float = 1.0
    rope_original_max_seq_len: int = 4096
    rope_partial_rotary_factor: float = 0.25
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    local_attention_window: int = 1024
    qwen3_next_full_attention_interval: int = 4
    attention_k_eq_v: bool = False
    per_layer_embedding_dim: int = 0
    final_logit_softcap: float = 0.0
    mtp_depth: int = 0
    mtp_loss_weight: float = 0.0

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        require(self.vocab_size > 0, "vocab_size must be > 0")
        require(self.dim > 0, "dim must be > 0")
        require(self.num_layers > 0, "num_layers must be > 0")
        require(self.num_heads > 0, "num_heads must be > 0")
        require(self.num_kv_heads > 0, "num_kv_heads must be > 0")
        require(self.max_seq_len > 0, "max_seq_len must be > 0")
        require(self.dim % self.num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")
        require(self.ffn_mult > 0, "ffn_mult must be > 0")
        require(self.connection_expansion > 0, "connection_expansion must be > 0")
        require(self.rope_base > 0, "rope_base must be > 0")
        require(self.rope_local_base > 0, "rope_local_base must be > 0")
        require(self.rope_global_base > 0, "rope_global_base must be > 0")
        require(self.rope_scaling_factor > 0, "rope_scaling_factor must be > 0")
        require(self.rope_original_max_seq_len > 0, "rope_original_max_seq_len must be > 0")
        require(0.0 < self.rope_partial_rotary_factor <= 1.0, "rope_partial_rotary_factor must be in (0, 1]")
        require(self.yarn_beta_fast > 0, "yarn_beta_fast must be > 0")
        require(self.yarn_beta_slow > 0, "yarn_beta_slow must be > 0")
        require(self.local_attention_window > 0, "local_attention_window must be > 0")
        require(self.qwen3_next_full_attention_interval > 0, "qwen3_next_full_attention_interval must be > 0")
        require(self.per_layer_embedding_dim >= 0, "per_layer_embedding_dim must be >= 0")
        require(self.final_logit_softcap >= 0, "final_logit_softcap must be >= 0")
        require(self.mtp_depth >= 0, "mtp_depth must be >= 0")
        require(self.mtp_loss_weight >= 0, "mtp_loss_weight must be >= 0")
        require((self.mtp_depth == 0) == (self.mtp_loss_weight == 0), (
            "mtp_depth and mtp_loss_weight must be enabled together"
        ))
        if _attention_uses_gqa(self.attention):
            require(self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
        if self.position in _ROPE_POSITIONS:
            require((self.dim // self.num_heads) % 2 == 0, "RoPE requires even head dimension")
        if self.position == "sinusoidal":
            require(self.dim % 2 == 0, "sinusoidal position requires even dim")
        if self.attention == "cosformer":
            require(self.position == "none", "cosFormer owns its positional reweighting; set position='none'")
        if self.attention == "lightning":
            require(self.position == "none", "Lightning Attention owns positional decay; set position='none'")
        if self.attention == "gemma3":
            require(self.position == "gemma3_rope", "Gemma 3 attention schedule requires position='gemma3_rope'")
        if self.attention == "gemma4":
            require(self.position == "gemma4_rope", "Gemma 4 attention schedule requires position='gemma4_rope'")
        if self.attention == "qwen3_next":
            require(self.position in {"qwen3_next_rope", "yarn_rope"}, (
                "Qwen3-Next schedule requires position='qwen3_next_rope' or 'yarn_rope'"
            ))
        if self.per_layer_embedding_dim > 0:
            require(self.connection == "residual", "per-layer embeddings require residual connections")
        if self.ffn in MOE_FFNS:
            require(self.num_experts > 0, "num_experts must be > 0")
            require(1 <= self.top_k_experts <= self.num_experts, "top_k_experts must be in [1, num_experts]")
            if self.ffn in TOP_ONE_MOE_FFNS:
                require(self.top_k_experts == 1, f"{self.ffn} requires top_k_experts=1")


PRESETS = {
    "gpt-tiny": {"dim": 128, "num_layers": 4, "num_heads": 4, "max_seq_len": 256},
    "gpt-small": {"dim": 256, "num_layers": 6, "num_heads": 8, "max_seq_len": 512},
    "gpt-medium": {"dim": 512, "num_layers": 12, "num_heads": 8, "max_seq_len": 1024},
    "gpt-large": {"dim": 768, "num_layers": 24, "num_heads": 12, "max_seq_len": 2048},
}


def gpt_preset(name, vocab_size):
    require(name in PRESETS, f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    preset = PRESETS[name]
    return GPTConfig(
        vocab_size=vocab_size,
        dim=preset["dim"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        max_seq_len=preset["max_seq_len"],
    )


def _build_transformer_ffn(config):
    ffn_hidden = int(config.dim * config.ffn_mult)
    if config.ffn in MOE_FFNS:
        return get_ffn(config.ffn)(
            config.dim,
            ffn_hidden,
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
        )
    return get_ffn(config.ffn)(config.dim, ffn_hidden)


def _build_transformer_attention(config, block_id):
    attention = _resolve_attention_name(config, block_id)
    attn_cls = get_attention(attention)
    if attention in GQA_ATTENTIONS:
        if attention in {"gqa_qknorm_partial_rope", "gated_gqa_qknorm_partial_rope"}:
            attn = attn_cls(
                config.dim,
                config.num_heads,
                config.num_kv_heads,
                config.dropout,
                rope_fraction=config.rope_partial_rotary_factor,
            )
        else:
            attn = attn_cls(config.dim, config.num_heads, config.num_kv_heads, config.dropout)
        if attention == "sliding_window_gqa_qknorm":
            attn.window_size = config.local_attention_window
        return attention, attn
    return attention, attn_cls(config.dim, config.num_heads, config.dropout)


def _build_gpt_position_modules(config):
    head_dim = config.dim // config.num_heads
    if config.position == "rope":
        return get_position("rope")(head_dim, config.max_seq_len, base=config.rope_base), None, None
    if config.position in {"gemma3_rope", "gemma4_rope", "qwen3_next_rope"}:
        local_pos_enc = get_position("rope")(head_dim, config.max_seq_len, base=config.rope_local_base)
        if config.position == "gemma4_rope":
            global_pos_enc = get_position("proportional_rope")(
                head_dim,
                config.max_seq_len,
                base=config.rope_global_base,
                partial_rotary_factor=config.rope_partial_rotary_factor,
                factor=config.rope_scaling_factor,
            )
        elif config.position == "qwen3_next_rope":
            global_pos_enc = get_position("proportional_rope")(
                head_dim,
                config.max_seq_len,
                base=config.rope_global_base,
                partial_rotary_factor=config.rope_partial_rotary_factor,
            )
        else:
            global_pos_enc = get_position("rope")(head_dim, config.max_seq_len, base=config.rope_global_base)
        return None, local_pos_enc, global_pos_enc
    if config.position == "yarn_rope":
        pos_enc = get_position("yarn_rope")(
            head_dim,
            config.max_seq_len,
            base=config.rope_base,
            factor=config.rope_scaling_factor,
            original_max_seq_len=config.rope_original_max_seq_len,
            beta_fast=config.yarn_beta_fast,
            beta_slow=config.yarn_beta_slow,
        )
        return pos_enc, None, None
    if config.position in _BIAS_POSITIONS:
        return get_position(config.position)(config.num_heads, config.max_seq_len), None, None
    return get_position(config.position)(config.dim, config.max_seq_len), None, None


def _build_per_layer_embeddings(config):
    if config.per_layer_embedding_dim == 0:
        return None, None, None, None, None
    embed = nn.Embedding(config.vocab_size, config.num_layers * config.per_layer_embedding_dim)
    projection = nn.Linear(config.dim, config.num_layers * config.per_layer_embedding_dim, bias=False)
    norm = get_norm(config.norm)(config.per_layer_embedding_dim)
    return embed, projection, norm, 1.0 / math.sqrt(2.0), config.dim ** -0.5


class TransformerBlock(nn.Module):
    def __init__(self, config, block_id):
        super().__init__()
        self.block_id = block_id
        self.per_layer_embedding_dim = config.per_layer_embedding_dim if block_id < config.num_layers else 0
        self.attn_norm = get_norm(config.norm)(config.dim)
        self.ffn_norm = get_norm(config.norm)(config.dim)
        self.post_norm = config.post_norm
        if self.post_norm:
            self.attn_post_norm = get_norm(config.norm)(config.dim)
            self.ffn_post_norm = get_norm(config.norm)(config.dim)
        if self.per_layer_embedding_dim > 0:
            self.per_layer_input_gate = nn.Linear(config.dim, self.per_layer_embedding_dim, bias=False)
            self.per_layer_projection = nn.Linear(self.per_layer_embedding_dim, config.dim, bias=False)
            self.post_per_layer_input_norm = get_norm(config.norm)(config.dim)
        self.drop = nn.Dropout(config.dropout)

        self.ffn = _build_transformer_ffn(config)
        self.attention_name, self.attn = _build_transformer_attention(config, block_id)

        conn_cls = get_connection(config.connection)
        if config.connection == "residual":
            self.attn_conn = conn_cls(config.dim)
            self.ffn_conn = conn_cls(config.dim)
        else:
            self.attn_conn = conn_cls(config.dim, config.connection_expansion, layer_id=2 * block_id)
            self.ffn_conn = conn_cls(config.dim, config.connection_expansion, layer_id=2 * block_id + 1)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False, per_layer_input=None):
        def attn_branch(h):
            out = self.attn(self.attn_norm(h), freqs_cis, attn_bias, is_causal)
            if self.post_norm:
                out = self.attn_post_norm(out)
            return self.drop(out)

        def ffn_branch(h):
            out = self.ffn(self.ffn_norm(h))
            if self.post_norm:
                out = self.ffn_post_norm(out)
            return self.drop(out)

        x = self.attn_conn(x, attn_branch)
        x = self.ffn_conn(x, ffn_branch)
        if self.per_layer_embedding_dim > 0:
            require(per_layer_input is not None, "per-layer embedding block requires per_layer_input")
            residual = x
            ple = F.gelu(self.per_layer_input_gate(x), approximate="tanh") * per_layer_input
            x = residual + self.post_per_layer_input_norm(self.per_layer_projection(ple))
        return x


def _resolve_attention_name(config, block_id):
    if config.attention == "gemma3":
        return "gqa_qknorm" if _is_gemma3_global_layer(block_id) else "sliding_window_gqa_qknorm"
    if config.attention == "gemma4":
        if _is_gemma4_global_layer(block_id):
            return "gqa_qknorm_kv_tied" if config.attention_k_eq_v else "gqa_qknorm"
        return "sliding_window_gqa_qknorm"
    if config.attention == "qwen3_next":
        if (block_id + 1) % config.qwen3_next_full_attention_interval == 0:
            return "gated_gqa_qknorm_partial_rope"
        return "gated_deltanet"
    return resolve_deepseek_v4_attention(config.attention, block_id)


def _attention_uses_gqa(attention):
    if attention in {"gemma3", "gemma4", "qwen3_next"}:
        return True
    return resolve_deepseek_v4_attention(attention, 0) in GQA_ATTENTIONS


def _is_gemma3_global_layer(block_id):
    return (block_id + 1) % 6 == 0


def _is_gemma4_global_layer(block_id):
    return (block_id + 1) % 6 == 0


class MultiTokenPredictionModule(nn.Module):
    """DeepSeek-V3 sequential MTP module with shared token embedding and LM head."""

    def __init__(self, config, block_id):
        super().__init__()
        self.hidden_norm = get_norm(config.norm)(config.dim)
        self.embed_norm = get_norm(config.norm)(config.dim)
        self.proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.block = TransformerBlock(config, block_id)

    def forward(self, hidden, future_emb, freqs_cis=None, attn_bias=None, is_causal=True):
        x = self.proj(torch.cat([self.hidden_norm(hidden), self.embed_norm(future_emb)], dim=-1))
        return self.block(x, freqs_cis=freqs_cis, attn_bias=attn_bias, is_causal=is_causal)


@register_model("gpt")
class GPT(BaseModel):
    config_class = GPTConfig
    provides_hidden_states = True

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_layers)])
        self.ln_f = get_norm(config.norm)(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        (
            self.embed_tokens_per_layer,
            self.per_layer_model_projection,
            self.per_layer_projection_norm,
            self.per_layer_input_scale,
            self.per_layer_model_projection_scale,
        ) = _build_per_layer_embeddings(config)
        self.mtp_modules = nn.ModuleList([
            MultiTokenPredictionModule(config, config.num_layers + i)
            for i in range(config.mtp_depth)
        ])

        self.pos_enc, self.local_pos_enc, self.global_pos_enc = _build_gpt_position_modules(config)

        self.apply(self._init_weights)
        if config.connection in ("hc", "mhc"):
            for block in self.blocks:
                block.attn_conn.reset_dynamic_parameters()
                block.ffn_conn.reset_dynamic_parameters()

    def muon_auxiliary_modules(self):
        modules = [self.tok_emb, self.lm_head]
        if self.embed_tokens_per_layer is not None:
            modules.append(self.embed_tokens_per_layer)
        if self.pos_enc is not None and any(p.requires_grad for p in self.pos_enc.parameters()):
            modules.append(self.pos_enc)
        if self.config.connection != "residual":
            for block in self.blocks:
                modules.extend((block.attn_conn, block.ffn_conn))
        return tuple(modules)

    def set_qk_clip_recording(self, enabled):
        for block in self.blocks:
            if block.attention_name in QK_CLIP_ATTENTIONS:
                block.attn.set_qk_clip_recording(enabled)
        for module in self.mtp_modules:
            block = module.block
            if block.attention_name in QK_CLIP_ATTENTIONS:
                block.attn.set_qk_clip_recording(enabled)

    def auxiliary_loss(self):
        return self._ffn_auxiliary_loss()

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        if self.config.ffn == "aux_free_moe":
            for block in self.blocks:
                block.ffn.commit_routing_bias_update()
            for module in self.mtp_modules:
                module.block.ffn.commit_routing_bias_update()
        if qk_clip_threshold <= 0:
            return
        for block in self.blocks:
            if block.attention_name in QK_CLIP_ATTENTIONS:
                block.attn.commit_qk_clip_update(qk_clip_threshold, qk_clip_balance)
        for module in self.mtp_modules:
            block = module.block
            if block.attention_name in QK_CLIP_ATTENTIONS:
                block.attn.commit_qk_clip_update(qk_clip_threshold, qk_clip_balance)

    def forward(self, idx, targets=None):
        logits, _, main_hidden = self.forward_hidden(idx, return_residual=True)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            loss = loss + self._mtp_loss(main_hidden, idx, targets)
            loss = loss + self._ffn_auxiliary_loss()

        return logits, loss

    def forward_hidden(self, idx, return_residual=False):
        _, T = idx.shape
        require(T <= self.config.max_seq_len, (
            f"GPT supports at most {self.config.max_seq_len} tokens, got {T}"
        ))
        x = self._cast_hidden(self.tok_emb(idx))
        per_layer_inputs = self._per_layer_inputs(idx, x)

        freqs_cis, attn_bias, is_causal = self._position_inputs(T)
        if self.pos_enc is not None and self.pos_enc.kind == "additive":
            x = x + self._cast_hidden(self.pos_enc(T))

        x = self.drop(x)

        if self.config.connection != "residual":
            x = expand_residual_stream(x, self.config.connection_expansion)

        for block in self.blocks:
            block_freqs = self._block_freqs(block, T, freqs_cis)
            per_layer_input = per_layer_inputs[:, :, block.block_id, :] if per_layer_inputs is not None else None
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, block_freqs, attn_bias, is_causal, per_layer_input, use_reentrant=False
                )
            else:
                x = block(
                    x,
                    freqs_cis=block_freqs,
                    attn_bias=attn_bias,
                    is_causal=is_causal,
                    per_layer_input=per_layer_input,
                )

        if self.config.connection != "residual":
            x = reduce_residual_stream(x)

        main_hidden = x
        x = self.ln_f(main_hidden)
        logits = self.lm_head(x)
        if self.config.final_logit_softcap > 0:
            logits = torch.tanh(logits / self.config.final_logit_softcap) * self.config.final_logit_softcap

        if return_residual:
            return logits, x, main_hidden
        return logits, x

    def _position_inputs(self, seq_len):
        freqs_cis, attn_bias, is_causal = None, None, True
        if self.pos_enc is not None:
            if self.pos_enc.kind == "rotary":
                freqs_cis = self.pos_enc(seq_len)
            elif self.pos_enc.kind == "bias":
                attn_bias = self.pos_enc(seq_len).unsqueeze(0)
                is_causal = False
            else:
                require(self.pos_enc.kind in {"additive", "none"}, f"Unknown position kind: {self.pos_enc.kind}")
        return freqs_cis, attn_bias, is_causal

    def _block_freqs(self, block, seq_len, default_freqs=None):
        if block.attention_name == "gated_deltanet":
            return None
        if self.config.position in {"gemma3_rope", "gemma4_rope", "qwen3_next_rope"}:
            if _is_global_attention_name(block.attention_name):
                return self.global_pos_enc(seq_len)
            return self.local_pos_enc(seq_len)
        return default_freqs

    def _per_layer_inputs(self, idx, inputs_embeds):
        if self.config.per_layer_embedding_dim == 0:
            return None
        B, T = idx.shape
        D = self.config.per_layer_embedding_dim
        token_inputs = self.embed_tokens_per_layer(idx).reshape(B, T, self.config.num_layers, D)
        token_inputs = token_inputs * math.sqrt(D)
        projected = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        projected = projected.reshape(B, T, self.config.num_layers, D)
        projected = self.per_layer_projection_norm(projected)
        return (projected + token_inputs) * self.per_layer_input_scale

    def _mtp_position_inputs(self, block, seq_len):
        freqs_cis, attn_bias, is_causal = self._position_inputs(seq_len)
        return self._block_freqs(block, seq_len, freqs_cis), attn_bias, is_causal

    def _mtp_logits(self, hidden):
        logits = self.lm_head(self.ln_f(hidden))
        if self.config.final_logit_softcap > 0:
            logits = torch.tanh(logits / self.config.final_logit_softcap) * self.config.final_logit_softcap
        return logits

    def _mtp_loss(self, main_hidden, idx, targets):
        if self.config.mtp_depth == 0 or self.config.mtp_loss_weight == 0:
            return main_hidden.sum() * 0.0
        mtp_losses = []
        mtp_aux = main_hidden.sum() * 0.0
        hidden = main_hidden
        for depth, module in enumerate(self.mtp_modules, start=1):
            if idx.size(1) <= depth:
                break
            current_hidden = hidden[:, : idx.size(1) - depth]
            future_emb = self._cast_hidden(self.tok_emb(idx[:, depth:]))
            seq_len = current_hidden.size(1)
            freqs_cis, attn_bias, is_causal = self._mtp_position_inputs(module.block, seq_len)
            hidden = module(current_hidden, future_emb, freqs_cis=freqs_cis, attn_bias=attn_bias, is_causal=is_causal)
            mtp_logits = self._mtp_logits(hidden)
            mtp_target = targets[:, depth:]
            mtp_losses.append(F.cross_entropy(
                mtp_logits.reshape(-1, mtp_logits.size(-1)),
                mtp_target.reshape(-1),
                ignore_index=-100,
            ))
            if self.config.ffn in MOE_FFNS:
                mtp_aux = mtp_aux + module.block.ffn.aux_loss
        if not mtp_losses:
            return main_hidden.sum() * 0.0
        return self.config.mtp_loss_weight * torch.stack(mtp_losses).mean() + mtp_aux

    def _ffn_auxiliary_loss(self):
        if self.config.ffn not in MOE_FFNS:
            return next(self.parameters()).sum() * 0.0
        loss = next(self.parameters()).sum() * 0.0
        for block in self.blocks:
            loss = loss + block.ffn.aux_loss
        return loss


def _is_global_attention_name(attention_name):
    return attention_name in {
        "gqa_qknorm",
        "gqa_qknorm_partial_rope",
        "gqa_qknorm_kv_tied",
        "gated_gqa_qknorm",
        "gated_gqa_qknorm_partial_rope",
    }
