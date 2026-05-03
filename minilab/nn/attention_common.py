import math

import torch

from minilab.checks import require


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    """q,k: (B,H,T,D), cos,sin: (T,D/2)."""
    cos = cos[None, None]
    sin = sin[None, None]
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    q_cos = cos.to(device=q.device, dtype=q.dtype)
    q_sin = sin.to(device=q.device, dtype=q.dtype)
    k_cos = cos.to(device=k.device, dtype=k.dtype)
    k_sin = sin.to(device=k.device, dtype=k.dtype)
    return (
        q * q_cos + rotate_half(q) * q_sin,
        k * k_cos + rotate_half(k) * k_sin,
    )


class _QKClipMixin:
    def _init_qk_clip(self, num_heads):
        self.register_buffer("_qk_clip_max_logits", torch.zeros(num_heads), persistent=False)
        self.register_buffer("_qk_clip_seen", torch.zeros((), dtype=torch.bool), persistent=False)
        self._qk_clip_recording = False

    @torch.no_grad()
    def set_qk_clip_recording(self, enabled):
        self._qk_clip_recording = enabled
        if not enabled:
            self._reset_qk_clip_stats()

    def _record_qk_clip_logits(self, q, k, attn_bias=None, is_causal=False, past_len=0):
        if not self._qk_clip_recording or not torch.is_grad_enabled():
            return
        with torch.no_grad():
            scores = torch.matmul(q.detach().float(), k.detach().float().transpose(-2, -1)) / math.sqrt(q.size(-1))
            scores = _mask_qk_scores_to_attention_support(scores, attn_bias, is_causal, past_len)
            max_logits = scores.amax(dim=(0, 2, 3)).to(self._qk_clip_max_logits.dtype)
            require(torch.isfinite(max_logits).all(), (
                "QK-Clip recording saw no finite attention logits for at least one head"
            ))
            current = torch.where(
                self._qk_clip_seen,
                torch.maximum(self._qk_clip_max_logits, max_logits),
                max_logits,
            )
            self._qk_clip_max_logits.copy_(current)
            self._qk_clip_seen.fill_(True)

    @torch.no_grad()
    def _qk_clip_gammas(self, threshold):
        require(threshold > 0, "qk_clip threshold must be > 0")
        if not bool(self._qk_clip_seen.item()):
            return None
        max_logits = self._qk_clip_max_logits.float().clamp(min=torch.finfo(torch.float32).tiny)
        return torch.minimum(torch.ones_like(max_logits), threshold / max_logits)

    @torch.no_grad()
    def _reset_qk_clip_stats(self):
        self._qk_clip_max_logits.zero_()
        self._qk_clip_seen.zero_()


class _QKNormClipMixin:
    @torch.no_grad()
    def commit_qk_clip_update(self, threshold, balance=0.5):
        require(0.0 <= balance <= 1.0, "qk_clip balance must be in [0, 1]")
        gammas = self._qk_clip_gammas(threshold)
        if gammas is None:
            return
        gamma = gammas.amin()
        self.q_norm.weight.mul_(gamma.pow(balance).to(self.q_norm.weight.device, self.q_norm.weight.dtype))
        self.k_norm.weight.mul_(gamma.pow(1.0 - balance).to(self.k_norm.weight.device, self.k_norm.weight.dtype))
        self._reset_qk_clip_stats()


def _scale_linear_heads(linear, num_heads, head_dim, scales):
    weight = linear.weight.view(num_heads, head_dim, linear.weight.size(1))
    weight.mul_(scales.to(device=weight.device, dtype=weight.dtype).view(num_heads, 1, 1))


def _mask_qk_scores_to_attention_support(scores, attn_bias, is_causal, past_len):
    if attn_bias is not None:
        support = _attention_support_from_bias(
            attn_bias.to(device=scores.device),
            scores.size(0),
            scores.size(1),
            scores.size(2),
            scores.size(3),
        )
        return scores.masked_fill(~support, float("-inf"))
    if is_causal:
        q_len = scores.size(2)
        kv_len = scores.size(3)
        query_pos = torch.arange(past_len, past_len + q_len, device=scores.device).view(q_len, 1)
        key_pos = torch.arange(kv_len, device=scores.device).view(1, kv_len)
        support = key_pos <= query_pos
        return scores.masked_fill(~support.view(1, 1, q_len, kv_len), float("-inf"))
    return scores


def _attention_support_from_bias(attn_bias, batch_size, num_heads, q_len, kv_len):
    require(attn_bias.size(-2) == q_len and attn_bias.size(-1) == kv_len, (
        f"attn_bias must end with shape ({q_len}, {kv_len}) for QK-Clip recording"
    ))
    support = attn_bias if attn_bias.dtype == torch.bool else torch.isfinite(attn_bias)
    if attn_bias.dim() == 2:
        return support.view(1, 1, q_len, kv_len)
    if attn_bias.dim() == 3:
        if attn_bias.size(0) == num_heads:
            return support.view(1, num_heads, q_len, kv_len)
        require(attn_bias.size(0) == batch_size, (
            "3D attn_bias must be shaped as (num_heads, q_len, kv_len) "
            "or (batch, q_len, kv_len) for QK-Clip recording"
        ))
        return support.view(batch_size, 1, q_len, kv_len)
    if attn_bias.dim() == 4:
        require(attn_bias.size(0) in {1, batch_size}, (
            "4D attn_bias batch dimension must be 1 or batch size for QK-Clip recording"
        ))
        require(attn_bias.size(1) in {1, num_heads}, (
            "4D attn_bias head dimension must be 1 or num_heads for QK-Clip recording"
        ))
        return support
    raise ValueError("attn_bias must have 2, 3, or 4 dimensions for QK-Clip recording")



def _local_attention_bias(T, window_size, device, dtype, is_causal):
    idx = torch.arange(T, device=device)
    delta = idx[:, None] - idx[None, :]
    if is_causal:
        allowed = (delta >= 0) & (delta < window_size)
    else:
        allowed = delta.abs() < window_size
    return _bool_to_additive_bias(allowed, dtype)


def _bool_to_additive_bias(allowed, dtype):
    bias = torch.zeros(allowed.shape, device=allowed.device, dtype=dtype)
    return bias.masked_fill(~allowed, float("-inf"))


def _merge_attention_bias(base_bias, extra_bias):
    if extra_bias is None:
        return base_bias
    if extra_bias.dtype == torch.bool:
        extra_bias = _bool_to_additive_bias(extra_bias.to(base_bias.device), base_bias.dtype)
    return base_bias + extra_bias


def _apply_tail_rotary(x, freqs_cis, positions, rope_dim=64, inverse=False):
    cos, sin = freqs_cis
    dim = min(x.size(-1), rope_dim, cos.size(-1) * 2)
    if dim % 2 == 1:
        dim -= 1
    if dim <= 0:
        return x
    half = dim // 2
    positions = positions.to(device=cos.device, dtype=torch.long)
    cos = cos.index_select(0, positions)[:, :half].to(device=x.device, dtype=x.dtype)
    sin = sin.index_select(0, positions)[:, :half].to(device=x.device, dtype=x.dtype)
    if inverse:
        sin = -sin
    view_shape = (1,) * (x.dim() - 2) + (positions.numel(), dim)
    cos = torch.cat([cos, cos], dim=-1).view(view_shape)
    sin = torch.cat([sin, sin], dim=-1).view(view_shape)
    plain, rotary = x[..., :-dim], x[..., -dim:]
    rotated = rotary * cos + rotate_half(rotary) * sin
    return torch.cat([plain, rotated], dim=-1)
