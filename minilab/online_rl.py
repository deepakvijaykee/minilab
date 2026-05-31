import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from minilab.alignment_common import (
    ReferenceCheckpointMixin,
    _generation_context_token_logp,
    _group_normalized_advantages,
    _load_reference_model,
    _masked_response_mean,
    _model_max_seq_len,
    _rollout_policy_train_loop,
    _trainer_reference_path,
    _validate_reference_tokenizer,
    _whiten_masked,
)
from minilab.base import BaseModel, unwrap_model
from minilab.checks import require, require_finite_number, require_integer_fields
from minilab.generation import generate
from minilab.registry import register_trainer
from minilab.rl_diagnostics import (
    append_jsonl,
    group_stats,
    ppo_stats,
    rollout_records,
    rollout_system_stats,
    scalar_record,
    stack_reward_results,
    split_reward_result,
    vpo_rollout_records,
)
from minilab.trainer import (
    TrainConfig,
    Trainer,
    commit_post_optimizer_updates,
    model_aux_loss,
    optimizer_decay_groups,
    set_seed,
)


ONLINE_RL_REFERENCE_ALGORITHMS = frozenset({"ppo", "grpo", "drgrpo", "gspo", "rloo"})


def online_rl_uses_reference(algorithm):
    return algorithm in ONLINE_RL_REFERENCE_ALGORITHMS


@dataclass
class PPOTrainConfig(TrainConfig):
    ppo_max_new_tokens: int = 128
    ppo_clip_ratio: float = 0.2
    ppo_value_clip: float = 0.2
    ppo_kl_coef: float = 0.1
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.0
    ppo_gamma: float = 1.0
    ppo_lam: float = 0.95
    ppo_inner_epochs: int = 4
    ppo_whiten_rewards: bool = True

    def __post_init__(self):
        super().__post_init__()
        require_integer_fields(self, ("ppo_max_new_tokens", "ppo_inner_epochs"))
        for name in (
            "ppo_clip_ratio", "ppo_value_clip", "ppo_kl_coef", "ppo_value_coef",
            "ppo_entropy_coef", "ppo_gamma", "ppo_lam",
        ):
            require_finite_number(getattr(self, name), name)
        require(self.ppo_max_new_tokens > 0, "ppo_max_new_tokens must be > 0")
        require(self.ppo_clip_ratio > 0, "ppo_clip_ratio must be > 0")
        require(self.ppo_value_clip > 0, "ppo_value_clip must be > 0")
        require(self.ppo_kl_coef >= 0, "ppo_kl_coef must be >= 0")
        require(self.ppo_value_coef >= 0, "ppo_value_coef must be >= 0")
        require(self.ppo_entropy_coef >= 0, "ppo_entropy_coef must be >= 0")
        require(0 < self.ppo_gamma <= 1, "ppo_gamma must be in (0, 1]")
        require(0 < self.ppo_lam <= 1, "ppo_lam must be in (0, 1]")
        require(self.ppo_inner_epochs >= 1, "ppo_inner_epochs must be >= 1")


@dataclass
class GRPOTrainConfig(TrainConfig):
    grpo_num_generations: int = 4
    grpo_max_new_tokens: int = 128
    grpo_clip_ratio: float = 0.2
    grpo_kl_coef: float = 0.1
    grpo_inner_epochs: int = 4

    def __post_init__(self):
        super().__post_init__()
        require_integer_fields(self, (
            "grpo_num_generations", "grpo_max_new_tokens", "grpo_inner_epochs",
        ))
        require_finite_number(self.grpo_clip_ratio, "grpo_clip_ratio")
        require_finite_number(self.grpo_kl_coef, "grpo_kl_coef")
        require(self.grpo_num_generations > 1, "GRPO requires grpo_num_generations > 1")
        require(self.grpo_max_new_tokens > 0, "grpo_max_new_tokens must be > 0")
        require(self.grpo_clip_ratio > 0, "grpo_clip_ratio must be > 0")
        require(self.grpo_kl_coef >= 0, "grpo_kl_coef must be >= 0")
        require(self.grpo_inner_epochs >= 1, "grpo_inner_epochs must be >= 1")


@dataclass
class DrGRPOTrainConfig(GRPOTrainConfig):
    """Dr.GRPO uses GRPO's loop knobs while changing advantage normalization."""


@dataclass
class TPOTrainConfig(GRPOTrainConfig):
    tpo_eta: float = 1.0
    tpo_anchor_old_policy: bool = True

    def __post_init__(self):
        super().__post_init__()
        require_finite_number(self.tpo_eta, "tpo_eta")
        require(self.grpo_clip_ratio == GRPOTrainConfig.grpo_clip_ratio, (
            "TPO fits a sampled target distribution and does not use GRPO clipping; "
            "leave grpo_clip_ratio at the inherited default"
        ))
        require(self.grpo_kl_coef == GRPOTrainConfig.grpo_kl_coef, (
            "TPO fits a sampled target distribution and does not use the GRPO KL penalty; "
            "leave grpo_kl_coef at the inherited default"
        ))
        require(self.tpo_eta > 0, "tpo_eta must be > 0")


@dataclass
class GroupPGTrainConfig(GRPOTrainConfig):
    def __post_init__(self):
        super().__post_init__()
        require(self.grpo_clip_ratio == GRPOTrainConfig.grpo_clip_ratio, (
            "GroupPG is an unclipped sequence policy-gradient ablation; "
            "leave grpo_clip_ratio at the inherited default"
        ))
        require(self.grpo_kl_coef == GRPOTrainConfig.grpo_kl_coef, (
            "GroupPG does not use the GRPO KL penalty; leave grpo_kl_coef at the inherited default"
        ))


@dataclass
class SequenceDGTrainConfig(GRPOTrainConfig):
    dg_eta: float = 1.0
    dg_keep_ratio: float = 0.5
    dg_uncertainty_threshold: float = 0.5
    dg_replay_capacity: int = 0
    dg_replay_min_age: int = 1
    dg_replay_age_decay: float = 0.0
    dg_staleness_delay: int = 0
    dg_drop_stale_after: int = 0

    def __post_init__(self):
        super().__post_init__()
        require_integer_fields(self, (
            "dg_replay_capacity", "dg_replay_min_age", "dg_staleness_delay",
            "dg_drop_stale_after",
        ))
        for name in (
            "dg_eta", "dg_keep_ratio", "dg_uncertainty_threshold",
            "dg_replay_age_decay",
        ):
            require_finite_number(getattr(self, name), name)
        require(self.grpo_clip_ratio == GRPOTrainConfig.grpo_clip_ratio, (
            "DG-family sequence objectives do not use GRPO token-level clipping; "
            "leave grpo_clip_ratio at the inherited default"
        ))
        require(self.dg_eta > 0, "dg_eta must be > 0")
        require(0 < self.dg_keep_ratio <= 1, "dg_keep_ratio must be in (0, 1]")
        require(self.dg_uncertainty_threshold >= 0, "dg_uncertainty_threshold must be >= 0")
        require(self.dg_replay_capacity >= 0, "dg_replay_capacity must be >= 0")
        require(self.dg_replay_min_age >= 0, "dg_replay_min_age must be >= 0")
        require(self.dg_replay_age_decay >= 0, "dg_replay_age_decay must be >= 0")
        require(self.dg_staleness_delay >= 0, "dg_staleness_delay must be >= 0")
        require(self.dg_drop_stale_after >= 0, "dg_drop_stale_after must be >= 0")
        require(
            self.dg_drop_stale_after == 0 or self.dg_drop_stale_after >= self.dg_staleness_delay,
            "dg_drop_stale_after must be 0 or >= dg_staleness_delay",
        )


@dataclass
class VPOTrainConfig(GRPOTrainConfig):
    vpo_num_candidates: int = 3
    vpo_num_scalarizations: int = 8
    vpo_dirichlet_alpha: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        require_integer_fields(self, ("vpo_num_candidates", "vpo_num_scalarizations"))
        require_finite_number(self.vpo_dirichlet_alpha, "vpo_dirichlet_alpha")
        require(self.grpo_kl_coef == GRPOTrainConfig.grpo_kl_coef, (
            "VPO uses set-level clipped policy updates and does not use the GRPO KL penalty; "
            "leave grpo_kl_coef at the inherited default"
        ))
        require(self.vpo_num_candidates > 1, "VPO requires vpo_num_candidates > 1")
        require(self.vpo_num_scalarizations > 0, "vpo_num_scalarizations must be > 0")
        require(self.vpo_dirichlet_alpha > 0, "vpo_dirichlet_alpha must be > 0")


@dataclass
class DAPOTrainConfig(GRPOTrainConfig):
    grpo_kl_coef: float = 0.0
    dapo_clip_ratio_low: float = 0.2
    dapo_clip_ratio_high: float = 0.28
    dapo_safe_length: int = 0
    dapo_length_penalty: float = 0.0
    dapo_max_resample: int = 5

    def __post_init__(self):
        super().__post_init__()
        require_integer_fields(self, ("dapo_safe_length", "dapo_max_resample"))
        require_finite_number(self.dapo_clip_ratio_low, "dapo_clip_ratio_low")
        require_finite_number(self.dapo_clip_ratio_high, "dapo_clip_ratio_high")
        require_finite_number(self.dapo_length_penalty, "dapo_length_penalty")
        require(self.grpo_kl_coef == 0, "DAPO removes the KL penalty; set grpo_kl_coef=0")
        require(self.grpo_clip_ratio == GRPOTrainConfig.grpo_clip_ratio, (
            "DAPO uses dapo_clip_ratio_low/high; leave grpo_clip_ratio at the inherited default"
        ))
        require(self.dapo_clip_ratio_low > 0, "dapo_clip_ratio_low must be > 0")
        require(self.dapo_clip_ratio_high > 0, "dapo_clip_ratio_high must be > 0")
        require(self.dapo_clip_ratio_high >= self.dapo_clip_ratio_low, (
            "DAPO Clip-Higher requires dapo_clip_ratio_high >= dapo_clip_ratio_low"
        ))
        require(self.dapo_safe_length >= 0, "dapo_safe_length must be >= 0")
        require(self.dapo_length_penalty >= 0, "dapo_length_penalty must be >= 0")
        require(self.dapo_max_resample >= 1, "dapo_max_resample must be >= 1")


@dataclass
class RLOOTrainConfig(GRPOTrainConfig):
    grpo_inner_epochs: int = 1

    def __post_init__(self):
        super().__post_init__()
        require(self.grpo_inner_epochs == 1, "RLOO is an on-policy REINFORCE estimator; set grpo_inner_epochs=1")
        require(self.grpo_clip_ratio == GRPOTrainConfig.grpo_clip_ratio, (
            "RLOO is unclipped REINFORCE leave-one-out; leave grpo_clip_ratio at the inherited default"
        ))


@dataclass
class GSPOTrainConfig(GRPOTrainConfig):
    grpo_clip_ratio: float = 4e-4


class PPOValueHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden):
        return self.proj(hidden).squeeze(-1)


@dataclass
class PPORollout:
    seqs: torch.Tensor
    labels: torch.Tensor
    completions: torch.Tensor
    completion_mask: torch.Tensor
    rewards: torch.Tensor
    reward_components: dict | None
    old_logp: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    mask: torch.Tensor
    generation_latency_ms: float = 0.0
    reward_latency_ms: float = 0.0
    generated_tokens: int = 0
    reward_calls: int = 0
    queue_depth: int = 0
    dropped_stale_count: int = 0
    staleness_delay: int = 0
    drop_stale_after: int = 0
    age: int = 0
    accepted_for_training: bool = True
    drop_reason: str = ""


@dataclass
class GroupRollout:
    seqs: list
    label_seqs: list
    old_token_logps: list
    completions: list
    completion_masks: list
    rewards: torch.Tensor
    adv: torch.Tensor
    reward_components: dict | None = None
    age: int = 0
    generation_latency_ms: float = 0.0
    reward_latency_ms: float = 0.0
    generated_tokens: int = 0
    reward_calls: int = 0
    queue_depth: int = 0
    dropped_stale_count: int = 0
    staleness_delay: int = 0
    drop_stale_after: int = 0
    accepted_for_training: bool = True
    drop_reason: str = ""


@dataclass
class VPORollout:
    seqs: list
    label_seqs: list
    old_token_logps: list
    completions: list
    completion_masks: list
    rewards: torch.Tensor
    adv: torch.Tensor
    candidate_rewards: torch.Tensor
    reward_components: dict
    generation_latency_ms: float = 0.0
    reward_latency_ms: float = 0.0
    generated_tokens: int = 0
    reward_calls: int = 0
    queue_depth: int = 0
    dropped_stale_count: int = 0
    staleness_delay: int = 0
    drop_stale_after: int = 0
    age: int = 0
    accepted_for_training: bool = True
    drop_reason: str = ""


def _sample_completion(model, prompt, max_new_tokens, context):
    require(max_new_tokens > 0, f"{context} requires room for at least one generated token")
    out = generate(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    gen = out[0, prompt.size(1):]
    require(gen.numel() > 0, f"{context} requires at least one generated token per prompt")
    return gen


def _pack_prompt_completions(prompt_ids, prompt_lens, completions, device):
    require(prompt_ids.dim() == 2, "prompt_ids must have shape (batch, seq)")
    B = prompt_ids.size(0)
    require(B > 0, "prompt_ids must contain at least one prompt")
    require(prompt_lens.shape == (B,), "prompt_lens must have shape (batch,)")
    require(prompt_lens.dtype == torch.long, "prompt_lens must use dtype torch.long")
    require(len(completions) == B, "completions must match prompt batch size")
    comp_lens = [gen.size(0) for gen in completions]
    max_clen = max(comp_lens)
    completion_pad = torch.zeros(B, max_clen, device=device, dtype=torch.long)
    completion_mask = torch.zeros(B, max_clen, device=device, dtype=torch.bool)
    seqs, labels = [], []

    for b, gen in enumerate(completions):
        require(gen.dim() == 1, "completions must be 1D token tensors")
        require(gen.dtype == torch.long, "completions must use dtype torch.long")
        plen = int(prompt_lens[b].item())
        require(0 < plen <= prompt_ids.size(1), "prompt_lens must be in [1, prompt sequence length]")
        clen = gen.size(0)
        require(clen > 0, "completions must contain at least one token")
        completion_pad[b, :clen] = gen
        completion_mask[b, :clen] = True
        full = torch.cat([prompt_ids[b, :plen], gen])
        lab = torch.full_like(full, -100)
        lab[plen - 1 : -1] = full[plen:]
        seqs.append(full)
        labels.append(lab)

    max_len = max(seq.size(0) for seq in seqs)
    seq_pad = torch.zeros(B, max_len, device=device, dtype=torch.long)
    label_pad = torch.full((B, max_len), -100, device=device, dtype=torch.long)
    for b in range(B):
        seq_pad[b, : seqs[b].size(0)] = seqs[b]
        label_pad[b, : labels[b].size(0)] = labels[b]
    return completion_pad, completion_mask, seq_pad, label_pad, comp_lens


def _group_centered_advantages(rewards):
    return rewards - rewards.mean(dim=1, keepdim=True)


def _tpo_skill(rewards):
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    std = centered.std(dim=1, unbiased=False, keepdim=True)
    return torch.where(std > 1e-6, centered / std.clamp(min=1e-6), centered)


def _safe_sequence_scores(token_logp, labels):
    mask = (labels != -100).float()
    require(mask.sum(dim=-1).gt(0).all(), "sequence policy scores require at least one supervised token")
    return (token_logp * mask).sum(dim=-1), mask


def _group_sequence_scores(model, rollout, device, return_aux=True):
    scores = []
    masks = []
    aux_total = torch.tensor(0.0, device=device)
    for seq, labels in zip(rollout.seqs, rollout.label_seqs, strict=True):
        if return_aux:
            logp, mask, aux = _generation_context_token_logp(model, seq, labels, return_aux=True)
            aux_total = aux_total + aux
        else:
            logp, mask = _generation_context_token_logp(model, seq, labels)
        score, _ = _safe_sequence_scores(logp, labels)
        scores.append(score)
        masks.append(mask)
    aux_mean = aux_total / max(1, len(scores))
    return torch.stack(scores, dim=1), masks, aux_mean


def _old_sequence_scores(rollout):
    return torch.stack([
        _safe_sequence_scores(old_logp, labels)[0]
        for old_logp, labels in zip(rollout.old_token_logps, rollout.label_seqs, strict=True)
    ], dim=1).detach()


def _vpo_set_rewards(candidate_rewards, components, num_scalarizations, dirichlet_alpha):
    require(candidate_rewards.dim() == 3, "VPO candidate rewards must have shape [batch, generations, candidates]")
    require(components, "VPO requires vector reward components; scalar-only rewards are not a VPO objective")
    require(num_scalarizations > 0, "VPO requires at least one scalarization")
    require(dirichlet_alpha > 0, "VPO Dirichlet alpha must be > 0")
    for key, value in components.items():
        require(value.shape == candidate_rewards.shape, (
            f"VPO reward component '{key}' must match candidate reward shape"
        ))
    vectors = torch.stack([value.float() for value in components.values()], dim=-1)
    dims = vectors.size(-1)
    concentration = torch.full((dims,), dirichlet_alpha, device=vectors.device, dtype=vectors.dtype)
    weights = torch.distributions.Dirichlet(concentration).sample(
        (vectors.size(0), num_scalarizations))
    scalarized = torch.einsum("bkcd,bsd->bkcs", vectors, weights)
    return scalarized.max(dim=2).values.mean(dim=-1)


@register_trainer("ppo")
class PPOTrainer(Trainer):
    """Actor-critic PPO for RLHF/RLVR.

    The reward function supplies one scalar terminal score per completion. The
    trainer adds the standard per-token KL shaping reward against a frozen
    reference policy, computes GAE with a learned value head, and optimizes the
    clipped PPO policy and value objectives over the generated response tokens.
    """

    _extra_critical_fields = (
        "ppo_max_new_tokens", "ppo_clip_ratio", "ppo_value_clip", "ppo_kl_coef",
        "ppo_value_coef", "ppo_entropy_coef", "ppo_gamma", "ppo_lam",
        "ppo_inner_epochs", "ppo_whiten_rewards",
    )

    def __init__(self, model, reward_fn, train_dataset, config, ref_model_path, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, PPOTrainConfig), "PPOTrainer requires PPOTrainConfig")
        require(config.optimizer == "adamw", "PPOTrainer uses AdamW for the policy/value-head optimizer")
        require(eval_dataset is None, "PPO has no LM-style eval loss; evaluate task reward post-training")
        require(config.eval_every == 0, "PPO has no LM-style eval loss; set eval_every=0")
        require(config.grad_accum_steps == 1, "PPO does not support grad_accum_steps > 1")
        model_core = unwrap_model(model)
        require(isinstance(model_core, BaseModel), "PPOTrainer requires a BaseModel policy")
        require(model_core.provides_hidden_states, "PPOTrainer requires a policy that declares hidden-state outputs")
        set_seed(config.seed)
        dim = model_core.config.dim
        self.value_head = PPOValueHead(dim)
        if config.resume_from:
            self._load_value_head_for_resume(config.resume_from)
        self.ref_model_path = _trainer_reference_path(ref_model_path, config, "PPO")
        _validate_reference_tokenizer(self.ref_model_path, tokenizer_sig, "PPO")
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.ref_model = _load_reference_model(self.model, self.ref_model_path, self.device, "PPO")
        self.reward_fn = reward_fn
        self.max_new_tokens = config.ppo_max_new_tokens
        self.clip_ratio = config.ppo_clip_ratio
        self.value_clip = config.ppo_value_clip
        self.kl_coef = config.ppo_kl_coef
        self.value_coef = config.ppo_value_coef
        self.entropy_coef = config.ppo_entropy_coef
        self.gamma = config.ppo_gamma
        self.lam = config.ppo_lam
        self.inner_epochs = config.ppo_inner_epochs
        self.whiten_rewards = config.ppo_whiten_rewards

    def _load_value_head_for_resume(self, resume_from):
        value_path = Path(resume_from) / "value_head.pt"
        require(value_path.exists(), f"PPO resume is missing {value_path}")
        self.value_head.load_state_dict(torch.load(value_path, map_location="cpu", weights_only=True))

    def _configured_scheduler_total_steps(self):
        return self.config.max_steps * self.config.ppo_inner_epochs

    def _build_optimizer(self):
        model = unwrap_model(self.model)
        self.value_head = self.value_head.to(self.device)
        params = list(self.model.parameters()) + list(self.value_head.parameters())
        groups = optimizer_decay_groups(model, params, self.config.weight_decay)
        return torch.optim.AdamW(groups, lr=self.config.lr, betas=(0.9, 0.95))

    def _optimizer_update(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        stepped = not self.scaler.is_enabled() or self.scaler.get_scale() >= old_scale
        if not stepped:
            raise FloatingPointError("AMP skipped the optimizer update; stopping before training step accounting advances")
        commit_post_optimizer_updates(self.model, self.config.qk_clip_threshold, self.config.qk_clip_balance)
        self.scheduler.step()
        return True

    def save_checkpoint(self):
        super().save_checkpoint()
        path = Path(self.config.save_dir) / f"step_{self.step}"
        (path / "ref_path.txt").write_text(self.ref_model_path)
        torch.save(self.value_head.state_dict(), path / "value_head.pt")

    def compute_loss(self, batch):
        raise NotImplementedError("PPO runs its own train loop; compute_loss is not called")

    def train(self):
        def loop():
            self.model.eval()
            self.value_head.eval()
            pbar = tqdm(range(self.step + 1, self.config.max_steps + 1), desc="Training")
            for self.step in pbar:
                batch = self._next_batch()
                rollout = self._rollout(batch)
                total_loss = 0.0
                for _ in range(self.inner_epochs):
                    self.model.eval()
                    self.value_head.train()
                    with torch.autocast(self.device, dtype=self.dtype, enabled=self.dtype != torch.float32):
                        loss = self._policy_loss(rollout)
                    self.scaler.scale(loss).backward()
                    self._optimizer_update()
                    total_loss += loss.item()

                lr = self.scheduler.get_last_lr()[0]
                if self.step % self.config.log_every == 0:
                    avg_loss = total_loss / self.inner_epochs
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                    if self.aim_run:
                        self.aim_run.track(avg_loss, name="loss", step=self.step)
                        self.aim_run.track(lr, name="lr", step=self.step)
                self._record_online_rollout(batch, rollout)
                self._save_checkpoint_if_due()
                self.model.eval()
                self.value_head.eval()

            self._save_final_checkpoint()

        self._run_train_loop_with_metrics(loop)

    def _rollout(self, batch):
        prompt_ids = batch["prompt_ids"]
        prompt_lens = batch["prompt_len"]
        require((prompt_lens > 0).all(), "PPO requires prompt_len > 0 for every prompt")
        max_ctx = _model_max_seq_len(self.model, "PPO rollout")
        B = prompt_ids.size(0)

        gens = []
        was_training = self.model.training
        value_was_training = self.value_head.training
        self.model.eval()
        self.value_head.eval()
        try:
            generation_started = time.perf_counter()
            with torch.no_grad():
                for b in range(B):
                    plen = int(prompt_lens[b].item())
                    new_tokens = min(self.max_new_tokens, max_ctx - plen)
                    gens.append(_sample_completion(
                        self.model,
                        prompt_ids[b : b + 1, :plen],
                        new_tokens,
                        "PPO",
                    ))
            generation_latency_ms = (time.perf_counter() - generation_started) * 1000.0

            completions, completion_mask, seq_pad, label_pad, _ = _pack_prompt_completions(
                prompt_ids,
                prompt_lens,
                gens,
                self.device,
            )

            reward_started = time.perf_counter()
            rewards, reward_components = split_reward_result(self.reward_fn(batch, completions, completion_mask))
            reward_latency_ms = (time.perf_counter() - reward_started) * 1000.0
            rewards = rewards.to(self.device)
            reward_components = {key: value.to(self.device) for key, value in reward_components.items()}
            with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype, enabled=self.dtype != torch.float32):
                old_logp, old_values, mask, _ = self._token_logp_values_entropy(self.model, seq_pad, label_pad)
                ref_logp, _ = _generation_context_token_logp(self.ref_model, seq_pad, label_pad)

            shaped = -self.kl_coef * (old_logp - ref_logp) * mask
            last_idx = mask.long().sum(dim=1) - 1
            active_positions = mask.bool().float().cumsum(dim=1) - 1
            for b in range(B):
                terminal = (active_positions[b].eq(last_idx[b]) & mask[b].bool()).nonzero(as_tuple=False).flatten()
                require(terminal.numel() == 1, "PPO terminal reward needs exactly one final response token")
                shaped[b, terminal.item()] = shaped[b, terminal.item()] + rewards[b]

            advantages, returns = self._gae(shaped, old_values, mask)
            if self.whiten_rewards:
                advantages = _whiten_masked(advantages, mask)

            return PPORollout(
                seqs=seq_pad,
                labels=label_pad,
                completions=completions,
                completion_mask=completion_mask,
                rewards=rewards.detach(),
                reward_components={key: value.detach() for key, value in reward_components.items()},
                old_logp=old_logp.detach(),
                old_values=old_values.detach(),
                advantages=advantages.detach(),
                returns=returns.detach(),
                mask=mask.detach(),
                generation_latency_ms=generation_latency_ms,
                reward_latency_ms=reward_latency_ms,
                generated_tokens=int(completion_mask.sum().item()),
                reward_calls=1,
            )
        finally:
            self.model.train(was_training)
            self.value_head.train(value_was_training)

    def _record_online_rollout(self, batch, rollout):
        if self.config.rl_metrics_every > 0 and self.step % self.config.rl_metrics_every == 0:
            metrics = ppo_stats(rollout.rewards, rollout.advantages, rollout.mask)
            metrics.update(rollout_system_stats(rollout))
            for key, value in (rollout.reward_components or {}).items():
                metrics[f"reward_component_{key}_mean"] = float(value.float().mean().item())
            append_jsonl(
                Path(self.config.save_dir) / "online_rl_metrics.jsonl",
                [scalar_record(type(self).__name__, self.step, "ppo", metrics)],
            )

    def _policy_loss(self, rollout):
        logp, values, mask, entropy = self._token_logp_values_entropy(self.model, rollout.seqs, rollout.labels)
        aux = model_aux_loss(self.model)
        ratio = (logp - rollout.old_logp).clamp(min=-20.0, max=20.0).exp()
        adv = rollout.advantages
        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        policy_loss = -_masked_response_mean(torch.min(surr1, surr2), mask, "PPO policy loss")

        value_clipped = rollout.old_values + (values - rollout.old_values).clamp(-self.value_clip, self.value_clip)
        value_loss = torch.max(
            (values - rollout.returns).square(),
            (value_clipped - rollout.returns).square(),
        )
        value_loss = 0.5 * _masked_response_mean(value_loss, mask, "PPO value loss")
        entropy_bonus = _masked_response_mean(entropy, mask, "PPO entropy")
        return policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus + aux

    def _token_logp_values_entropy(self, model, input_ids, labels):
        logits, hidden = unwrap_model(model).forward_hidden(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        mask = (labels != -100).float()
        safe_targets = labels.where(mask.bool(), torch.zeros_like(labels))
        token_logp = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        values = self.value_head(hidden)
        return token_logp, values, mask, entropy

    def _gae(self, rewards, values, mask):
        advantages = torch.zeros_like(rewards)
        last_adv = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)
        for t in range(rewards.size(1) - 1, -1, -1):
            active = mask[:, t]
            if t + 1 < rewards.size(1):
                next_value = values[:, t + 1]
                next_active = mask[:, t + 1]
            else:
                next_value = torch.zeros_like(last_adv)
                next_active = torch.zeros_like(last_adv)
            delta = rewards[:, t] + self.gamma * next_value * next_active - values[:, t]
            last_adv = delta + self.gamma * self.lam * next_active * last_adv
            last_adv = torch.where(active > 0, last_adv, torch.zeros_like(last_adv))
            advantages[:, t] = last_adv
        returns = advantages + values
        return advantages, returns


@register_trainer("grpo")
class GRPOTrainer(ReferenceCheckpointMixin, Trainer):
    """Per step: sample K completions per prompt under the current (old) policy,
    freeze their log-probs, then run grpo_inner_epochs clipped-surrogate updates
    against those frozen old log-probs. This is what makes the PPO-style ratio
    clip actually active: without inner epochs, old_logp == logp exactly on the
    first pass and the clip is a no-op."""

    _extra_critical_fields = (
        "grpo_num_generations", "grpo_max_new_tokens",
        "grpo_clip_ratio", "grpo_kl_coef", "grpo_inner_epochs",
    )
    _uses_reference_model = True

    def __init__(self, model, reward_fn, train_dataset, config, ref_model_path=None, *, signature, tokenizer_sig="", eval_dataset=None):
        # PromptDataset yields prompts, not (input, label) pairs — so the generic
        # LM-style eval loss is undefined here. The paper metric is held-out task
        # accuracy, which scripts/grpo.py runs post-training. Reject eval hooks
        # explicitly so this can't silently call compute_loss during training.
        require(isinstance(config, GRPOTrainConfig), "GRPOTrainer requires GRPOTrainConfig")
        require(eval_dataset is None, "GRPO has no LM-style eval loss; evaluate accuracy post-training")
        require(config.eval_every == 0, "GRPO has no LM-style eval loss; set eval_every=0")
        # Gradient accumulation is not implemented in the GRPO inner-epoch loop.
        # Each inner epoch steps the optimizer; accumulating would reshape the
        # update schedule in a way that silently changes the algorithm.
        require(config.grad_accum_steps == 1, "GRPO does not support grad_accum_steps > 1")
        if self._uses_reference_model:
            self.ref_model_path = _trainer_reference_path(ref_model_path, config, self._algorithm_name().upper())
            _validate_reference_tokenizer(self.ref_model_path, tokenizer_sig, self._algorithm_name().upper())
        else:
            require(ref_model_path in {None, ""}, (
                f"{type(self).__name__} is reference-free; do not pass ref_model_path"
            ))
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        if self._uses_reference_model:
            self.ref_model = _load_reference_model(
                self.model,
                self.ref_model_path,
                self.device,
                self._algorithm_name().upper(),
            )
        self.reward_fn = reward_fn
        self.K = config.grpo_num_generations
        self.inner_epochs = config.grpo_inner_epochs
        self.max_new_tokens = config.grpo_max_new_tokens
        self.clip_ratio = config.grpo_clip_ratio
        self.kl_coef = config.grpo_kl_coef
        self._last_policy_metrics = {}

    def _configured_scheduler_total_steps(self):
        return self.config.max_steps * self.config.grpo_inner_epochs

    def save_checkpoint(self):
        if self._uses_reference_model:
            super().save_checkpoint()
        else:
            Trainer.save_checkpoint(self)

    def train(self):
        _rollout_policy_train_loop(self, self.inner_epochs)

    def compute_loss(self, batch):
        # Unused — GRPO overrides train() because old_logps must be frozen once
        # per rollout and reused across inner epochs. compute_loss takes a single
        # batch with no notion of "rollout vs. update", which does not match the
        # PPO-style update structure.
        raise NotImplementedError("GRPO runs its own train loop; compute_loss is not called")

    def _rollout(self, batch):
        """Sample K completions per prompt under the current policy, compute
        group-relative advantages, and freeze the old-policy log-probs that the
        clipped surrogate ratio will be measured against."""
        prompt_ids = batch["prompt_ids"]
        prompt_lens = batch["prompt_len"]
        require((prompt_lens > 0).all(), "GRPO requires prompt_len > 0 for every prompt")
        B = prompt_ids.size(0)

        completions, completion_masks, seqs, label_seqs = [], [], [], []
        was_training = self.model.training
        self.model.eval()
        try:
            generation_started = time.perf_counter()
            with torch.no_grad():
                for _ in range(self.K):
                    gens = []
                    for b in range(B):
                        plen = int(prompt_lens[b].item())
                        # GRPO ratios below are full-softmax token ratios, so the
                        # behavior policy must not apply top-k/top-p/repetition filters.
                        gens.append(_sample_completion(
                            self.model,
                            prompt_ids[b : b + 1, :plen],
                            self.max_new_tokens,
                            "GRPO",
                        ))
                    padded, completion_mask, seq_pad, label_pad, _ = _pack_prompt_completions(
                        prompt_ids,
                        prompt_lens,
                        gens,
                        self.device,
                    )
                    require(completion_mask.any(dim=1).all(), "GRPO requires at least one generated token per prompt")
                    completions.append(padded)
                    completion_masks.append(completion_mask)
                    seqs.append(seq_pad)
                    label_seqs.append(label_pad)
            generation_latency_ms = (time.perf_counter() - generation_started) * 1000.0

            reward_started = time.perf_counter()
            rewards, reward_components = stack_reward_results(
                [
                    self.reward_fn(batch, c, m)
                    for c, m in zip(completions, completion_masks, strict=True)
                ],
                self.device,
            )
            reward_latency_ms = (time.perf_counter() - reward_started) * 1000.0
            adv = _group_normalized_advantages(rewards)

            old_token_logps = []
            for k in range(self.K):
                with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype, enabled=self.dtype != torch.float32):
                    old_token_logps.append(_generation_context_token_logp(self.model, seqs[k], label_seqs[k])[0])

            return GroupRollout(
                seqs=seqs,
                label_seqs=label_seqs,
                old_token_logps=old_token_logps,
                completions=completions,
                completion_masks=completion_masks,
                rewards=rewards,
                adv=adv,
                reward_components=reward_components,
                generation_latency_ms=generation_latency_ms,
                reward_latency_ms=reward_latency_ms,
                generated_tokens=int(sum(mask.sum().item() for mask in completion_masks)),
                reward_calls=len(completion_masks),
            )
        finally:
            self.model.train(was_training)

    def _record_online_rollout(self, batch, rollout):
        if self.config.rl_metrics_every > 0 and self.step % self.config.rl_metrics_every == 0:
            metrics = group_stats(rollout.rewards, rollout.adv, rollout.completion_masks, rollout.reward_components)
            metrics.update(rollout_system_stats(rollout))
            metrics.update(self._last_policy_metrics)
            append_jsonl(
                Path(self.config.save_dir) / "online_rl_metrics.jsonl",
                [scalar_record(type(self).__name__, self.step, self._algorithm_name(), metrics)],
            )
        if self.config.rl_trace_samples > 0:
            append_jsonl(
                Path(self.config.save_dir) / "trajectories.jsonl",
                rollout_records(
                    type(self).__name__,
                    self.step,
                    self._algorithm_name(),
                    batch,
                    rollout,
                    self.config.rl_trace_samples,
                ),
            )

    def _algorithm_name(self):
        return "grpo"

    def _policy_loss(self, rollout):
        total = torch.tensor(0.0, device=self.device)
        ratio_values = []
        kl_values = []
        for k in range(self.K):
            logp, mask, aux = _generation_context_token_logp(
                self.model,
                rollout.seqs[k],
                rollout.label_seqs[k],
                return_aux=True,
            )
            ratio = (logp - rollout.old_token_logps[k]).clamp(min=-20.0, max=20.0).exp()
            adv_k = rollout.adv[:, k : k + 1]
            surr1 = ratio * adv_k
            surr2 = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * adv_k
            policy_loss = -_masked_response_mean(torch.min(surr1, surr2), mask, "GRPO policy loss")
            with torch.no_grad():
                ref_logp, _ = _generation_context_token_logp(self.ref_model, rollout.seqs[k], rollout.label_seqs[k])
            delta = ref_logp - logp
            kl = _masked_response_mean(delta.exp() - delta - 1, mask, "GRPO KL")
            ratio_values.append(ratio.detach()[mask.bool()])
            kl_values.append((delta.exp() - delta - 1).detach()[mask.bool()])
            total = total + policy_loss + self.kl_coef * kl + aux
        if ratio_values:
            ratio_flat = torch.cat(ratio_values)
            kl_flat = torch.cat(kl_values)
            self._last_policy_metrics = {
                "ratio_mean": float(ratio_flat.mean().item()),
                "clip_fraction": float(((ratio_flat < 1 - self.clip_ratio) | (ratio_flat > 1 + self.clip_ratio)).float().mean().item()),
                "kl_mean": float(kl_flat.mean().item()),
            }
        return total / self.K


@register_trainer("drgrpo")
class DrGRPOTrainer(GRPOTrainer):
    """Dr.GRPO-style group-centered trainer without reward-std normalization.

    This scoped path isolates the normalization ablation in Minilab's LM rollout
    contract. Length-bias corrections are not part of this class; DAPO owns the
    length-shaping path.
    """

    def __init__(self, model, reward_fn, train_dataset, config, ref_model_path, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, DrGRPOTrainConfig), "DrGRPOTrainer requires DrGRPOTrainConfig")
        super().__init__(model, reward_fn, train_dataset, config, ref_model_path, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)

    def _rollout(self, batch):
        rollout = super()._rollout(batch)
        return replace(rollout, adv=_group_centered_advantages(rollout.rewards))

    def _algorithm_name(self):
        return "drgrpo"


@register_trainer("tpo")
class TPOTrainer(GRPOTrainer):
    """Target Policy Optimization over LM rollout groups.

    For each prompt, the sampled completions form a candidate simplex. The
    target distribution is proportional to the old policy's sequence
    probabilities times an exponentiated standardized reward skill, then the
    current policy is fit to that target over the sampled candidates.
    """

    _extra_critical_fields = GRPOTrainer._extra_critical_fields + ("tpo_eta", "tpo_anchor_old_policy")
    _uses_reference_model = False

    def __init__(self, model, reward_fn, train_dataset, config, ref_model_path=None, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, TPOTrainConfig), "TPOTrainer requires TPOTrainConfig")
        super().__init__(model, reward_fn, train_dataset, config, ref_model_path, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.eta = config.tpo_eta
        self.anchor_old_policy = config.tpo_anchor_old_policy

    def _policy_loss(self, rollout):
        current_scores, _masks, aux = _group_sequence_scores(self.model, rollout, self.device, return_aux=True)
        old_scores = _old_sequence_scores(rollout)
        skill = _tpo_skill(rollout.rewards)
        if self.anchor_old_policy:
            target_logits = F.log_softmax(old_scores, dim=1) + skill / self.eta
        else:
            target_logits = skill / self.eta
        q = F.softmax(target_logits, dim=1).detach()
        log_p = F.log_softmax(current_scores, dim=1)
        loss = -(q * log_p).sum(dim=1).mean() + aux
        self._last_policy_metrics = {
            "tpo_target_entropy": float((-(q * q.clamp_min(1e-12).log()).sum(dim=1)).mean().item()),
            "tpo_target_top_prob": float(q.max(dim=1).values.mean().item()),
        }
        return loss

    def _algorithm_name(self):
        return "tpo" if self.anchor_old_policy else "tpo_no_anchor"


@register_trainer("tpo_no_anchor")
class TPONoAnchorTrainer(TPOTrainer):
    def __init__(self, model, reward_fn, train_dataset, config, ref_model_path=None, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, TPOTrainConfig), "TPONoAnchorTrainer requires TPOTrainConfig")
        require(not config.tpo_anchor_old_policy, "TPONoAnchorTrainer requires tpo_anchor_old_policy=False")
        super().__init__(model, reward_fn, train_dataset, config, ref_model_path, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)


@register_trainer("group_pg")
class GroupPGTrainer(GRPOTrainer):
    """Reward-skill weighted sequence policy-gradient ablation for TPO."""

    _uses_reference_model = False

    def __init__(self, model, reward_fn, train_dataset, config, ref_model_path=None, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, GroupPGTrainConfig), "GroupPGTrainer requires GroupPGTrainConfig")
        super().__init__(model, reward_fn, train_dataset, config, ref_model_path, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)

    def _policy_loss(self, rollout):
        current_scores, _masks, aux = _group_sequence_scores(self.model, rollout, self.device, return_aux=True)
        skill = _tpo_skill(rollout.rewards).detach()
        self._last_policy_metrics = {"group_pg_skill_abs_mean": float(skill.abs().mean().item())}
        return -(skill * current_scores).sum(dim=1).mean() / self.K + aux

    def _algorithm_name(self):
        return "group_pg"


@register_trainer("vpo")
class VPOTrainer(GRPOTrainer):
    """Vector Policy Optimization over LM candidate sets.

    A rollout group contains K sets, each set contains C sampled completions,
    and component rewards define the vector reward. Each prompt group shares a
    batch of Dirichlet scalarizations, and the set reward is the mean best
    candidate under those scalarizations.
    """

    _extra_critical_fields = GRPOTrainer._extra_critical_fields + (
        "vpo_num_candidates", "vpo_num_scalarizations", "vpo_dirichlet_alpha",
    )
    _uses_reference_model = False

    def __init__(self, model, reward_fn, train_dataset, config, ref_model_path=None, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, VPOTrainConfig), "VPOTrainer requires VPOTrainConfig")
        super().__init__(model, reward_fn, train_dataset, config, ref_model_path, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.C = config.vpo_num_candidates
        self.S = config.vpo_num_scalarizations
        self.dirichlet_alpha = config.vpo_dirichlet_alpha

    def _rollout(self, batch):
        prompt_ids = batch["prompt_ids"]
        prompt_lens = batch["prompt_len"]
        require((prompt_lens > 0).all(), "VPO requires prompt_len > 0 for every prompt")
        B = prompt_ids.size(0)

        flat_completions, flat_masks, flat_seqs, flat_labels = [], [], [], []
        candidate_rewards = []
        component_rows = {}
        generation_latency_ms = 0.0
        reward_latency_ms = 0.0
        generated_tokens = 0
        reward_calls = 0
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for _ in range(self.K):
                    set_rewards = []
                    set_components = {}
                    for _candidate in range(self.C):
                        generation_started = time.perf_counter()
                        gens = []
                        for b in range(B):
                            plen = int(prompt_lens[b].item())
                            gens.append(_sample_completion(
                                self.model,
                                prompt_ids[b : b + 1, :plen],
                                self.max_new_tokens,
                                "VPO",
                            ))
                        padded, completion_mask, seq_pad, label_pad, _ = _pack_prompt_completions(
                            prompt_ids,
                            prompt_lens,
                            gens,
                            self.device,
                        )
                        generation_latency_ms += (time.perf_counter() - generation_started) * 1000.0
                        generated_tokens += int(completion_mask.sum().item())
                        reward_started = time.perf_counter()
                        reward, components = stack_reward_results([self.reward_fn(batch, padded, completion_mask)], self.device)
                        reward_latency_ms += (time.perf_counter() - reward_started) * 1000.0
                        reward_calls += 1
                        flat_completions.append(padded)
                        flat_masks.append(completion_mask)
                        flat_seqs.append(seq_pad)
                        flat_labels.append(label_pad)
                        set_rewards.append(reward[:, 0])
                        for key, value in components.items():
                            set_components.setdefault(key, []).append(value[:, 0])
                    candidate_rewards.append(torch.stack(set_rewards, dim=1))
                    for key, values in set_components.items():
                        component_rows.setdefault(key, []).append(torch.stack(values, dim=1))

            candidate_rewards = torch.stack(candidate_rewards, dim=1)
            components = {
                key: torch.stack(values, dim=1)
                for key, values in component_rows.items()
            }
            rewards = _vpo_set_rewards(candidate_rewards, components, self.S, self.dirichlet_alpha)
            adv = _group_normalized_advantages(rewards)

            old_token_logps = []
            for seq, label in zip(flat_seqs, flat_labels, strict=True):
                with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype, enabled=self.dtype != torch.float32):
                    old_token_logps.append(_generation_context_token_logp(self.model, seq, label)[0])

            return VPORollout(
                seqs=flat_seqs,
                label_seqs=flat_labels,
                old_token_logps=old_token_logps,
                completions=flat_completions,
                completion_masks=flat_masks,
                rewards=rewards,
                adv=adv,
                candidate_rewards=candidate_rewards,
                reward_components=components,
                generation_latency_ms=generation_latency_ms,
                reward_latency_ms=reward_latency_ms,
                generated_tokens=generated_tokens,
                reward_calls=reward_calls,
            )
        finally:
            self.model.train(was_training)

    def _policy_loss(self, rollout):
        total = torch.tensor(0.0, device=self.device)
        ratios = []
        flat_index = 0
        for k in range(self.K):
            for _candidate in range(self.C):
                logp, mask, aux = _generation_context_token_logp(
                    self.model,
                    rollout.seqs[flat_index],
                    rollout.label_seqs[flat_index],
                    return_aux=True,
                )
                ratio = (logp - rollout.old_token_logps[flat_index]).clamp(min=-20.0, max=20.0).exp()
                adv = rollout.adv[:, k : k + 1]
                surr1 = ratio * adv
                surr2 = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                policy_loss = -_masked_response_mean(torch.min(surr1, surr2), mask, "VPO policy loss")
                total = total + policy_loss + aux
                ratios.append(ratio.detach()[mask.bool()])
                flat_index += 1
        if ratios:
            ratio_flat = torch.cat(ratios)
            self._last_policy_metrics = {
                "ratio_mean": float(ratio_flat.mean().item()),
                "clip_fraction": float(((ratio_flat < 1 - self.clip_ratio) | (ratio_flat > 1 + self.clip_ratio)).float().mean().item()),
                "best_at_set_mean": float(rollout.candidate_rewards.max(dim=2).values.mean().item()),
                "pool_diversity_l1": float((rollout.candidate_rewards.max(dim=2).values - rollout.candidate_rewards.min(dim=2).values).abs().mean().item()),
            }
        return total / (self.K * self.C)

    def _record_online_rollout(self, batch, rollout):
        if self.config.rl_metrics_every > 0 and self.step % self.config.rl_metrics_every == 0:
            metrics = group_stats(rollout.rewards, rollout.adv)
            metrics.update(rollout_system_stats(rollout))
            metrics.update(self._last_policy_metrics)
            for key, value in rollout.reward_components.items():
                metrics[f"reward_component_{key}_mean"] = float(value.float().mean().item())
                metrics[f"vpo_best_{key}_mean"] = float(value.float().max(dim=2).values.mean().item())
                metrics[f"vpo_diversity_{key}_mean"] = float((value.float().max(dim=2).values - value.float().min(dim=2).values).mean().item())
            append_jsonl(
                Path(self.config.save_dir) / "online_rl_metrics.jsonl",
                [scalar_record(type(self).__name__, self.step, self._algorithm_name(), metrics)],
            )
        if self.config.rl_trace_samples > 0:
            append_jsonl(
                Path(self.config.save_dir) / "trajectories.jsonl",
                vpo_rollout_records(
                    type(self).__name__,
                    self.step,
                    self._algorithm_name(),
                    rollout,
                    self.config.rl_trace_samples,
                ),
            )

    def _algorithm_name(self):
        return "vpo"


@register_trainer("dg")
class DGTrainer(GRPOTrainer):
    """Sequence-level Delightful Policy Gradient over LM rollouts.

    Minilab's scoped LM version uses the rollout group's leave-mean reward as
    the baseline because exact actor expected reward is unavailable for
    free-form generation.
    """

    _extra_critical_fields = GRPOTrainer._extra_critical_fields + (
        "dg_eta", "dg_keep_ratio", "dg_uncertainty_threshold",
        "dg_replay_capacity", "dg_replay_min_age", "dg_replay_age_decay",
        "dg_staleness_delay", "dg_drop_stale_after",
    )
    _uses_reference_model = False
    _uses_grpo_kl_as_ratio_variance = False

    def __init__(self, model, reward_fn, train_dataset, config, ref_model_path=None, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, SequenceDGTrainConfig), f"{type(self).__name__} requires SequenceDGTrainConfig")
        if not self._uses_grpo_kl_as_ratio_variance:
            require(config.grpo_kl_coef == GRPOTrainConfig.grpo_kl_coef, (
                f"{type(self).__name__} does not use the R2VPO ratio-variance coefficient; "
                "leave grpo_kl_coef at the inherited default"
            ))
        super().__init__(model, reward_fn, train_dataset, config, ref_model_path, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.dg_eta = config.dg_eta
        self.keep_ratio = config.dg_keep_ratio
        self.uncertainty_threshold = config.dg_uncertainty_threshold
        self.replay_capacity = config.dg_replay_capacity
        self.replay_min_age = config.dg_replay_min_age
        self.replay_age_decay = config.dg_replay_age_decay
        self.staleness_delay = config.dg_staleness_delay
        self.drop_stale_after = config.dg_drop_stale_after
        self._replay = []

    def _sequence_advantage(self, rollout):
        return rollout.rewards - rollout.rewards.mean(dim=1, keepdim=True)

    def _sequence_uncertainty(self, rollout):
        return rollout.rewards.std(dim=1, keepdim=True, unbiased=False).expand_as(rollout.rewards)

    def _gate_and_advantage(self, seq_logp, old_seq_logp, rollout):
        advantage = self._sequence_advantage(rollout)
        surprisal = -old_seq_logp.detach()
        gate = torch.sigmoid(advantage * surprisal / self.dg_eta)
        return gate, advantage, {}

    def _policy_loss(self, rollout):
        seq_logp, _masks, aux = _group_sequence_scores(self.model, rollout, self.device, return_aux=True)
        old_seq_logp = _old_sequence_scores(rollout)
        gate, advantage, metrics = self._gate_and_advantage(seq_logp, old_seq_logp, rollout)
        weight = (gate * advantage).detach()
        self._last_policy_metrics = {
            "gate_mean": float(gate.mean().item()),
            **metrics,
        }
        return -(seq_logp * weight).mean() + aux

    def _algorithm_name(self):
        return "dg"


@register_trainer("kondo")
class KondoTrainer(DGTrainer):
    """Compute-screening DG ablation using a smooth keep-ratio gate."""

    def _gate_and_advantage(self, seq_logp, old_seq_logp, rollout):
        advantage = self._sequence_advantage(rollout)
        delight = advantage * (-old_seq_logp.detach())
        flat = delight.reshape(-1)
        if self.keep_ratio >= 1:
            screen = torch.ones_like(delight)
            threshold = flat.min()
        else:
            threshold = torch.quantile(flat, 1 - self.keep_ratio)
            screen = torch.sigmoid((delight - threshold) / self.dg_eta)
        gate = torch.sigmoid(delight / self.dg_eta) * screen
        return gate, advantage, {
            "kept_frac": float((screen > 0.5).float().mean().item()),
            "gate_prob_mean": float(screen.mean().item()),
        }

    def _algorithm_name(self):
        return "kondo"


@register_trainer("uncertainty_dg")
class UncertaintyDGTrainer(DGTrainer):
    def _gate_and_advantage(self, seq_logp, old_seq_logp, rollout):
        advantage = self._sequence_advantage(rollout)
        uncertainty = self._sequence_uncertainty(rollout)
        surprisal = -old_seq_logp.detach()
        gate = torch.sigmoid((advantage * surprisal - uncertainty) / self.dg_eta)
        return gate, advantage, {"uncertainty_mean": float(uncertainty.mean().item())}

    def _algorithm_name(self):
        return "uncertainty_dg"


@register_trainer("filtered_dg")
class FilteredDGTrainer(DGTrainer):
    def _gate_and_advantage(self, seq_logp, old_seq_logp, rollout):
        advantage = self._sequence_advantage(rollout)
        uncertainty = self._sequence_uncertainty(rollout)
        keep = (uncertainty <= self.uncertainty_threshold).float()
        gate = torch.sigmoid(advantage * (-old_seq_logp.detach()) / self.dg_eta) * keep
        return gate, advantage, {
            "uncertainty_mean": float(uncertainty.mean().item()),
            "kept_frac": float(keep.mean().item()),
        }

    def _algorithm_name(self):
        return "filtered_dg"


@register_trainer("reward_variance_dg")
class RewardVarianceDGTrainer(DGTrainer):
    def _gate_and_advantage(self, seq_logp, old_seq_logp, rollout):
        advantage = self._sequence_advantage(rollout)
        uncertainty = self._sequence_uncertainty(rollout)
        effective = advantage / (1 + uncertainty)
        gate = torch.sigmoid(effective * (-old_seq_logp.detach()) / self.dg_eta)
        return gate, effective, {"uncertainty_mean": float(uncertainty.mean().item())}

    def _algorithm_name(self):
        return "reward_variance_dg"


@register_trainer("aspo")
class ASPOTrainer(DGTrainer):
    """Asymmetric sequence-ratio policy gradient."""

    def _policy_loss(self, rollout):
        seq_logp, _masks, aux = _group_sequence_scores(self.model, rollout, self.device, return_aux=True)
        old_seq_logp = _old_sequence_scores(rollout)
        advantage = self._sequence_advantage(rollout)
        log_ratio = (seq_logp - old_seq_logp).clamp(min=-20, max=20)
        flipped = torch.where(advantage > 0, -log_ratio, log_ratio)
        ratio = flipped.exp()
        effective = ratio * advantage
        self._last_policy_metrics = {"ratio_mean": float(ratio.mean().item())}
        return -(seq_logp * effective.detach()).mean() + aux

    def _algorithm_name(self):
        return "aspo"


@register_trainer("r2vpo")
class R2VPOTrainer(DGTrainer):
    """Ratio-variance regularized sequence policy gradient."""

    _uses_grpo_kl_as_ratio_variance = True

    def _policy_loss(self, rollout):
        seq_logp, _masks, aux = _group_sequence_scores(self.model, rollout, self.device, return_aux=True)
        old_seq_logp = _old_sequence_scores(rollout)
        advantage = self._sequence_advantage(rollout)
        ratio = (seq_logp - old_seq_logp).clamp(min=-20, max=20).exp()
        penalty = 2 * self.config.grpo_kl_coef * (ratio - 1)
        effective = ratio * (advantage - penalty)
        self._last_policy_metrics = {
            "ratio_mean": float(ratio.mean().item()),
            "var_penalty_mean": float(penalty.mean().item()),
        }
        return -(seq_logp * effective.detach()).mean() + aux

    def _algorithm_name(self):
        return "r2vpo"


@register_trainer("replay_dg")
class ReplayDGTrainer(DGTrainer):
    """In-process stale-rollout DG, mirroring replay as a local systems probe."""

    def _rollout(self, batch):
        fresh = replace(super()._rollout(batch), age=0)
        capacity = max(self.replay_capacity, self.staleness_delay + 1 if self.staleness_delay > 0 else 0)
        if capacity <= 0:
            return replace(fresh, queue_depth=0)
        self._replay = [replace(item, age=item.age + 1) for item in self._replay]
        self._replay.append(fresh)
        if len(self._replay) > capacity:
            self._replay.pop(0)
        dropped = 0
        if self.drop_stale_after > 0:
            kept = []
            for item in self._replay:
                if item.age > self.drop_stale_after:
                    dropped += 1
                else:
                    kept.append(item)
            self._replay = kept
        delay = max(self.replay_min_age, self.staleness_delay)
        ready = [item for item in self._replay if item.age >= delay]
        selected = min(ready, key=lambda item: item.age) if ready else fresh
        return replace(
            selected,
            queue_depth=len(self._replay),
            dropped_stale_count=dropped,
            staleness_delay=delay,
            drop_stale_after=self.drop_stale_after,
            drop_reason="" if ready or delay == 0 else "warming_up_for_staleness",
        )

    def _algorithm_name(self):
        return "replay_dg"


@register_trainer("fresh_dg")
class FreshDGTrainer(ReplayDGTrainer):
    def _gate_and_advantage(self, seq_logp, old_seq_logp, rollout):
        gate, advantage, metrics = super()._gate_and_advantage(seq_logp, old_seq_logp, rollout)
        freshness = torch.tensor(math.exp(-self.replay_age_decay * rollout.age), device=gate.device)
        return gate * freshness, advantage, {**metrics, "freshness_weight": float(freshness.item()), "batch_age": float(rollout.age)}

    def _algorithm_name(self):
        return "fresh_dg"


@register_trainer("rloo")
class RLOOTrainer(GRPOTrainer):
    """REINFORCE Leave-One-Out with optional sequence KL reward penalty."""

    _extra_critical_fields = (
        "grpo_num_generations", "grpo_max_new_tokens",
        "grpo_kl_coef", "grpo_inner_epochs",
    )

    def __init__(self, model, reward_fn, train_dataset, config, ref_model_path, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, RLOOTrainConfig), "RLOOTrainer requires RLOOTrainConfig")
        super().__init__(model, reward_fn, train_dataset, config, ref_model_path, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)

    def _rollout(self, batch):
        rollout = super()._rollout(batch)
        seq_rewards = []
        old_seq_logps = []
        for k in range(self.K):
            old_logp = (rollout.old_token_logps[k] * (rollout.label_seqs[k] != -100).float()).sum(dim=-1)
            with torch.no_grad():
                ref_logp, mask = _generation_context_token_logp(self.ref_model, rollout.seqs[k], rollout.label_seqs[k])
                kl = ((rollout.old_token_logps[k] - ref_logp) * mask).sum(dim=-1)
            seq_rewards.append(rollout.rewards[:, k] - self.kl_coef * kl)
            old_seq_logps.append(old_logp)

        rewards = torch.stack(seq_rewards, dim=1)
        baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (self.K - 1)
        return replace(rollout, old_token_logps=old_seq_logps, rewards=rewards, adv=rewards - baseline)

    def _policy_loss(self, rollout):
        total = torch.tensor(0.0, device=self.device)
        seq_logps = []
        for k in range(self.K):
            logp, mask, aux = _generation_context_token_logp(
                self.model,
                rollout.seqs[k],
                rollout.label_seqs[k],
                return_aux=True,
            )
            seq_logp = (logp * mask).sum(dim=-1)
            seq_logps.append(seq_logp.detach())
            total = total - (seq_logp * rollout.adv[:, k].detach()).mean() + aux
        if seq_logps:
            self._last_policy_metrics = {"seq_logp_mean": float(torch.stack(seq_logps, dim=1).mean().item())}
        return total / self.K

    def _algorithm_name(self):
        return "rloo"


@register_trainer("gspo")
class GSPOTrainer(GRPOTrainer):
    """Group Sequence Policy Optimization with sequence-level ratios."""

    _extra_critical_fields = (
        "grpo_num_generations", "grpo_max_new_tokens",
        "grpo_clip_ratio", "grpo_kl_coef", "grpo_inner_epochs",
    )

    def __init__(self, model, reward_fn, train_dataset, config, ref_model_path, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, GSPOTrainConfig), "GSPOTrainer requires GSPOTrainConfig")
        super().__init__(model, reward_fn, train_dataset, config, ref_model_path, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)

    def _policy_loss(self, rollout):
        total = torch.tensor(0.0, device=self.device)
        ratios = []
        kls = []
        for k in range(self.K):
            logp, mask, aux = _generation_context_token_logp(
                self.model,
                rollout.seqs[k],
                rollout.label_seqs[k],
                return_aux=True,
            )
            token_counts = mask.sum(dim=-1).clamp(min=1.0)
            seq_log_ratio = ((logp - rollout.old_token_logps[k]) * mask).sum(dim=-1) / token_counts
            ratio = seq_log_ratio.clamp(min=-20.0, max=20.0).exp()
            adv = rollout.adv[:, k]
            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            with torch.no_grad():
                ref_logp, _ = _generation_context_token_logp(self.ref_model, rollout.seqs[k], rollout.label_seqs[k])
            delta = ref_logp - logp
            kl = _masked_response_mean(delta.exp() - delta - 1, mask, "GSPO KL")
            ratios.append(ratio.detach())
            kls.append((delta.exp() - delta - 1).detach()[mask.bool()])
            total = total + policy_loss + self.kl_coef * kl + aux
        if ratios:
            ratio_flat = torch.cat([r.reshape(-1) for r in ratios])
            kl_flat = torch.cat(kls)
            self._last_policy_metrics = {
                "ratio_mean": float(ratio_flat.mean().item()),
                "clip_fraction": float(((ratio_flat < 1 - self.clip_ratio) | (ratio_flat > 1 + self.clip_ratio)).float().mean().item()),
                "kl_mean": float(kl_flat.mean().item()),
            }
        return total / self.K

    def _algorithm_name(self):
        return "gspo"


@register_trainer("dapo")
class DAPOTrainer(GRPOTrainer):
    """DAPO: decoupled clipping, dynamic sampling, token-level aggregation, length shaping."""

    _uses_reference_model = False
    _extra_critical_fields = (
        "grpo_num_generations", "grpo_max_new_tokens",
        "grpo_inner_epochs",
        "dapo_clip_ratio_low", "dapo_clip_ratio_high",
        "dapo_safe_length", "dapo_length_penalty", "dapo_max_resample",
    )

    def __init__(self, model, reward_fn, train_dataset, config, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, DAPOTrainConfig), "DAPOTrainer requires DAPOTrainConfig")
        require(eval_dataset is None, "DAPO has no LM-style eval loss; evaluate accuracy post-training")
        require(config.eval_every == 0, "DAPO has no LM-style eval loss; set eval_every=0")
        require(config.grad_accum_steps == 1, "DAPO does not support grad_accum_steps > 1")
        Trainer.__init__(
            self,
            model,
            train_dataset,
            config,
            signature=signature,
            tokenizer_sig=tokenizer_sig,
            eval_dataset=eval_dataset,
        )
        self.reward_fn = reward_fn
        self.K = config.grpo_num_generations
        self.inner_epochs = config.grpo_inner_epochs
        self.max_new_tokens = config.grpo_max_new_tokens
        self.clip_ratio_low = config.dapo_clip_ratio_low
        self.clip_ratio_high = config.dapo_clip_ratio_high
        self.safe_length = config.dapo_safe_length
        self.length_penalty = config.dapo_length_penalty
        self.max_resample = config.dapo_max_resample

    def save_checkpoint(self):
        # DAPO is reference-free, but inherits GRPO rollout helpers. Bypass
        # ReferenceCheckpointMixin so checkpoints do not require ref_path.txt.
        Trainer.save_checkpoint(self)

    def compute_loss(self, batch):
        raise NotImplementedError("DAPO runs its own train loop; compute_loss is not called")

    def _rollout(self, batch):
        target_groups = batch["prompt_ids"].size(0)
        selected = []
        selected_groups = 0
        candidate_batch = batch

        for _ in range(self.max_resample):
            rollout = super()._rollout(candidate_batch)
            raw_std = rollout.rewards.std(dim=1, keepdim=True, unbiased=False)
            valid = (raw_std.squeeze(1) > 0).nonzero(as_tuple=False).flatten()
            if valid.numel() > 0:
                remaining = target_groups - selected_groups
                keep = valid[:remaining]
                selected.append(_select_group_rollout_rows(rollout, keep))
                selected_groups += keep.numel()
                if selected_groups >= target_groups:
                    break
            if selected_groups < target_groups:
                candidate_batch = self._next_batch()

        if selected_groups < target_groups:
            raise ValueError(
                "DAPO dynamic sampling exhausted dapo_max_resample without enough "
                "non-degenerate prompt groups to fill the batch"
            )

        rollout = _merge_group_rollouts(selected)

        rewards = rollout.rewards + torch.stack([
            self._length_reward(m) for m in rollout.completion_masks
        ], dim=1)
        components = dict(rollout.reward_components or {})
        if self.safe_length > 0 and self.length_penalty > 0:
            components["length_penalty"] = rewards - rollout.rewards
        adv = _group_normalized_advantages(rewards)
        return replace(rollout, rewards=rewards, adv=adv, reward_components=components)

    def _length_reward(self, completion_mask):
        if self.safe_length == 0 or self.length_penalty == 0:
            return torch.zeros(completion_mask.size(0), device=self.device)
        lengths = completion_mask.sum(dim=1).to(torch.float32)
        over = (lengths - self.safe_length).clamp(min=0)
        budget = max(1, self.max_new_tokens - self.safe_length)
        return -(over / budget).clamp(max=1.0) * self.length_penalty

    def _policy_loss(self, rollout):
        total = torch.tensor(0.0, device=self.device)
        token_total = torch.tensor(0.0, device=self.device)
        aux_total = torch.tensor(0.0, device=self.device)
        ratios = []
        for k in range(self.K):
            logp, mask, aux = _generation_context_token_logp(
                self.model,
                rollout.seqs[k],
                rollout.label_seqs[k],
                return_aux=True,
            )
            ratio = (logp - rollout.old_token_logps[k]).clamp(min=-20.0, max=20.0).exp()
            adv = rollout.adv[:, k : k + 1]
            clipped = ratio.clamp(1 - self.clip_ratio_low, 1 + self.clip_ratio_high)
            policy_loss = -torch.min(ratio * adv, clipped * adv) * mask
            ratios.append(ratio.detach()[mask.bool()])
            total = total + policy_loss.sum()
            token_total = token_total + mask.sum()
            aux_total = aux_total + aux
        if ratios:
            ratio_flat = torch.cat(ratios)
            self._last_policy_metrics = {
                "ratio_mean": float(ratio_flat.mean().item()),
                "clip_fraction": float(((ratio_flat < 1 - self.clip_ratio_low) | (ratio_flat > 1 + self.clip_ratio_high)).float().mean().item()),
            }
        return total / token_total.clamp(min=1.0) + aux_total / self.K

    def _algorithm_name(self):
        return "dapo"


def _select_group_rollout_rows(rollout, rows):
    return replace(
        rollout,
        seqs=[seq.index_select(0, rows.to(seq.device)) for seq in rollout.seqs],
        label_seqs=[labels.index_select(0, rows.to(labels.device)) for labels in rollout.label_seqs],
        old_token_logps=[logp.index_select(0, rows.to(logp.device)) for logp in rollout.old_token_logps],
        completions=[completion.index_select(0, rows.to(completion.device)) for completion in rollout.completions],
        completion_masks=[mask.index_select(0, rows.to(mask.device)) for mask in rollout.completion_masks],
        rewards=rollout.rewards.index_select(0, rows.to(rollout.rewards.device)),
        adv=rollout.adv.index_select(0, rows.to(rollout.adv.device)),
        reward_components={
            key: value.index_select(0, rows.to(value.device))
            for key, value in (rollout.reward_components or {}).items()
        },
    )


def _merge_group_rollouts(chunks):
    require(chunks, "DAPO dynamic sampling requires at least one selected rollout chunk")
    K = len(chunks[0].seqs)
    require(K > 0, "DAPO dynamic sampling selected an empty rollout")
    component_keys = set(chunks[0].reward_components or {})
    for chunk in chunks:
        require(len(chunk.seqs) == K, "DAPO rollout chunks must have the same number of generations")
        require(len(chunk.label_seqs) == K, "DAPO rollout chunks must have matching label generations")
        require(len(chunk.old_token_logps) == K, "DAPO rollout chunks must have matching old log-prob generations")
        require(len(chunk.completions) == K, "DAPO rollout chunks must have matching completion generations")
        require(len(chunk.completion_masks) == K, "DAPO rollout chunks must have matching completion masks")
        require(set(chunk.reward_components or {}) == component_keys, (
            "DAPO rollout chunks must have matching reward component keys"
        ))
    return GroupRollout(
        seqs=[
            _cat_padded([chunk.seqs[k] for chunk in chunks], pad_value=0)
            for k in range(K)
        ],
        label_seqs=[
            _cat_padded([chunk.label_seqs[k] for chunk in chunks], pad_value=-100)
            for k in range(K)
        ],
        old_token_logps=[
            _cat_padded([chunk.old_token_logps[k] for chunk in chunks], pad_value=0.0)
            for k in range(K)
        ],
        completions=[
            _cat_padded([chunk.completions[k] for chunk in chunks], pad_value=0)
            for k in range(K)
        ],
        completion_masks=[
            _cat_padded([chunk.completion_masks[k] for chunk in chunks], pad_value=False)
            for k in range(K)
        ],
        rewards=torch.cat([chunk.rewards for chunk in chunks], dim=0),
        adv=torch.cat([chunk.adv for chunk in chunks], dim=0),
        reward_components={
            key: torch.cat([chunk.reward_components[key] for chunk in chunks], dim=0)
            for key in component_keys
        },
        generation_latency_ms=sum(chunk.generation_latency_ms for chunk in chunks),
        reward_latency_ms=sum(chunk.reward_latency_ms for chunk in chunks),
        generated_tokens=sum(chunk.generated_tokens for chunk in chunks),
        reward_calls=sum(chunk.reward_calls for chunk in chunks),
    )


def _cat_padded(tensors, pad_value):
    require(tensors, "cannot concatenate an empty tensor list")
    max_len = max(t.size(1) for t in tensors)
    padded = []
    for tensor in tensors:
        if tensor.size(1) == max_len:
            padded.append(tensor)
            continue
        pad = tensor.new_full((tensor.size(0), max_len - tensor.size(1)), pad_value)
        padded.append(torch.cat([tensor, pad], dim=1))
    return torch.cat(padded, dim=0)
