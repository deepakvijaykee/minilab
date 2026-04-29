from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from minilab.alignment_common import (
    _generation_context_token_logp,
    _load_reference_model,
    _masked_response_mean,
    _model_max_seq_len,
    _trainer_reference_path,
    _validate_reference_tokenizer,
    _whiten_masked,
)
from minilab.base import BaseModel, unwrap_model
from minilab.checks import require
from minilab.generation import generate
from minilab.registry import register_trainer
from minilab.trainer import (
    TrainConfig,
    Trainer,
    commit_post_optimizer_updates,
    model_aux_loss,
    optimizer_decay_groups,
    set_seed,
)


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
        require(self.grpo_num_generations > 1, "GRPO requires grpo_num_generations > 1")
        require(self.grpo_max_new_tokens > 0, "grpo_max_new_tokens must be > 0")
        require(self.grpo_clip_ratio > 0, "grpo_clip_ratio must be > 0")
        require(self.grpo_kl_coef >= 0, "grpo_kl_coef must be >= 0")
        require(self.grpo_inner_epochs >= 1, "grpo_inner_epochs must be >= 1")


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

    def __post_init__(self):
        super().__post_init__()


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
    old_logp: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    mask: torch.Tensor


@dataclass
class GroupRollout:
    seqs: list
    label_seqs: list
    old_token_logps: list
    completions: list
    completion_masks: list
    rewards: torch.Tensor
    adv: torch.Tensor


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
    B = prompt_ids.size(0)
    comp_lens = [gen.size(0) for gen in completions]
    max_clen = max(comp_lens)
    completion_pad = torch.zeros(B, max_clen, device=device, dtype=torch.long)
    completion_mask = torch.zeros(B, max_clen, device=device, dtype=torch.bool)
    seqs, labels = [], []

    for b, gen in enumerate(completions):
        plen = int(prompt_lens[b].item())
        clen = gen.size(0)
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
            self._save_checkpoint_if_due()
            self.model.eval()
            self.value_head.eval()

        self._save_final_checkpoint()

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

        completions, completion_mask, seq_pad, label_pad, _ = _pack_prompt_completions(
            prompt_ids,
            prompt_lens,
            gens,
            self.device,
        )

        rewards = self.reward_fn(batch, completions, completion_mask).to(self.device)
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

        self.model.train(was_training)
        self.value_head.train(value_was_training)
        return PPORollout(
            seqs=seq_pad,
            labels=label_pad,
            old_logp=old_logp.detach(),
            old_values=old_values.detach(),
            advantages=advantages.detach(),
            returns=returns.detach(),
            mask=mask.detach(),
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
class GRPOTrainer(Trainer):
    """Per step: sample K completions per prompt under the current (old) policy,
    freeze their log-probs, then run grpo_inner_epochs clipped-surrogate updates
    against those frozen old log-probs. This is what makes the PPO-style ratio
    clip actually active: without inner epochs, old_logp == logp exactly on the
    first pass and the clip is a no-op."""

    _extra_critical_fields = (
        "grpo_num_generations", "grpo_max_new_tokens",
        "grpo_clip_ratio", "grpo_kl_coef", "grpo_inner_epochs",
    )

    def __init__(self, model, reward_fn, train_dataset, config, ref_model_path, *, signature, tokenizer_sig="", eval_dataset=None):
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
        self.ref_model_path = _trainer_reference_path(ref_model_path, config, "GRPO")
        _validate_reference_tokenizer(self.ref_model_path, tokenizer_sig, "GRPO")
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.ref_model = _load_reference_model(self.model, self.ref_model_path, self.device, "GRPO")
        self.reward_fn = reward_fn
        self.K = config.grpo_num_generations
        self.inner_epochs = config.grpo_inner_epochs
        self.max_new_tokens = config.grpo_max_new_tokens
        self.clip_ratio = config.grpo_clip_ratio
        self.kl_coef = config.grpo_kl_coef

    def _configured_scheduler_total_steps(self):
        return self.config.max_steps * self.config.grpo_inner_epochs

    def save_checkpoint(self):
        super().save_checkpoint()
        (Path(self.config.save_dir) / f"step_{self.step}" / "ref_path.txt").write_text(self.ref_model_path)

    def compute_loss(self, batch):
        # Unused — GRPO overrides train() because old_logps must be frozen once
        # per rollout and reused across inner epochs. compute_loss takes a single
        # batch with no notion of "rollout vs. update", which does not match the
        # PPO-style update structure.
        raise NotImplementedError("GRPO runs its own train loop; compute_loss is not called")

    def train(self):
        # GRPO samples and scores the policy in eval mode. Gradients still flow
        # during the update pass, while train-time stochastic layers stay out of
        # the PPO ratio.
        self.model.eval()
        pbar = tqdm(range(self.step + 1, self.config.max_steps + 1), desc="Training")
        for self.step in pbar:
            batch = self._next_batch()
            rollout = self._rollout(batch)
            total_loss = 0.0
            for _ in range(self.inner_epochs):
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
            self._save_checkpoint_if_due()

        self._save_final_checkpoint()

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

        rewards = torch.stack([
            self.reward_fn(batch, c, m)
            for c, m in zip(completions, completion_masks, strict=True)
        ], dim=1).to(self.device)
        adv = rewards - rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        adv = torch.where(std > 0, adv / std, torch.zeros_like(adv))

        old_token_logps = []
        for k in range(self.K):
            with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype, enabled=self.dtype != torch.float32):
                old_token_logps.append(_generation_context_token_logp(self.model, seqs[k], label_seqs[k])[0])

        self.model.train(was_training)
        return GroupRollout(
            seqs=seqs,
            label_seqs=label_seqs,
            old_token_logps=old_token_logps,
            completions=completions,
            completion_masks=completion_masks,
            rewards=rewards,
            adv=adv,
        )

    def _policy_loss(self, rollout):
        total = torch.tensor(0.0, device=self.device)
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
            total = total + policy_loss + self.kl_coef * kl + aux
        return total / self.K


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
        for k in range(self.K):
            logp, mask, aux = _generation_context_token_logp(
                self.model,
                rollout.seqs[k],
                rollout.label_seqs[k],
                return_aux=True,
            )
            seq_logp = (logp * mask).sum(dim=-1)
            total = total - (seq_logp * rollout.adv[:, k].detach()).mean() + aux
        return total / self.K


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
            total = total + policy_loss + self.kl_coef * kl + aux
        return total / self.K


@register_trainer("dapo")
class DAPOTrainer(GRPOTrainer):
    """DAPO: decoupled clipping, dynamic sampling, token-level aggregation, length shaping."""

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
        adv = rewards - rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        adv = torch.where(std > 0, adv / std, torch.zeros_like(adv))
        return replace(rollout, rewards=rewards, adv=adv)

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
            total = total + policy_loss.sum()
            token_total = token_total + mask.sum()
            aux_total = aux_total + aux
        return total / token_total.clamp(min=1.0) + aux_total / self.K


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
    )


def _merge_group_rollouts(chunks):
    require(chunks, "DAPO dynamic sampling requires at least one selected rollout chunk")
    K = len(chunks[0].seqs)
    require(K > 0, "DAPO dynamic sampling selected an empty rollout")
    for chunk in chunks:
        require(len(chunk.seqs) == K, "DAPO rollout chunks must have the same number of generations")
        require(len(chunk.label_seqs) == K, "DAPO rollout chunks must have matching label generations")
        require(len(chunk.old_token_logps) == K, "DAPO rollout chunks must have matching old log-prob generations")
        require(len(chunk.completions) == K, "DAPO rollout chunks must have matching completion generations")
        require(len(chunk.completion_masks) == K, "DAPO rollout chunks must have matching completion masks")
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
