import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from minilab.alignment_common import (
    _diffusion_loss_per_example,
    _group_normalized_advantages,
    _load_reference_model,
    _rollout_policy_train_loop,
    _trainer_reference_path,
    _validate_reference_tokenizer,
)
from minilab.base import unwrap_model
from minilab.checks import require
from minilab.diffusion import forward_process_signature
from minilab.diffusion_sampling import (
    d3pm_reverse_timesteps,
    sample_categorical,
    sedd_absorbing_step_probs,
)
from minilab.models.d3pm import absorbing_posterior_log_probs
from minilab.online_rl import GRPOTrainConfig
from minilab.preference_alignment import DPOTrainConfig
from minilab.registry import register_trainer
from minilab.trainer import Trainer, _validate_diffusion_trainer_contract, model_aux_loss


@dataclass
class DiffusionGRPOTrainConfig(GRPOTrainConfig):
    diffusion_num_steps: int = 128
    diffusion_temperature: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        require(self.diffusion_num_steps > 0, "diffusion_num_steps must be > 0")
        require(self.diffusion_temperature == 1.0, (
            "Diffusion GRPO uses exact model-policy trajectory ratios; keep diffusion_temperature=1.0"
        ))


@dataclass
class DiffusionVRPOTrainConfig(DPOTrainConfig):
    vrpo_num_samples: int = 4
    vrpo_antithetic: bool = True

    def __post_init__(self):
        super().__post_init__()
        require(self.vrpo_num_samples > 0, "vrpo_num_samples must be > 0")


@dataclass
class DiffusionGRPORollout:
    traces: list
    old_logps: list
    adv: torch.Tensor


@dataclass
class DiffusionRolloutSample:
    trace: list
    logp: torch.Tensor
    completions: torch.Tensor
    completion_mask: torch.Tensor


@dataclass
class DiffusionTraceStep:
    kind: str
    z: torch.Tensor
    t: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor
    t_next: torch.Tensor | None = None
    t_prev: torch.Tensor | None = None


def _save_diffusion_alignment_checkpoint(save_dir, step, ref_model_path, forward_process):
    path = Path(save_dir) / f"step_{step}"
    (path / "ref_path.txt").write_text(ref_model_path)
    forward_process.save(path / "forward_process.json")


class _DiffusionAlignmentCheckpointMixin:
    def save_checkpoint(self):
        super().save_checkpoint()
        _save_diffusion_alignment_checkpoint(self.config.save_dir, self.step, self.ref_model_path, self.fwd)


def _diffusion_pair_losses(
    model,
    fwd,
    chosen_ids,
    chosen_mask,
    t,
    chosen_z,
    chosen_noised,
    rejected_ids,
    rejected_mask,
    rejected_z,
    rejected_noised,
):
    chosen = _diffusion_loss_per_example(
        model, fwd, chosen_ids, chosen_mask, t, chosen_z, chosen_noised
    )
    aux_loss = model_aux_loss(model)
    rejected = _diffusion_loss_per_example(
        model, fwd, rejected_ids, rejected_mask, t, rejected_z, rejected_noised
    )
    aux_loss = aux_loss + model_aux_loss(model)
    return chosen, rejected, 0.5 * aux_loss


def _validate_reference_forward_process(ref_model_path, forward_process, algorithm):
    path = Path(ref_model_path) / "forward_process.json"
    require(path.exists(), (
        f"{algorithm} frozen reference checkpoint is missing {path}; "
        "cannot validate diffusion forward process identity"
    ))
    saved = json.loads(path.read_text())
    require(isinstance(saved, dict), (
        f"{algorithm} frozen reference forward process must be a JSON object"
    ))
    expected = forward_process.to_state()
    missing = set(expected) - set(saved)
    unknown = set(saved) - set(expected)
    require(not missing, f"{algorithm} frozen reference forward process is missing fields: {sorted(missing)}")
    require(not unknown, f"{algorithm} frozen reference forward process has unknown fields: {sorted(unknown)}")
    mismatches = {
        key: (saved[key], expected[key])
        for key in expected
        if saved[key] != expected[key]
    }
    require(not mismatches, (
        f"{algorithm} frozen reference forward process mismatch: "
        f"saved/current={mismatches}"
    ))


@register_trainer("diffusion_dpo")
class DiffusionDPOTrainer(_DiffusionAlignmentCheckpointMixin, Trainer):
    """Diffusion-DPO with an ELBO/loss proxy for log-likelihood.

    For a response pair, the policy is preferred when it assigns lower diffusion
    loss (higher approximate log-probability) to the chosen response than to the
    rejected response, relative to the frozen reference. The policy/reference
    branches share the same sampled timestep; each response is corrupted under
    its own conditional forward process sample.
    """

    _extra_critical_fields = ("dpo_beta",)

    def __init__(self, model, forward_process, train_dataset, config, ref_model_path, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, DPOTrainConfig), "DiffusionDPOTrainer requires DPOTrainConfig")
        if not config.resume_from:
            _validate_diffusion_trainer_contract(model, forward_process)
        fwd_signature = forward_process_signature(forward_process)
        signature = hashlib.sha256((signature + fwd_signature).encode()).hexdigest()
        self.ref_model_path = _trainer_reference_path(ref_model_path, config, "Diffusion DPO")
        _validate_reference_tokenizer(self.ref_model_path, tokenizer_sig, "Diffusion DPO")
        _validate_reference_forward_process(self.ref_model_path, forward_process, "Diffusion DPO")
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.fwd = forward_process
        _validate_diffusion_trainer_contract(self.model, forward_process)
        self.ref_model = _load_reference_model(self.model, self.ref_model_path, self.device, "Diffusion DPO")
        _validate_diffusion_trainer_contract(self.ref_model, forward_process)
        self.beta = config.dpo_beta

    def compute_loss(self, batch):
        model = unwrap_model(self.model)
        chosen_ids = batch["chosen_ids"]
        rejected_ids = batch["rejected_ids"]
        chosen_mask = batch["chosen_loss_mask"]
        rejected_mask = batch["rejected_loss_mask"]
        chosen_valid = batch["chosen_valid_mask"]
        rejected_valid = batch["rejected_valid_mask"]

        chosen_z, chosen_noised, t, _ = model.diffusion_conditional_training_state(
            self.fwd,
            chosen_ids,
            chosen_mask,
            chosen_valid,
            self.device,
        )
        rejected_z, rejected_noised, _, _ = model.diffusion_conditional_training_state(
            self.fwd,
            rejected_ids,
            rejected_mask,
            rejected_valid,
            self.device,
            t=t,
        )

        policy_chosen, policy_rejected, aux_loss = _diffusion_pair_losses(
            self.model,
            self.fwd,
            chosen_ids,
            chosen_mask,
            t,
            chosen_z,
            chosen_noised,
            rejected_ids,
            rejected_mask,
            rejected_z,
            rejected_noised,
        )

        with torch.no_grad():
            ref_chosen = _diffusion_loss_per_example(
                self.ref_model, self.fwd, chosen_ids, chosen_mask, t, chosen_z, chosen_noised
            )
            ref_rejected = _diffusion_loss_per_example(
                self.ref_model, self.fwd, rejected_ids, rejected_mask, t, rejected_z, rejected_noised
            )

        # log p ≈ -diffusion loss. The reference-relative chosen-vs-rejected
        # margin is the DPO classification logit.
        policy_margin = (-policy_chosen) - (-policy_rejected)
        ref_margin = (-ref_chosen) - (-ref_rejected)
        return -F.logsigmoid(self.beta * (policy_margin - ref_margin)).mean() + aux_loss


@register_trainer("diffusion_vrpo")
class DiffusionVRPOTrainer(DiffusionDPOTrainer):
    """Variance-reduced diffusion preference optimization.

    This trainer keeps the Diffusion-DPO objective but averages multiple shared
    ELBO timestep estimates per pair. When enabled, antithetic sampling pairs a
    timestep `t` with `1 - t`, matching VRPO's unbiased variance-reduction
    strategy for diffusion preference gradients.
    """

    _extra_critical_fields = ("dpo_beta", "vrpo_num_samples", "vrpo_antithetic")

    def __init__(self, model, forward_process, train_dataset, config, ref_model_path, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, DiffusionVRPOTrainConfig), "DiffusionVRPOTrainer requires DiffusionVRPOTrainConfig")
        super().__init__(
            model,
            forward_process,
            train_dataset,
            config,
            ref_model_path,
            signature=signature,
            tokenizer_sig=tokenizer_sig,
            eval_dataset=eval_dataset,
        )
        self.num_samples = config.vrpo_num_samples
        self.antithetic = config.vrpo_antithetic

    def compute_loss(self, batch):
        model = unwrap_model(self.model)
        chosen_ids = batch["chosen_ids"]
        rejected_ids = batch["rejected_ids"]
        chosen_mask = batch["chosen_loss_mask"]
        rejected_mask = batch["rejected_loss_mask"]
        chosen_valid = batch["chosen_valid_mask"]
        rejected_valid = batch["rejected_valid_mask"]

        total = torch.tensor(0.0, device=self.device)
        aux_total = torch.tensor(0.0, device=self.device)
        paired_t = None
        for sample_id in range(self.num_samples):
            store_antithetic_pair = False
            if self.antithetic and paired_t is not None:
                t = 1.0 - paired_t
                paired_t = None
            else:
                t = None
                if self.antithetic:
                    store_antithetic_pair = True

            chosen_z, chosen_noised, t, _ = model.diffusion_conditional_training_state(
                self.fwd,
                chosen_ids,
                chosen_mask,
                chosen_valid,
                self.device,
                t=t,
            )
            if store_antithetic_pair:
                paired_t = t
            rejected_z, rejected_noised, _, _ = model.diffusion_conditional_training_state(
                self.fwd,
                rejected_ids,
                rejected_mask,
                rejected_valid,
                self.device,
                t=t,
            )
            policy_chosen, policy_rejected, aux_loss = _diffusion_pair_losses(
                self.model,
                self.fwd,
                chosen_ids,
                chosen_mask,
                t,
                chosen_z,
                chosen_noised,
                rejected_ids,
                rejected_mask,
                rejected_z,
                rejected_noised,
            )
            aux_total = aux_total + aux_loss
            with torch.no_grad():
                ref_chosen = _diffusion_loss_per_example(
                    self.ref_model, self.fwd, chosen_ids, chosen_mask, t, chosen_z, chosen_noised
                )
                ref_rejected = _diffusion_loss_per_example(
                    self.ref_model, self.fwd, rejected_ids, rejected_mask, t, rejected_z, rejected_noised
                )
            policy_margin = (-policy_chosen) - (-policy_rejected)
            ref_margin = (-ref_chosen) - (-ref_rejected)
            total = total + -F.logsigmoid(self.beta * (policy_margin - ref_margin)).mean()
        return total / self.num_samples + aux_total / self.num_samples


@register_trainer("diffusion_grpo")
class DiffusionGRPOTrainer(_DiffusionAlignmentCheckpointMixin, Trainer):
    """GRPO for diffusion LMs using reverse-chain trajectory log-prob ratios."""

    _extra_critical_fields = (
        "grpo_num_generations", "grpo_max_new_tokens",
        "grpo_clip_ratio", "grpo_kl_coef", "grpo_inner_epochs",
        "diffusion_num_steps", "diffusion_temperature",
    )

    def __init__(self, model, forward_process, reward_fn, train_dataset, config, ref_model_path, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, DiffusionGRPOTrainConfig), "DiffusionGRPOTrainer requires DiffusionGRPOTrainConfig")
        require(eval_dataset is None, "Diffusion GRPO has no LM-style eval loss; evaluate task accuracy post-training")
        require(config.eval_every == 0, "Diffusion GRPO has no LM-style eval loss; set eval_every=0")
        require(config.grad_accum_steps == 1, "Diffusion GRPO does not support grad_accum_steps > 1")
        if not config.resume_from:
            _validate_diffusion_trainer_contract(model, forward_process)
        fwd_signature = forward_process_signature(forward_process)
        signature = hashlib.sha256((signature + fwd_signature).encode()).hexdigest()
        self.ref_model_path = _trainer_reference_path(ref_model_path, config, "Diffusion GRPO")
        _validate_reference_tokenizer(self.ref_model_path, tokenizer_sig, "Diffusion GRPO")
        _validate_reference_forward_process(self.ref_model_path, forward_process, "Diffusion GRPO")
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.fwd = forward_process
        _validate_diffusion_trainer_contract(self.model, forward_process)
        self.ref_model = _load_reference_model(self.model, self.ref_model_path, self.device, "Diffusion GRPO")
        _validate_diffusion_trainer_contract(self.ref_model, forward_process)
        require(config.diffusion_num_steps <= forward_process.num_timesteps, (
            "diffusion_num_steps must be <= forward_process.num_timesteps"
        ))
        self.reward_fn = reward_fn
        self.K = config.grpo_num_generations
        self.inner_epochs = config.grpo_inner_epochs
        self.max_new_tokens = config.grpo_max_new_tokens
        self.clip_ratio = config.grpo_clip_ratio
        self.kl_coef = config.grpo_kl_coef
        self.num_steps = config.diffusion_num_steps
        self.temperature = config.diffusion_temperature

    def _configured_scheduler_total_steps(self):
        return self.config.max_steps * self.config.grpo_inner_epochs

    def train(self):
        _rollout_policy_train_loop(self, self.inner_epochs)

    def compute_loss(self, batch):
        raise NotImplementedError("Diffusion GRPO runs its own train loop; compute_loss is not called")

    def _rollout(self, batch):
        prompt_ids = batch["prompt_ids"]
        prompt_lens = batch["prompt_len"]
        require((prompt_lens > 0).all(), "Diffusion GRPO requires prompt_len > 0 for every prompt")

        traces, old_logps, completions, completion_masks = [], [], [], []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.K):
                rollout = _diffusion_rollout_once(
                    self.model,
                    self.fwd,
                    prompt_ids,
                    prompt_lens,
                    self.max_new_tokens,
                    self.num_steps,
                    self.temperature,
                )
                traces.append(rollout.trace)
                old_logps.append(rollout.logp)
                completions.append(rollout.completions)
                completion_masks.append(rollout.completion_mask)

        rewards = torch.stack([
            self.reward_fn(batch, c, m)
            for c, m in zip(completions, completion_masks, strict=True)
        ], dim=1).to(self.device)
        adv = _group_normalized_advantages(rewards)

        self.model.train(was_training)
        return DiffusionGRPORollout(
            traces=traces,
            old_logps=old_logps,
            adv=adv,
        )

    def _policy_loss(self, rollout):
        total = torch.tensor(0.0, device=self.device)
        for k in range(self.K):
            logp, aux = _diffusion_trajectory_logp(self.model, self.fwd, rollout.traces[k], include_aux=True)
            ratio = (logp - rollout.old_logps[k]).clamp(min=-20.0, max=20.0).exp()
            adv_k = rollout.adv[:, k]
            surr1 = ratio * adv_k
            surr2 = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * adv_k
            policy_loss = -torch.min(surr1, surr2).mean()
            with torch.no_grad():
                ref_logp, _ = _diffusion_trajectory_logp(self.ref_model, self.fwd, rollout.traces[k])
            delta = ref_logp - logp
            kl = (delta.exp() - delta - 1).mean()
            total = total + policy_loss + self.kl_coef * kl + aux
        return total / self.K


def _diffusion_rollout_once(model, fwd, prompt_ids, prompt_lens, max_new_tokens, num_steps, temperature):
    core = unwrap_model(model)
    parameterization = core.reverse_parameterization
    require(parameterization in {"clean_logits", "sedd_log_scores", "d3pm_x0_logits"}, (
        f"Diffusion GRPO unsupported reverse_parameterization={parameterization!r}"
    ))
    require(core.supports_unconditional_diffusion_sampling(), (
        "Diffusion GRPO reverse rollouts require a model that can score denoising "
        "steps without clean x_0 context"
    ))
    require(prompt_ids.size(1) <= core.config.max_seq_len, "Diffusion GRPO prompt tensor exceeds model max_seq_len")
    if parameterization == "clean_logits":
        require(fwd.has_terminal_mask_prior(), (
            "Diffusion GRPO with clean_logits starts response slots from [MASK] and requires alpha[-1] = 0"
        ))

    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    prompt_lens = prompt_lens.to(device)
    B, T = prompt_ids.shape
    mask_id = fwd.mask_token_id
    tokens = prompt_ids.clone()
    response_mask = torch.zeros_like(tokens, dtype=torch.bool)
    valid_mask = torch.zeros_like(tokens, dtype=torch.bool)
    completion_mask = torch.zeros(B, max_new_tokens, device=device, dtype=torch.bool)
    completions = torch.zeros(B, max_new_tokens, device=device, dtype=torch.long)
    for b in range(B):
        start = int(prompt_lens[b].item())
        end = min(T, start + max_new_tokens)
        require(end > start, "Diffusion GRPO requires at least one response slot per prompt")
        valid_mask[b, :end] = True
        response_mask[b, start:end] = True
        completion_mask[b, : end - start] = True
    tokens[~valid_mask] = mask_id
    tokens[response_mask] = mask_id

    if parameterization == "clean_logits":
        z, trace, logp = _rollout_clean_logits(model, fwd, tokens, response_mask, num_steps, temperature)
    elif parameterization == "sedd_log_scores":
        z, trace, logp = _rollout_sedd(model, fwd, tokens, response_mask, num_steps, temperature)
    else:
        z, trace, logp = _rollout_d3pm(model, fwd, tokens, response_mask, num_steps, temperature)

    for b in range(B):
        start = int(prompt_lens[b].item())
        count = int(completion_mask[b].sum().item())
        completions[b, :count] = z[b, start : start + count]
    require(trace, "Diffusion GRPO rollout produced no stochastic denoising actions")
    return DiffusionRolloutSample(
        trace=trace,
        logp=logp,
        completions=completions,
        completion_mask=completion_mask,
    )


def _rollout_clean_logits(model, fwd, tokens, response_mask, num_steps, temperature):
    device = tokens.device
    B = tokens.size(0)
    mask_id = fwd.mask_token_id
    z = tokens.clone()
    trace = []
    total_logp = torch.zeros(B, device=device)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        masked = response_mask & (z == mask_id)
        if not masked.any():
            break
        t_now, t_next = timesteps[i], timesteps[i + 1]
        log_probs = _clean_logits_log_probs(model(z, t_now.expand(B)), mask_id, temperature)
        actions = _sample_from_log_probs(log_probs)
        alpha_now = fwd.alpha_at(t_now.unsqueeze(0)).item()
        alpha_next = fwd.alpha_at(t_next.unsqueeze(0)).item()
        unmask_prob = (alpha_next - alpha_now) / (1.0 - alpha_now) if alpha_now < 1.0 else alpha_next
        action_mask = masked & (torch.rand_like(tokens, dtype=torch.float) < unmask_prob)
        if action_mask.any():
            total_logp = total_logp + _sum_action_logp(log_probs, actions, action_mask)
            trace.append(_trace_step("clean", z, t_now, actions, action_mask))
            z = torch.where(action_mask, actions, z)

    still_masked = response_mask & (z == mask_id)
    if still_masked.any():
        t_zero = torch.zeros((), device=device)
        log_probs = _clean_logits_log_probs(model(z, t_zero.expand(B)), mask_id, temperature)
        actions = _sample_from_log_probs(log_probs)
        total_logp = total_logp + _sum_action_logp(log_probs, actions, still_masked)
        trace.append(_trace_step("clean", z, t_zero, actions, still_masked))
        z = torch.where(still_masked, actions, z)
    return z, trace, total_logp


def _rollout_sedd(model, fwd, tokens, response_mask, num_steps, temperature):
    device = tokens.device
    B = tokens.size(0)
    mask_id = fwd.mask_token_id
    z = tokens.clone()
    trace = []
    total_logp = torch.zeros(B, device=device)
    eps = 1.0 / fwd.num_timesteps
    timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)
    for i in range(num_steps):
        masked = response_mask & (z == mask_id)
        if not masked.any():
            break
        t_now, t_next = timesteps[i], timesteps[i + 1]
        log_scores = model(z, t_now.expand(B))
        dsigma = fwd.get_sigma(t_now.unsqueeze(0)).to(device) - fwd.get_sigma(t_next.unsqueeze(0)).to(device)
        probs = sedd_absorbing_step_probs(log_scores, z, dsigma, mask_id, temperature)
        log_probs = _safe_log_probs(probs)
        actions = sample_categorical(probs)
        total_logp = total_logp + _sum_action_logp(log_probs, actions, masked)
        trace.append(_trace_step("sedd", z, t_now, actions, masked, t_next=t_next))
        z = torch.where(response_mask, actions, tokens)

    still_masked = response_mask & (z == mask_id)
    if still_masked.any():
        t_eps = timesteps[-1]
        log_scores = model(z, t_eps.expand(B))
        sigma = fwd.get_sigma(t_eps.unsqueeze(0)).to(device)
        probs = sedd_absorbing_step_probs(log_scores, z, sigma, mask_id, temperature, drop_mask=True)
        log_probs = _safe_log_probs(probs)
        actions = sample_categorical(probs)
        total_logp = total_logp + _sum_action_logp(log_probs, actions, still_masked)
        trace.append(_trace_step("sedd_final", z, t_eps, actions, still_masked))
        z = torch.where(still_masked, actions, z)
    return z, trace, total_logp


def _rollout_d3pm(model, fwd, tokens, response_mask, num_steps, temperature):
    device = tokens.device
    B = tokens.size(0)
    mask_id = fwd.mask_token_id
    z = tokens.clone()
    trace = []
    total_logp = torch.zeros(B, device=device)
    timesteps = d3pm_reverse_timesteps(fwd, num_steps, device)
    for i in range(num_steps):
        masked = response_mask & (z == mask_id)
        if not masked.any():
            break
        t_now, t_prev = timesteps[i], timesteps[i + 1]
        logits = model(z, t_now.expand(B))
        log_probs = absorbing_posterior_log_probs(logits, z, t_now.expand(B), t_prev.expand(B), fwd, mask_id)
        log_probs = _temperature_log_probs(log_probs, temperature)
        actions = _sample_from_log_probs(log_probs)
        total_logp = total_logp + _sum_action_logp(log_probs, actions, masked)
        trace.append(_trace_step("d3pm", z, t_now, actions, masked, t_prev=t_prev))
        z = torch.where(masked, actions, z)

    still_masked = response_mask & (z == mask_id)
    require(not still_masked.any(), "D3PM reverse chain left masked tokens at t=0")
    return z, trace, total_logp


def _diffusion_trajectory_logp(model, fwd, trace, include_aux=False):
    B = trace[0].z.size(0)
    device = trace[0].z.device
    total_logp = torch.zeros(B, device=device)
    total_aux = torch.tensor(0.0, device=device)
    aux_steps = 0
    for step in trace:
        z = step.z
        t = step.t
        actions = step.actions
        action_mask = step.action_mask
        if step.kind == "clean":
            log_probs = _clean_logits_log_probs(model(z, t.expand(B)), fwd.mask_token_id, temperature=1.0)
        elif step.kind == "sedd":
            require(step.t_next is not None, "SEDD diffusion trace step requires t_next")
            log_scores = model(z, t.expand(B))
            dsigma = fwd.get_sigma(t.unsqueeze(0)).to(device) - fwd.get_sigma(step.t_next.unsqueeze(0)).to(device)
            probs = sedd_absorbing_step_probs(log_scores, z, dsigma, fwd.mask_token_id, temperature=1.0)
            log_probs = _safe_log_probs(probs)
        elif step.kind == "sedd_final":
            log_scores = model(z, t.expand(B))
            sigma = fwd.get_sigma(t.unsqueeze(0)).to(device)
            probs = sedd_absorbing_step_probs(log_scores, z, sigma, fwd.mask_token_id, temperature=1.0, drop_mask=True)
            log_probs = _safe_log_probs(probs)
        elif step.kind == "d3pm":
            require(step.t_prev is not None, "D3PM diffusion trace step requires t_prev")
            logits = model(z, t.expand(B))
            log_probs = absorbing_posterior_log_probs(
                logits, z, t.expand(B), step.t_prev.expand(B), fwd, fwd.mask_token_id
            )
        else:
            raise ValueError(f"Unknown diffusion trace step kind: {step.kind!r}")
        total_logp = total_logp + _sum_action_logp(log_probs, actions, action_mask)
        if include_aux:
            total_aux = total_aux + model_aux_loss(model)
            aux_steps += 1
    if include_aux:
        total_aux = total_aux / aux_steps
    return total_logp, total_aux


def _trace_step(kind, z, t, actions, action_mask, t_next=None, t_prev=None):
    return DiffusionTraceStep(
        kind=kind,
        z=z.detach().clone(),
        t=t.detach().clone(),
        actions=actions.detach().clone(),
        action_mask=action_mask.detach().clone(),
        t_next=t_next.detach().clone() if t_next is not None else None,
        t_prev=t_prev.detach().clone() if t_prev is not None else None,
    )


def _clean_logits_log_probs(logits, mask_id, temperature):
    require(temperature > 0, "trajectory log-prob scoring requires temperature > 0")
    logits = logits.clone()
    logits[:, :, mask_id] = float("-inf")
    return F.log_softmax(logits / temperature, dim=-1)


def _temperature_log_probs(log_probs, temperature):
    require(temperature > 0, "trajectory log-prob scoring requires temperature > 0")
    if temperature == 1.0:
        return log_probs
    scaled = log_probs / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def _safe_log_probs(probs):
    tiny = torch.finfo(probs.dtype).tiny
    return probs.clamp(min=tiny).log()


def _sample_from_log_probs(log_probs):
    return sample_categorical(log_probs.exp())


def _sum_action_logp(log_probs, actions, action_mask):
    selected = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return (selected * action_mask.to(selected.dtype)).sum(dim=-1)
