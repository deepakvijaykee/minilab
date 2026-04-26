import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from minilab.base import BaseModel, unwrap_model
from minilab.checks import require
from minilab.generation import generate
from minilab.registry import register_trainer
from minilab.trainer import TrainConfig, Trainer


def _validated_reference_path(ref_model_path, algorithm):
    require(ref_model_path is not None, f"{algorithm} requires a frozen reference checkpoint")
    ref_model_path = str(ref_model_path).strip()
    require(ref_model_path, f"{algorithm} requires a frozen reference checkpoint")
    path = Path(ref_model_path).expanduser().resolve()
    require(path.exists(), f"{algorithm} frozen reference checkpoint does not exist: {path}")
    missing = [name for name in ("config.json", "model.pt") if not (path / name).exists()]
    require(not missing, f"{algorithm} frozen reference checkpoint {path} is missing: {missing}")
    return str(path)


def _resume_reference_path(resume_from, algorithm):
    path = Path(resume_from) / "ref_path.txt"
    require(path.exists(), f"{algorithm} resume is missing {path}; cannot restore frozen reference")
    ref_path = path.read_text().strip()
    require(ref_path, f"{algorithm} resume has empty frozen reference path in {path}")
    return _validated_reference_path(ref_path, algorithm)


def resolve_reference_path(checkpoint, resume_from, algorithm):
    if resume_from:
        return _resume_reference_path(resume_from, algorithm)
    return _validated_reference_path(checkpoint, algorithm)


def _trainer_reference_path(ref_model_path, config, algorithm):
    ref_path = _validated_reference_path(ref_model_path, algorithm)
    if config.resume_from:
        saved_ref_path = _resume_reference_path(config.resume_from, algorithm)
        require(ref_path == saved_ref_path, (
            f"{algorithm} resume reference mismatch: checkpoint expects {saved_ref_path}, "
            f"caller supplied {ref_path}"
        ))
    return ref_path


def _validate_reference_tokenizer(ref_model_path, tokenizer_sig, algorithm):
    require(tokenizer_sig, f"{algorithm} requires tokenizer_sig to validate the frozen reference tokenizer")
    meta_path = Path(ref_model_path) / "run_meta.json"
    require(meta_path.exists(), (
        f"{algorithm} frozen reference checkpoint is missing {meta_path}; "
        "cannot validate tokenizer identity"
    ))
    saved_meta = json.loads(meta_path.read_text())
    require("tokenizer_signature" in saved_meta, (
        f"{algorithm} frozen reference checkpoint is missing tokenizer_signature in {meta_path}"
    ))
    require(saved_meta["tokenizer_signature"] == tokenizer_sig, (
        f"{algorithm} frozen reference tokenizer mismatch: "
        f"saved={saved_meta['tokenizer_signature'][:12]}... current={tokenizer_sig[:12]}..."
    ))


def _load_reference_model(model, ref_model_path, device, algorithm):
    model = unwrap_model(model)
    require(isinstance(model, BaseModel), (
        f"{algorithm} requires a BaseModel trainable model so the frozen reference "
        f"can be loaded from the validated checkpoint path"
    ))
    ref_model = type(model).load(ref_model_path, device=device).eval()
    require(ref_model.config.to_dict() == model.config.to_dict(), (
        f"{algorithm} frozen reference config does not match the trainable model config"
    ))
    for p in ref_model.parameters():
        p.requires_grad = False
    return ref_model


@dataclass
class DPOTrainConfig(TrainConfig):
    dpo_beta: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        require(self.dpo_beta > 0, "dpo_beta must be > 0")


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


@register_trainer("sft")
class SFTTrainer(Trainer):
    def compute_loss(self, batch):
        _, loss = self.model(batch["input_ids"], batch["labels"])
        return loss


@register_trainer("dpo")
class DPOTrainer(Trainer):
    _extra_critical_fields = ("dpo_beta",)

    def __init__(self, model, train_dataset, config, ref_model_path, *, signature, tokenizer_sig="", eval_dataset=None):
        """ref_model_path points at the original (pre-DPO) policy. The trainer loads
        the frozen reference from that validated path, so callers cannot accidentally
        pair a correct path with the wrong in-memory object."""
        require(isinstance(config, DPOTrainConfig), "DPOTrainer requires DPOTrainConfig")
        self.ref_model_path = _trainer_reference_path(ref_model_path, config, "DPO")
        _validate_reference_tokenizer(self.ref_model_path, tokenizer_sig, "DPO")
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.ref_model = _load_reference_model(self.model, self.ref_model_path, self.device, "DPO")
        self.beta = config.dpo_beta

    def save_checkpoint(self):
        super().save_checkpoint()
        (Path(self.config.save_dir) / f"step_{self.step}" / "ref_path.txt").write_text(self.ref_model_path)

    def compute_loss(self, batch):
        chosen_logp = _seq_logp(self.model, batch["chosen_ids"], batch["chosen_labels"])
        rejected_logp = _seq_logp(self.model, batch["rejected_ids"], batch["rejected_labels"])

        with torch.no_grad():
            ref_chosen_logp = _seq_logp(self.ref_model, batch["chosen_ids"], batch["chosen_labels"])
            ref_rejected_logp = _seq_logp(self.ref_model, batch["rejected_ids"], batch["rejected_labels"])

        chosen_reward = self.beta * (chosen_logp - ref_chosen_logp)
        rejected_reward = self.beta * (rejected_logp - ref_rejected_logp)
        return -F.logsigmoid(chosen_reward - rejected_reward).mean()


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

        completions, completion_masks, comp_lens = [], [], []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.K):
                gens = []
                for b in range(B):
                    plen = prompt_lens[b].item()
                    # GRPO ratios below are full-softmax token ratios, so the
                    # behavior policy must not apply top-k/top-p/repetition filters.
                    out = generate(
                        self.model,
                        prompt_ids[b : b + 1, :plen],
                        max_new_tokens=self.max_new_tokens,
                        temperature=1.0,
                        top_k=0,
                        top_p=1.0,
                        repetition_penalty=1.0,
                    )
                    gens.append(out[0, plen:])
                lens = [g.size(0) for g in gens]
                max_clen = max(lens)
                padded = torch.zeros(B, max_clen, device=self.device, dtype=torch.long)
                completion_mask = torch.zeros(B, max_clen, device=self.device, dtype=torch.bool)
                for b, g in enumerate(gens):
                    padded[b, : g.size(0)] = g
                    completion_mask[b, : g.size(0)] = True
                require(completion_mask.any(dim=1).all(), "GRPO requires at least one generated token per prompt")
                completions.append(padded)
                completion_masks.append(completion_mask)
                comp_lens.append(lens)

        rewards = torch.stack([
            self.reward_fn(batch, c, m)
            for c, m in zip(completions, completion_masks, strict=True)
        ], dim=1).to(self.device)
        adv = rewards - rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        adv = torch.where(std > 0, adv / std, torch.zeros_like(adv))

        seqs, label_seqs, old_token_logps = [], [], []
        for k in range(self.K):
            full_list, label_list = [], []
            for b in range(B):
                plen = prompt_lens[b].item()
                clen = comp_lens[k][b]
                full_b = torch.cat([prompt_ids[b, :plen], completions[k][b, :clen]])
                lab_b = torch.full_like(full_b, -100)
                lab_b[plen - 1 : -1] = full_b[plen:]
                full_list.append(full_b)
                label_list.append(lab_b)
            max_len = max(f.size(0) for f in full_list)
            full_pad = torch.zeros(B, max_len, device=self.device, dtype=torch.long)
            lab_pad = torch.full((B, max_len), -100, device=self.device, dtype=torch.long)
            for b in range(B):
                full_pad[b, : full_list[b].size(0)] = full_list[b]
                lab_pad[b, : label_list[b].size(0)] = label_list[b]
            seqs.append(full_pad)
            label_seqs.append(lab_pad)
            with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype, enabled=self.dtype != torch.float32):
                old_token_logps.append(_generation_context_token_logp(self.model, full_pad, lab_pad)[0])

        self.model.train(was_training)
        return {
            "seqs": seqs,
            "label_seqs": label_seqs,
            "old_token_logps": old_token_logps,
            "completion_masks": completion_masks,
            "adv": adv,
        }

    def _policy_loss(self, rollout):
        total = torch.tensor(0.0, device=self.device)
        for k in range(self.K):
            logp, mask = _generation_context_token_logp(self.model, rollout["seqs"][k], rollout["label_seqs"][k])
            ratio = (logp - rollout["old_token_logps"][k]).exp()
            adv_k = rollout["adv"][:, k : k + 1]
            surr1 = ratio * adv_k
            surr2 = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * adv_k
            policy_loss = -_masked_response_mean(torch.min(surr1, surr2), mask, "GRPO policy loss")
            with torch.no_grad():
                ref_logp, _ = _generation_context_token_logp(self.ref_model, rollout["seqs"][k], rollout["label_seqs"][k])
            delta = ref_logp - logp
            kl = _masked_response_mean(delta.exp() - delta - 1, mask, "GRPO KL")
            total = total + policy_loss + self.kl_coef * kl
        return total / self.K


def _seq_logp(model, input_ids, labels):
    token_logp, mask = _token_logp(model, input_ids, labels)
    return (token_logp * mask).sum(dim=-1)


def _token_logp(model, input_ids, labels):
    logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    mask = (labels != -100).float()
    safe_targets = labels.where(mask.bool(), torch.zeros_like(labels))
    token_logp = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return token_logp, mask


def _masked_response_mean(values, mask, context):
    token_counts = mask.sum(dim=-1)
    require((token_counts > 0).all(), f"{context} requires at least one generated token per response")
    return ((values * mask).sum(dim=-1) / token_counts).mean()


def _generation_context_token_logp(model, input_ids, labels):
    """Token log-probs under the same cropped context contract used by generate()."""
    max_ctx = _model_max_seq_len(model, "GRPO token log-prob scoring")
    if input_ids.size(1) <= max_ctx:
        return _token_logp(model, input_ids, labels)

    mask = labels != -100
    token_logp = None
    safe_targets = labels.where(mask, torch.zeros_like(labels))
    for pos in range(input_ids.size(1)):
        active = mask[:, pos]
        if not active.any():
            continue
        start = max(0, pos + 1 - max_ctx)
        context = input_ids[active, start : pos + 1]
        logits, _ = model(context)
        selected = F.log_softmax(logits[:, -1], dim=-1).gather(
            -1,
            safe_targets[active, pos].unsqueeze(-1),
        ).squeeze(-1)
        if token_logp is None:
            token_logp = torch.zeros(labels.shape, device=input_ids.device, dtype=selected.dtype)
        token_logp[active, pos] = selected

    if token_logp is None:
        token_logp = torch.zeros(labels.shape, device=input_ids.device, dtype=torch.float32)
    return token_logp, mask.float()


def _model_max_seq_len(model, context):
    model = unwrap_model(model)
    require(isinstance(model, BaseModel), f"{context} requires a BaseModel")
    max_seq_len = model.config.max_seq_len
    require(max_seq_len > 0, f"{context} requires model.config.max_seq_len > 0")
    return max_seq_len
