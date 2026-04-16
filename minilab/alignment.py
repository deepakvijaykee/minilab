from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from minilab.generation import generate
from minilab.registry import register_trainer
from minilab.trainer import Trainer


@register_trainer("sft")
class SFTTrainer(Trainer):
    def compute_loss(self, batch):
        _, loss = self.model(batch["input_ids"], batch["labels"])
        return loss


@register_trainer("dpo")
class DPOTrainer(Trainer):
    _extra_critical_fields = ("dpo_beta",)

    def __init__(self, model, train_dataset, config, ref_model, ref_model_path, *, signature, eval_dataset=None):
        """ref_model/ref_model_path are required and must point at the original (pre-DPO)
        policy. The earlier fallback of deepcopy(model) silently drifted the reference on
        resume — the resumed policy is not the reference the DPO objective is defined
        against. The path is persisted in save_checkpoint so resume restores it."""
        super().__init__(model, train_dataset, config, signature=signature, eval_dataset=eval_dataset)
        self.ref_model = ref_model.to(self.device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model_path = ref_model_path
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

    def __init__(self, model, reward_fn, train_dataset, config, ref_model, ref_model_path, *, signature, eval_dataset=None):
        # PromptDataset yields prompts, not (input, label) pairs — so the generic
        # LM-style eval loss is undefined here. The paper metric is held-out task
        # accuracy, which scripts/grpo.py runs post-training. Reject eval hooks
        # explicitly so this can't silently call compute_loss during training.
        assert eval_dataset is None, "GRPO has no LM-style eval loss; evaluate accuracy post-training"
        assert config.eval_every == 0, "GRPO has no LM-style eval loss; set eval_every=0"
        # PPO-style ratio semantics require that old_logp and new_logp come from
        # the same stochastic graph. Dropout (and any other train-mode stochasticity)
        # contaminates the ratio independently of the parameter update.
        assert getattr(model.config, "dropout", 0.0) == 0.0, \
            "GRPO requires dropout=0.0 so the PPO ratio is not contaminated by dropout noise"
        # Gradient accumulation is not implemented in the GRPO inner-epoch loop.
        # Each inner epoch steps the optimizer; accumulating would reshape the
        # update schedule in a way that silently changes the algorithm.
        assert config.grad_accum_steps == 1, "GRPO does not support grad_accum_steps > 1"
        super().__init__(model, train_dataset, config, signature=signature, eval_dataset=eval_dataset)
        self.ref_model = ref_model.to(self.device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model_path = ref_model_path
        self.reward_fn = reward_fn
        self.K = config.grpo_num_generations
        assert self.K > 1, "GRPO requires grpo_num_generations > 1"
        self.inner_epochs = config.grpo_inner_epochs
        assert self.inner_epochs >= 1, "grpo_inner_epochs must be >= 1"
        self.max_new_tokens = config.grpo_max_new_tokens
        self.clip_ratio = config.grpo_clip_ratio
        self.kl_coef = config.grpo_kl_coef

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
        self.model.train()
        pbar = tqdm(range(self.step + 1, self.config.max_steps + 1), desc="Training")
        for self.step in pbar:
            batch = self._next_batch()
            rollout = self._rollout(batch)
            total_loss = 0.0
            for _ in range(self.inner_epochs):
                with torch.autocast(self.device, dtype=self.dtype):
                    loss = self._policy_loss(rollout)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item()
            self.scheduler.step()

            lr = self.scheduler.get_last_lr()[0]
            if self.step % self.config.log_every == 0:
                avg_loss = total_loss / self.inner_epochs
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                if self.aim_run:
                    self.aim_run.track(avg_loss, name="loss", step=self.step)
                    self.aim_run.track(lr, name="lr", step=self.step)
            if self.config.save_every > 0 and self.step % self.config.save_every == 0:
                self.save_checkpoint()

        if self.config.save_every > 0:
            self.save_checkpoint()

    def _rollout(self, batch):
        """Sample K completions per prompt under the current policy, compute
        group-relative advantages, and freeze the old-policy log-probs that the
        clipped surrogate ratio will be measured against."""
        prompt_ids = batch["prompt_ids"]
        prompt_lens = batch["prompt_len"]
        B = prompt_ids.size(0)

        completions, comp_lens = [], []
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.K):
                gens = []
                for b in range(B):
                    plen = prompt_lens[b].item()
                    out = generate(self.model, prompt_ids[b : b + 1, :plen], max_new_tokens=self.max_new_tokens, temperature=1.0)
                    gens.append(out[0, plen:])
                lens = [g.size(0) for g in gens]
                max_clen = max(lens)
                padded = torch.zeros(B, max_clen, device=self.device, dtype=torch.long)
                for b, g in enumerate(gens):
                    padded[b, : g.size(0)] = g
                completions.append(padded)
                comp_lens.append(lens)
        self.model.train()

        rewards = torch.stack([self.reward_fn(batch, c) for c in completions], dim=1).to(self.device)
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
            with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype):
                old_token_logps.append(_token_logp(self.model, full_pad, lab_pad)[0])

        return {"seqs": seqs, "label_seqs": label_seqs, "old_token_logps": old_token_logps, "adv": adv}

    def _policy_loss(self, rollout):
        total = torch.tensor(0.0, device=self.device)
        for k in range(self.K):
            logp, mask = _token_logp(self.model, rollout["seqs"][k], rollout["label_seqs"][k])
            ratio = (logp - rollout["old_token_logps"][k]).exp()
            adv_k = rollout["adv"][:, k : k + 1]
            surr1 = ratio * adv_k
            surr2 = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * adv_k
            policy_loss = -(torch.min(surr1, surr2) * mask).sum() / mask.sum()
            with torch.no_grad():
                ref_logp, _ = _token_logp(self.ref_model, rollout["seqs"][k], rollout["label_seqs"][k])
            delta = ref_logp - logp
            kl = ((delta.exp() - delta - 1) * mask).sum() / mask.sum()
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
