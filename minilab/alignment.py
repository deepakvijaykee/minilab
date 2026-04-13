import copy

import torch
import torch.nn.functional as F

from minilab.registry import register_trainer
from minilab.trainer import Trainer


@register_trainer("sft")
class SFTTrainer(Trainer):
    def compute_loss(self, batch):
        _, loss = self.model(batch["input_ids"], batch["labels"])
        return loss


@register_trainer("dpo")
class DPOTrainer(Trainer):

    def __init__(self, model, train_dataset, config, eval_dataset=None, ref_model=None):
        super().__init__(model, train_dataset, config, eval_dataset)
        self.ref_model = (ref_model or copy.deepcopy(model)).to(self.device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.beta = config.dpo_beta

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
    """Generate K completions per prompt, compute group-relative advantage,
    optimize with clipped surrogate. No critic needed."""

    def __init__(self, model, reward_fn, train_dataset, config, eval_dataset=None, ref_model=None):
        super().__init__(model, train_dataset, config, eval_dataset)
        self.ref_model = (ref_model or copy.deepcopy(model)).to(self.device).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.reward_fn = reward_fn
        self.K = config.grpo_num_generations
        self.max_new_tokens = config.grpo_max_new_tokens
        self.clip_ratio = config.grpo_clip_ratio
        self.kl_coef = config.grpo_kl_coef

    def compute_loss(self, batch):
        from minilab.generation import generate

        prompt_ids = batch["prompt_ids"]
        prompt_len = prompt_ids.size(1)

        completions = []
        with torch.no_grad():
            for _ in range(self.K):
                out = generate(self.model, prompt_ids, max_new_tokens=self.max_new_tokens, temperature=1.0)
                completions.append(out[:, prompt_len:])

        rewards = torch.stack([self.reward_fn(batch, c) for c in completions], dim=1).to(self.device)

        adv = rewards - rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True)
        adv = torch.where(std > 0, adv / std, torch.zeros_like(adv))

        total_loss = torch.tensor(0.0, device=self.device)
        for k in range(self.K):
            full_ids = torch.cat([prompt_ids, completions[k]], dim=1)
            labels = torch.full_like(full_ids, -100)
            labels[:, prompt_len - 1 : -1] = full_ids[:, prompt_len:]

            logp = _seq_logp(self.model, full_ids, labels)
            with torch.no_grad():
                old_logp = _seq_logp(self.ref_model, full_ids, labels)

            ratio = (logp - old_logp).exp()
            a = adv[:, k]
            surr1 = ratio * a
            surr2 = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * a
            total_loss = total_loss - torch.min(surr1, surr2).mean() + self.kl_coef * (old_logp - logp).mean()

        return total_loss / self.K


def _seq_logp(model, input_ids, labels):
    logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != -100
    safe_targets = labels.where(mask, torch.zeros_like(labels))
    token_logp = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)
