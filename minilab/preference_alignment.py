from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from minilab.alignment_common import (
    ReferenceCheckpointMixin,
    _kto_kl_batch,
    _load_reference_model,
    _log1mexp,
    _seq_avg_logp,
    _seq_logp,
    _trainer_reference_path,
    _validate_reference_tokenizer,
)
from minilab.checks import require
from minilab.data import KTOBalancedBatchSampler, KTODataset
from minilab.registry import register_trainer
from minilab.trainer import TrainConfig, Trainer, model_aux_loss, supervised_lm_batch_loss


@dataclass
class DPOTrainConfig(TrainConfig):
    dpo_beta: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        require(self.dpo_beta > 0, "dpo_beta must be > 0")


@dataclass
class CPOTrainConfig(DPOTrainConfig):
    cpo_alpha: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        require(self.cpo_alpha >= 0, "cpo_alpha must be >= 0")


@dataclass
class SimPOTrainConfig(DPOTrainConfig):
    simpo_gamma: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        require(self.simpo_gamma >= 0, "simpo_gamma must be >= 0")


@dataclass
class RePOTrainConfig(TrainConfig):
    repo_margin: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        require(self.repo_margin >= 0, "repo_margin must be >= 0")


@dataclass
class ORPOTrainConfig(TrainConfig):
    orpo_beta: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        require(self.orpo_beta > 0, "orpo_beta must be > 0")


@dataclass
class KTOTrainConfig(DPOTrainConfig):
    kto_desirable_weight: float = 1.0
    kto_undesirable_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        require(self.batch_size > 1, "KTO requires batch_size > 1 to estimate the KL term")
        require(self.batch_size % 2 == 0, "KTO requires an even batch_size for balanced minibatches")
        require(self.kto_desirable_weight > 0, "kto_desirable_weight must be > 0")
        require(self.kto_undesirable_weight > 0, "kto_undesirable_weight must be > 0")


@register_trainer("sft")
class SFTTrainer(Trainer):
    def compute_loss(self, batch):
        return supervised_lm_batch_loss(self.model, batch)


def _preference_pair_logps(model, batch, *, average=False):
    score = _seq_avg_logp if average else _seq_logp
    chosen = score(model, batch["chosen_ids"], batch["chosen_labels"])
    aux_loss = model_aux_loss(model)
    rejected = score(model, batch["rejected_ids"], batch["rejected_labels"])
    aux_loss = aux_loss + model_aux_loss(model)
    return chosen, rejected, 0.5 * aux_loss


@register_trainer("dpo")
class DPOTrainer(ReferenceCheckpointMixin, Trainer):
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

    def compute_loss(self, batch):
        chosen_logp, rejected_logp, aux_loss = _preference_pair_logps(self.model, batch)

        with torch.no_grad():
            ref_chosen_logp = _seq_logp(self.ref_model, batch["chosen_ids"], batch["chosen_labels"])
            ref_rejected_logp = _seq_logp(self.ref_model, batch["rejected_ids"], batch["rejected_labels"])

        chosen_reward = self.beta * (chosen_logp - ref_chosen_logp)
        rejected_reward = self.beta * (rejected_logp - ref_rejected_logp)
        return -F.logsigmoid(chosen_reward - rejected_reward).mean() + aux_loss


@register_trainer("ipo")
class IPOTrainer(ReferenceCheckpointMixin, Trainer):
    """Identity Preference Optimization with sequence log-probability margins."""

    _extra_critical_fields = ("dpo_beta",)

    def __init__(self, model, train_dataset, config, ref_model_path, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, DPOTrainConfig), "IPOTrainer requires DPOTrainConfig")
        self.ref_model_path = _trainer_reference_path(ref_model_path, config, "IPO")
        _validate_reference_tokenizer(self.ref_model_path, tokenizer_sig, "IPO")
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.ref_model = _load_reference_model(self.model, self.ref_model_path, self.device, "IPO")
        self.beta = config.dpo_beta

    def compute_loss(self, batch):
        chosen_logp, rejected_logp, aux_loss = _preference_pair_logps(self.model, batch)
        with torch.no_grad():
            ref_chosen = _seq_logp(self.ref_model, batch["chosen_ids"], batch["chosen_labels"])
            ref_rejected = _seq_logp(self.ref_model, batch["rejected_ids"], batch["rejected_labels"])
        logit = (chosen_logp - rejected_logp) - (ref_chosen - ref_rejected)
        return (logit - 1.0 / (2.0 * self.beta)).square().mean() + aux_loss


@register_trainer("cpo")
class CPOTrainer(Trainer):
    """Contrastive Preference Optimization without a frozen reference model."""

    _extra_critical_fields = ("dpo_beta", "cpo_alpha")

    def __init__(self, model, train_dataset, config, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, CPOTrainConfig), "CPOTrainer requires CPOTrainConfig")
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.beta = config.dpo_beta
        self.alpha = config.cpo_alpha

    def compute_loss(self, batch):
        chosen_logp, rejected_logp, aux_loss = _preference_pair_logps(self.model, batch)
        pref_loss = -F.logsigmoid(self.beta * (chosen_logp - rejected_logp)).mean()
        bc_loss = -chosen_logp.mean()
        return pref_loss + self.alpha * bc_loss + aux_loss


@register_trainer("simpo")
class SimPOTrainer(Trainer):
    """Simple Preference Optimization with length-normalized, reference-free rewards."""

    _extra_critical_fields = ("dpo_beta", "simpo_gamma")

    def __init__(self, model, train_dataset, config, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, SimPOTrainConfig), "SimPOTrainer requires SimPOTrainConfig")
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.beta = config.dpo_beta
        self.gamma = config.simpo_gamma

    def compute_loss(self, batch):
        chosen_avg, rejected_avg, aux_loss = _preference_pair_logps(self.model, batch, average=True)
        return -F.logsigmoid(self.beta * (chosen_avg - rejected_avg) - self.gamma).mean() + aux_loss


@register_trainer("orpo")
class ORPOTrainer(Trainer):
    """Odds Ratio Preference Optimization: chosen NLL plus odds-ratio penalty."""

    _extra_critical_fields = ("orpo_beta",)

    def __init__(self, model, train_dataset, config, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, ORPOTrainConfig), "ORPOTrainer requires ORPOTrainConfig")
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.beta = config.orpo_beta

    def compute_loss(self, batch):
        chosen_avg, rejected_avg, aux_loss = _preference_pair_logps(self.model, batch, average=True)
        log_odds = (chosen_avg - rejected_avg) - (_log1mexp(chosen_avg) - _log1mexp(rejected_avg))
        odds_loss = -self.beta * F.logsigmoid(log_odds).mean()
        chosen_nll = -chosen_avg.mean()
        return chosen_nll + odds_loss + aux_loss


@register_trainer("repo")
class RePOTrainer(Trainer):
    """ReLU-based Preference Optimization.

    RePO is the max-margin, reference-free limiting case of SimPO: it optimizes
    the length-normalized chosen-vs-rejected log-probability gap with one margin
    hyperparameter and a ReLU loss.
    """

    _extra_critical_fields = ("repo_margin",)

    def __init__(self, model, train_dataset, config, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, RePOTrainConfig), "RePOTrainer requires RePOTrainConfig")
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.margin = config.repo_margin

    def compute_loss(self, batch):
        chosen_avg, rejected_avg, aux_loss = _preference_pair_logps(self.model, batch, average=True)
        return F.relu(self.margin - (chosen_avg - rejected_avg)).mean() + aux_loss


@register_trainer("kto")
class KTOTrainer(ReferenceCheckpointMixin, Trainer):
    """Kahneman-Tversky Optimization from binary desirable/undesirable examples."""

    _extra_critical_fields = ("dpo_beta", "kto_desirable_weight", "kto_undesirable_weight")

    def __init__(self, model, train_dataset, config, ref_model_path, *, signature, tokenizer_sig="", eval_dataset=None):
        require(isinstance(config, KTOTrainConfig), "KTOTrainer requires KTOTrainConfig")
        self.ref_model_path = _trainer_reference_path(ref_model_path, config, "KTO")
        _validate_reference_tokenizer(self.ref_model_path, tokenizer_sig, "KTO")
        super().__init__(model, train_dataset, config, signature=signature, tokenizer_sig=tokenizer_sig, eval_dataset=eval_dataset)
        self.ref_model = _load_reference_model(self.model, self.ref_model_path, self.device, "KTO")
        self.beta = config.dpo_beta
        self.desirable_weight = config.kto_desirable_weight
        self.undesirable_weight = config.kto_undesirable_weight

    def _build_data_loaders(self, train_dataset, eval_dataset):
        require(isinstance(train_dataset, KTODataset), "KTOTrainer requires KTODataset")
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=KTOBalancedBatchSampler(
                train_dataset,
                self.config.batch_size,
                generator=self.loader_generator,
                shuffle=True,
            ),
        )
        eval_loader = None
        if eval_dataset is not None:
            require(isinstance(eval_dataset, KTODataset), "KTOTrainer eval_dataset must be KTODataset")
            eval_loader = DataLoader(
                eval_dataset,
                batch_sampler=KTOBalancedBatchSampler(
                    eval_dataset,
                    self.config.batch_size,
                    shuffle=False,
                ),
            )
        return train_loader, eval_loader

    def compute_loss(self, batch):
        labels = batch["preference_label"].bool()
        require(labels.any() and (~labels).any(), "KTO minibatch requires both desirable and undesirable examples")

        policy_logp = _seq_logp(self.model, batch["input_ids"], batch["labels"])
        aux_loss = model_aux_loss(self.model)
        with torch.no_grad():
            ref_logp = _seq_logp(self.ref_model, batch["input_ids"], batch["labels"])
            kl_ids, kl_labels = _kto_kl_batch(batch)
            policy_kl = _seq_logp(self.model, kl_ids, kl_labels)
            ref_kl = _seq_logp(self.ref_model, kl_ids, kl_labels)
            kl = (policy_kl - ref_kl).mean().clamp(min=0)

        reward = policy_logp - ref_logp
        desirable_loss = 1.0 - torch.sigmoid(self.beta * (reward[labels] - kl))
        undesirable_loss = 1.0 - torch.sigmoid(self.beta * (kl - reward[~labels]))
        loss = torch.cat([
            self.desirable_weight * desirable_loss,
            self.undesirable_weight * undesirable_loss,
        ]).mean()
        return loss + aux_loss
