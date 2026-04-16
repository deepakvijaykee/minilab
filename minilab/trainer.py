import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from minilab.config import BaseConfig
from minilab.nn.optimizers import Lion, Muon
from minilab.registry import register_trainer


@dataclass
class TrainConfig(BaseConfig):
    max_steps: int = 10000
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    dtype: str = "bfloat16"
    optimizer: str = "adamw"
    lr_schedule: str = "cosine"
    compile: bool = False
    seed: int = 42
    aim_repo: str = ""  # path to aim repo, e.g. "." or "runs/aim"
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    save_dir: str = "checkpoints"
    eval_steps: int = 50
    dpo_beta: float = 0.1
    grpo_num_generations: int = 4
    grpo_max_new_tokens: int = 128
    grpo_clip_ratio: float = 0.2
    grpo_kl_coef: float = 0.1
    grpo_inner_epochs: int = 4
    resume_from: str = ""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Fields on TrainConfig whose value must match at resume time: changing them
# silently redefines the training trajectory. Anything else (log_every, save_every,
# aim_repo, save_dir, resume_from, compile, dtype) is free to differ.
# `max_steps` is intentionally excluded: extending the training horizon on resume is a
# normal workflow. The LR profile relative to max_steps does shift when it changes, but
# that is a property of the cosine schedule being parameterized by max_steps, not a
# correctness issue we can guard against without blocking all extension resumes.
_RESUME_CRITICAL_CONFIG_FIELDS = (
    "batch_size", "lr", "weight_decay", "warmup_steps",
    "grad_accum_steps", "optimizer", "lr_schedule", "seed",
)


def run_signature(tokenizer, dataset_desc, seq_len):
    """Hash of resume-critical inputs owned by the caller (tokenizer identity,
    dataset identity, tokenization length). Persisted with the checkpoint and
    asserted on resume — a mismatch means the resumed run would silently be a
    different experiment. `dataset_desc` is a caller-provided dict like
    {"name": "tinystories", "split": "train", "max_examples": 50000}."""
    payload = json.dumps({
        "tokenizer": tokenizer._get_state(),
        "dataset": dataset_desc,
        "seq_len": seq_len,
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class Trainer:
    # Subclass-declared TrainConfig fields that are critical for *that* trainer's
    # objective (e.g. DPO's beta, GRPO's clip ratio) and must match on resume.
    # Generic base fields live in _RESUME_CRITICAL_CONFIG_FIELDS above.
    _extra_critical_fields: tuple = ()

    def __init__(self, model, train_dataset, config, *, signature, eval_dataset=None):
        """`signature` is a caller-owned hash of resume-critical inputs not captured
        by TrainConfig (tokenizer identity, dataset identity, seq_len). Build it with
        run_signature(...). On resume it is asserted equal to the saved value."""
        self.config = config
        self.signature = signature
        set_seed(config.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

        if config.compile:
            self.model = torch.compile(self.model)

        DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        assert config.dtype in DTYPES, f"Unknown dtype: '{config.dtype}'. Available: {sorted(DTYPES)}"
        self.dtype = DTYPES[config.dtype]
        self.scaler = torch.amp.GradScaler(enabled=config.dtype == "float16")
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        # Dedicated generator for the shuffle RNG so resume can restore batch order
        # independently of the global torch RNG (which advances with every forward pass).
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(config.seed)
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False,
                                       generator=self.loader_generator)
        self.eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, drop_last=False) if eval_dataset else None
        self.step = 0
        self.batches_consumed = 0
        # Snapshot of loader_generator state taken immediately before the current
        # epoch's iter() call. That is the state that must be restored on resume so
        # the new iter() produces the same shuffle for the epoch we're in the middle of.
        self.loader_rng_epoch_start = self.loader_generator.get_state()
        self.aim_run = None

        if config.resume_from:
            # Validate resume-critical inputs BEFORE loading anything — a mismatch
            # means the checkpoint belongs to a different experiment and silently
            # continuing would produce results that look like the old run but aren't.
            meta_path = Path(config.resume_from) / "run_meta.json"
            assert meta_path.exists(), f"Missing run_meta.json at {meta_path}; cannot validate resume"
            saved_meta = json.loads(meta_path.read_text())
            assert saved_meta["signature"] == signature, (
                f"Resume signature mismatch: checkpoint was built with a different "
                f"tokenizer/dataset/seq_len. Saved={saved_meta['signature'][:12]}... "
                f"Current={signature[:12]}..."
            )
            saved_cfg = saved_meta["config"]
            critical = _RESUME_CRITICAL_CONFIG_FIELDS + type(self)._extra_critical_fields
            mismatches = [(k, saved_cfg[k], getattr(config, k))
                          for k in critical if saved_cfg[k] != getattr(config, k)]
            assert not mismatches, f"Resume config mismatch on critical fields: {mismatches}"
            state = torch.load(Path(config.resume_from) / "trainer_state.pt", map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            if "scaler" in state:
                self.scaler.load_state_dict(state["scaler"])
            self.step = state["step"]
            # Faithful continuation: restore RNG states + loader generator + fast-forward
            # the iterator so the next batch is the one that would have come next.
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if state.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
            self.loader_rng_epoch_start = state["loader_rng_epoch_start"].cpu()
            self.batches_consumed = state["batches_consumed"]
            self.loader_generator.set_state(self.loader_rng_epoch_start)
            self.train_iter = iter(self.train_loader)
            for _ in range(self.batches_consumed % len(self.train_loader)):
                next(self.train_iter)
            print(f"Resumed from {config.resume_from} at step {self.step} (batches_consumed={self.batches_consumed})")
        else:
            self.loader_rng_epoch_start = self.loader_generator.get_state()
            self.train_iter = iter(self.train_loader)

        if config.aim_repo:
            from aim import Run
            self.aim_run = Run(repo=config.aim_repo)
            self.aim_run["config"] = config.to_dict()

    def _build_optimizer(self):
        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": self.config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(groups, lr=self.config.lr, betas=(0.9, 0.95))
        if self.config.optimizer == "lion":
            return Lion(groups, lr=self.config.lr, weight_decay=self.config.weight_decay)
        if self.config.optimizer == "muon":
            # NS orthogonalization on hidden-layer matrices only; skip for vocab-sized embeddings
            embed_keys = {"tok_emb", "lm_head", "score_head"}
            hidden = [p for n, p in self.model.named_parameters() if p.dim() >= 2 and not any(k in n for k in embed_keys)]
            embeds = [p for n, p in self.model.named_parameters() if p.dim() >= 2 and any(k in n for k in embed_keys)]
            biases = [p for p in self.model.parameters() if p.dim() < 2]
            return Muon([
                {"params": hidden},
                {"params": embeds, "ns_iters": 0},
                {"params": biases, "ns_iters": 0},
            ], lr=self.config.lr)
        raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _build_scheduler(self):
        warmup = self.config.warmup_steps
        total = self.config.max_steps
        schedule = self.config.lr_schedule
        assert total > warmup, f"max_steps ({total}) must exceed warmup_steps ({warmup})"

        def lr_fn(step):
            # LambdaLR is constructed with last_epoch=0, and we call scheduler.step()
            # AFTER each optimizer.step(). Without this +1 shift, optimizer step k
            # would use lr_fn(k-1); in particular the very first update uses lr_fn(0)
            # which is 0 during warmup — a no-op first step and the entire profile
            # is shifted by one. Treat `step` as the 1-indexed optimizer step about
            # to run, so warmup_steps=W gives LRs (1/W, 2/W, ..., 1.0) over the
            # first W updates.
            step = step + 1
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / (total - warmup)
            if schedule == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            if schedule == "linear":
                return 1.0 - progress
            if schedule == "constant":
                return 1.0
            if schedule == "wsd":
                if progress < 0.8:
                    return 1.0
                return 0.5 * (1.0 + math.cos(math.pi * (progress - 0.8) / 0.2))
            raise ValueError(f"Unknown lr_schedule: {schedule}")

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_fn)

    def _next_batch(self):
        batch = next(self.train_iter, None)
        if batch is None:
            # Defensive fallback: normally the eager rollover below advances to the
            # next epoch the moment the current one exhausts, so this branch only
            # fires the first time through if the loader was empty, which we assert.
            assert False, "train_loader produced no batches"
        self.batches_consumed += 1
        # Eager rollover: if we just consumed the last batch of the epoch, snapshot
        # the next epoch's RNG start and rebuild the iterator now. This keeps the
        # invariant "loader_rng_epoch_start + train_iter point at the NEXT batch"
        # true at every save point — including exactly on an epoch boundary, where
        # a lazy rollover would resume to a replay of the finished epoch.
        if self.batches_consumed % len(self.train_loader) == 0:
            self.loader_rng_epoch_start = self.loader_generator.get_state()
            self.train_iter = iter(self.train_loader)
        return {k: v.to(self.device) for k, v in batch.items()}

    def compute_loss(self, batch):
        raise NotImplementedError

    def train(self):
        self.model.train()
        pbar = tqdm(range(self.step + 1, self.config.max_steps + 1), desc="Training")

        for self.step in pbar:
            total_loss = 0.0
            for _ in range(self.config.grad_accum_steps):
                batch = self._next_batch()
                with torch.autocast(self.device, dtype=self.dtype):
                    loss = self.compute_loss(batch) / self.config.grad_accum_steps
                self.scaler.scale(loss).backward()
                total_loss += loss.item()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            lr = self.scheduler.get_last_lr()[0]
            if self.step % self.config.log_every == 0:
                pbar.set_postfix(loss=f"{total_loss:.4f}", lr=f"{lr:.2e}")
                if self.aim_run:
                    self.aim_run.track(total_loss, name="loss", step=self.step)
                    self.aim_run.track(lr, name="lr", step=self.step)

            if self.eval_loader and self.config.eval_every > 0 and self.step % self.config.eval_every == 0:
                eval_loss = self.evaluate()
                print(f"\n  step {self.step} eval loss: {eval_loss:.4f}")
                if self.aim_run:
                    self.aim_run.track(eval_loss, name="eval_loss", step=self.step)
                self.model.train()

            if self.config.save_every > 0 and self.step % self.config.save_every == 0:
                self.save_checkpoint()

        if self.config.save_every > 0:
            self.save_checkpoint()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total, count = 0.0, 0
        for i, batch in enumerate(self.eval_loader):
            if i >= self.config.eval_steps:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.autocast(self.device, dtype=self.dtype):
                total += self.compute_loss(batch).item()
            count += 1
        assert count > 0, "Eval loader produced no batches"
        return total / count

    def save_checkpoint(self):
        path = Path(self.config.save_dir) / f"step_{self.step}"
        self.model.save(path)
        torch.save(
            {
                "step": self.step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loader_rng_epoch_start": self.loader_rng_epoch_start,
                "batches_consumed": self.batches_consumed,
            },
            path / "trainer_state.pt",
        )
        (path / "run_meta.json").write_text(json.dumps({
            "signature": self.signature,
            "config": self.config.to_dict(),
        }, indent=2))
        print(f"\n  saved {path}")


@register_trainer("lm")
class LMTrainer(Trainer):
    def compute_loss(self, batch):
        _, loss = self.model(batch["input_ids"], batch["labels"])
        return loss


@register_trainer("diffusion")
class DiffusionTrainer(Trainer):
    def __init__(self, model, forward_process, train_dataset, config, *, signature, eval_dataset=None):
        # Bind the noise schedule into the resume signature so a checkpoint built
        # with one schedule cannot be silently resumed under another.
        signature = hashlib.sha256((signature + forward_process.schedule).encode()).hexdigest()
        super().__init__(model, train_dataset, config, signature=signature, eval_dataset=eval_dataset)
        self.fwd = forward_process

    def save_checkpoint(self):
        super().save_checkpoint()
        self.fwd.save(Path(self.config.save_dir) / f"step_{self.step}" / "forward_process.json")

    def compute_loss(self, batch):
        x_0 = batch["input_ids"]
        # Resample t until at least one position is masked. With tiny batch_size and
        # small alpha_t samples, q_sample can produce an all-clean batch; running the
        # loss on that would return a detached zero whose backward() raises.
        for _ in range(10):
            t = torch.rand(x_0.size(0), device=self.device) * 0.999 + 0.001
            z_t, mask = self.fwd.q_sample(x_0, t)
            if mask.any():
                break
        assert mask.any(), "q_sample produced no masked positions after 10 retries"
        output = self.model(z_t, t)
        return self.model.compute_loss(output, x_0, mask, t, self.fwd)
