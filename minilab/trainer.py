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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, model, train_dataset, config, eval_dataset=None):
        self.config = config
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
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        self.eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, drop_last=True) if eval_dataset else None
        self.train_iter = iter(self.train_loader)
        self.step = 0
        self.aim_run = None

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
            return Muon(groups, lr=self.config.lr)
        raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _build_scheduler(self):
        warmup = self.config.warmup_steps
        total = self.config.max_steps
        schedule = self.config.lr_schedule
        assert total > warmup, f"max_steps ({total}) must exceed warmup_steps ({warmup})"

        def lr_fn(step):
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
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
        return {k: v.to(self.device) for k, v in batch.items()}

    def compute_loss(self, batch):
        raise NotImplementedError

    def train(self):
        self.model.train()
        pbar = tqdm(range(self.step, self.config.max_steps), desc="Training")

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

            if self.eval_loader and self.config.eval_every > 0 and self.step > 0 and self.step % self.config.eval_every == 0:
                eval_loss = self.evaluate()
                print(f"\n  step {self.step} eval loss: {eval_loss:.4f}")
                if self.aim_run:
                    self.aim_run.track(eval_loss, name="eval_loss", step=self.step)
                self.model.train()

            if self.config.save_every > 0 and self.step > 0 and self.step % self.config.save_every == 0:
                self.save_checkpoint()

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
            {"step": self.step, "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()},
            path / "trainer_state.pt",
        )
        print(f"\n  saved {path}")


@register_trainer("lm")
class LMTrainer(Trainer):
    def compute_loss(self, batch):
        _, loss = self.model(batch["input_ids"], batch["labels"])
        return loss


@register_trainer("diffusion")
class DiffusionTrainer(Trainer):
    def __init__(self, model, forward_process, train_dataset, config, eval_dataset=None):
        super().__init__(model, train_dataset, config, eval_dataset)
        self.fwd = forward_process

    def compute_loss(self, batch):
        x_0 = batch["input_ids"]
        t = torch.rand(x_0.size(0), device=self.device) * 0.999 + 0.001
        z_t, mask = self.fwd.q_sample(x_0, t)
        output = self.model(z_t, t)
        return self.model.compute_loss(output, x_0, mask, t, self.fwd)
