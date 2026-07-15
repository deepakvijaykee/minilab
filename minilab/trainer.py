import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from minilab.base import BaseModel, unwrap_model
from minilab.checks import require, require_finite_number, require_integer, require_integer_fields
from minilab.config import BaseConfig
from minilab.nn.optimizers import DEFAULT_SOFT_MUON_POWER, Lion, Muon
from minilab.registry import register_trainer


_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
_OPTIMIZERS = {"adamw", "lion", "muon", "soft_muon"}
_LR_SCHEDULES = {"cosine", "linear", "constant", "wsd"}
_DECAYING_LR_SCHEDULES = {"cosine", "linear", "wsd"}


def autocast_context(device, dtype):
    if isinstance(dtype, str):
        require(dtype in _DTYPES, f"Unknown dtype: '{dtype}'. Available: {sorted(_DTYPES)}")
        dtype = _DTYPES[dtype]
    device_type = torch.device(device).type
    return torch.autocast(device_type, dtype=dtype, enabled=dtype != torch.float32)


@dataclass
class TrainConfig(BaseConfig):
    max_steps: int = 10000
    batch_size: int = 32
    lr: float = 3e-4
    muon_lr: float = 0.02
    soft_muon_power: float = DEFAULT_SOFT_MUON_POWER
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    dtype: str = "bfloat16"
    optimizer: str = "adamw"
    lr_schedule: str = "cosine"
    qk_clip_threshold: float = 0.0
    qk_clip_balance: float = 0.5
    compile: bool = False
    seed: int = 42
    aim_repo: str = ""  # path to aim repo, e.g. "." or "runs/aim"
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    save_dir: str = "checkpoints"
    eval_steps: int = 50
    resume_from: str = ""
    token_superposition_size: int = 1
    token_superposition_steps: int = 0
    rl_metrics_every: int = 0
    rl_trace_samples: int = 0

    def __post_init__(self):
        require_integer_fields(self, (
            "max_steps", "batch_size", "warmup_steps", "grad_accum_steps", "seed",
            "log_every", "eval_every", "save_every", "eval_steps",
            "token_superposition_size", "token_superposition_steps",
            "rl_metrics_every", "rl_trace_samples",
        ))
        for name in (
            "lr", "muon_lr", "soft_muon_power", "weight_decay", "max_grad_norm",
            "qk_clip_threshold", "qk_clip_balance",
        ):
            require_finite_number(getattr(self, name), name)
        require(self.max_steps > 0, "max_steps must be > 0")
        require(self.batch_size > 0, "batch_size must be > 0")
        require(self.lr >= 0, "lr must be >= 0")
        require(self.muon_lr >= 0, "muon_lr must be >= 0")
        require(self.soft_muon_power == DEFAULT_SOFT_MUON_POWER, (
            "soft_muon_power currently supports the fixed p=0.4 coefficient profile"
        ))
        require(self.weight_decay >= 0, "weight_decay must be >= 0")
        require(self.warmup_steps >= 0, "warmup_steps must be >= 0")
        require(self.max_grad_norm > 0, "max_grad_norm must be > 0")
        require(self.grad_accum_steps > 0, "grad_accum_steps must be > 0")
        require(self.dtype in _DTYPES, f"Unknown dtype: '{self.dtype}'. Available: {sorted(_DTYPES)}")
        require(self.optimizer in _OPTIMIZERS, f"Unknown optimizer: '{self.optimizer}'. Available: {sorted(_OPTIMIZERS)}")
        require(self.lr_schedule in _LR_SCHEDULES, f"Unknown lr_schedule: '{self.lr_schedule}'. Available: {sorted(_LR_SCHEDULES)}")
        require(self.qk_clip_threshold >= 0, "qk_clip_threshold must be >= 0")
        require(0.0 <= self.qk_clip_balance <= 1.0, "qk_clip_balance must be in [0, 1]")
        require(self.log_every > 0, "log_every must be > 0")
        require(self.eval_every >= 0, "eval_every must be >= 0")
        require(self.save_every >= 0, "save_every must be >= 0")
        require(self.eval_steps > 0, "eval_steps must be > 0")
        require(self.token_superposition_size > 0, "token_superposition_size must be > 0")
        require(self.token_superposition_steps >= 0, "token_superposition_steps must be >= 0")
        require(self.rl_metrics_every >= 0, "rl_metrics_every must be >= 0")
        require(self.rl_trace_samples >= 0, "rl_trace_samples must be >= 0")
        require(self.token_superposition_size == 1 or self.token_superposition_steps > 0, (
            "token_superposition_size > 1 requires token_superposition_steps > 0"
        ))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _decay_progress(update, warmup, total):
    if warmup == 0:
        progress = 0.0 if total == 1 else (update - 1) / (total - 1)
    else:
        progress = (update - warmup) / (total - warmup)
    return min(1.0, max(0.0, progress))


def optimizer_decay_groups(model, params, weight_decay):
    core = unwrap_model(model)
    params = list(params)
    no_weight_decay_names = set(core.no_weight_decay_parameter_names())
    weight_decay_names = set(core.weight_decay_parameter_names())
    no_weight_decay_ids = {
        id(param)
        for name, param in core.named_parameters()
        if name in no_weight_decay_names
    }
    weight_decay_ids = {
        id(param)
        for name, param in core.named_parameters()
        if name in weight_decay_names
    }
    decay = [
        p for p in params
        if id(p) not in no_weight_decay_ids and (p.dim() >= 2 or id(p) in weight_decay_ids)
    ]
    no_decay = [
        p for p in params
        if id(p) in no_weight_decay_ids or (p.dim() < 2 and id(p) not in weight_decay_ids)
    ]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# Fields on TrainConfig whose value must match at resume time: changing them
# silently redefines the training trajectory. Anything else (log_every, save_every,
# aim_repo, save_dir, resume_from, compile) is free to differ.
# `max_steps` is intentionally excluded: callers may stop earlier or later within
# the saved scheduler horizon. Decaying schedules reject horizon extension on
# resume because continuing past the saved horizon would train at the schedule
# endpoint.
_RESUME_CRITICAL_CONFIG_FIELDS = (
    "batch_size", "lr", "weight_decay", "warmup_steps",
    "max_grad_norm", "grad_accum_steps", "dtype", "optimizer", "lr_schedule", "seed",
    "muon_lr", "soft_muon_power", "qk_clip_threshold", "qk_clip_balance",
    "token_superposition_size", "token_superposition_steps",
)

def tokenizer_signature(tokenizer):
    payload = json.dumps(tokenizer._get_state(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def run_signature(tokenizer, dataset_desc, seq_len):
    """Hash of resume-critical inputs owned by the caller (tokenizer identity,
    dataset identity, tokenization length). Persisted with the checkpoint and
    asserted on resume — a mismatch means the resumed run would silently be a
    different experiment. `dataset_desc` is a caller-provided dict like
    {"name": "tinystories", "split": "train", "max_examples": 50000}."""
    payload = json.dumps({
        "tokenizer": tokenizer_signature(tokenizer),
        "dataset": dataset_desc,
        "seq_len": seq_len,
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _fill_legacy_resume_defaults(saved_cfg, current_values):
    if "soft_muon_power" in saved_cfg:
        return
    if current_values["soft_muon_power"] != DEFAULT_SOFT_MUON_POWER:
        return
    if "optimizer" in saved_cfg and saved_cfg["optimizer"] == "soft_muon":
        return
    saved_cfg["soft_muon_power"] = DEFAULT_SOFT_MUON_POWER


def validate_checkpoint_tokenizer(checkpoint, tokenizer):
    meta_path = Path(checkpoint) / "run_meta.json"
    require(meta_path.exists(), f"Missing run_meta.json at {meta_path}; cannot validate tokenizer identity")
    saved_meta = json.loads(meta_path.read_text())
    require(isinstance(saved_meta, dict), f"Checkpoint run metadata at {meta_path} must be a JSON object")
    require("tokenizer_signature" in saved_meta, (
        f"Missing tokenizer_signature in {meta_path}; checkpoint cannot be safely used with an external tokenizer"
    ))
    current = tokenizer_signature(tokenizer)
    require(saved_meta["tokenizer_signature"] == current, (
        f"Tokenizer mismatch for checkpoint {checkpoint}: "
        f"saved={saved_meta['tokenizer_signature'][:12]}... current={current[:12]}..."
    ))


class Trainer:
    # Subclass-declared TrainConfig fields that are critical for *that* trainer's
    # objective (e.g. DPO's beta, GRPO's clip ratio) and must match on resume.
    # Generic base fields live in _RESUME_CRITICAL_CONFIG_FIELDS above.
    _extra_critical_fields: tuple = ()

    def __init__(self, model, train_dataset, config, *, signature, tokenizer_sig="", eval_dataset=None):
        """`signature` is a caller-owned hash of resume-critical inputs not captured
        by TrainConfig (tokenizer identity, dataset identity, seq_len). Build it with
        run_signature(...). On resume it is asserted equal to the saved value."""
        self.config = config
        self.signature = signature
        self.tokenizer_signature = tokenizer_sig

        # Trainer owns training-loop RNG. Callers that construct fresh models set
        # this same seed before model construction so initialization is reproducible.
        set_seed(config.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        require(isinstance(unwrap_model(model), BaseModel), "Trainer requires a BaseModel")
        self.model = self._prepare_model(model)
        self.dtype = _DTYPES[config.dtype]
        self.scaler = torch.amp.GradScaler(self.device, enabled=config.dtype == "float16")

        self._resume_scheduler_total_steps = None
        if config.resume_from:
            self._validate_resume_metadata()

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        # Dedicated generator for the shuffle RNG so resume can restore batch order
        # independently of the global torch RNG (which advances with every forward pass).
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(config.seed)
        self.train_loader, self.eval_loader = self._build_data_loaders(train_dataset, eval_dataset)
        self.step = 0
        self.batches_consumed = 0
        self._last_checkpoint_step = None
        self._run_metrics_active = False
        self._run_metrics = None
        self._run_metrics_started_at = None
        self._run_metrics_started_at_iso = ""
        self._run_metrics_start_step = 0
        self._run_metrics_start_batches = 0
        self._run_metrics_observed_examples = 0
        self._run_metrics_observed_token_slots = 0
        # Snapshot of loader_generator state taken immediately before the current
        # epoch's iter() call. That is the state that must be restored on resume so
        # the new iter() produces the same shuffle for the epoch we're in the middle of.
        self.loader_rng_epoch_start = self.loader_generator.get_state()
        self.aim_run = None

        if config.resume_from:
            self._restore_training_state()
        else:
            self.train_iter = iter(self.train_loader)

        self._start_aim_run()

    def _prepare_model(self, model):
        model = model.to(self.device)
        if self.config.resume_from:
            model = self._load_model_for_resume(model)
        if self.config.qk_clip_threshold > 0:
            require(unwrap_model(model).supports_qk_clip(), (
                "qk_clip_threshold requires a model with QK-Clip-capable attention"
            ))
        unwrap_model(model).set_qk_clip_recording(self.config.qk_clip_threshold > 0)
        if self.config.compile:
            model = torch.compile(model)
        return model

    def _validate_resume_metadata(self):
        # Validate resume-critical inputs BEFORE loading optimizer/scheduler
        # state. A mismatch means the checkpoint belongs to a different
        # experiment and continuing would silently redefine the run.
        meta_path = Path(self.config.resume_from) / "run_meta.json"
        require(meta_path.exists(), f"Missing run_meta.json at {meta_path}; cannot validate resume")
        saved_meta = json.loads(meta_path.read_text())
        require(isinstance(saved_meta, dict), f"Resume metadata at {meta_path} must be a JSON object")
        required_meta = {"signature", "scheduler_total_steps", "config"}
        missing_meta = required_meta - set(saved_meta)
        require(not missing_meta, f"Missing resume metadata fields: {sorted(missing_meta)}")
        require(saved_meta["signature"] == self.signature, (
            f"Resume signature mismatch: checkpoint was built with a different "
            f"tokenizer/dataset/seq_len. Saved={saved_meta['signature'][:12]}... "
            f"Current={self.signature[:12]}..."
        ))
        self._resume_scheduler_total_steps = saved_meta["scheduler_total_steps"]
        require_integer(self._resume_scheduler_total_steps, "scheduler_total_steps")
        require(self._resume_scheduler_total_steps > 0, "scheduler_total_steps must be > 0")
        saved_cfg = saved_meta["config"]
        require(isinstance(saved_cfg, dict), f"Resume config metadata at {meta_path} must be a JSON object")
        critical = _RESUME_CRITICAL_CONFIG_FIELDS + type(self)._extra_critical_fields
        current_values = self.config.to_dict()
        _fill_legacy_resume_defaults(saved_cfg, current_values)
        missing_critical = [k for k in critical if k not in saved_cfg]
        require(not missing_critical, f"Resume metadata missing critical config fields: {missing_critical}")
        mismatches = []
        for k in critical:
            saved_value = saved_cfg[k]
            current_value = current_values[k]
            if saved_value != current_value:
                mismatches.append((k, saved_value, current_value))
        require(not mismatches, f"Resume config mismatch on critical fields: {mismatches}")
        self._validate_resume_scheduler_horizon()

    def _build_data_loaders(self, train_dataset, eval_dataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            generator=self.loader_generator,
        )
        eval_loader = (
            DataLoader(eval_dataset, batch_size=self.config.batch_size, drop_last=False)
            if eval_dataset is not None
            else None
        )
        return train_loader, eval_loader

    def _restore_training_state(self):
        state = torch.load(Path(self.config.resume_from) / "trainer_state.pt", map_location=self.device, weights_only=False)
        required_state = {
            "step", "optimizer", "scheduler", "scaler", "python_rng", "numpy_rng",
            "torch_rng", "cuda_rng", "loader_rng_epoch_start", "batches_consumed",
        }
        missing_state = required_state - set(state)
        require(not missing_state, f"Missing trainer state fields: {sorted(missing_state)}")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.step = state["step"]
        # Faithful continuation: restore RNG states + loader generator + fast-forward
        # the iterator so the next batch is the one that would have come next.
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        self.loader_rng_epoch_start = state["loader_rng_epoch_start"].cpu()
        self.batches_consumed = state["batches_consumed"]
        self.loader_generator.set_state(self.loader_rng_epoch_start)
        self.train_iter = iter(self.train_loader)
        for _ in range(self.batches_consumed % len(self.train_loader)):
            next(self.train_iter)
        print(f"Resumed from {self.config.resume_from} at step {self.step} (batches_consumed={self.batches_consumed})")

    def _start_aim_run(self):
        if self.config.aim_repo:
            # Aim is an optional logging extra, so importing it belongs at the
            # logging boundary instead of making trainer import require the extra.
            from aim import Run
            self.aim_run = Run(repo=self.config.aim_repo)
            self.aim_run["config"] = self.config.to_dict()

    def _load_model_for_resume(self, model):
        model = unwrap_model(model)
        gradient_checkpointing = model._gradient_checkpointing
        loaded = type(model).load(self.config.resume_from, device=self.device)
        if gradient_checkpointing:
            loaded.gradient_checkpointing_enable()
        return loaded

    def _build_optimizer(self):
        model = unwrap_model(self.model)
        if self.config.optimizer in {"muon", "soft_muon"}:
            hidden, aux_matrices, biases = model.muon_parameter_groups()
            soft_muon = self.config.optimizer == "soft_muon"
            return Muon([
                {
                    "params": hidden,
                    "use_muon": True,
                    "soft_muon": soft_muon,
                    "soft_muon_power": self.config.soft_muon_power,
                    "lr": self.config.muon_lr,
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": aux_matrices,
                    "use_muon": False,
                    "soft_muon": False,
                    "lr": self.config.lr,
                    "weight_decay": self.config.weight_decay,
                },
                {
                    "params": biases,
                    "use_muon": False,
                    "soft_muon": False,
                    "lr": self.config.lr,
                    "weight_decay": 0.0,
                },
            ], lr=self.config.muon_lr, soft_muon=soft_muon, soft_muon_power=self.config.soft_muon_power)
        groups = optimizer_decay_groups(model, self.model.parameters(), self.config.weight_decay)
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(groups, lr=self.config.lr, betas=(0.9, 0.95))
        if self.config.optimizer == "lion":
            return Lion(groups, lr=self.config.lr, weight_decay=self.config.weight_decay)
        raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _build_scheduler(self):
        warmup = self.config.warmup_steps
        total = self._scheduler_total_steps()
        schedule = self.config.lr_schedule
        require(warmup >= 0, f"warmup_steps ({warmup}) must be >= 0")
        require(total > 0, f"scheduler total steps ({total}) must be > 0")
        require(total > warmup, f"scheduler total steps ({total}) must exceed warmup_steps ({warmup})")

        def lr_fn(step):
            # LambdaLR is initialized before the first optimizer update, and we
            # call scheduler.step() only after an optimizer update actually runs.
            # Treat this callback's 0-indexed value as the 1-indexed update whose
            # LR is about to be used.
            update = step + 1
            if warmup > 0 and update <= warmup:
                return update / warmup
            if schedule == "cosine":
                progress = _decay_progress(update, warmup, total)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            if schedule == "linear":
                progress = _decay_progress(update, warmup, total)
                return 1.0 - progress
            if schedule == "constant":
                return 1.0
            if schedule == "wsd":
                progress = _decay_progress(update, warmup, total)
                if progress < 0.8:
                    return 1.0
                return 0.5 * (1.0 + math.cos(math.pi * (progress - 0.8) / 0.2))
            raise ValueError(f"Unknown lr_schedule: {schedule}")

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_fn)

    def _scheduler_total_steps(self):
        if self._resume_scheduler_total_steps is not None:
            return self._resume_scheduler_total_steps
        return self._configured_scheduler_total_steps()

    def _configured_scheduler_total_steps(self):
        return self.config.max_steps

    def _validate_resume_scheduler_horizon(self):
        configured_total = self._configured_scheduler_total_steps()
        if self.config.lr_schedule in _DECAYING_LR_SCHEDULES:
            require(configured_total <= self._resume_scheduler_total_steps, (
                f"Cannot extend {self.config.lr_schedule} scheduler on resume: "
                f"configured scheduler total steps ({configured_total}) exceeds saved horizon "
                f"({self._resume_scheduler_total_steps}). Start a new run for a longer decay."
            ))

    def _cuda_metrics_enabled(self):
        return self.device == "cuda" and torch.cuda.is_available()

    def _cuda_device_index(self):
        return torch.cuda.current_device()

    def _cuda_device_summary(self):
        if not self._cuda_metrics_enabled():
            return {
                "available": False,
                "measurement": "unavailable",
            }

        idx = self._cuda_device_index()
        props = torch.cuda.get_device_properties(idx)
        summary = {
            "available": True,
            "measurement": "torch.cuda allocator stats",
            "device_index": idx,
            "device_name": props.name,
            "total_memory_bytes": props.total_memory,
            "total_memory_gb": props.total_memory / 1024 ** 3,
        }
        if hasattr(torch.cuda, "get_allocator_backend"):
            summary["allocator_backend"] = torch.cuda.get_allocator_backend()
        return summary

    def _begin_run_metrics(self):
        self._run_metrics_active = True
        self._run_metrics_started_at = time.perf_counter()
        self._run_metrics_started_at_iso = datetime.now(timezone.utc).isoformat()
        self._run_metrics_start_step = self.step
        self._run_metrics_start_batches = self.batches_consumed
        self._run_metrics_observed_examples = 0
        self._run_metrics_observed_token_slots = 0
        if self._cuda_metrics_enabled():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def _record_batch_metrics(self, batch):
        if not self._run_metrics_active:
            return

        for value in batch.values():
            if torch.is_tensor(value) and value.dim() > 0:
                self._run_metrics_observed_examples += value.size(0)
                break

        token_keys = ("input_ids", "labels", "chosen_ids", "rejected_ids", "prompt_ids")
        for key in token_keys:
            value = batch.get(key)
            if torch.is_tensor(value) and value.dim() >= 2:
                self._run_metrics_observed_token_slots += value.numel()

    def _finish_run_metrics(self, status, error=None):
        if not self._run_metrics_active:
            return None

        if self._cuda_metrics_enabled():
            torch.cuda.synchronize()

        duration = time.perf_counter() - self._run_metrics_started_at
        metrics = {
            "status": status,
            "error": error,
            "started_at": self._run_metrics_started_at_iso,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
            "device": self.device,
            "trainer": type(self).__name__,
            "model": type(unwrap_model(self.model)).__name__,
            "parameters": unwrap_model(self.model).num_parameters(),
            "step": {
                "start": self._run_metrics_start_step,
                "end": self.step,
                "measured": max(0, self.step - self._run_metrics_start_step),
                "configured_max": self.config.max_steps,
            },
            "batches": {
                "start": self._run_metrics_start_batches,
                "end": self.batches_consumed,
                "measured": max(0, self.batches_consumed - self._run_metrics_start_batches),
            },
            "observed_examples": self._run_metrics_observed_examples,
            "observed_token_slots": self._run_metrics_observed_token_slots,
            "token_slots_per_second": (
                self._run_metrics_observed_token_slots / duration if duration > 0 else None
            ),
            "config": self.config.to_dict(),
            "cuda": self._cuda_device_summary(),
        }

        if self._cuda_metrics_enabled():
            metrics["cuda"].update({
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "max_memory_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
                "memory_allocated_bytes": torch.cuda.memory_allocated(),
                "memory_reserved_bytes": torch.cuda.memory_reserved(),
            })

        self._run_metrics = metrics
        self._run_metrics_active = False
        self._write_run_metrics(metrics)
        self._print_run_metrics(metrics)
        return metrics

    def _write_run_metrics(self, metrics):
        save_dir = Path(self.config.save_dir)
        checkpoint_dir = save_dir / f"step_{self.step}"
        save_dir.mkdir(parents=True, exist_ok=True)
        targets = [save_dir / "run_metrics.json"]
        if checkpoint_dir.exists():
            targets.append(checkpoint_dir / "run_metrics.json")
        for target in targets:
            target.write_text(json.dumps(metrics, indent=2) + "\n")

    def _print_run_metrics(self, metrics):
        duration = metrics["duration_seconds"]
        print(f"\n  run metrics: duration={duration:.1f}s")
        cuda = metrics["cuda"]
        if cuda.get("available"):
            alloc = cuda["max_memory_allocated_gb"]
            reserved = cuda["max_memory_reserved_gb"]
            print(f"  peak CUDA memory: allocated={alloc:.2f} GB reserved={reserved:.2f} GB")
        if Path(self.config.save_dir, "run_metrics.json").exists():
            print(f"  wrote {Path(self.config.save_dir) / 'run_metrics.json'}")

    def _finish_failed_run_metrics(self, exc):
        return self._finish_run_metrics(
            "failed",
            {"type": type(exc).__name__, "message": str(exc)},
        )

    def _run_train_loop_with_metrics(self, loop_fn):
        self._begin_run_metrics()
        try:
            loop_fn()
        except KeyboardInterrupt as exc:
            self._finish_failed_run_metrics(exc)
            raise
        except Exception as exc:
            self._finish_failed_run_metrics(exc)
            raise
        return self._finish_run_metrics("completed")

    def _next_batch(self):
        batch = next(self.train_iter, None)
        require(batch is not None, "train_loader produced no batches")
        self.batches_consumed += 1
        # Eager rollover: if we just consumed the last batch of the epoch, snapshot
        # the next epoch's RNG start and rebuild the iterator now. This keeps the
        # invariant "loader_rng_epoch_start + train_iter point at the NEXT batch"
        # true at every save point — including exactly on an epoch boundary, where
        # a lazy rollover would resume to a replay of the finished epoch.
        if self.batches_consumed % len(self.train_loader) == 0:
            self.loader_rng_epoch_start = self.loader_generator.get_state()
            self.train_iter = iter(self.train_loader)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        self._record_batch_metrics(batch)
        return batch

    def compute_loss(self, batch):
        raise NotImplementedError

    def _optimizer_update(self):
        """Apply one optimizer update and advance scheduler state with that update."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        stepped = not self.scaler.is_enabled() or self.scaler.get_scale() >= old_scale
        if not stepped:
            raise FloatingPointError(
                "AMP skipped the optimizer update; stopping before training step accounting advances"
            )
        commit_post_optimizer_updates(self.model, self.config.qk_clip_threshold, self.config.qk_clip_balance)
        self.scheduler.step()
        return True

    def _save_checkpoint_for_step(self):
        self.save_checkpoint()
        self._last_checkpoint_step = self.step

    def _save_checkpoint_if_due(self):
        if self.config.save_every > 0 and self.step % self.config.save_every == 0:
            self._save_checkpoint_for_step()

    def _save_final_checkpoint(self):
        if self.config.save_every > 0 and self._last_checkpoint_step != self.step:
            self._save_checkpoint_for_step()

    def train(self):
        def loop():
            self.model.train()
            pbar = tqdm(range(self.step + 1, self.config.max_steps + 1), desc="Training")

            for self.step in pbar:
                unwrap_model(self.model).set_training_step(self.step)
                total_loss = 0.0
                for _ in range(self.config.grad_accum_steps):
                    batch = self._next_batch()
                    with autocast_context(self.device, self.dtype):
                        loss = self.compute_loss(batch) / self.config.grad_accum_steps
                    self.scaler.scale(loss).backward()
                    total_loss += loss.item()

                self._optimizer_update()

                lr = self.scheduler.get_last_lr()[0]
                if self.step % self.config.log_every == 0:
                    pbar.set_postfix(loss=f"{total_loss:.4f}", lr=f"{lr:.2e}")
                    if self.aim_run:
                        self.aim_run.track(total_loss, name="loss", step=self.step)
                        self.aim_run.track(lr, name="lr", step=self.step)

                should_eval = (
                    self.eval_loader is not None
                    and self.config.eval_every > 0
                    and self.step % self.config.eval_every == 0
                )
                if should_eval:
                    eval_loss = self.evaluate()
                    print(f"\n  step {self.step} eval loss: {eval_loss:.4f}")
                    if self.aim_run:
                        self.aim_run.track(eval_loss, name="eval_loss", step=self.step)
                    self.model.train()

                self._save_checkpoint_if_due()

            self._save_final_checkpoint()

        self._run_train_loop_with_metrics(loop)

    @torch.no_grad()
    def evaluate(self):
        require(self.eval_loader is not None, "evaluate requires an eval_dataset")
        self.model.eval()
        total, count = 0.0, 0
        for i, batch in enumerate(self.eval_loader):
            if i >= self.config.eval_steps:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with autocast_context(self.device, self.dtype):
                total += self.compute_loss(batch).item()
            count += 1
        require(count > 0, "Eval loader produced no batches")
        return total / count

    def save_checkpoint(self):
        path = Path(self.config.save_dir) / f"step_{self.step}"
        unwrap_model(self.model).save(path)
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
        run_meta = {
            "signature": self.signature,
            "config": self.config.to_dict(),
            "scheduler_total_steps": self._scheduler_total_steps(),
        }
        if self.tokenizer_signature:
            run_meta["tokenizer_signature"] = self.tokenizer_signature
        (path / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
        print(f"\n  saved {path}")


@register_trainer("lm")
class LMTrainer(Trainer):
    def compute_loss(self, batch):
        if (
            self.model.training
            and self.config.token_superposition_size > 1
            and self.step <= self.config.token_superposition_steps
        ):
            return unwrap_model(self.model).token_superposition_loss(
                batch["input_ids"],
                batch["labels"],
                self.config.token_superposition_size,
            )
        return supervised_lm_batch_loss(self.model, batch)


def model_aux_loss(model):
    return unwrap_model(model).auxiliary_loss()


def supervised_lm_batch_loss(model, batch):
    _, loss = model(batch["input_ids"], batch["labels"])
    return loss


def commit_post_optimizer_updates(model, qk_clip_threshold, qk_clip_balance):
    unwrap_model(model).post_optimizer_step(qk_clip_threshold, qk_clip_balance)
