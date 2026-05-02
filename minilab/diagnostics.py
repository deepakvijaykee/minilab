import time

from minilab.base import unwrap_model
from minilab.checks import require


def parameter_count(model, trainable_only=False):
    params = unwrap_model(model).parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def parameter_summary(model):
    total = parameter_count(model)
    trainable = parameter_count(model, trainable_only=True)
    return {
        "parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def optimizer_state_bytes(model, optimizer="adamw", dtype_bytes=4):
    require(dtype_bytes > 0, "dtype_bytes must be > 0")
    params = parameter_count(model, trainable_only=True)
    if optimizer == "adamw":
        return 2 * params * dtype_bytes
    if optimizer == "lion":
        return params * dtype_bytes
    if optimizer == "sgd":
        return 0
    raise ValueError(f"Unknown optimizer: {optimizer!r}")


def training_memory_bytes(model, dtype_bytes=2, optimizer="adamw"):
    require(dtype_bytes > 0, "dtype_bytes must be > 0")
    params = parameter_count(model)
    trainable = parameter_count(model, trainable_only=True)
    return {
        "parameter_bytes": params * dtype_bytes,
        "gradient_bytes": trainable * dtype_bytes,
        "optimizer_state_bytes": optimizer_state_bytes(model, optimizer=optimizer, dtype_bytes=4),
    }


def transformer_flops_per_token(config, seq_len=None):
    """Rough dense-transformer training FLOPs per token for planning runs."""
    require(config.dim > 0, "config.dim must be > 0")
    require(config.num_layers > 0, "config.num_layers must be > 0")
    if seq_len is None:
        seq_len = config.max_seq_len
    require(seq_len > 0, "seq_len must be > 0")
    ffn_mult = config.ffn_mult
    attn = 4 * config.dim * config.dim
    ffn = 2 * config.dim * int(config.dim * ffn_mult)
    attention_scores = 2 * seq_len * config.dim
    return 6 * config.num_layers * (attn + ffn + attention_scores)


def tokens_per_second(num_tokens, elapsed_seconds):
    require(num_tokens >= 0, "num_tokens must be >= 0")
    require(elapsed_seconds > 0, "elapsed_seconds must be > 0")
    return num_tokens / elapsed_seconds


def model_flops_utilization(num_tokens, flops_per_token, elapsed_seconds, peak_flops):
    require(flops_per_token > 0, "flops_per_token must be > 0")
    require(peak_flops > 0, "peak_flops must be > 0")
    achieved = tokens_per_second(num_tokens, elapsed_seconds) * flops_per_token
    return achieved / peak_flops


class ThroughputTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
