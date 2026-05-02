import json
from pathlib import Path

import torch
import torch.nn as nn

from minilab.checks import require
from minilab.losses import causal_lm_cross_entropy
from minilab.nn.norm import ZeroCenteredRMSNorm


def apply_conditional_diffusion_mask(
    z_t,
    noised_mask,
    x_0,
    loss_mask,
    valid_mask,
    mask_token_id,
    context="conditional diffusion",
):
    require(z_t.shape == x_0.shape, f"{context} noised tokens must match input_ids")
    require(noised_mask.shape == x_0.shape, f"{context} noised mask must match input_ids")
    require(noised_mask.dtype == torch.bool, f"{context} noised mask must be bool")
    require(loss_mask is not None, f"{context} loss_mask is required")
    require(loss_mask.shape == x_0.shape, f"{context} loss_mask must match input_ids")
    require(loss_mask.dtype == torch.bool, f"{context} loss_mask must be bool")
    loss_mask = loss_mask.to(x_0.device)
    require(loss_mask.any(dim=-1).all(), f"{context} requires at least one supervised token per example")
    if valid_mask is not None:
        require(valid_mask.shape == x_0.shape, f"{context} valid_mask must match input_ids")
        require(valid_mask.dtype == torch.bool, f"{context} valid_mask must be bool")
        valid_mask = valid_mask.to(x_0.device)
        require((loss_mask & ~valid_mask).sum().item() == 0, (
            f"{context} loss_mask must be contained in valid_mask"
        ))

    z_t = torch.where(loss_mask, z_t, x_0)
    if valid_mask is not None:
        z_t = torch.where(valid_mask, z_t, torch.full_like(z_t, mask_token_id))
    return z_t, noised_mask.to(x_0.device) & loss_mask


class BaseModel(nn.Module):
    config_class = None
    forward_process_type = None
    reverse_parameterization = None
    requires_terminal_mask_prior = False
    provides_hidden_states = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def muon_auxiliary_modules(self):
        """Modules whose matrix parameters should stay on Muon's AdamW path."""
        return ()

    def no_weight_decay_parameter_names(self):
        return ()

    def weight_decay_parameter_names(self):
        return tuple(
            f"{module_name}.weight"
            for module_name, module in self.named_modules()
            if isinstance(module, ZeroCenteredRMSNorm)
        )

    def auxiliary_loss(self):
        for param in self.parameters():
            return torch.tensor(0.0, device=param.device)
        for buffer in self.buffers():
            return torch.tensor(0.0, device=buffer.device)
        return torch.tensor(0.0)

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        return None

    def set_qk_clip_recording(self, enabled):
        return None

    def supports_qk_clip(self):
        return False

    def supports_kv_cache(self):
        return False

    def supports_unconditional_diffusion_sampling(self):
        return True

    def diffusion_forward_kwargs(self, x_0):
        return {}

    def diffusion_training_state(self, forward_process, x_0, device):
        t = forward_process.sample_time(x_0.size(0), device, mode=self.config.time_sampling)
        z_t, mask = forward_process.q_sample(x_0, t)
        return z_t, mask, t, self.diffusion_forward_kwargs(x_0)

    def diffusion_conditional_training_state(self, forward_process, x_0, loss_mask, valid_mask, device, t=None):
        if t is None:
            t = forward_process.sample_time(x_0.size(0), device, mode=self.config.time_sampling)
        else:
            t = t.to(device)
            require(t.shape == (x_0.size(0),), "generic conditional diffusion time must have shape (batch,)")
        z_t, mask = forward_process.q_sample(x_0, t)
        z_t, mask = apply_conditional_diffusion_mask(
            z_t,
            mask,
            x_0,
            loss_mask,
            valid_mask,
            forward_process.mask_token_id,
        )
        return z_t, mask, t, self.diffusion_forward_kwargs(x_0)

    def muon_parameter_groups(self):
        auxiliary_ids = {
            id(param)
            for module in self.muon_auxiliary_modules()
            for param in module.parameters()
        }
        no_decay_ids = {
            id(param)
            for name, param in self.named_parameters()
            if name in self.no_weight_decay_parameter_names()
        }
        weight_decay_ids = {
            id(param)
            for name, param in self.named_parameters()
            if name in self.weight_decay_parameter_names()
        }
        hidden, auxiliary, biases = [], [], []
        seen = set()
        for param in self.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            if param_id in no_decay_ids:
                biases.append(param)
            elif param_id in auxiliary_ids or param_id in weight_decay_ids:
                auxiliary.append(param)
            elif param.dim() < 2:
                biases.append(param)
            else:
                hidden.append(param)
        return hidden, auxiliary, biases

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model = unwrap_model(self)
        torch.save(model.state_dict(), path / "model.pt")
        model.config.save(path / "config.json")
        # Bind the checkpoint to the model family. Config validation catches stale
        # fields, while this catches same-shaped configs loaded under the wrong
        # objective. Sibling file, not a config field, keeps configs free of meta concerns.
        (path / "model_type.txt").write_text(type(model).__name__)

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        raw_config = json.loads((path / "config.json").read_text())
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)

        type_path = path / "model_type.txt"
        require(type_path.exists(), f"Missing {type_path}; checkpoint must declare its model family")
        saved_type = type_path.read_text().strip()
        require(saved_type == cls.__name__, (
            f"Checkpoint was saved as {saved_type}, cannot load as {cls.__name__}. "
            f"Loading across model families silently reinterprets the objective."
        ))
        config = cls.config_class.from_dict(raw_config)
        model = cls(config)
        model.load_state_dict(state)
        return model.to(device)

    def _cast_hidden(self, x):
        if torch.is_autocast_enabled(x.device.type):
            return x.to(torch.get_autocast_dtype(x.device.type))
        return x

    def _causal_lm_forward(self, idx, targets=None, include_auxiliary_loss=False):
        logits, _ = self.forward_hidden(idx)
        loss = None
        if targets is not None:
            loss = causal_lm_cross_entropy(logits, targets)
            if include_auxiliary_loss:
                loss = loss + self.auxiliary_loss()
        return logits, loss


def unwrap_model(model):
    """Return the real module behind torch.compile's optimized wrapper."""
    if isinstance(model, torch._dynamo.OptimizedModule):
        return model._orig_mod
    return model


class BaseTokenizer:

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._get_state(), indent=2))

    @classmethod
    def load(cls, path):
        tok = cls()
        state = json.loads(Path(path).read_text())
        require(isinstance(state, dict), "Tokenizer state must be a JSON object")
        expected_state = tok._get_state()
        if "type" in expected_state:
            require("type" in state, "Tokenizer state is missing required field: type")
            require(state["type"] == expected_state["type"], (
                f"Tokenizer state was saved as {state['type']!r}, cannot load as {cls.__name__}."
            ))
        tok._set_state(state)
        return tok

    def _get_state(self):
        raise NotImplementedError

    def _set_state(self, state):
        raise NotImplementedError
