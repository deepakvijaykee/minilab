import json
from pathlib import Path

import torch
import torch.nn as nn

from minilab.checks import require


class BaseModel(nn.Module):
    config_class = None
    forward_process_type = None
    reverse_parameterization = None
    requires_terminal_mask_prior = False

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

    def muon_parameter_groups(self):
        auxiliary_ids = {
            id(param)
            for module in self.muon_auxiliary_modules()
            for param in module.parameters()
        }
        hidden, auxiliary, biases = [], [], []
        seen = set()
        for param in self.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            if param.dim() < 2:
                biases.append(param)
            elif param_id in auxiliary_ids:
                auxiliary.append(param)
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


def unwrap_model(model):
    """Return the real module behind torch.compile's OptimizedModule wrapper."""
    return getattr(model, "_orig_mod", model)


class BaseTokenizer:

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._get_state(), indent=2))

    @classmethod
    def load(cls, path):
        tok = cls()
        tok._set_state(json.loads(Path(path).read_text()))
        return tok

    def _get_state(self):
        raise NotImplementedError

    def _set_state(self, state):
        raise NotImplementedError
