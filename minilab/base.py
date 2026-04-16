import json
from pathlib import Path

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    config_class = None

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

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model = getattr(self, "_orig_mod", self)
        torch.save(model.state_dict(), path / "model.pt")
        model.config.save(path / "config.json")
        # Bind the checkpoint to the model family. Without this, BaseConfig.from_dict
        # silently drops unknown fields and state_dict-compatible families (e.g. MDLM
        # and D3PM share the backbone) cross-load under the wrong objective. Sibling
        # file, not a config field, so configs stay free of meta concerns.
        (path / "model_type.txt").write_text(type(model).__name__)

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        type_path = path / "model_type.txt"
        assert type_path.exists(), f"Missing {type_path}; cannot validate model family"
        saved_type = type_path.read_text().strip()
        assert saved_type == cls.__name__, (
            f"Checkpoint was saved as {saved_type}, cannot load as {cls.__name__}. "
            f"Loading across model families silently reinterprets the objective."
        )
        config = cls.config_class.load(path / "config.json")
        model = cls(config)
        state = torch.load(path / "model.pt", map_location=device, weights_only=True)
        if state and next(iter(state)).startswith("_orig_mod."):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)
        return model

    def _cast_hidden(self, x):
        if torch.is_autocast_enabled(x.device.type):
            return x.to(torch.get_autocast_dtype(x.device.type))
        return x


class BaseTokenizer:

    def save(self, path):
        Path(path).write_text(json.dumps(self._get_state(), indent=2))

    @classmethod
    def load(cls, path):
        tok = cls()
        tok._set_state(json.loads(Path(path).read_text()))
        return tok

    def _get_state(self):
        raise NotImplementedError

    def _set_state(self, state):
        raise NotImplementedError
