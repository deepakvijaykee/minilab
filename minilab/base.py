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
        torch.save(self.state_dict(), path / "model.pt")
        self.config.save(path / "config.json")

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        config = cls.config_class.load(path / "config.json")
        model = cls(config)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
        return model


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
