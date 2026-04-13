import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class BaseConfig:

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
