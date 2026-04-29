import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from minilab.checks import require


@dataclass
class BaseConfig:

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        require(isinstance(d, dict), f"{cls.__name__} config must be a JSON object")
        valid = {f.name for f in fields(cls)}
        provided = set(d)
        unknown = provided - valid
        missing = valid - provided
        require(not unknown, f"Unknown {cls.__name__} fields: {sorted(unknown)}")
        require(not missing, f"Missing {cls.__name__} fields: {sorted(missing)}")
        return cls(**d)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
