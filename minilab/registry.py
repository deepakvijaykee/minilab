_REGISTRIES: dict[str, dict[str, type]] = {}


def register(kind, name):
    def decorator(cls):
        _REGISTRIES.setdefault(kind, {})[name] = cls
        return cls
    return decorator


def get(kind, name):
    assert kind in _REGISTRIES, f"Unknown registry kind: '{kind}'"
    registry = _REGISTRIES[kind]
    assert name in registry, f"Unknown {kind}: '{name}'. Available: {sorted(registry.keys())}"
    return registry[name]


def list_available(kind):
    assert kind in _REGISTRIES, f"Unknown registry kind: '{kind}'"
    return sorted(_REGISTRIES[kind].keys())


def register_model(name): return register("model", name)
def register_attention(name): return register("attention", name)
def register_position(name): return register("position", name)
def register_norm(name): return register("norm", name)
def register_ffn(name): return register("ffn", name)
def register_connection(name): return register("connection", name)
def register_tokenizer(name): return register("tokenizer", name)
def register_scheduler(name): return register("scheduler", name)
def register_sampler(name): return register("sampler", name)
def register_trainer(name): return register("trainer", name)

def get_model(name): return get("model", name)
def get_attention(name): return get("attention", name)
def get_position(name): return get("position", name)
def get_norm(name): return get("norm", name)
def get_ffn(name): return get("ffn", name)
def get_connection(name): return get("connection", name)
def get_tokenizer(name): return get("tokenizer", name)
def get_scheduler(name): return get("scheduler", name)
def get_sampler(name): return get("sampler", name)
def get_trainer(name): return get("trainer", name)
