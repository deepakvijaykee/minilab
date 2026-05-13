import os
from pathlib import Path


def configure_hf_cache(*, include_datasets=False):
    cache_root = Path.cwd() / ".cache" / "huggingface"

    if not any(os.environ.get(name) for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")):
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_root)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")

    if include_datasets and not os.environ.get("HF_DATASETS_CACHE"):
        root = Path(os.environ.get("HF_HOME", cache_root))
        dataset_cache = root / "datasets"
        dataset_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(dataset_cache)
