# 01 Generate with HF model

This recipe generates text through the Hugging Face `generate()` API
rather than through Minilab's own sampler. The point of staging it
between inspection and import is to confirm that a curated preset
actually produces sensible text in its native runtime before any
weight mapping is attempted. If a preset fails to generate cleanly
here, the right place to investigate is upstream of the importer
rather than inside it.

```bash
bash recipes/hf_to_native/01_generate/run.sh
```

By default the recipe runs `smollm2-135m-instruct` on `DEVICE=auto`
with the prompt "Explain gravity in one paragraph.", generating up to
128 tokens at `temperature=0.7`, `top_p=0.95`, and `top_k=50`. Any of
these are overridable from the command line:

```bash
MODEL=qwen3-0.6b DEVICE=cuda PROMPT="Solve 12+37." bash recipes/hf_to_native/01_generate/run.sh
```

The `DTYPE` variable is intentionally empty by default, which lets the
script fall through to whatever dtype Transformers picks for the
preset. Setting it (for example `DTYPE=bfloat16`) is only useful when
that automatic choice needs to be overridden, typically to match the
dtype a downstream import or training run will use.
