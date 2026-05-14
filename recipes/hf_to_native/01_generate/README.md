# 01 Generate with HF model

Generates text through the HuggingFace `generate()` API, not through
Minilab's own sampler. Useful for sanity-checking a curated preset before
importing it.

```bash
bash recipes/hf_to_native/01_generate/run.sh
```

By default it runs `smollm2-135m-instruct` on `DEVICE=auto` with the prompt
"Explain gravity in one paragraph.", generating up to 128 tokens at
`temperature=0.7`, `top_p=0.95`, `top_k=50`.

```bash
MODEL=qwen3-0.6b DEVICE=cuda PROMPT="Solve 12+37." bash recipes/hf_to_native/01_generate/run.sh
```

`DTYPE` is empty by default so the script falls through to whatever
Transformers picks for the preset. Set it (e.g. `DTYPE=bfloat16`) only
when you need to override that.
