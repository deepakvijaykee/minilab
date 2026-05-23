# 00 Inspect HF model

This recipe prints the config, parameter count, and tokenizer metadata
for a curated Hugging Face preset without running the model. It is the
cheapest way to verify that a preset alias resolves correctly and that
the tokenizer and architecture line up with what the import path
expects, which is the right thing to confirm before any download-heavy
recipe runs.

```bash
bash recipes/hf_to_native/00_inspect/run.sh
```

The default model is `smollm2-135m`. Any curated preset can be
substituted by setting `MODEL=`:

```bash
MODEL=qwen3-0.6b bash recipes/hf_to_native/00_inspect/run.sh
```

Setting `LOAD=1` pulls the weights into memory and reports the actual
parameter-memory footprint instead of just the announced parameter
count. Without it the script reads `config.json` only, which is enough
to verify that the preset and its tokenizer resolve and is what makes
the recipe safe to call as a sanity check without committing to a
download.
