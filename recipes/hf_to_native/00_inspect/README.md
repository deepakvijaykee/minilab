# 00 Inspect HF model

Prints config, parameter count, and tokenizer metadata for a curated HF
preset without running the model. Cheap dry run; safe to call before any
import.

```bash
bash recipes/hf_to_native/00_inspect/run.sh
```

Default model is `smollm2-135m`. Override with `MODEL=`:

```bash
MODEL=qwen3-0.6b bash recipes/hf_to_native/00_inspect/run.sh
```

Set `LOAD=1` to actually pull weights into memory and report parameter
memory; otherwise the script reads `config.json` only.
