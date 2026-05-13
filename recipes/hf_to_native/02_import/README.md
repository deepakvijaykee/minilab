# 02 Import HF To Native

Import a compatible Hugging Face model into Minilab's native GPT checkpoint
format, then train it with the existing Minilab scripts.

```bash
bash recipes/hf_to_native/02_import/run.sh
```

After import:

```bash
python scripts/sft.py \
  --tokenizer checkpoints/imported/smollm2-135m/tokenizer.json \
  --checkpoint checkpoints/imported/smollm2-135m \
  --save-dir checkpoints/imported/smollm2-135m-sft
```

Current importer scope is intentionally narrow: Llama-compatible small models
such as SmolLM2. Qwen3/Gemma need separate mapping validation before import.
