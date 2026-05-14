# 02 Import HF to native

Imports a compatible HF causal LM into Minilab's native GPT checkpoint
format. After this step the model loads through the same code path as a
checkpoint trained from scratch.

```bash
bash recipes/hf_to_native/02_import/run.sh
```

Out of the box it pulls `smollm2-135m`, writes to
`checkpoints/imported/smollm2-135m`, sets the native context length to 512,
runs on CPU, and verifies logits. The verify step compares HF and native
logits on a short prompt and prints the max and mean absolute difference;
values around `1e-5` or smaller mean the mapping is clean.

After import the same SFT / preference / GRPO scripts work on the result:

```bash
python scripts/sft.py \
  --tokenizer checkpoints/imported/smollm2-135m/tokenizer.json \
  --checkpoint checkpoints/imported/smollm2-135m \
  --save-dir checkpoints/imported/smollm2-135m-sft
```

The importer accepts only Llama-compatible HF models (SmolLM2 today).
Qwen3 and Gemma3 trip the model-type guard in `scripts/import_hf.py` and
need separate weight-mapping work before they can come through this path.
