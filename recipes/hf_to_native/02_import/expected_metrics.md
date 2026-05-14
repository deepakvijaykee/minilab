# Expected signals

The importer writes a self-contained native checkpoint directory:

- `model.pt`, `config.json`, `model_type.txt` for the native GPT
  checkpoint (`model_type.txt` reads `GPT`).
- `tokenizer.json` plus an `hf_tokenizer/` directory holding the original
  HF tokenizer; the native tokenizer is a thin wrapper around that.
- `run_meta.json` with the tokenizer signature and the HF source metadata.
  The trainer reads the signature on later runs to refuse mismatched
  tokenizers; without it you can silently load a checkpoint under the
  wrong tokenization and get garbage loss.
- `import_meta.json` with the native config and, with `VERIFY=1`, the
  logit-check result.

With `VERIFY=1` the script forward-passes both the HF model and the native
mapped model on a short prompt and reports max/mean absolute logit
difference. SmolLM2-135M usually lands `max_abs_diff` around `1e-5`;
values above `1e-3` mean the mapping has a bug (typically a transposed
projection or a forgotten norm-scale). Below `1e-5` the difference is
mostly fp32 accumulation order; not worth debugging.
