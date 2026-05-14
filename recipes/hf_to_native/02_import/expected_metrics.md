# Expected signals

The importer writes a self-contained native checkpoint directory:

- `model.pt`, `config.json`, `model_type.txt` for the GPT checkpoint
  (`model_type.txt` should read `GPT`).
- `tokenizer.json` pointing at the saved HF tokenizer under
  `hf_tokenizer/`.
- `run_meta.json` with the tokenizer signature and the HF source
  (`source.repo`, `source.alias`, `source.model_type`). The trainer reads
  the signature on later runs to refuse mismatched tokenizers.
- `import_meta.json` with the native config plus the verification block
  if `VERIFY=1` was set.

With `VERIFY=1` the printed `max_abs_diff` and `mean_abs_diff` are the
quantitative check that the weight mapping is correct. SmolLM2-135M
typically prints `max_abs_diff` around `1e-5`; values above `1e-3`
indicate the mapping is broken.
