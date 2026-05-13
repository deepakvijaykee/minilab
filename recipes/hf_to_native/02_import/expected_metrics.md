# Expected Metrics

The importer writes:

- `model.pt`, `config.json`, and `model_type.txt` for the native GPT checkpoint
- `tokenizer.json` pointing at the saved local HF tokenizer
- `run_meta.json` with tokenizer identity for native trainer validation
- `import_meta.json` with source and native config details

With `VERIFY=1`, the HF and native logits should be very close on the short
verification prompt.
