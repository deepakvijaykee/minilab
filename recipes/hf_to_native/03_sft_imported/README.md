# 03 SFT Imported Model

Run native Minilab SFT on a Hugging Face model that was imported with
`02_import`.

```bash
bash recipes/hf_to_native/03_sft_imported/run.sh
```

Default input:

```text
checkpoints/imported/smollm2-135m
```

Useful override:

```bash
MODEL=smollm2-360m MAX_STEPS=200 bash recipes/hf_to_native/03_sft_imported/run.sh
```
