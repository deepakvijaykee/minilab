# 03 SFT imported model

Runs native Minilab SFT on a checkpoint imported with recipe 02. Uses the
same `scripts/sft.py` as the local-training track; only the input
checkpoint and learning rate differ.

```bash
bash recipes/hf_to_native/03_sft_imported/run.sh
```

Defaults: `MODEL=smollm2-135m`,
`CHECKPOINT=checkpoints/imported/smollm2-135m`, `--max-steps 100`,
`--batch-size 1`, `--lr 2e-5`, `--max-examples 500`. Output:
`checkpoints/imported/smollm2-135m-sft`.

The learning rate is much lower than the from-scratch SFT recipe (1e-4 vs
2e-5) because the imported model already has competent representations;
larger steps tend to wipe them out.

```bash
MODEL=smollm2-360m MAX_STEPS=200 bash recipes/hf_to_native/03_sft_imported/run.sh
```
