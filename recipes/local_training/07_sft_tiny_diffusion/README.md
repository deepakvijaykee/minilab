# 07 SFT tiny diffusion

Instruction SFT for the diffusion LM. The prompt tokens are held clean and
only the response tokens are noised and supervised, so the model learns to
denoise an answer conditioned on the question instead of being treated as a
left-to-right predictor.

```bash
bash recipes/local_training/07_sft_tiny_diffusion/run.sh
```

Defaults: `DIFFUSION_CHECKPOINT=.../diffusion/step_1000`, `--dataset alpaca`,
`--max-steps 500`, `--batch-size 2`, `--lr 1e-4`, `--max-examples 2000`,
`--sample-new-tokens 80`, gradient checkpointing on. Output lands in
`checkpoints/local_training/diffusion_sft`.

Useful overrides:

```bash
MAX_STEPS=1000 MAX_EXAMPLES=5000 bash recipes/local_training/07_sft_tiny_diffusion/run.sh
DATASET=dolly bash recipes/local_training/07_sft_tiny_diffusion/run.sh
```

The forward noise process is loaded from the base checkpoint's
`forward_process.json`. Switching schedules between recipe 06 and this one
is not supported; the file copies through.
