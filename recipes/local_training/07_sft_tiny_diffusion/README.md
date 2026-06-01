# 07 SFT tiny diffusion

This recipe runs instruction SFT on the diffusion language model. The
trainer holds the prompt tokens clean at every diffusion timestep and
noises only the response tokens, supervising the model on denoising
the answer conditional on the question. The model is never reduced to a
left-to-right next-token predictor for alignment, so the inductive bias
of the diffusion objective carries straight through into the SFT stage
rather than being discarded for convenience.

```bash
bash recipes/local_training/07_sft_tiny_diffusion/run.sh
```

The defaults are `DIFFUSION_CHECKPOINT=.../diffusion/step_1000`,
`--dataset alpaca`, `--max-steps 500`, `--batch-size 2`, `--lr 1e-4`,
`--max-examples 2000`, and `--sample-new-tokens 80`, with gradient
checkpointing on. Output lands in
`checkpoints/local_training/diffusion_sft`.

Useful overrides:

```bash
MAX_STEPS=1000 MAX_EXAMPLES=5000 bash recipes/local_training/07_sft_tiny_diffusion/run.sh
DATASET=dolly bash recipes/local_training/07_sft_tiny_diffusion/run.sh
```

The forward noise process is loaded from the base checkpoint's
`forward_process.json`, and this file copies through unchanged. The
recipe does not support switching the schedule between recipe 06 and
recipe 07, because the SFT objective is defined against the same
forward process the base was trained with, and changing the schedule
would silently break that contract.
