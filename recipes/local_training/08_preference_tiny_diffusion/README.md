# 08 Preference tiny diffusion

Diffusion preference optimization on the diffusion SFT checkpoint. Default
is diffusion DPO over the model's denoising-loss proxy; `ALGORITHM=vrpo`
swaps in the variance-reduced variant that averages multiple shared
diffusion estimates per pair.

```bash
bash recipes/local_training/08_preference_tiny_diffusion/run.sh
```

Defaults: `--algorithm dpo`, `--dataset hh-rlhf`,
`DIFFUSION_SFT_CHECKPOINT=.../diffusion_sft/step_500`, `--max-steps 300`,
`--batch-size 2`, `--lr 1e-5`, `--beta 0.1`, `--max-examples 1000`,
`--sample-new-tokens 80`. The save directory is named
`checkpoints/local_training/diffusion_<algorithm>`.

Run the variance-reduced variant:

```bash
ALGORITHM=vrpo bash recipes/local_training/08_preference_tiny_diffusion/run.sh
```

VRPO adds `--vrpo-num-samples` (defaulting to 4) and bumps memory use; the
estimator switches method to `diffusion_vrpo` automatically.

The diffusion preference loaders accept dataset names `hh-rlhf` and
`ultrafeedback`. The AR counterpart (recipe 03) uses `hh` instead of
`hh-rlhf`; both eventually call the same HH-RLHF loader, just through
different registry entries.
