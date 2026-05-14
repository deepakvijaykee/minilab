# Sample output

```text
Estimated VRAM (rough planning estimate)
  ...
Trainable: checkpoints/local_training/diffusion_sft/step_500 (mdlm, 27,xxx,xxx params, schedule=cosine)
Frozen reference: checkpoints/local_training/diffusion_sft/step_500
hh-rlhf: 1000 diffusion preference pairs
  step 50 loss=...
  ...
  saved checkpoints/local_training/diffusion_dpo/step_300
  wrote checkpoints/local_training/diffusion_dpo/step_300/run_metrics.json

--- After Diffusion DPO ---
  Q: What makes a good friend?
  A: ...
```

For `ALGORITHM=vrpo` the header reads `--- After Diffusion VRPO ---` and
the save directory becomes `diffusion_vrpo`.
