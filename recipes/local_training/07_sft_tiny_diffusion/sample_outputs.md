# Sample output

```text
Estimated VRAM (rough planning estimate)
  ...
Loaded checkpoints/local_training/diffusion/step_1000 (mdlm, 27,xxx,xxx params)
alpaca: 2000 diffusion SFT examples
  step 100 loss=...
  ...
  saved checkpoints/local_training/diffusion_sft/step_500
  wrote checkpoints/local_training/diffusion_sft/step_500/run_metrics.json

--- After Diffusion SFT ---
  Q: Give three tips for staying healthy.
  A: ...
```

Response text comes out of reverse diffusion (the script calls
`sample_with_prompt`), so the quality is bounded by the base diffusion
checkpoint, not by how many SFT steps you ran.
