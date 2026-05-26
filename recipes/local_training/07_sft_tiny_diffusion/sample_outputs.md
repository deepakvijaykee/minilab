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

The response text comes out of reverse diffusion through
`sample_with_prompt`, not from an autoregressive sampler. Response
quality is therefore bounded above by the base diffusion checkpoint
rather than by the SFT step count, and running more SFT steps will
not extract better text from a base that does not yet denoise to
coherent output.
