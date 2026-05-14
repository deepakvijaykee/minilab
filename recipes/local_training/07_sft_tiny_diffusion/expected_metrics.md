# Expected signals

- Diffusion SFT loss is the same denoising loss as recipe 06, just masked
  to response tokens. Step-to-step jitter is larger than AR SFT because
  every step samples a different timestep.
- The save directory should contain `model.pt`, `config.json`,
  `model_type.txt`, and `forward_process.json`. If `forward_process.json`
  is missing, recipe 08 and 09 will fail to load this checkpoint.
- The `--- After Diffusion SFT ---` block answers three fixed prompts
  through reverse-diffusion infilling rather than autoregressive sampling.
  Responses look response-shaped only if recipe 06 was trained long enough
  to produce text-like unconditional samples; with the default 1000-step
  base they are usually short and choppy.
