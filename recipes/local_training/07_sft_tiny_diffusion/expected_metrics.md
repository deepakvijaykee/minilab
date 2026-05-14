# Expected signals

- The loss is recipe 06's denoising objective restricted to response
  tokens: prompt tokens stay clean at every timestep, only the response
  is noised and supervised. Step-to-step jitter is larger than AR SFT
  because each step samples a single timestep `t` per example and the
  loss weight depends on `t`; this is the diffusion-side analogue of
  AR SFT's noise-free loss.
- The checkpoint directory contains `model.pt`, `config.json`,
  `model_type.txt`, and `forward_process.json`. Drop the last file and
  recipes 08/09 refuse to load this checkpoint, because the forward
  process is what defines the diffusion loss they need to compute.
- `--- After Diffusion SFT ---` generates by reverse diffusion with the
  prompt held clean, which is infilling rather than left-to-right
  sampling. Quality is gated by recipe 06: SFT can shift *which* response
  distribution the model denoises toward, but cannot teach it to denoise
  text-like outputs when the base does not already produce them. With
  the default 1000-step base, expect short, choppy answers.
