# Expected Signals

- Diffusion GRPO should complete with finite policy loss.
- The held-out GSM8K subset accuracy prints at the end.
- Short defaults may show little or no accuracy gain; the primary goal is to
  validate the diffusion RLVR loop locally.
- If memory or runtime is tight, reduce `NUM_GENERATIONS`, `MAX_NEW_TOKENS`, or
  `DIFFUSION_STEPS`.
