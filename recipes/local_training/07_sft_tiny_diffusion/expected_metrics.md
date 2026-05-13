# Expected Signals

- Diffusion SFT loss should stay finite.
- The final checkpoint should include both model files and `forward_process.json`.
- The printed infill samples should show response-like behavior if the base
  diffusion model was trained long enough.

Short defaults mainly validate mechanics. Increase pretraining and SFT steps
before judging generation quality.
