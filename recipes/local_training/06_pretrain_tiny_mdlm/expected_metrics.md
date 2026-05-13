# Expected Signals

- Diffusion training loss should remain finite and generally trend downward.
- The checkpoint directory should include `forward_process.json`.
- Unconditional samples should become more text-like as `MAX_STEPS` increases.

The default is a short sanity run. Replace these notes with measured loss,
runtime, peak VRAM, and sample quality after running on a target GPU.
