# 09 GRPO Tiny Diffusion Math

Run diffusion GRPO with GSM8K verifier rewards.
The trainer samples completions through the reverse denoising chain, scores
them with the numeric GSM8K verifier, and updates the model with trajectory
log-probability ratios.

```bash
bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```

The defaults keep rollout cost small: `batch_size=1`, `num_generations=2`,
`max_new_tokens=64`, and `diffusion_steps=64`.

Useful overrides:

```bash
MAX_STEPS=300 NUM_GENERATIONS=4 bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
DIFFUSION_STEPS=128 MAX_NEW_TOKENS=96 bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```
