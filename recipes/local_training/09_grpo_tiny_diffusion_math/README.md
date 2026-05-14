# 09 GRPO tiny diffusion math

Diffusion RLVR on GSM8K. The trainer samples completions through the
reverse denoising chain, scores them with the GSM8K numeric verifier, and
updates the policy using trajectory log-probability ratios.

```bash
bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```

Defaults: `DIFFUSION_SFT_CHECKPOINT=.../diffusion_sft/step_500`,
`--max-steps 100`, `--batch-size 1`, `--num-generations 2`,
`--max-new-tokens 64`, `--diffusion-steps 64`, `--inner-epochs 4`,
`--max-examples 500`, `--eval-examples 50`. Output lands in
`checkpoints/local_training/diffusion_grpo`.

Useful overrides:

```bash
MAX_STEPS=300 NUM_GENERATIONS=4 bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
DIFFUSION_STEPS=128 MAX_NEW_TOKENS=96 bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```

Rollout cost is roughly `batch_size * num_generations * diffusion_steps *
max_new_tokens`. The AR analogue (recipe 04) only multiplies by the first
three of those, so a "modest" diffusion GRPO run is still substantially
slower per step than its AR counterpart.
