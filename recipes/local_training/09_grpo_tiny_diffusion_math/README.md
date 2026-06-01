# 09 GRPO tiny diffusion math

This recipe runs diffusion RLVR on GSM8K. The trainer samples
completions by running the reverse denoising chain, scores each
sampled completion with the GSM8K numeric verifier, and updates the
policy using trajectory log-probability ratios. The structural
difference from recipe 04 is that the policy log-probability is the
sum of per-step denoising log-probabilities along the reverse chain
rather than the next-token sum of an autoregressive sampler, so what
GRPO scores here is a trajectory through latent state space rather
than a sequence of tokens.

```bash
bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```

The defaults are `DIFFUSION_SFT_CHECKPOINT=.../diffusion_sft/step_500`,
`--max-steps 100`, `--batch-size 1`, `--num-generations 2`,
`--max-new-tokens 64`, `--diffusion-steps 64`, `--inner-epochs 4`,
`--max-examples 500`, and `--eval-examples 50`. Output lands in
`checkpoints/local_training/diffusion_grpo`.

Useful overrides:

```bash
MAX_STEPS=300 NUM_GENERATIONS=4 bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
DIFFUSION_STEPS=128 MAX_NEW_TOKENS=96 bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```

Rollout cost scales as
`batch_size * num_generations * diffusion_steps * max_new_tokens`.
The autoregressive analogue in recipe 04 only multiplies by the first
three of those factors, so even a modest diffusion GRPO run is
substantially slower per step than its autoregressive counterpart.
That extra factor is why the wall-time table in the root README is
dominated by rollouts rather than gradient steps.
