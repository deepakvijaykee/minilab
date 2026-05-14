# Expected signals

- Diffusion GRPO scores entire reverse-denoising trajectories, not
  individual tokens. The policy log-prob is the sum over the chain of
  per-step denoising log-probs, so two trajectories that diverge only
  in the last few denoising steps look almost identical to the policy
  gradient. Most of the trainable signal is in the early (high-noise)
  steps where the answer structure forms.
- Trajectory-level scoring also explains the wall-time shape: one outer
  step needs `batch * num_generations * diffusion_steps` reverse passes.
  At defaults that is 1 * 2 * 64 = 128 forwards before the policy
  update, which is why the cost table for this recipe is dominated by
  rollouts rather than by gradient steps.
- The eval block prints up to five `Q/A/(predicted, expected, OK|WRONG)`
  lines and a `GSM8K test subset (50 of full split) accuracy: ...` line.
  At default scale the reward is sparse and the within-group z-score
  noisy, so accuracy hovers near the SFT baseline regardless of step count.

A common failure: the verifier returns 0 for every rollout because the
diffusion base never produces a numeric answer. The loss is well-defined
but the gradient is zero. The gating constraint is base capacity, not RL
hyperparameters; push recipe 06 further before chasing GRPO numbers here.
