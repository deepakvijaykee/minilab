# Expected signals

Diffusion GRPO scores entire reverse-denoising trajectories rather
than individual tokens. The policy log-probability is the sum over
the chain of per-step denoising log-probabilities, so two
trajectories that diverge only in their last few denoising steps look
almost identical from the perspective of the policy gradient. Most of
the trainable signal therefore lives in the early, high-noise steps
where the answer structure forms, while the low-noise steps near the
end of the chain contribute relatively little to the gradient even
though they produce the visible output.

Trajectory-level scoring also explains the wall-time shape of this
recipe. One outer step needs
`batch_size * num_generations * diffusion_steps` reverse passes
before the policy update can be computed. At the default settings
that comes to `1 * 2 * 64 = 128` forward passes before any gradient
step, which is why the cost table is dominated by rollouts rather
than by gradient steps. Adjusting `diffusion_steps` downward is the
single largest lever on wall time, and the trade-off is sample
quality rather than gradient quality.

The eval block prints up to five `Q/A/(predicted, expected, OK|WRONG)`
lines followed by a summary line of the form
`GSM8K test subset (50 of full split) accuracy: ...`. At the default
budget the reward is sparse and the within-group z-score is noisy, so
accuracy hovers near the SFT baseline regardless of step count. What
this recipe demonstrates at this scale is the trajectory-scoring
machinery itself, not a movement in the accuracy number.

The most common dead-run pattern is the verifier returning zero on
every rollout because the diffusion base never produces a numeric
answer. The loss is then well-defined but the gradient is identically
zero. The gating constraint here is base capacity, not RL
hyperparameters. Push recipe 06 further before chasing GRPO numbers
in this recipe.
