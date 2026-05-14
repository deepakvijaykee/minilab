# Expected signals

- The trainer prints `Trainable: <path> (mdlm, <N> params,
  schedule=<schedule>)` and `Frozen reference: <path>`. Both should point
  at the diffusion SFT checkpoint by default.
- The dataset line is `GSM8K: train=<N> test=<N>`. With the default
  `MAX_EXAMPLES=500` and `EVAL_EXAMPLES=50`, those are the counts after
  filtering for sequences that fit `--seq-len`.
- Up to five rollout samples print `Q: ... A: ... (predicted=<pred>,
  expected=<exp>, OK|WRONG)`, then a single summary line: `GSM8K test
  subset (50 of full split) accuracy: <correct>/<total> = <pct>%`. With
  `EVAL_EXAMPLES=0` the label is `GSM8K test` over the full split.
- Loss is noisy; the relevant signal is whether accuracy nudges above 0
  on average across seeds at higher `MAX_STEPS` and `NUM_GENERATIONS`.

A common failure: GSM8K verifier returns 0 for every rollout because the
diffusion base never produces a number in its response. In that case the
loss is well-defined but the gradient signal is zero. Train recipe 06
longer before chasing diffusion GRPO numbers.
