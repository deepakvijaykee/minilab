# Expected signals

- `Trainable:` and `Frozen reference:` lines print at startup. For
  `ALGORITHM=dapo` only the trainable line prints (DAPO has no reference).
- The eval block looks the same as recipe 04 in `local_training/`: up to
  five `Q/A/predicted/expected` lines, then a summary
  `GSM8K test subset (20 of full split) accuracy: <correct>/<total> = <pct>%`.
- `run_metrics.json` is written under
  `checkpoints/imported/<model>-grpo/step_<MAX_STEPS>/`.
- 25 steps with 2 generations is below the noise floor for accuracy. The
  point is to confirm the rollout/verifier loop runs cleanly on an
  imported checkpoint, not to claim a real RLVR result.

If completions are empty across all rollouts the policy checkpoint is
generating EOS immediately; increase `--max-new-tokens` and verify recipe
04 actually changed the policy distribution.
