# Expected signals

- Policy loss stays finite and oscillates; RL loss curves are not monotone.
- A `Frozen reference: <path>` line prints for every algorithm except DAPO
  (DAPO does not use a reference model).
- The end-of-run block prints `--- After <ALGO> (held-out GSM8K test) ---`,
  then up to five `Q/A/predicted/expected/(OK|WRONG)` lines, and finally
  one summary line: `GSM8K test subset (50 of full split) accuracy:
  <correct>/<total> = <pct>%`. With `EVAL_EXAMPLES=0` the label becomes
  `GSM8K test` (full split) instead.
- At the default 100 steps and 2 generations per prompt, GSM8K accuracy is
  usually a single-digit percent and noisy. The job here is to confirm the
  rollout/verifier/update loop runs end to end, not to show RLVR gains.

If accuracy is exactly 0% across many seeds, check the SFT base produces
anything answer-shaped: GRPO cannot bootstrap reward when no completion ever
hits the verifier.
