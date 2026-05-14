# Expected signals

- Policy loss oscillates because GRPO's advantage is group-relative: the
  within-group z-score of the verifier rewards. With `num_generations=2`,
  the advantage collapses to zero whenever both completions get the same
  score, which at this scale is most of the time. The non-trivial loss
  steps are the prompts where the two rollouts disagreed.
- `Frozen reference:` prints for every algorithm except DAPO. DAPO drops
  the KL penalty (and the reference along with it) and relies on
  asymmetric `clip_ratio_low/high` to keep updates near the rollout
  distribution; RLOO drops the clip entirely and uses an unclipped
  REINFORCE estimator with leave-one-out baseline.
- The eval block prints up to five `Q/A/(predicted, expected, OK|WRONG)`
  lines and a summary: `GSM8K test subset (50 of full split) accuracy:
  <correct>/<total> = <pct>%`. `EVAL_EXAMPLES=0` drops the subset note
  and runs the full split.
- Single-digit accuracy at 100 steps x 2 generations is what the rollout
  budget predicts. The group-relative signal only exists where the two
  generations disagree; lifting `NUM_GENERATIONS` to 4 or 8 multiplies
  the density of non-zero gradients per step far more than `MAX_STEPS` does.

If accuracy is exactly 0% across seeds, the SFT base never produces a
parseable answer. GRPO is non-bootstrapping: with zero reward density
the gradient is zero and the policy drifts under whatever KL the schedule
imposes, not learning.
