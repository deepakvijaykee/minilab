# Expected signals

- `Trainable:` and `Frozen reference:` print at startup; DAPO drops the
  reference line because it has no reference model.
- The eval block matches recipe 04 in `local_training/`: up to five
  `Q/A/(predicted, expected, OK|WRONG)` lines, then
  `GSM8K test subset (20 of full split) accuracy: <correct>/<total> = <pct>%`.
- `run_metrics.json` is written under
  `checkpoints/imported/<model>-grpo/step_<MAX_STEPS>/`.
- 25 outer steps with 2 generations per prompt is below the noise floor
  for GRPO accuracy. The signal-to-noise of group-relative advantages
  scales with `num_generations`, and `2` is the minimum where any group
  signal exists at all. This recipe is for confirming the loop runs on
  imported weights, not for claiming an RLVR result.

If completions are empty across all rollouts, the policy is emitting EOS
immediately. Either the SimPO checkpoint collapsed the response
distribution, or `max_new_tokens` is too small for the prompt template
the model was instruction-tuned on. Bump `MAX_NEW_TOKENS` and confirm
recipe 04 actually shifted the policy.
