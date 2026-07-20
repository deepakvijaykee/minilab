# Expected signals

The `Trainable:` and `Frozen reference:` lines print at startup. DAPO
omits the reference line because it has no reference model to load, and
that absence is the visible signature of having dropped the reference KL
in favor of asymmetric ratio clipping.

The eval block mirrors the local-training GRPO recipe. It prints up
to five `Q/A/(predicted, expected, OK|WRONG)` lines followed by a
summary line of the form
`GSM8K test subset (20 of full split) accuracy: <correct>/<total> = <pct>%`.
The `run_metrics.json` file is written under
`checkpoints/imported/<model>-grpo/step_<MAX_STEPS>/`, alongside the
model artifacts.

Twenty-five outer steps with two generations per prompt sits below
the noise floor for any meaningful GRPO accuracy estimate. The
signal-to-noise of group-relative advantages scales with
`num_generations`, and two is the minimum group size at which any
within-group signal exists at all. So the recipe firmly establishes
just one thing, that the RLVR loop runs cleanly on imported weights,
and the accuracy after twenty-five steps carries no signal beyond
that.

Empty completions across all rollouts point at the policy emitting
EOS immediately. Two causes are typical: either the
SimPO checkpoint collapsed the response distribution onto an empty
generation, or `max_new_tokens` is too small for the prompt template
the model was instruction-tuned against. The fix is to raise
`MAX_NEW_TOKENS` and to check that preference tuning shifted the
policy. Either way the cause sits upstream of the RL loop, in the
policy the rollouts are sampling from.
