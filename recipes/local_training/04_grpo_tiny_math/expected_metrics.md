# Expected signals

Policy loss oscillates rather than descending smoothly, because GRPO's
advantage is group-relative: it is the within-group z-score of the
verifier rewards. With `num_generations=2`, the advantage collapses to
zero whenever both rollouts in the group get the same verifier
score, which at this scale happens on most prompts. The non-trivial
loss steps are the prompts where the two rollouts genuinely disagreed,
and those are the only steps where the policy learns anything from the
verifier signal.

`Frozen reference:` prints at startup for every algorithm except DAPO.
DAPO removes the KL penalty entirely, and with it the reference model,
and relies on asymmetric `clip_ratio_low` and `clip_ratio_high` to
keep policy updates near the rollout distribution. RLOO sits between
the two: it drops the clip entirely and uses an unclipped REINFORCE
estimator with an unbiased leave-one-out baseline.

The eval block at the end of the run prints up to five
`Q/A/(predicted, expected, OK|WRONG)` lines and then a summary line of
the form `GSM8K test subset (50 of full split) accuracy: <correct>/<total>
= <pct>%`. Setting `EVAL_EXAMPLES=0` drops the subset annotation and
evaluates on the full split, which is slower but more representative.

Single-digit accuracy after a hundred steps with two generations per
prompt is what the rollout budget predicts. The group-relative signal
only exists on prompts where the two rollouts disagree, and the way to
increase it is to lift `NUM_GENERATIONS` to four or eight rather than to
take more steps at group size two. Each step
becomes more expensive, but each step also carries far more gradient
information per unit of compute.

A clean zero across seeds means the SFT base is never producing an
answer the verifier can parse. GRPO is non-bootstrapping in that
situation: the reward is zero on every rollout, the advantage is
zero, the gradient is zero, and the policy drifts only under
whatever KL the schedule imposes. The fix lives upstream, in recipe
02. Train SFT longer before tuning anything inside this recipe.
