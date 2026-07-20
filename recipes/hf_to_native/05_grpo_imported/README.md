# 05 GRPO imported model

This recipe runs native Minilab RLVR with the GSM8K verifier reward
on an imported and aligned checkpoint. By default it starts from the
SimPO output of recipe 04. The preference step can be skipped by
pointing `POLICY_CHECKPOINT` at the SFT checkpoint from recipe 03
instead, which is occasionally useful when comparing the effect of
preference tuning on downstream RLVR signal.

```bash
bash recipes/hf_to_native/05_grpo_imported/run.sh
```

The defaults are an `smollm2-135m` SimPO policy, GRPO for 25 outer
steps at `batch_size=1` with two generations per prompt, 64 maximum
new tokens, 100 training examples, and 20 evaluation examples. Output
is written to `checkpoints/imported/smollm2-135m-grpo`.

These defaults stay small because RLVR on a 135M model is much heavier
per outer step than on the local `gpt-10m`. Rollout-phase memory scales
with the policy size and the number of generations, so the per-step
cost grows with the model, roughly the ratio of 135M to the local 7.5M.
Expect noticeably longer wall time per step than the local-training
GRPO recipe.

Lifting `MAX_STEPS` alone or `NUM_GENERATIONS` alone produces noise
rather than signal. The two interact: more steps without more
generations per prompt simply takes more steps at the same near-zero
rate of mixed-reward groups, while more generations without more steps
multiplies the per-step cost without giving the optimizer enough
updates to use the richer gradient. Both should move together when
looking for a real RLVR effect on an imported baseline.
