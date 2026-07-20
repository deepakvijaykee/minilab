# 04 GRPO tiny math

This recipe runs verifier-reward RLVR on GSM8K. The trainer loads the
SFT checkpoint from recipe 03 (or, if preference tuning was skipped,
recipe 02), samples short completions for each problem, scores each
completion against the GSM8K numeric answer with the deterministic
verifier, and then updates the policy using the GRPO objective. The
purpose of staging RLVR here is to expose the central reality of RL
with a sparse outcome reward at tiny scale: gradients exist only on
the small fraction of prompts where the group of rollouts disagrees on
the verifier's verdict.

```bash
bash recipes/local_training/04_grpo_tiny_math/run.sh
```

The defaults are `--algorithm grpo`, `--max-steps 100`, `--batch-size 1`,
`--num-generations 2`, `--max-new-tokens 64`, `--max-examples 500`, and
`--eval-examples 50`. The save directory is named after the algorithm,
which keeps the artifacts of different RL variants from colliding:
`checkpoints/local_training/<algorithm>`.

The defaults are deliberately conservative because RLVR memory cost
scales as `batch_size * num_generations * (seq_len + max_new_tokens)`.
Every rollout in the group carries its own KV cache, so doubling
`num_generations` doubles activation memory for the rollout phase
rather than the smaller marginal cost one might expect from a single
extra forward pass. The `run.sh` wrapper calls `estimate_vram.py`
first. Pushing past `batch_size=1` is only sensible once that estimate
comfortably fits the available VRAM.

Switch algorithms with `ALGORITHM=`:

```bash
ALGORITHM=rloo bash recipes/local_training/04_grpo_tiny_math/run.sh
ALGORITHM=dapo bash recipes/local_training/04_grpo_tiny_math/run.sh
```

DAPO is structurally different from the GRPO and RLOO branches. It
removes the KL penalty entirely (`--kl-coef 0` is enforced) and takes
asymmetric clipping with `--clip-ratio-low` and `--clip-ratio-high`
in place of a single symmetric ratio. The asymmetric clip is what
plays the role the reference KL plays in GRPO: it constrains how far
the policy can move from the rollout distribution. RLOO is closer to
GRPO in shape but runs one inner epoch instead of four, which trades
sample reuse for fresher rollouts at the cost of more wall time per
unit of effective batch.

The group size, controlled by `NUM_GENERATIONS`, matters more for RLVR
signal than the total step count does. The advantage in GRPO is a
within-group z-score of the verifier rewards, and that z-score is zero
whenever all completions in the group tie. At two generations most
prompts tie, so most gradients are zero on most steps. Lifting the
group size to four or eight multiplies the density of non-zero
gradients per step, which is a different kind of improvement than
adding more steps at group size two and is the first knob to reach
for when the run looks dead in the water.
