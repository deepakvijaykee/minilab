# 04 GRPO tiny math

Verifier-reward RLVR on GSM8K. Loads the SFT checkpoint, rolls out short
completions, scores them against the GSM8K numeric answer, and updates the
policy with GRPO.

```bash
bash recipes/local_training/04_grpo_tiny_math/run.sh
```

Defaults: `--algorithm grpo`, `--max-steps 100`, `--batch-size 1`,
`--num-generations 2`, `--max-new-tokens 64`, `--max-examples 500`,
`--eval-examples 50`. The save directory follows the algorithm name:
`checkpoints/local_training/<algorithm>`.

The defaults are small for a reason. RLVR memory cost scales with
`batch_size * num_generations * (seq_len + max_new_tokens)` because every
rollout in the group has its own KV cache. Run `estimate_vram.py` (the
run.sh already does) and only push past `batch_size=1` after that number
fits comfortably.

Switch algorithms with `ALGORITHM=`:

```bash
ALGORITHM=rloo bash recipes/local_training/04_grpo_tiny_math/run.sh
ALGORITHM=dapo bash recipes/local_training/04_grpo_tiny_math/run.sh
```

DAPO is special: it removes the KL penalty (`--kl-coef 0` is enforced) and
takes its own `--clip-ratio-low/--clip-ratio-high` instead of a single ratio.
RLOO uses one inner epoch instead of four.
