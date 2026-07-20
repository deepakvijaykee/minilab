# Local training recipes

These recipes form one end-to-end GPT pipeline, from an untrained
tokenizer to a verifier-scored policy, run entirely on a single device.
The defaults are small so each stage finishes in minutes, the scale at
which you can watch one checkpoint change the next and form intuition
about what each stage contributes. They are sized
for that, not for quality. Once the path runs cleanly, scale it up through
`MAX_STEPS`, `MAX_EXAMPLES`, `PRESET`, and the batch settings rather than
by changing the structure of the pipeline.

Install the data extra first, then run the stages in order from the
repository root:

```bash
python -m pip install -e ".[data]"
```

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
bash recipes/local_training/02_sft_tiny_instruct/run.sh
bash recipes/local_training/03_preference_tiny/run.sh
bash recipes/local_training/04_grpo_tiny_math/run.sh
bash recipes/local_training/05_eval_all/run.sh
```

Pretraining feeds SFT; preference tuning and RLVR each branch from the
SFT checkpoint, and evaluation covers all four:

| Recipe | Output |
|---|---|
| `00_train_tokenizer` | `checkpoints/local_training/tokenizer.json` |
| `01_pretrain_tiny_gpt` | `checkpoints/local_training/lm/step_1000` |
| `02_sft_tiny_instruct` | `checkpoints/local_training/sft/step_500` |
| `03_preference_tiny` | `checkpoints/local_training/preference_dpo/step_300` |
| `04_grpo_tiny_math` | `checkpoints/local_training/grpo/step_100` |
| `05_eval_all` | consolidated evaluation across the four checkpoints |

Every stage takes environment overrides, the way to scale a run up
without touching the script:

```bash
MAX_STEPS=3000 PRESET=gpt-25m bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
ALGORITHM=simpo bash recipes/local_training/03_preference_tiny/run.sh
NUM_GENERATIONS=8 MAX_NEW_TOKENS=128 bash recipes/local_training/04_grpo_tiny_math/run.sh
```

Estimate memory before a larger run:

```bash
python scripts/estimate_vram.py --model gpt-25m --method grpo --seq-len 512 --batch-size 1 --num-generations 4
```

Every recipe writes `run_metrics.json` into its final checkpoint directory
and copies it to the recipe save root. On CUDA that file records
`max_memory_allocated_gb` and `max_memory_reserved_gb` from PyTorch's peak
memory statistics. On CPU those keys are simply absent rather than zeroed.
Across these recipes, `run_metrics.json` records the numbers a run
produced, while each recipe's `expected_metrics.md` describes the shape
those numbers should take and how to read them.
