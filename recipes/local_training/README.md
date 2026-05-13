# Local Training Recipes

This track is the end-to-end Minilab path for local, single-device training.
It starts with tokenizer training, then branches into autoregressive GPT and
diffusion MDLM training paths with SFT, preference optimization, RLVR, and
evaluation or sampling.

The defaults favor short, inspectable runs over leaderboard numbers. Increase
`MAX_STEPS`, `MAX_EXAMPLES`, `PRESET`, and batch settings after the full path
works on your machine.

Install the data extra before running the track:

```bash
python -m pip install -e ".[data]"
```

## Autoregressive Run Order

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
bash recipes/local_training/02_sft_tiny_instruct/run.sh
bash recipes/local_training/03_preference_tiny/run.sh
bash recipes/local_training/04_grpo_tiny_math/run.sh
bash recipes/local_training/05_eval_all/run.sh
```

## Diffusion Run Order

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
bash recipes/local_training/07_sft_tiny_diffusion/run.sh
bash recipes/local_training/08_preference_tiny_diffusion/run.sh
bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```

The diffusion branch mirrors the AR branch without pretending diffusion models
are next-token models:

| Recipe | What it trains | Default objective |
| --- | --- | --- |
| `06_pretrain_tiny_mdlm` | base MDLM | masked denoising |
| `07_sft_tiny_diffusion` | instruction-tuned MDLM | denoise response tokens while prompt tokens stay fixed |
| `08_preference_tiny_diffusion` | preference-tuned MDLM | diffusion DPO, or VRPO with `ALGORITHM=vrpo` |
| `09_grpo_tiny_diffusion_math` | verifier-reward MDLM | diffusion GRPO over reverse denoising trajectories |

## Default Artifacts

- `checkpoints/local_training/tokenizer.json`
- `checkpoints/local_training/lm/step_1000`
- `checkpoints/local_training/sft/step_500`
- `checkpoints/local_training/preference_dpo/step_300`
- `checkpoints/local_training/grpo/step_100`
- `checkpoints/local_training/diffusion/step_1000`
- `checkpoints/local_training/diffusion_sft/step_500`
- `checkpoints/local_training/diffusion_dpo/step_300`
- `checkpoints/local_training/diffusion_grpo/step_100`

## Useful Overrides

```bash
PRESET=gpt-25m MAX_STEPS=3000 bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
ALGORITHM=simpo bash recipes/local_training/03_preference_tiny/run.sh
NUM_GENERATIONS=4 MAX_NEW_TOKENS=128 bash recipes/local_training/04_grpo_tiny_math/run.sh
MAX_STEPS=3000 bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
ALGORITHM=vrpo bash recipes/local_training/08_preference_tiny_diffusion/run.sh
DIFFUSION_STEPS=128 bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```

Before longer runs, estimate memory:

```bash
python scripts/estimate_vram.py --model gpt-25m --method grpo --seq-len 512 --batch-size 1 --num-generations 4
python scripts/estimate_vram.py --model mdlm-25m --method diffusion_grpo --seq-len 512 --batch-size 1 --num-generations 2
```

Each training recipe writes `run_metrics.json` automatically. The file is
written to the final checkpoint directory and copied to the recipe save
directory. On CUDA, the metrics use PyTorch allocator stats:
`max_memory_allocated_gb` and `max_memory_reserved_gb`.

Replace the sanity checks in each `expected_metrics.md` with measured numbers
from your own hardware when publishing benchmark-style tables.
