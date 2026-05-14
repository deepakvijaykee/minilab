# Local training recipes

The end-to-end path for local, single-device training. Start with the
tokenizer, then branch into either the autoregressive GPT track or the
diffusion MDLM track. Both branches go through SFT, preference optimization,
RLVR, and evaluation.

Defaults are sized for a laptop GPU and a short run. They are good for
checking that the loop works on your machine, not for chasing quality. Bump
`MAX_STEPS`, `MAX_EXAMPLES`, `PRESET`, and the batch settings once the full
path runs end to end.

Install the data extra before running the track:

```bash
python -m pip install -e ".[data]"
```

## Autoregressive run order

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
bash recipes/local_training/01_pretrain_tiny_gpt/run.sh
bash recipes/local_training/02_sft_tiny_instruct/run.sh
bash recipes/local_training/03_preference_tiny/run.sh
bash recipes/local_training/04_grpo_tiny_math/run.sh
bash recipes/local_training/05_eval_all/run.sh
```

## Diffusion run order

```bash
bash recipes/local_training/00_train_tokenizer/run.sh
bash recipes/local_training/06_pretrain_tiny_mdlm/run.sh
bash recipes/local_training/07_sft_tiny_diffusion/run.sh
bash recipes/local_training/08_preference_tiny_diffusion/run.sh
bash recipes/local_training/09_grpo_tiny_diffusion_math/run.sh
```

The diffusion branch mirrors the AR branch stage for stage but stays diffusion
native; nothing is treated as a next-token model.

| Recipe | What it trains | Default objective |
| --- | --- | --- |
| `06_pretrain_tiny_mdlm` | base MDLM | masked denoising |
| `07_sft_tiny_diffusion` | instruction-tuned MDLM | denoise response tokens while prompt tokens stay fixed |
| `08_preference_tiny_diffusion` | preference-tuned MDLM | diffusion DPO, or VRPO with `ALGORITHM=vrpo` |
| `09_grpo_tiny_diffusion_math` | verifier-reward MDLM | diffusion GRPO over reverse denoising trajectories |

## Default artefacts

- `checkpoints/local_training/tokenizer.json`
- `checkpoints/local_training/lm/step_1000`
- `checkpoints/local_training/sft/step_500`
- `checkpoints/local_training/preference_dpo/step_300`
- `checkpoints/local_training/grpo/step_100`
- `checkpoints/local_training/diffusion/step_1000`
- `checkpoints/local_training/diffusion_sft/step_500`
- `checkpoints/local_training/diffusion_dpo/step_300`
- `checkpoints/local_training/diffusion_grpo/step_100`

## Useful overrides

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

Every training recipe writes `run_metrics.json` to the final checkpoint
directory and copies it into the recipe save directory. On CUDA the file
includes `max_memory_allocated_gb` and `max_memory_reserved_gb` from
`torch.cuda` peak memory stats; on CPU those keys are absent. Open
`run_metrics.json` after a run for actual numbers; the per-recipe
`expected_metrics.md` files describe what to look for, not measured
benchmarks.
