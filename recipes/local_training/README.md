# Local training recipes

This directory holds the end-to-end path for local, single-device
training. Every run begins with the tokenizer in recipe 00, then branches
into either the autoregressive GPT track or the diffusion MDLM track.
Both branches pass through pretraining, SFT, preference optimization,
and RLVR before landing in evaluation, so reading the two tracks side by
side is the cleanest way to see how the same alignment idea changes
shape when the underlying density model changes.

The defaults are deliberately small. They make the full loop runnable in
a few minutes per stage, which is the right scale for confirming that
the pipeline works end to end on a given machine and for forming
intuition about how each stage changes the previous one. They are not
sized for quality. Once the path runs cleanly, the right next move is
to scale `MAX_STEPS`, `MAX_EXAMPLES`, `PRESET`, and the batch settings
up rather than to change the structure of the pipeline.

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

The diffusion branch mirrors the autoregressive branch stage for stage,
but keeps a diffusion objective at every stage rather than reducing the
model to a next-token predictor for alignment. This is the more
informative comparison: it isolates whether a behavior depends on the
alignment algorithm or on the underlying density model.

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

Every training recipe writes `run_metrics.json` into the final
checkpoint directory and copies it into the recipe save directory. On
CUDA the file records `max_memory_allocated_gb` and
`max_memory_reserved_gb` from `torch.cuda` peak memory statistics. On
CPU those keys are simply absent rather than zeroed. The convention
across the lab is that `run_metrics.json` carries the actual measured
numbers, while each recipe's `expected_metrics.md` describes the shape
of the curve and what to read from it. The two documents complement
each other: one says what happened, the other says what should have
happened and why.
