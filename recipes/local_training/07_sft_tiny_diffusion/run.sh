#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

TOKENIZER="${TOKENIZER:-checkpoints/local_training/tokenizer.json}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-checkpoints/local_training/diffusion/step_1000}"
SAVE_DIR="${SAVE_DIR:-checkpoints/local_training/diffusion_sft}"
PRESET="${PRESET:-mdlm-25m}"
DATASET="${DATASET:-alpaca}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-500}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-2000}"
SAMPLE_NEW_TOKENS="${SAMPLE_NEW_TOKENS:-80}"

python scripts/estimate_vram.py \
  --model "$PRESET" \
  --method diffusion_sft \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE" \
  --grad-checkpoint

python scripts/sft_diffusion.py \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$DIFFUSION_CHECKPOINT" \
  --save-dir "$SAVE_DIR" \
  --dataset "$DATASET" \
  --seq-len "$SEQ_LEN" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-examples "$MAX_EXAMPLES" \
  --sample-new-tokens "$SAMPLE_NEW_TOKENS" \
  --grad-checkpoint
