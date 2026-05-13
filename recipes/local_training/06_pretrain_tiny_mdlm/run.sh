#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

TOKENIZER="${TOKENIZER:-checkpoints/local_training/tokenizer.json}"
PRESET="${PRESET:-mdlm-25m}"
SAVE_DIR="${SAVE_DIR:-checkpoints/local_training/diffusion}"
DATASET="${DATASET:-tinystories}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-3e-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-10000}"

python scripts/estimate_vram.py \
  --model "$PRESET" \
  --method diffusion_pretrain \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE" \
  --grad-checkpoint

python scripts/pretrain_diffusion.py \
  --tokenizer "$TOKENIZER" \
  --preset "$PRESET" \
  --dataset "$DATASET" \
  --seq-len "$SEQ_LEN" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-examples "$MAX_EXAMPLES" \
  --save-dir "$SAVE_DIR" \
  --grad-checkpoint
