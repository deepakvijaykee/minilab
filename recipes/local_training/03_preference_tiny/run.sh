#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

TOKENIZER="${TOKENIZER:-checkpoints/local_training/tokenizer.json}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-checkpoints/local_training/sft/step_500}"
ALGORITHM="${ALGORITHM:-dpo}"
DATASET="${DATASET:-hh}"
SAVE_DIR="${SAVE_DIR:-checkpoints/local_training/preference_${ALGORITHM}}"
PRESET="${PRESET:-gpt-10m}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-300}"
WARMUP_STEPS="${WARMUP_STEPS:-30}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-5}"
BETA="${BETA:-0.1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"

python scripts/estimate_vram.py \
  --model "$PRESET" \
  --method "$ALGORITHM" \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE"

python scripts/preference.py \
  --algorithm "$ALGORITHM" \
  --dataset "$DATASET" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$SFT_CHECKPOINT" \
  --save-dir "$SAVE_DIR" \
  --seq-len "$SEQ_LEN" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --beta "$BETA" \
  --max-examples "$MAX_EXAMPLES"
