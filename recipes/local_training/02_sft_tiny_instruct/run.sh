#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

TOKENIZER="${TOKENIZER:-checkpoints/local_training/tokenizer.json}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-checkpoints/local_training/lm/step_1000}"
SAVE_DIR="${SAVE_DIR:-checkpoints/local_training/sft}"
PRESET="${PRESET:-gpt-10m}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-500}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-2000}"

python scripts/estimate_vram.py \
  --model "$PRESET" \
  --method sft \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE"

python scripts/sft.py \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$BASE_CHECKPOINT" \
  --save-dir "$SAVE_DIR" \
  --seq-len "$SEQ_LEN" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-examples "$MAX_EXAMPLES"
