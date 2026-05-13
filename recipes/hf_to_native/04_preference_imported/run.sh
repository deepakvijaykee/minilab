#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

MODEL="${MODEL:-smollm2-135m}"
ALGORITHM="${ALGORITHM:-simpo}"
DATASET="${DATASET:-hh}"
TOKENIZER="${TOKENIZER:-checkpoints/imported/${MODEL}/tokenizer.json}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-checkpoints/imported/${MODEL}-sft/step_100}"
SAVE_DIR="${SAVE_DIR:-checkpoints/imported/${MODEL}-${ALGORITHM}}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
BETA="${BETA:-0.1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"

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
