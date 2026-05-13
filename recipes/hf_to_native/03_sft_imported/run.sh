#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

MODEL="${MODEL:-smollm2-135m}"
IMPORT_DIR="${IMPORT_DIR:-checkpoints/imported/${MODEL}}"
TOKENIZER="${TOKENIZER:-${IMPORT_DIR}/tokenizer.json}"
CHECKPOINT="${CHECKPOINT:-${IMPORT_DIR}}"
SAVE_DIR="${SAVE_DIR:-checkpoints/imported/${MODEL}-sft}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-2e-5}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"

python scripts/sft.py \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$CHECKPOINT" \
  --save-dir "$SAVE_DIR" \
  --seq-len "$SEQ_LEN" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-examples "$MAX_EXAMPLES"
