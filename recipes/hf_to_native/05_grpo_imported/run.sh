#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

MODEL="${MODEL:-smollm2-135m}"
ALGORITHM="${ALGORITHM:-grpo}"
TOKENIZER="${TOKENIZER:-checkpoints/imported/${MODEL}/tokenizer.json}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-checkpoints/imported/${MODEL}-simpo/step_50}"
SAVE_DIR="${SAVE_DIR:-checkpoints/imported/${MODEL}-${ALGORITHM}}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-25}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-5e-6}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-20}"

python scripts/grpo.py \
  --algorithm "$ALGORITHM" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$POLICY_CHECKPOINT" \
  --save-dir "$SAVE_DIR" \
  --seq-len "$SEQ_LEN" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --num-generations "$NUM_GENERATIONS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-examples "$MAX_EXAMPLES" \
  --eval-examples "$EVAL_EXAMPLES"
