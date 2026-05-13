#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

TOKENIZER="${TOKENIZER:-checkpoints/local_training/tokenizer.json}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-checkpoints/local_training/sft/step_500}"
ALGORITHM="${ALGORITHM:-grpo}"
SAVE_DIR="${SAVE_DIR:-checkpoints/local_training/${ALGORITHM}}"
PRESET="${PRESET:-gpt-10m}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-50}"

python scripts/estimate_vram.py \
  --model "$PRESET" \
  --method "$ALGORITHM" \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE" \
  --num-generations "$NUM_GENERATIONS" \
  --max-new-tokens "$MAX_NEW_TOKENS"

cmd=(
  python scripts/grpo.py
  --algorithm "$ALGORITHM"
  --tokenizer "$TOKENIZER"
  --checkpoint "$SFT_CHECKPOINT"
  --save-dir "$SAVE_DIR"
  --seq-len "$SEQ_LEN"
  --max-steps "$MAX_STEPS"
  --warmup-steps "$WARMUP_STEPS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --max-examples "$MAX_EXAMPLES"
  --eval-examples "$EVAL_EXAMPLES"
)

if [[ "$ALGORITHM" != "ppo" ]]; then
  cmd+=(--num-generations "$NUM_GENERATIONS")
fi

"${cmd[@]}"
