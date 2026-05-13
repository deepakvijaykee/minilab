#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

TOKENIZER="${TOKENIZER:-checkpoints/local_training/tokenizer.json}"
DIFFUSION_SFT_CHECKPOINT="${DIFFUSION_SFT_CHECKPOINT:-checkpoints/local_training/diffusion_sft/step_500}"
SAVE_DIR="${SAVE_DIR:-checkpoints/local_training/diffusion_grpo}"
PRESET="${PRESET:-mdlm-25m}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
INNER_EPOCHS="${INNER_EPOCHS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-64}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-50}"

python scripts/estimate_vram.py \
  --model "$PRESET" \
  --method diffusion_grpo \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE" \
  --num-generations "$NUM_GENERATIONS" \
  --max-new-tokens "$MAX_NEW_TOKENS"

python scripts/grpo_diffusion.py \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$DIFFUSION_SFT_CHECKPOINT" \
  --save-dir "$SAVE_DIR" \
  --seq-len "$SEQ_LEN" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --num-generations "$NUM_GENERATIONS" \
  --inner-epochs "$INNER_EPOCHS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --diffusion-steps "$DIFFUSION_STEPS" \
  --max-examples "$MAX_EXAMPLES" \
  --eval-examples "$EVAL_EXAMPLES"
