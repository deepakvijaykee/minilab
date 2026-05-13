#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

TOKENIZER="${TOKENIZER:-checkpoints/local_training/tokenizer.json}"
DIFFUSION_SFT_CHECKPOINT="${DIFFUSION_SFT_CHECKPOINT:-checkpoints/local_training/diffusion_sft/step_500}"
ALGORITHM="${ALGORITHM:-dpo}"
DATASET="${DATASET:-hh-rlhf}"
SAVE_DIR="${SAVE_DIR:-checkpoints/local_training/diffusion_${ALGORITHM}}"
PRESET="${PRESET:-mdlm-25m}"
SEQ_LEN="${SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-300}"
WARMUP_STEPS="${WARMUP_STEPS:-30}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-5}"
BETA="${BETA:-0.1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"
SAMPLE_NEW_TOKENS="${SAMPLE_NEW_TOKENS:-80}"
VRPO_NUM_SAMPLES="${VRPO_NUM_SAMPLES:-4}"

ESTIMATE_METHOD="diffusion_dpo"
if [[ "$ALGORITHM" == "vrpo" ]]; then
  ESTIMATE_METHOD="diffusion_vrpo"
fi

python scripts/estimate_vram.py \
  --model "$PRESET" \
  --method "$ESTIMATE_METHOD" \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE"

cmd=(
  python scripts/dpo_diffusion.py
  --algorithm "$ALGORITHM"
  --tokenizer "$TOKENIZER"
  --checkpoint "$DIFFUSION_SFT_CHECKPOINT"
  --save-dir "$SAVE_DIR"
  --dataset "$DATASET"
  --seq-len "$SEQ_LEN"
  --max-steps "$MAX_STEPS"
  --warmup-steps "$WARMUP_STEPS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --beta "$BETA"
  --max-examples "$MAX_EXAMPLES"
  --sample-new-tokens "$SAMPLE_NEW_TOKENS"
)

if [[ "$ALGORITHM" == "vrpo" ]]; then
  cmd+=(--vrpo-num-samples "$VRPO_NUM_SAMPLES")
fi

"${cmd[@]}"
