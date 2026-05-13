#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

TOKENIZER="${TOKENIZER:-checkpoints/local_training/tokenizer.json}"
DATASET="${DATASET:-tinystories}"
SEQ_LEN="${SEQ_LEN:-512}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-checkpoints/local_training/lm/step_1000}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-checkpoints/local_training/sft/step_500}"
PREFERENCE_CHECKPOINT="${PREFERENCE_CHECKPOINT:-checkpoints/local_training/preference_dpo/step_300}"
GRPO_CHECKPOINT="${GRPO_CHECKPOINT:-checkpoints/local_training/grpo/step_100}"

CHECKPOINTS=(
  "base:${BASE_CHECKPOINT}"
  "sft:${SFT_CHECKPOINT}"
  "preference:${PREFERENCE_CHECKPOINT}"
  "grpo:${GRPO_CHECKPOINT}"
)

for item in "${CHECKPOINTS[@]}"; do
  label="${item%%:*}"
  checkpoint="${item#*:}"
  if [[ ! -d "$checkpoint" ]]; then
    echo "Skipping ${label}: missing ${checkpoint}"
    continue
  fi

  echo
  echo "== ${label}: ${checkpoint} =="
  python scripts/evaluate.py \
    --tokenizer "$TOKENIZER" \
    --checkpoint "$checkpoint" \
    --dataset "$DATASET" \
    --seq-len "$SEQ_LEN" \
    --num-samples "$NUM_SAMPLES"
done
