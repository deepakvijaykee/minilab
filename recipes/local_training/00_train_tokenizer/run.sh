#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

TOKENIZER="${TOKENIZER:-checkpoints/local_training/tokenizer.json}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-bpe}"
DATASET="${DATASET:-tinystories}"
VOCAB_SIZE="${VOCAB_SIZE:-4096}"
NUM_TEXTS="${NUM_TEXTS:-5000}"

mkdir -p "$(dirname "$TOKENIZER")"

python scripts/train_tokenizer.py \
  --type "$TOKENIZER_TYPE" \
  --dataset "$DATASET" \
  --vocab-size "$VOCAB_SIZE" \
  --num-texts "$NUM_TEXTS" \
  --save "$TOKENIZER"
