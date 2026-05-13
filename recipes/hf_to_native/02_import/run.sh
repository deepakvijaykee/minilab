#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

MODEL="${MODEL:-smollm2-135m}"
SAVE_DIR="${SAVE_DIR:-checkpoints/imported/${MODEL}}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
DEVICE="${DEVICE:-cpu}"
VERIFY="${VERIFY:-1}"

cmd=(
  python scripts/import_hf.py
  --model "$MODEL"
  --save-dir "$SAVE_DIR"
  --max-seq-len "$MAX_SEQ_LEN"
  --device "$DEVICE"
)

if [[ "$VERIFY" == "1" ]]; then
  cmd+=(--verify)
fi

"${cmd[@]}"
