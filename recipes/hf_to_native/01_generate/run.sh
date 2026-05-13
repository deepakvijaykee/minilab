#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

MODEL="${MODEL:-smollm2-135m-instruct}"
PROMPT="${PROMPT:-Explain gravity in one paragraph.}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"

cmd=(
  python scripts/hf_generate.py
  --model "$MODEL"
  --prompt "$PROMPT"
  --device "$DEVICE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
)

if [[ -n "$DTYPE" ]]; then
  cmd+=(--dtype "$DTYPE")
fi

"${cmd[@]}"
