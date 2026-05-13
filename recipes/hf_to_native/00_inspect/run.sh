#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

MODEL="${MODEL:-smollm2-135m}"
LOAD="${LOAD:-0}"

cmd=(python scripts/hf_inspect.py --model "$MODEL")
if [[ "$LOAD" == "1" ]]; then
  cmd+=(--load)
fi

"${cmd[@]}"
