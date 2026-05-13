#!/usr/bin/env bash
set -euo pipefail

ROOT="${MINILAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"

PATTERN="${PATTERN:-checkpoints/imported/smollm2-135m-*}"
JSON="${JSON:-0}"

cmd=(python scripts/summarize_runs.py "$PATTERN")
if [[ "$JSON" == "1" ]]; then
  cmd+=(--json)
fi

"${cmd[@]}"
