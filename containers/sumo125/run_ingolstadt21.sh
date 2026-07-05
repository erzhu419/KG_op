#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-kgop-sumo:1.25}"
RUNTIME="${RUNTIME:-docker}"

if ! command -v "$RUNTIME" >/dev/null 2>&1; then
  echo "Container runtime '$RUNTIME' not found." >&2
  exit 127
fi

"$RUNTIME" run --rm -it \
  -v "$ROOT:/workspace" \
  -w /workspace/Final_Submission/GPR_KG_Code \
  "$IMAGE" \
  python -m experiments.ingolstadt21.validate_oos_feasibility "$@"
