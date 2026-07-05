#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-kgop-sumo:1.25}"
RUNTIME="${RUNTIME:-docker}"

if ! command -v "$RUNTIME" >/dev/null 2>&1; then
  echo "Container runtime '$RUNTIME' not found." >&2
  echo "Install/enable Docker Desktop for WSL, or run with RUNTIME=podman if available." >&2
  exit 127
fi

"$RUNTIME" build \
  -t "$IMAGE" \
  -f "$ROOT/containers/sumo125/Dockerfile" \
  "$ROOT"
