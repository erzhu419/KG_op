#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-kgop-sumo:1.25}"
RUNTIME="${RUNTIME:-docker}"

if ! command -v "$RUNTIME" >/dev/null 2>&1; then
  echo "Container runtime '$RUNTIME' not found." >&2
  exit 127
fi

"$RUNTIME" run --rm \
  -v "$ROOT:/workspace" \
  -w /workspace/Final_Submission/GPR_KG_Code \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    python --version
    sumo --version | head -4
    python - <<'"'"'PY'"'"'
import shutil
import sys
import numpy
import scipy
import sumolib
import traci
import libsumo

print("python", sys.version.split()[0])
print("sumo_bin", shutil.which("sumo"))
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("sumolib", getattr(sumolib, "__version__", "1.25.0"), sumolib.__file__)
print("traci", getattr(traci, "__version__", "1.25.0"), traci.__file__)
print("libsumo", getattr(libsumo, "__version__", "1.25.0"), libsumo.__file__)
PY
    python -m experiments.ingolstadt21.validate_oos_feasibility --dry-run >/tmp/ingolstadt21_dry_run.txt
    head -5 /tmp/ingolstadt21_dry_run.txt
    echo DONE
  '
