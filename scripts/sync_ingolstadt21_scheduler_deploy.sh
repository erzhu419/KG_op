#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="${DEPLOY:-/home/erzhu419/mine_code/KG_op_scheduler_deploy}"
REMOTE="${REMOTE:-zhengliang01@202.197.46.16}"
PROXY="${PROXY:-jtl110gpu}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy}"
SSH_OPTS=(
  -o ConnectTimeout=30
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=6
  -o BatchMode=yes
  -J "$PROXY"
)

retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$@"; then
      return 0
    fi
    echo "retry $attempt: $*" >&2
    sleep 3
  done
  return 1
}

tar_to_remote() {
  local src_dir="$1"
  local rel="$2"
  local dst_dir="$3"
  retry bash -lc \
    "tar -C '$src_dir' -cf - '$rel' | ssh ${SSH_OPTS[*]@Q} '$REMOTE' 'mkdir -p \"$dst_dir\" && tar -C \"$dst_dir\" -xf -'"
}

rm -rf "$DEPLOY"
mkdir -p "$DEPLOY/Final_Submission/GPR_KG_Code/results/ingolstadt21"

rsync -a --delete \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='results/' \
  --exclude='figures/' \
  "$ROOT/Final_Submission/GPR_KG_Code/" \
  "$DEPLOY/Final_Submission/GPR_KG_Code/"

mkdir -p "$DEPLOY/Final_Submission/GPR_KG_Code/results/ingolstadt21"
cp -a "$ROOT/Final_Submission/GPR_KG_Code/results/ingolstadt21/baseline.json" \
  "$DEPLOY/Final_Submission/GPR_KG_Code/results/ingolstadt21/"
cp -a "$ROOT/Final_Submission/GPR_KG_Code/results/ingolstadt21/decision_space.json" \
  "$DEPLOY/Final_Submission/GPR_KG_Code/results/ingolstadt21/"

python3 - <<'PY'
from pathlib import Path
import shutil

root = Path("/home/erzhu419/mine_code/KG_op")
src = root / "Final_Submission/GPR_KG_Code/results/ingolstadt21"
dst = Path("/home/erzhu419/mine_code/KG_op_scheduler_deploy/Final_Submission/GPR_KG_Code/results/ingolstadt21")
count = 0
for path in src.glob("*/summary.json"):
    out = dst / path.parent.name / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out)
    count += 1
print(f"copied original summary files: {count}")
PY

rsync -a --delete \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='profiles/' \
  --exclude='results/' \
  --exclude='checkpoints/' \
  "$ROOT/SC-OLH-KG/" \
  "$DEPLOY/SC-OLH-KG/"

ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$REMOTE_ROOT'"
tar_to_remote "$DEPLOY" "." "$REMOTE_ROOT"

ssh "${SSH_OPTS[@]}" "$REMOTE" \
  "du -sh '$REMOTE_ROOT'; test -f '$REMOTE_ROOT/Final_Submission/GPR_KG_Code/results/ingolstadt21/baseline.json'; test -f '$REMOTE_ROOT/SC-OLH-KG/performance/benchmark_traffic_ingolstadt21.py'"

echo "local deploy:  $DEPLOY"
echo "remote deploy: $REMOTE_ROOT"
