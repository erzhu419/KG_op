#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="${DEPLOY:-/home/erzhu419/mine_code/KG_op_scheduler_deploy}"
REMOTE="${REMOTE:-zhengliang01@202.197.46.16}"
PROXY="${PROXY:-jtl110gpu}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy}"

EXCLUDES=(
  --exclude='profiles/'
  --exclude='results/'
  --exclude='checkpoints/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
)

mkdir -p "$DEPLOY/SC-OLH-KG"
rsync -a --delete "${EXCLUDES[@]}" \
  "$ROOT/SC-OLH-KG/" "$DEPLOY/SC-OLH-KG/"

ssh -o ConnectTimeout=30 -o BatchMode=yes -J "$PROXY" "$REMOTE" \
  "mkdir -p '$REMOTE_ROOT/SC-OLH-KG'"
rsync -a --delete "${EXCLUDES[@]}" \
  -e "ssh -o ConnectTimeout=30 -o BatchMode=yes -J $PROXY" \
  "$DEPLOY/SC-OLH-KG/" "$REMOTE:$REMOTE_ROOT/SC-OLH-KG/"

echo "synced SC-OLH-KG: $ROOT -> $REMOTE:$REMOTE_ROOT"
