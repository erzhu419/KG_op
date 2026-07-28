#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="${DEPLOY:-/home/erzhu419/mine_code/KG_op_scheduler_deploy}"
REMOTE="${REMOTE:-zhengliang01@202.197.46.16}"
PROXY="${PROXY:-jtl110gpu}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy}"

REPOS=(
  jonasrothfuss__f-pacoh-torch
  machinelearningnuremberg__FSBO
  boschresearch__MALIBO
  boschresearch__MetaBO
  google-research__hyperbo
  boschresearch__transfergpbo
)
EXCLUDES=(
  --exclude='.git/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='hpob-data/'
  --exclude='benchmark_results/'
  --exclude='checkpoints/'
  --exclude='results/'
)

mkdir -p "$DEPLOY/external_repos"
for repo in "${REPOS[@]}"; do
  rsync -a "${EXCLUDES[@]}" \
    "$ROOT/repo/clones/$repo/" "$DEPLOY/external_repos/$repo/"
done

ssh -o ConnectTimeout=30 -o BatchMode=yes -J "$PROXY" "$REMOTE" \
  "mkdir -p '$REMOTE_ROOT/external_repos'"
rsync -a "${EXCLUDES[@]}" \
  -e "ssh -o ConnectTimeout=30 -o BatchMode=yes -J $PROXY" \
  "$DEPLOY/external_repos/" "$REMOTE:$REMOTE_ROOT/external_repos/"

echo "synced transfer repos: $ROOT/repo/clones -> $REMOTE:$REMOTE_ROOT/external_repos"
