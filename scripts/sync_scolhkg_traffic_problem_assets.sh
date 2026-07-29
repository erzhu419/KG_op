#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="${DEPLOY:-/home/erzhu419/mine_code/KG_op_scheduler_deploy}"
REMOTE="${REMOTE:-zhengliang01@202.197.46.16}"
PROXY="${PROXY:-jtl110gpu}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy}"
GPU_HOSTS="${GPU_HOSTS:-jtl110gpu jtl110gpu2}"

RELATIVE="Final_Submission/GPR_KG_Code/results/ingolstadt21/decision_space.json"
SOURCE="$ROOT/$RELATIVE"
EXPECTED_SHA256="86bdb6504ce9f1fd31388ad4d30b37312e7f136d67aad181cdc34362aa4659a0"

if [[ ! -f "$SOURCE" ]]; then
  echo "missing traffic problem asset: $SOURCE" >&2
  exit 1
fi
OBSERVED_SHA256="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [[ "$OBSERVED_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "traffic problem asset hash changed: $OBSERVED_SHA256" >&2
  exit 1
fi

LOCAL_PARENT="$(dirname "$DEPLOY/$RELATIVE")"
mkdir -p "$LOCAL_PARENT"
rsync -a "$SOURCE" "$LOCAL_PARENT/"

for host in $GPU_HOSTS; do
  remote_path="/home/erzhu419/mine_code/KG_op_scheduler_deploy/$RELATIVE"
  ssh -o BatchMode=yes -o ConnectTimeout=30 "$host" \
    "mkdir -p '$(dirname "$remote_path")'"
  rsync -a \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=30" \
    "$SOURCE" "$host:$(dirname "$remote_path")/"
  remote_hash="$(
    ssh -o BatchMode=yes -o ConnectTimeout=30 "$host" \
      "sha256sum '$remote_path'" | awk '{print $1}'
  )"
  if [[ "$remote_hash" != "$EXPECTED_SHA256" ]]; then
    echo "traffic asset verification failed on $host" >&2
    exit 1
  fi
done

hpc_path="$REMOTE_ROOT/$RELATIVE"
ssh -o BatchMode=yes -o ConnectTimeout=30 -J "$PROXY" "$REMOTE" \
  "mkdir -p '$(dirname "$hpc_path")'"
rsync -a \
  -e "ssh -o BatchMode=yes -o ConnectTimeout=30 -J $PROXY" \
  "$SOURCE" "$REMOTE:$(dirname "$hpc_path")/"
hpc_hash="$(
  ssh -o BatchMode=yes -o ConnectTimeout=30 -J "$PROXY" "$REMOTE" \
    "sha256sum '$hpc_path'" | awk '{print $1}'
)"
if [[ "$hpc_hash" != "$EXPECTED_SHA256" ]]; then
  echo "traffic asset verification failed on the HPC shared deploy" >&2
  exit 1
fi

echo "synced traffic problem asset: $EXPECTED_SHA256"
