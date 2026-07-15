#!/usr/bin/env bash
set -euo pipefail

# Run on jtl110gpu or jtl110gpu2, which can reach PyPI.  The interpreter
# running pip may be Python 3.11; pip is instructed to resolve CPython 3.10
# manylinux wheels for the offline zndx environment.
HOST="$(hostname -s)"
case "$HOST" in
  jtl110gpu|jtl110gpu2) ;;
  *)
    echo "refusing network download on $HOST; run on jtl110gpu or jtl110gpu2" >&2
    exit 2
    ;;
esac

PYTHON="${PYTHON:-$HOME/.venvs/scheduleurm-torch-bench/bin/python}"
WHEEL_ROOT="${WHEEL_ROOT:-$HOME/.cache/scolhkg_transfer_wheels}"
TRANSFER_WHEELS="$WHEEL_ROOT/transfergpbo_py310"
HYPERBO_WHEELS="$WHEEL_ROOT/hyperbo_py310"
TARGET=(
  --only-binary=:all:
  --platform manylinux2014_x86_64
  --python-version 310
  --implementation cp
  --abi cp310
)

mkdir -p "$TRANSFER_WHEELS" "$HYPERBO_WHEELS"
"$PYTHON" -m pip download --dest "$TRANSFER_WHEELS" "${TARGET[@]}" \
  'numpy==1.26.4' \
  'scipy==1.12.0' \
  'GPy==1.13.2'
"$PYTHON" -m pip download --dest "$TRANSFER_WHEELS" "${TARGET[@]}" \
  --no-deps 'emukit==0.4.9' 'emcee==3.1.6'

"$PYTHON" -m pip download --dest "$HYPERBO_WHEELS" "${TARGET[@]}" \
  'jax==0.5.3' \
  'jaxlib==0.5.3' \
  'flax==0.10.6' \
  'optax==0.2.4' \
  'jaxopt==0.8.5' \
  'clu==0.0.12' \
  'tensorflow-probability==0.25.0'

echo "downloaded CPython 3.10 transfer wheels to $WHEEL_ROOT"
