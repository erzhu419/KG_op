#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python}"
PKG_ROOT="${PKG_ROOT:-/home/zhengliang01/scheduleurm_work/python_pkgs}"
WHEEL_ROOT="${WHEEL_ROOT:-/home/zhengliang01/scheduleurm_work/wheelhouse/scolhkg_transfer_wheels}"
TRANSFER_OVERLAY="$PKG_ROOT/transfergpbo_py310"
HYPERBO_OVERLAY="$PKG_ROOT/hyperbo_py310"
TRANSFER_WHEELS="$WHEEL_ROOT/transfergpbo_py310"
HYPERBO_WHEELS="$WHEEL_ROOT/hyperbo_py310"

mkdir -p "$TRANSFER_OVERLAY" "$HYPERBO_OVERLAY"
test -d "$TRANSFER_WHEELS"
test -d "$HYPERBO_WHEELS"

"$PYTHON" -m pip install --no-index --upgrade \
  --find-links "$TRANSFER_WHEELS" --target "$TRANSFER_OVERLAY" \
  'numpy==1.26.4' \
  'scipy==1.12.0' \
  'GPy==1.13.2'
"$PYTHON" -m pip install --no-index --upgrade --no-deps \
  --find-links "$TRANSFER_WHEELS" --target "$TRANSFER_OVERLAY" \
  'emukit==0.4.9' 'emcee==3.1.6'

"$PYTHON" -m pip install --no-index --upgrade \
  --find-links "$HYPERBO_WHEELS" --target "$HYPERBO_OVERLAY" \
  'jax==0.5.3' \
  'jaxlib==0.5.3' \
  'flax==0.10.6' \
  'optax==0.2.4' \
  'jaxopt==0.8.5' \
  'clu==0.0.12' \
  'tensorflow-probability==0.25.0'

PYTHONPATH="$TRANSFER_OVERLAY" "$PYTHON" -c \
  'import GPy, emukit, numpy, scipy; print("transfer overlay ok", numpy.__version__, scipy.__version__)'
PYTHONPATH="$TRANSFER_OVERLAY:$HYPERBO_OVERLAY" "$PYTHON" -c \
  'import jax, flax, optax; print("hyperbo overlay ok", jax.__version__, flax.__version__)'
