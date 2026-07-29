#!/usr/bin/env bash
set -euo pipefail

# node007 has the shared Python/SUMO 1.25 installation.  jtl110gpu has the
# same SUMO release in the offline-sumo environment and the audited BoTorch
# runtime in scheduleurm-torch-bench.  Force TraCI on both hosts so execution
# semantics do not depend on a Python-version-specific libsumo extension.
if [[ -x /home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python ]]; then
    python_bin=/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python
    sumo_root=/home/zhengliang01/scheduleurm_work/python_pkgs/eclipse_sumo_1_25
    sumo_bin="${sumo_root}/sumo/bin/sumo"
    sumo_tools="${sumo_root}/sumo/tools"
    export LD_LIBRARY_PATH="${sumo_root}/libsumo.libs:${sumo_root}/eclipse_sumo.libs:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="/home/zhengliang01/scheduleurm_work/python_pkgs/botorch_overlay_py310:${sumo_root}:${sumo_tools}:${PYTHONPATH:-}"
elif [[ -x /home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python ]] \
    && [[ -x /home/erzhu419/.conda/envs/offline-sumo/bin/sumo ]]; then
    python_bin=/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python
    sumo_root=/home/erzhu419/.conda/envs/offline-sumo/lib/python3.8/site-packages/sumo
    sumo_bin=/home/erzhu419/.conda/envs/offline-sumo/bin/sumo
    sumo_tools="${sumo_root}/tools"
    export PYTHONPATH="${sumo_tools}:${PYTHONPATH:-}"
else
    echo "No audited CUDA + SUMO 1.25 runtime is available on $(hostname)" >&2
    exit 127
fi

sumo_version="$("${sumo_bin}" --version 2>&1 | head -n 1)"
if [[ "${sumo_version}" != *"sumo 1.25.0"* ]]; then
    echo "Expected SUMO 1.25.0, observed: ${sumo_version}" >&2
    exit 64
fi

export SUMO_HOME="${sumo_root}"
export PATH="$(dirname "${sumo_bin}"):/usr/local/bin:/usr/bin:/bin"
export INGOLSTADT21_SUMO_BACKEND=traci
exec "${python_bin}" "$@"
