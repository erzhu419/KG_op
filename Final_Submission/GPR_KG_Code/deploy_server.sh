#!/usr/bin/env bash
# Server-side deploy & run script for Phase 2 of the InTAS case study.
#
# Prerequisites (install once on the server):
#   - Python >= 3.10 with packages: numpy, scipy, matplotlib, pymoo
#   - SUMO >= 1.18 with libsumo Python bindings
#   - InTAS repository cloned locally
#
# Usage (on server):
#   1. scp/rsync this entire directory to server, e.g.
#        scp -r GPR_KG_Code/ user@server:~/intas_experiment/
#   2. scp InTAS repo to server (or clone fresh):
#        git clone https://github.com/silaslobo/InTAS ~/InTAS
#   3. Edit the two path variables below, then:
#        cd ~/intas_experiment/GPR_KG_Code
#        bash deploy_server.sh

set -euo pipefail

# ==== User-editable paths ==================================================
SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
INTAS_ROOT="${INTAS_ROOT:-$HOME/InTAS}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# ============================================================================

echo "================================================================"
echo "Server-side Phase 2 launcher"
echo "  SUMO_HOME   = $SUMO_HOME"
echo "  INTAS_ROOT  = $INTAS_ROOT"
echo "  PYTHON_BIN  = $PYTHON_BIN"
echo "================================================================"

export SUMO_HOME
export INTAS_ROOT
export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"

# Sanity checks
if [ ! -d "$SUMO_HOME" ]; then
    echo "ERROR: SUMO_HOME=$SUMO_HOME does not exist"
    exit 1
fi
if [ ! -f "$INTAS_ROOT/scenario/ingolstadt.net.xml" ]; then
    echo "ERROR: InTAS net file not found at $INTAS_ROOT/scenario/ingolstadt.net.xml"
    exit 1
fi
"$PYTHON_BIN" -c "import libsumo; print('libsumo OK:', libsumo.__file__)" || {
    echo "ERROR: libsumo import failed"
    exit 1
}
"$PYTHON_BIN" -c "import numpy, scipy, matplotlib, pymoo; print('deps OK')" || {
    echo "ERROR: python dependency missing"
    exit 1
}

# Make sure results dir exists and Phase 0/1 artifacts were transferred
for f in results/intas/baseline.json \
         results/intas/hetero_test.json \
         results/intas/decision_space.json ; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f missing — transfer Phase 0/1 results before running"
        exit 1
    fi
done

mkdir -p logs

echo ""
echo "Smoke test: run 1 simulation to verify pipeline..."
"$PYTHON_BIN" -u -c "
import json, numpy as np, sys, time
sys.path.insert(0, '.')
from experiments.intas.sumo_sim import simulate
from experiments.intas.config import RESULTS_DIR
ds = json.load(open(RESULTS_DIR + '/decision_space.json'))
var_map = [tuple(v) for v in ds['var_map']]
default_x = np.array(ds['defaults'])
t0 = time.time()
y = simulate(var_map, default_x, None, T0=1.0, A0=1.0, E0=1.0, seed=42)
print(f'Smoke: avg_tt={y[0]:.1f}s atkinson={y[1]:.4f} CO2={y[2]:.0f}kg  dt={time.time()-t0:.0f}s')
"

echo ""
echo "Launching Phase 2 — 2 methods in parallel (if CPUs allow)"
echo ""

# GPR-KG (with VEPM)
nohup "$PYTHON_BIN" -u -m experiments.intas.run_main \
    --method GPR-KG --n0 100 --N 300 --seed 100 \
    > logs/phase2_gprkg.log 2>&1 &
PID_GPR=$!
echo "  GPR-KG    started: PID=$PID_GPR   log: logs/phase2_gprkg.log"

# GPR-KG-nV (ablation)
nohup "$PYTHON_BIN" -u -m experiments.intas.run_main \
    --method GPR-KG-nV --n0 100 --N 300 --seed 100 \
    > logs/phase2_gprkgnv.log 2>&1 &
PID_NV=$!
echo "  GPR-KG-nV started: PID=$PID_NV   log: logs/phase2_gprkgnv.log"

echo ""
echo "Monitor with:"
echo "  tail -f logs/phase2_gprkg.log"
echo "  tail -f logs/phase2_gprkgnv.log"
echo ""
echo "Each method runs ~77h (400 sims x ~11.5 min).  Checkpoint is"
echo "saved after every iteration to results/intas/{METHOD}_run/checkpoint.pkl,"
echo "so an interrupted run can be resumed by re-running the same command."
echo ""
echo "When both PIDs exit, collect the results from:"
echo "  results/intas/GPR_KG_run/    (checkpoint.pkl, snapshots.jsonl, summary.json)"
echo "  results/intas/GPR_KG_nV_run/ (same layout)"
