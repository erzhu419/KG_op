# Phase 2 Server Deployment — Step-by-Step

Server ID: `3112080`  (tracked by the user)

## 1. Transfer files FROM Windows local TO server

Run the following from **your Windows machine** (PowerShell or Git Bash):

```powershell
# Replace USER and HOST with your SSH credentials
$DEST_USER = "user"
$DEST_HOST = "3112080.server.hostname"
$DEST_PATH = "~/intas_experiment"

# 1. Copy the code (entire GPR_KG_Code tree, excluding huge __pycache__)
scp -r `
    "E:/从Dell移动到X1/科研同行/谭真/OR投稿/Final_Submission/GPR_KG_Code" `
    "${DEST_USER}@${DEST_HOST}:${DEST_PATH}/"

# 2. Copy the InTAS repo  (if server does NOT yet have it)
scp -r "C:/InTAS" "${DEST_USER}@${DEST_HOST}:~/InTAS"

# Alternative for step 2 (faster on server):
# ssh user@host "git clone https://github.com/silaslobo/InTAS ~/InTAS"
```

**Critical files that must arrive at server** (already in GPR_KG_Code):

| File | Purpose | Why it must be transferred (cannot be regenerated on server) |
|------|---------|---|
| `results/intas/baseline.json` | Phase 0 output (T0, A0, E0) | Required for normalisation in Phase 2 |
| `results/intas/hetero_test.json` | Phase 1 output (variance test) | Used to pick reference plan for VEPM tracking figure |
| `results/intas/hetero_test.partial.jsonl` | Phase 1 raw per-rep data | For possible post-hoc re-analysis |
| `results/intas/hetero_test.plans.json` | The 20 LHS plans used in Phase 1 | For picking `x_ref` in Phase 2 |
| `results/intas/decision_space.json` | Decision bounds and var_map | Re-derivable but avoids re-parse of 16 MB net.xml |

## 2. Install prerequisites on server (one-time)

```bash
# On server (SSH in first)
ssh user@3112080.server.hostname

# Python 3.10+
python3 --version   # must be >= 3.10

# Python packages (venv recommended)
python3 -m venv ~/venv_intas
source ~/venv_intas/bin/activate
pip install numpy scipy matplotlib pymoo

# SUMO
#   On Ubuntu:   sudo apt install sumo sumo-tools sumo-doc
#   On CentOS:   yum install sumo
#   Or build from source: https://sumo.dlr.de/docs/Installing/Linux_Build.html
# After install, verify:
which sumo
python3 -c "import libsumo; print(libsumo.__file__)"
```

## 3. Launch Phase 2 on server

```bash
ssh user@3112080.server.hostname
cd ~/intas_experiment/GPR_KG_Code

# Activate your Python env
source ~/venv_intas/bin/activate

# Export paths (or edit inside deploy_server.sh)
export SUMO_HOME=/usr/share/sumo
export INTAS_ROOT=~/InTAS

# Launch both methods in parallel  (smoke test + nohup background)
bash deploy_server.sh
```

`deploy_server.sh` will:
1. Verify SUMO, libsumo, all Python deps, and Phase 0/1 artifacts
2. Run one smoke-test simulation (~11 min) to confirm the pipeline works
3. Launch **GPR-KG** (with VEPM) in background  — log: `logs/phase2_gprkg.log`
4. Launch **GPR-KG-nV** (ablation) in background — log: `logs/phase2_gprkgnv.log`

Both methods run 400 sims each (~77 h each on a single core). If the
server has ≥ 2 CPU cores, they run truly in parallel so wall time is
still ~77 h (not 154 h).

## 4. Monitor progress

From the server (ssh in any time):

```bash
cd ~/intas_experiment/GPR_KG_Code

# Live tail both logs
tail -f logs/phase2_gprkg.log
tail -f logs/phase2_gprkgnv.log

# Check per-iteration snapshot (updated every iteration)
wc -l results/intas/GPR_KG_run/snapshots.jsonl
tail -5 results/intas/GPR_KG_run/snapshots.jsonl | python3 -m json.tool

# Check Python processes
ps -ef | grep 'experiments.intas.run_main' | grep -v grep
```

## 5. If a run crashes or the server reboots

The algorithm **checkpoints after every iteration** to
`results/intas/{METHOD}_run/checkpoint.pkl`. To resume:

```bash
# Just re-run the same command — it auto-detects the checkpoint
nohup python3 -u -m experiments.intas.run_main \
    --method GPR-KG --n0 100 --N 300 --seed 100 \
    > logs/phase2_gprkg_resume.log 2>&1 &
```

The script prints `[resume] GPR-KG: presampling_done=True main_iter_completed=156/300`
when it picks up from a checkpoint.

## 6. When Phase 2 finishes — transfer results back

From **your Windows machine**:

```powershell
scp -r `
    "${DEST_USER}@${DEST_HOST}:${DEST_PATH}/GPR_KG_Code/results/intas/GPR_KG_run" `
    "E:/从Dell移动到X1/科研同行/谭真/OR投稿/Final_Submission/GPR_KG_Code/results/intas/"

scp -r `
    "${DEST_USER}@${DEST_HOST}:${DEST_PATH}/GPR_KG_Code/results/intas/GPR_KG_nV_run" `
    "E:/从Dell移动到X1/科研同行/谭真/OR投稿/Final_Submission/GPR_KG_Code/results/intas/"
```

After the transfer, locally run Phase 3 (LNS post-hoc) and Phase 4
(figure/table generation).

## 7. Estimated timeline

- Smoke test + setup verification: **10–15 min**
- Phase 2 (both methods, parallel): **~77 hours ≈ 3.2 days**
- Transfer results back: **< 5 min**

Total: **~3.3 days of server wall time** from launch.
