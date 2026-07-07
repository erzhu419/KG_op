#!/usr/bin/env python3
"""Submit fresh trajectory-log generation for traffic recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from scheduler_node_policy import allowed_node_flags, parse_cpu_nodes


DEFAULT_SCHEDULER = Path.home() / ".claude/skills/scheduler/scheduler.py"
DEFAULT_DEPLOY = Path("/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy")
SUMO_PKG = "/home/zhengliang01/scheduleurm_work/python_pkgs/eclipse_sumo_1_25"
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"


def parse_csv(text):
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def run_cmd(cmd, dry_run=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    if dry_run:
        return ""
    return subprocess.check_output([str(c) for c in cmd], text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", default=str(DEFAULT_SCHEDULER))
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--nodes", default="node001,node002,node003,node004,node005,node006")
    parser.add_argument("--summary-glob", required=True)
    parser.add_argument("--source-indexes", default="0")
    parser.add_argument("--max-policies", type=int, default=20)
    parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=900000)
    parser.add_argument("--trajectory-interval", type=int, default=60)
    parser.add_argument("--wait-for-data-sec", type=float, default=0.0)
    parser.add_argument("--wait-interval-sec", type=float, default=30.0)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    nodes = parse_cpu_nodes(args.nodes)
    node = nodes[0]
    cwd = args.deploy / "SC-OLH-KG"
    out_root = args.deploy / "Final_Submission/GPR_KG_Code/results/ingolstadt21"
    out_csv = out_root / f"traffic_trajectory_{run_id}.csv"
    out_json = out_root / f"traffic_trajectory_{run_id}.json"
    out_sim_csv = out_root / f"traffic_trajectory_{run_id}_simulations.csv"
    parts = [
        "export LC_ALL=C LANG=C",
        f"export SUMO_PKG={SUMO_PKG}",
        "export SUMO_HOME=$SUMO_PKG/sumo",
        "export PYTHONPATH=$SUMO_PKG:$SUMO_PKG/sumo/tools:$PYTHONPATH",
        "export PATH=$SUMO_HOME/bin:$PATH",
        "export LD_LIBRARY_PATH=$SUMO_PKG/libsumo.libs:$SUMO_PKG/eclipse_sumo.libs:$LD_LIBRARY_PATH",
        (
            f"{PYTHON} performance/generate_traffic_trajectory_logs.py "
            f"--summary-glob '{args.summary_glob}' "
            f"--source-indexes {args.source_indexes} "
            f"--max-policies {args.max_policies} --R {args.R} "
            f"--seed-start {args.seed_start} "
            f"--trajectory-interval {args.trajectory_interval} "
            f"--wait-for-data-sec {args.wait_for_data_sec} "
            f"--wait-interval-sec {args.wait_interval_sec} "
            f"--out-csv {out_csv} --out-json {out_json} "
            f"--out-sim-csv {out_sim_csv} --resume"
        ),
        "echo DONE",
    ]
    cmd = "; ".join(parts)
    out = run_cmd([
        sys.executable, args.scheduler, "submit",
        "--description", f"SC-OLH-KG traffic trajectory logs {run_id}",
        "--cmd", cmd,
        "--cwd", str(cwd),
        "--signature", f"KG_op/traffic_trajectory/{run_id}",
        "--project", "KG-SUMO",
        "--vram", "0",
        "--cpu", str(args.cpu),
        "--ram-mb", str(args.ram_mb),
        "--require-node", node,
        *allowed_node_flags(nodes),
        "--allow-no-ckpt",
        "--allow-no-resume",
        "--allow-duplicate",
    ], dry_run=args.dry_run)
    task_ids = []
    if out:
        print(out, end="")
        task_ids.append(out.split()[1])
    if args.dispatch:
        run_cmd([sys.executable, args.scheduler, "dispatch"], dry_run=args.dry_run)
    print({
        "run_id": run_id,
        "task_ids": task_ids,
        "out_csv": str(out_csv),
        "out_json": str(out_json),
    })


if __name__ == "__main__":
    main()
