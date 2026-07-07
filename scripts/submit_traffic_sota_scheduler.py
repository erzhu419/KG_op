#!/usr/bin/env python3
"""Submit live SUMO traffic SOTA baselines as CPU shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from scheduler_node_policy import parse_cpu_nodes


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


def sumo_env_prefix():
    return "; ".join([
        "export LC_ALL=C LANG=C",
        f"export SUMO_PKG={SUMO_PKG}",
        "export SUMO_HOME=$SUMO_PKG/sumo",
        "export PYTHONPATH=$SUMO_PKG:$SUMO_PKG/sumo/tools:$PYTHONPATH",
        "export PATH=$SUMO_HOME/bin:$PATH",
        "export LD_LIBRARY_PATH=$SUMO_PKG/libsumo.libs:$SUMO_PKG/eclipse_sumo.libs:$LD_LIBRARY_PATH",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", default=str(DEFAULT_SCHEDULER))
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--nodes", default="node001,node002,node003,node004,node005,node006")
    parser.add_argument("--methods", default="sobol,random,hetgp_lite,rahbo_lite,safeopt_lite,legacy_vepm_lite,botorch_turbo,botorch_scbo,botorch_saasbo")
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--n0", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--traffic-anchor-policy", default="strict_none")
    parser.add_argument("--botorch-fallback", choices=("lite", "error"), default="error")
    parser.add_argument("--botorch-timeout-sec", type=float, default=60.0)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    cwd = args.deploy / "SC-OLH-KG"
    paper_results = args.deploy / "Final_Submission/GPR_KG_Code/results/ingolstadt21"
    nodes = parse_cpu_nodes(args.nodes)
    node_csv = ",".join(nodes)
    task_groups = []
    for method in parse_csv(args.methods):
        tag = args.tag or run_id
        command = (
            f"{sumo_env_prefix()}; "
            f"{PYTHON} performance/benchmark_traffic_sota_ingolstadt21.py "
            f"--methods {method} --N {args.N} --n0 {args.n0} "
            f"--seed_start {{start}} --n_seeds {{items}} "
            f"--paper_results_dir {paper_results} --resume "
            f"--traffic_anchor_policy {args.traffic_anchor_policy} "
            f"--botorch_fallback {args.botorch_fallback} "
            f"--botorch_timeout_sec {args.botorch_timeout_sec} "
            f"--tag {tag}; echo DONE"
        )
        out = run_cmd([
            sys.executable, args.scheduler, "submit-cpu-batch",
            "--items", str(args.n_seeds),
            "--cmd-template", command,
            "--cwd", str(cwd),
            "--signature", f"KG_op/traffic_sota/{run_id}/{method}/{{node}}",
            "--description", f"SC-OLH-KG traffic SOTA {method} {run_id} {{node}}",
            "--nodes", node_csv,
            "--ram-mb", str(args.ram_mb),
            "--project", "KG-SUMO",
            "--allow-no-ckpt",
            "--allow-no-resume",
            "--allow-duplicate",
            "--json",
        ], dry_run=args.dry_run)
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
            task_groups.append(method)
    if args.dispatch:
        run_cmd([sys.executable, args.scheduler, "dispatch"], dry_run=args.dry_run)
    print({"run_id": run_id, "submitted_methods": task_groups})


if __name__ == "__main__":
    main()
