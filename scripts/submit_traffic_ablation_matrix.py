#!/usr/bin/env python3
"""Submit the strict no-history traffic ablation matrix as CPU shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


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
    parser.add_argument("--nodes", default="node001,node002,node003,node004,node005,node006")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--N", type=int, default=12)
    parser.add_argument("--n0", type=int, default=5)
    parser.add_argument("--K1", type=int, default=12)
    parser.add_argument("--final-revalidation-candidates", type=int, default=8)
    parser.add_argument("--final-revalidation-replications", type=int, default=20)
    parser.add_argument(
        "--encoder-kinds",
        default="synthetic,pca_manifold,kernel_manifold,ssl_masked,ssl_transformer",
    )
    parser.add_argument("--trajectory-log", default="")
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    cwd = args.deploy / "SC-OLH-KG"
    paper_results = args.deploy / "Final_Submission/GPR_KG_Code/results/ingolstadt21"
    common = (
        f"{PYTHON} performance/benchmark_traffic_ingolstadt21.py "
        f"--N {args.N} --n0 {args.n0} --K1 {args.K1} --K2 0 "
        f"--posterior_pool_size 200 --posterior_keep 10 "
        f"--axis_candidate_count 0 --structured_candidate_count 0 "
        f"--seed_start {{start}} --n_seeds {{items}} "
        f"--paper_results_dir {paper_results} --resume "
        f"--traffic_anchor_policy strict_none "
        f"--final_revalidation_candidates {args.final_revalidation_candidates} "
        f"--final_revalidation_replications {args.final_revalidation_replications} "
        f"--recommendation_infeasible_strategy min_margin "
        f"--recommendation_safety_z 0.5 --coupling_safety_z 0.5"
    )
    specs = [
        ("raw_olh_hvd", common + " --variants olhkg --variance_modes pooled,class,orthogonal,factor --state_candidate_count 0"),
        ("sc_statecand_hvd", common + " --variants sc_olhkg --variance_modes pooled,class,orthogonal,factor --state_candidate_count 20"),
        ("sc_statebasis_factor", common + " --variants sc_olhkg --variance_modes factor --state_candidate_count 0 --use_state_basis"),
        ("sc_rawstate_factor", common + " --variants sc_olhkg --variance_modes factor --state_candidate_count 20 --use_state_basis"),
        (
            "anchor_only",
            common.replace("--traffic_anchor_policy strict_none", "--traffic_anchor_policy only")
            + " --variants anchor_only --variance_modes factor --state_candidate_count 0",
        ),
    ]
    for encoder_kind in parse_csv(args.encoder_kinds):
        basis_mode = "raw+state" if encoder_kind == "synthetic" else "raw+manifold"
        trajectory_arg = f" --trajectory_log {args.trajectory_log}" if args.trajectory_log else ""
        specs.append((
            f"sc_repr_{encoder_kind}",
            (
                common
                + " --variants sc_olhkg --variance_modes factor --state_candidate_count 20 "
                + f"--use_state_basis --state_basis_mode {basis_mode} "
                + f"--encoder_kind {encoder_kind}"
                + trajectory_arg
            ),
        ))
    task_ids = []
    for tag, command in specs:
        cmd_template = (
            f"{sumo_env_prefix()}; {command} --tag {tag}_{run_id}; echo DONE"
        )
        out = run_cmd([
            sys.executable, args.scheduler, "submit-cpu-batch",
            "--items", str(args.n_seeds),
            "--cmd-template", cmd_template,
            "--cwd", str(cwd),
            "--signature", f"KG_op/traffic_ablation/{run_id}/{tag}/{{node}}",
            "--description", f"SC-OLH-KG traffic ablation {tag} {run_id} {{node}}",
            "--nodes", args.nodes,
            "--ram-mb", str(args.ram_mb),
            "--project", "KG-SUMO",
            "--allow-no-ckpt",
            "--allow-no-resume",
            "--allow-duplicate",
            "--json",
        ], dry_run=args.dry_run)
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
            task_ids.append(out)
    if args.dispatch:
        run_cmd([sys.executable, args.scheduler, "dispatch"], dry_run=args.dry_run)
    print({"run_id": run_id, "submitted_specs": [tag for tag, _ in specs]})


if __name__ == "__main__":
    main()
