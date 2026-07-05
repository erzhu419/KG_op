#!/usr/bin/env python3
"""Submit SC-OLH-KG traffic optimization runs to node001-node006."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_SCHEDULER = Path.home() / ".claude/skills/scheduler/scheduler.py"
DEFAULT_DEPLOY = Path("/home/erzhu419/mine_code/KG_op_scheduler_deploy")
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
    parser.add_argument("--tag", default="")
    parser.add_argument("--nodes", default="node001,node002,node003,node004,node005,node006")
    parser.add_argument("--variants", default="olhkg,sc_olhkg")
    parser.add_argument("--variance-modes", default="factor")
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--K1", type=int, default=20)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--posterior-pool-size", type=int, default=200)
    parser.add_argument("--posterior-keep", type=int, default=10)
    parser.add_argument("--axis-candidate-count", type=int, default=0)
    parser.add_argument("--structured-candidate-count", type=int, default=12)
    parser.add_argument("--state-candidate-count", type=int, default=20)
    parser.add_argument("--state-inverse-pool-size", type=int, default=300)
    parser.add_argument("--state-inverse-neighbors", type=int, default=2)
    parser.add_argument("--lambda-feas", type=float, default=0.25)
    parser.add_argument("--lambda-var", type=float, default=0.25)
    parser.add_argument("--lambda-mean", type=float, default=0.10)
    parser.add_argument("--lambda-coupling", type=float, default=0.10)
    parser.add_argument("--beta-g", type=float, default=2.0)
    parser.add_argument("--recommendation-infeasible-penalty", type=float, default=5.0)
    parser.add_argument("--recommendation-infeasible-strategy", default="min_margin")
    parser.add_argument("--recommendation-safety-z", type=float, default=0.5)
    parser.add_argument("--coupling-safety-z", type=float, default=0.5)
    parser.add_argument("--evaluate-interval", type=int, default=0)
    parser.add_argument("--true-replications", type=int, default=2)
    parser.add_argument("--sigma-replications", type=int, default=3)
    parser.add_argument(
        "--traffic-anchor-policy",
        default="historical",
        choices=["historical", "none", "strict_none", "only"],
    )
    parser.add_argument("--keep-observed-candidates", type=int, default=0)
    parser.add_argument("--final-revalidation-candidates", type=int, default=0)
    parser.add_argument("--final-revalidation-replications", type=int, default=0)
    parser.add_argument("--final-revalidation-seed-start", type=int, default=70000)
    parser.add_argument("--final-revalidation-include-refinement", action="store_true")
    parser.add_argument("--acquisition-mode", default="additive")
    parser.add_argument("--exact-kg-mc-samples", type=int, default=0)
    parser.add_argument("--use-state-basis", action="store_true")
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=28800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    cwd = args.deploy / "SC-OLH-KG"
    paper_results = args.deploy / "Final_Submission/GPR_KG_Code/results/ingolstadt21"
    nodes = parse_csv(args.nodes)
    variants = parse_csv(args.variants)
    modes = parse_csv(args.variance_modes)
    task_ids = []
    job = 0
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        for variant in variants:
            for mode in modes:
                node = nodes[job % len(nodes)]
                job += 1
                parts = [
                    "export LC_ALL=C LANG=C",
                    f"export SUMO_PKG={SUMO_PKG}",
                    "export SUMO_HOME=$SUMO_PKG/sumo",
                    "export PYTHONPATH=$SUMO_PKG:$SUMO_PKG/sumo/tools:$PYTHONPATH",
                    "export PATH=$SUMO_HOME/bin:$PATH",
                    "export LD_LIBRARY_PATH=$SUMO_PKG/libsumo.libs:$SUMO_PKG/eclipse_sumo.libs:$LD_LIBRARY_PATH",
                    (
                        f"{PYTHON} performance/benchmark_traffic_ingolstadt21.py "
                        f"--variants {variant} --variance_modes {mode} "
                        f"--N {args.N} --n0 {args.n0} --K1 {args.K1} --K2 {args.K2} "
                        f"--posterior_pool_size {args.posterior_pool_size} "
                        f"--posterior_keep {args.posterior_keep} "
                        f"--axis_candidate_count {args.axis_candidate_count} "
                        f"--structured_candidate_count {args.structured_candidate_count} "
                        f"--state_candidate_count {args.state_candidate_count} "
                        f"--state_inverse_pool_size {args.state_inverse_pool_size} "
                        f"--state_inverse_neighbors {args.state_inverse_neighbors} "
                        f"--seed_start {seed} --n_seeds 1 "
                        f"--paper_results_dir {paper_results} --resume "
                        f"--evaluate_interval {args.evaluate_interval} "
                        f"--true_replications {args.true_replications} "
                        f"--sigma_replications {args.sigma_replications} "
                        f"--traffic_anchor_policy {args.traffic_anchor_policy} "
                        f"--keep_observed_candidates {args.keep_observed_candidates} "
                        f"--final_revalidation_candidates {args.final_revalidation_candidates} "
                        f"--final_revalidation_replications {args.final_revalidation_replications} "
                        f"--final_revalidation_seed_start {args.final_revalidation_seed_start} "
                        f"--acquisition_mode {args.acquisition_mode} "
                        f"--exact_kg_mc_samples {args.exact_kg_mc_samples} "
                        f"--lambda_feas {args.lambda_feas} "
                        f"--lambda_var {args.lambda_var} "
                        f"--lambda_mean {args.lambda_mean} "
                        f"--lambda_coupling {args.lambda_coupling} "
                        f"--beta_g {args.beta_g} "
                        f"--recommendation_infeasible_penalty {args.recommendation_infeasible_penalty} "
                        f"--recommendation_infeasible_strategy {args.recommendation_infeasible_strategy} "
                        f"--recommendation_safety_z {args.recommendation_safety_z} "
                        f"--coupling_safety_z {args.coupling_safety_z}"
                    ),
                    "echo DONE",
                ]
                if args.tag:
                    parts[-2] += f" --tag {args.tag}"
                if args.final_revalidation_include_refinement:
                    parts[-2] += " --final_revalidation_include_refinement"
                if args.use_state_basis:
                    parts[-2] += " --use_state_basis"
                cmd = "; ".join(parts)
                desc = f"SC-OLH-KG ingolstadt21 {variant} {mode} seed={seed} {run_id}"
                out = run_cmd([
                    sys.executable, args.scheduler, "submit",
                    "--description", desc,
                    "--cmd", cmd,
                    "--cwd", str(cwd),
                    "--signature", f"KG_op/sc_traffic/{run_id}/{variant}/{mode}/seed{seed}",
                    "--project", "KG-SUMO",
                    "--vram", "0",
                    "--cpu", str(args.cpu),
                    "--ram-mb", str(args.ram_mb),
                    "--require-node", node,
                    "--allow-no-ckpt",
                    "--allow-no-resume",
                    "--allow-duplicate",
                ], dry_run=args.dry_run)
                if out:
                    print(out, end="")
                    task_ids.append(out.split()[1])
    if args.dispatch:
        run_cmd([sys.executable, args.scheduler, "dispatch"], dry_run=args.dry_run)
    if args.wait and task_ids:
        run_cmd(
            [sys.executable, args.scheduler, "wait-for", "--task-id", *task_ids,
             "--poll", str(args.poll), "--timeout", str(args.timeout), "--verbose"],
            dry_run=args.dry_run,
        )
    print({"run_id": run_id, "task_ids": task_ids})


if __name__ == "__main__":
    main()
