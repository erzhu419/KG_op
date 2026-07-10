#!/usr/bin/env python3
"""Submit non-SUMO SC-OLH-KG paper suites to node001-node006."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from scheduler_node_policy import allowed_node_flags, parse_cpu_nodes


DEFAULT_SCHEDULER = Path.home() / ".claude/skills/scheduler/scheduler.py"
DEFAULT_DEPLOY = Path("/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy")
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"
BOTORCH_OVERLAY = "/home/zhengliang01/scheduleurm_work/python_pkgs/botorch_overlay_py310"


def parse_csv(text):
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def run_cmd(cmd, dry_run=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    if dry_run:
        return ""
    return subprocess.check_output([str(c) for c in cmd], text=True)


def suite_commands(args, run_id):
    out = []
    ckpt_root = args.deploy / "SC-OLH-KG" / "checkpoints" / run_id
    checkpoint_common = (
        f"--exact_kg_jobs {args.exact_kg_jobs} "
        f"--checkpoint_resume --checkpoint_interval 1 "
        f"--checkpoint_keep_last {args.checkpoint_keep_last}"
    )
    base = (
        f"--N {args.N} --n0 {args.n0} --K1 {args.K1} --K2 {args.K2} "
        f"--posterior_pool_size {args.posterior_pool_size} "
        f"--posterior_keep {args.posterior_keep} "
        f"--state_candidate_count {args.state_candidate_count} "
        f"--state_inverse_pool_size {args.state_inverse_pool_size} "
        f"--state_inverse_neighbors {args.state_inverse_neighbors} "
        f"--eval_pool_size {args.eval_pool_size} "
        f"--n_seeds {args.n_seeds} --jobs {args.jobs_per_suite} "
        f"{'--disable_recommendation_calibration ' if args.disable_recommendation_calibration else ''}"
        f"{'--disable_recommendation_axis_oracle ' if args.disable_recommendation_axis_oracle else ''}"
        f"{'--disable_problem_initial_samples ' if args.disable_problem_initial_samples else ''}"
        f"{'--disable_boundary_initial_samples ' if args.disable_boundary_initial_samples else ''}"
        f"{'--disable_recommendation_refinement ' if args.disable_recommendation_refinement else ''}"
        f"{'--disable_lf_os_problem_state_anchor ' if args.disable_lf_os_problem_state_anchor else ''}"
        f"{checkpoint_common}"
    )
    if args.use_state_basis:
        base += f" --use_state_basis --state_basis_mode {args.state_basis_mode}"
    if "exact" in args.suites:
        for problem in parse_csv(args.problems):
            out.append((
                f"exact/{problem}",
                (
                    f"{PYTHON} performance/benchmark_exact_kg.py "
                    f"--problem {problem} --variance_mode factor "
                    f"--methods additive,blend0.25,blend0.5,exact "
                    f"--exact_mc_samples {args.exact_mc_samples} "
                    f"--exact_kg_jobs {args.exact_kg_jobs} "
                    f"--checkpoint_dir {ckpt_root / 'exact'} "
                    f"--checkpoint_resume --checkpoint_interval 1 "
                    f"--checkpoint_keep_last {args.checkpoint_keep_last} "
                    f"--use_state_coupling --N {args.exact_N} --n0 {args.n0} "
                    f"--K1 {args.exact_K1} --K2 0 --posterior_pool_size {args.posterior_pool_size} "
                    f"--posterior_keep {args.posterior_keep} --state_candidate_count {args.state_candidate_count} "
                    f"--state_inverse_pool_size {args.state_inverse_pool_size} "
                    f"--state_inverse_neighbors {args.state_inverse_neighbors} "
                    f"--eval_pool_size {args.eval_pool_size} --n_seeds {args.n_seeds} "
                    f"--jobs {args.jobs_per_suite} --out_prefix paper_exact_{run_id}_{problem.lower()}"
                ),
            ))
    if "sota" in args.suites:
        out.append((
            "sota",
            (
                f"{PYTHON} performance/benchmark_sota_suite.py "
                f"--problems {args.problems} --variance_mode factor "
                f"--baselines {args.baselines} {base} "
                f"--botorch_fallback {args.botorch_fallback} "
                f"--botorch_raw_samples {args.botorch_raw_samples} "
                f"--botorch_num_restarts {args.botorch_num_restarts} "
                f"--botorch_maxiter {args.botorch_maxiter} "
                f"--botorch_timeout_sec {args.botorch_timeout_sec} "
                f"--embedding_dim {args.embedding_dim} "
                f"--embedding_dim_max {args.embedding_dim_max} "
                f"--checkpoint_dir {ckpt_root / 'sota'} "
                f"--out_prefix paper_sota_{run_id}"
            ),
        ))
    if "hvd" in args.suites:
        out.append((
            "hvd",
            (
                f"{PYTHON} performance/benchmark_hvd_suite.py "
                f"--problems {args.hvd_problems} --modes pooled,class,orthogonal,factor "
                f"--sc_modes factor {base} --checkpoint_dir {ckpt_root / 'hvd'} "
                f"--out_prefix paper_hvd_{run_id}"
            ),
        ))
    if "encoder" in args.suites:
        out.append((
            "encoder",
            (
                f"{PYTHON} performance/benchmark_encoder_suite.py "
                f"--problem StatePolicyRZDT1 --encoder_kinds {args.encoder_kinds} "
                f"--N {args.N} --n0 {args.n0} --K1 {args.K1} --K2 {args.K2} "
                f"--posterior_pool_size {args.posterior_pool_size} "
                f"--posterior_keep {args.posterior_keep} "
                f"--state_candidate_count {args.state_candidate_count} "
                f"--state_inverse_pool_size {args.state_inverse_pool_size} "
                f"--state_inverse_neighbors {args.state_inverse_neighbors} "
                f"--use_state_basis --state_basis_mode {args.state_basis_mode} "
                f"--n_seeds {args.n_seeds} --exact_kg_jobs {args.exact_kg_jobs} "
                f"--checkpoint_dir {ckpt_root / 'encoder'} --checkpoint_resume "
                f"--checkpoint_interval 1 --checkpoint_keep_last {args.checkpoint_keep_last} "
                f"--out_prefix paper_encoder_{run_id}"
            ),
        ))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", default=str(DEFAULT_SCHEDULER))
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--nodes", default="node001,node002,node003,node004,node005,node006")
    parser.add_argument("--suites", default="exact,sota,hvd,encoder")
    parser.add_argument("--problems", default="RegimeRZDT1,RZDT2,FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl,StatePolicyRZDT1,PaperRZDT1,PaperRZDT2,PaperRZDT5_RR")
    parser.add_argument("--hvd-problems", default="RegimeRZDT1,RZDT2,FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl,StatePolicyRZDT1")
    parser.add_argument("--baselines", default="sobol,random,hetgp_lite,rahbo_lite,safeopt_lite,legacy_vepm_lite,rembo_lite,baxus_lite,botorch_turbo,botorch_scbo,botorch_saasbo")
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--exact-N", type=int, default=40)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--K1", type=int, default=20)
    parser.add_argument("--exact-K1", type=int, default=12)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--posterior-pool-size", type=int, default=300)
    parser.add_argument("--posterior-keep", type=int, default=15)
    parser.add_argument("--state-candidate-count", type=int, default=24)
    parser.add_argument("--state-inverse-pool-size", type=int, default=600)
    parser.add_argument("--state-inverse-neighbors", type=int, default=2)
    parser.add_argument("--eval-pool-size", type=int, default=500)
    parser.add_argument("--disable-recommendation-calibration", action="store_true")
    parser.add_argument("--disable-recommendation-axis-oracle", action="store_true")
    parser.add_argument("--disable-problem-initial-samples", action="store_true")
    parser.add_argument("--disable-boundary-initial-samples", action="store_true")
    parser.add_argument("--disable-recommendation-refinement", action="store_true")
    parser.add_argument("--disable-lf-os-problem-state-anchor", action="store_true")
    parser.add_argument("--use-state-basis", dest="use_state_basis", action="store_true", default=True)
    parser.add_argument("--disable-state-basis", dest="use_state_basis", action="store_false")
    parser.add_argument(
        "--state-basis-mode",
        default="raw+state",
        choices=["raw", "state", "raw+state", "manifold", "raw+manifold"],
    )
    parser.add_argument(
        "--encoder-kinds",
        default=(
            "synthetic,pca_manifold,kernel_manifold,graph_laplacian,ssl_masked,"
            "ssl_contrastive,ssl_next_risk,ssl_transformer"
        ),
    )
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--jobs-per-suite", type=int, default=10)
    parser.add_argument("--exact-mc-samples", type=int, default=8)
    parser.add_argument("--exact-kg-jobs", type=int, default=1)
    parser.add_argument("--checkpoint-keep-last", type=int, default=2)
    parser.add_argument("--botorch-fallback", choices=("lite", "error"), default="error")
    parser.add_argument("--botorch-overlay", default=BOTORCH_OVERLAY)
    parser.add_argument("--botorch-raw-samples", type=int, default=128)
    parser.add_argument("--botorch-num-restarts", type=int, default=8)
    parser.add_argument("--botorch-maxiter", type=int, default=80)
    parser.add_argument("--botorch-timeout-sec", type=float, default=45.0)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--embedding-dim-max", type=int, default=32)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.suites = set(parse_csv(args.suites))
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    cwd = args.deploy / "SC-OLH-KG"
    nodes = parse_cpu_nodes(args.nodes)
    task_ids = []
    for idx, (name, command) in enumerate(suite_commands(args, run_id)):
        node = nodes[idx % len(nodes)]
        cmd = "; ".join([
            "export LC_ALL=C LANG=C",
            "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
            f"export PYTHONPATH={args.botorch_overlay}:$PYTHONPATH",
            f"{command} && echo DONE",
        ])
        out = run_cmd([
            sys.executable, args.scheduler, "submit",
            "--description", f"SC-OLH-KG synthetic suite {name} {run_id}",
            "--cmd", cmd,
            "--cwd", str(cwd),
            "--signature", f"KG_op/scolhkg_synth/{run_id}/{name}",
            "--project", "KG-SYNTH",
            "--vram", "0",
            "--cpu", str(args.cpu),
            "--ram-mb", str(args.ram_mb),
            "--require-node", node,
            *allowed_node_flags(nodes),
            "--allow-no-ckpt",
            "--allow-no-resume",
            "--allow-duplicate",
        ], dry_run=args.dry_run)
        if out:
            print(out, end="")
            task_ids.append(out.split()[1])
    if args.dispatch:
        run_cmd([sys.executable, args.scheduler, "dispatch"], dry_run=args.dry_run)
    print({"run_id": run_id, "task_ids": task_ids})


if __name__ == "__main__":
    main()
