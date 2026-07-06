#!/usr/bin/env python3
"""Submit LODO learned meta-prior SC-OLH-KG suites to CPU scheduler nodes."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_SCHEDULER = Path.home() / ".claude/skills/scheduler/scheduler.py"
DEFAULT_DEPLOY = Path("/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy")
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"


LOSS_PRESETS = {
    "calibrated": {
        "meta_boundary_weight": 1.5,
        "meta_boundary_temperature": 1.0,
        "meta_variance_weight": 0.75,
        "meta_feasible_penalty": 7.0,
        "meta_feasible_bonus": 0.20,
        "meta_elite_fraction": 0.40,
        "meta_boundary_fraction": 0.35,
    },
    "strong_boundary": {
        "meta_boundary_weight": 3.0,
        "meta_boundary_temperature": 1.25,
        "meta_variance_weight": 1.0,
        "meta_feasible_penalty": 8.0,
        "meta_feasible_bonus": 0.15,
        "meta_elite_fraction": 0.30,
        "meta_boundary_fraction": 0.50,
    },
    "feasible_guard": {
        "meta_boundary_weight": 1.0,
        "meta_boundary_temperature": 0.75,
        "meta_variance_weight": 0.75,
        "meta_feasible_penalty": 12.0,
        "meta_feasible_bonus": 0.35,
        "meta_elite_fraction": 0.55,
        "meta_boundary_fraction": 0.25,
    },
}


def parse_csv(text, cast=str):
    return [cast(x.strip()) for x in str(text or "").split(",") if x.strip()]


def run_cmd(cmd, dry_run=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    if dry_run:
        return ""
    return subprocess.check_output([str(c) for c in cmd], text=True)


def preset_flags(name):
    if name not in LOSS_PRESETS:
        raise ValueError(f"unknown loss preset {name!r}; choices={sorted(LOSS_PRESETS)}")
    return " ".join(f"--{key} {value}" for key, value in LOSS_PRESETS[name].items())


def suite_command(args, run_id, N, preset):
    ckpt_root = args.deploy / "SC-OLH-KG" / "checkpoints" / run_id / "lodo_meta"
    prefix = f"lodo_meta_{run_id}_N{N}_{preset}"
    return (
        f"{PYTHON} performance/benchmark_lodo_meta_prior.py "
        f"--domains {args.domains} --heldouts {args.heldouts} --lines {args.lines} "
        f"--d {args.d} --L {args.L} --sigma {args.sigma} --alpha {args.alpha} "
        f"--weights {args.weights} --N {N} --n0 {args.n0} --K1 {args.K1} --K2 {args.K2} "
        f"--posterior_pool_size {args.posterior_pool_size} "
        f"--posterior_keep {args.posterior_keep} "
        f"--axis_candidate_count {args.axis_candidate_count} "
        f"--structured_candidate_count {args.structured_candidate_count} "
        f"--state_candidate_count {args.state_candidate_count} "
        f"--state_inverse_pool_size {args.state_inverse_pool_size} "
        f"--state_inverse_neighbors {args.state_inverse_neighbors} "
        f"--n_thr {args.n_thr} --eval_pool_size {args.eval_pool_size} "
        f"--variance_mode factor --lambda_feas {args.lambda_feas} "
        f"--lambda_var {args.lambda_var} --lambda_mean {args.lambda_mean} "
        f"--lambda_coupling {args.lambda_coupling} --beta_g {args.beta_g} "
        f"--certification_mode theory --use_state_basis "
        f"--state_basis_mode {args.state_basis_mode} --raw_basis_dim {args.raw_basis_dim} "
        f"--raw_projection_seed {args.raw_projection_seed} "
        f"--numeric_backend {args.numeric_backend} "
        f"--numeric_backend_device {args.numeric_backend_device} "
        f"--torch_dtype {args.torch_dtype} --torch_min_rows {args.torch_min_rows} "
        f"--acquisition_mode {args.acquisition_mode} "
        f"--exact_kg_mc_samples {args.exact_kg_mc_samples} "
        f"--exact_kg_jobs {args.exact_kg_jobs} "
        f"--exact_kg_blend {args.exact_kg_blend} "
        f"--source_records_per_domain {args.source_records_per_domain} "
        f"--meta_local_dim {args.meta_local_dim} "
        f"--meta_shared_dim {args.meta_shared_dim} "
        f"--meta_anchor_count {args.meta_anchor_count} "
        f"--meta_kmeans_iters {args.meta_kmeans_iters} "
        f"--meta_soft_temperature {args.meta_soft_temperature} "
        f"--meta_ridge {args.meta_ridge} "
        f"{preset_flags(preset)} "
        f"--meta_seed {args.meta_seed} "
        f"--meta_proposal_pool_size {args.meta_proposal_pool_size} "
        f"--meta_refinement_count {args.meta_refinement_count} "
        f"--seed_start {args.seed_start} --n_seeds {args.n_seeds} "
        f"--jobs {args.jobs_per_suite} "
        f"--checkpoint_path {ckpt_root / (prefix + '.jsonl')} --resume_completed "
        f"--out_prefix {prefix}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", default=str(DEFAULT_SCHEDULER))
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--nodes", default="node001,node002,node003,node004,node005,node006")
    parser.add_argument("--N-values", default="40,80")
    parser.add_argument("--loss-presets", default="calibrated")
    parser.add_argument(
        "--domains",
        default="FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl",
    )
    parser.add_argument("--heldouts", default="FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl")
    parser.add_argument("--lines", default="strict,lodo,domain")
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--K1", type=int, default=20)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--posterior-pool-size", type=int, default=300)
    parser.add_argument("--posterior-keep", type=int, default=15)
    parser.add_argument("--axis-candidate-count", type=int, default=-1)
    parser.add_argument("--structured-candidate-count", type=int, default=0)
    parser.add_argument("--state-candidate-count", type=int, default=24)
    parser.add_argument("--state-inverse-pool-size", type=int, default=600)
    parser.add_argument("--state-inverse-neighbors", type=int, default=2)
    parser.add_argument("--n-thr", type=int, default=5)
    parser.add_argument("--eval-pool-size", type=int, default=500)
    parser.add_argument("--lambda-feas", type=float, default=0.25)
    parser.add_argument("--lambda-var", type=float, default=0.25)
    parser.add_argument("--lambda-mean", type=float, default=0.10)
    parser.add_argument("--lambda-coupling", type=float, default=0.05)
    parser.add_argument("--beta-g", type=float, default=2.0)
    parser.add_argument("--state-basis-mode", default="raw+state")
    parser.add_argument("--raw-basis-dim", type=int, default=32)
    parser.add_argument("--raw-projection-seed", type=int, default=314159)
    parser.add_argument("--numeric-backend", default="numpy")
    parser.add_argument("--numeric-backend-device", default="auto")
    parser.add_argument("--torch-dtype", default="float64")
    parser.add_argument("--torch-min-rows", type=int, default=128)
    parser.add_argument("--acquisition-mode", default="additive")
    parser.add_argument("--exact-kg-mc-samples", type=int, default=0)
    parser.add_argument("--exact-kg-jobs", type=int, default=1)
    parser.add_argument("--exact-kg-blend", type=float, default=0.0)
    parser.add_argument("--source-records-per-domain", type=int, default=256)
    parser.add_argument("--meta-local-dim", type=int, default=3)
    parser.add_argument("--meta-shared-dim", type=int, default=3)
    parser.add_argument("--meta-anchor-count", type=int, default=32)
    parser.add_argument("--meta-kmeans-iters", type=int, default=35)
    parser.add_argument("--meta-soft-temperature", type=float, default=0.75)
    parser.add_argument("--meta-ridge", type=float, default=1e-4)
    parser.add_argument("--meta-seed", type=int, default=20260706)
    parser.add_argument("--meta-proposal-pool-size", type=int, default=1024)
    parser.add_argument("--meta-refinement-count", type=int, default=192)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--jobs-per-suite", type=int, default=10)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    cwd = args.deploy / "SC-OLH-KG"
    nodes = parse_csv(args.nodes)
    task_ids = []
    suites = [
        (N, preset)
        for N in parse_csv(args.N_values, int)
        for preset in parse_csv(args.loss_presets)
    ]
    for idx, (N, preset) in enumerate(suites):
        node = nodes[idx % len(nodes)]
        command = suite_command(args, run_id, N, preset)
        cmd = "; ".join([
            "export LC_ALL=C LANG=C",
            "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
            command,
            "echo DONE",
        ])
        out = run_cmd([
            sys.executable, args.scheduler, "submit",
            "--description", f"SC-OLH-KG LODO meta-prior N={N} {preset} {run_id}",
            "--cmd", cmd,
            "--cwd", str(cwd),
            "--signature", f"KG_op/scolhkg_lodo_meta/{run_id}/N{N}/{preset}",
            "--project", "KG-SYNTH",
            "--vram", "0",
            "--cpu", str(args.cpu),
            "--ram-mb", str(args.ram_mb),
            "--require-node", node,
            "--allow-duplicate",
        ], dry_run=args.dry_run)
        if out:
            print(out, end="")
            parts = out.split()
            if len(parts) > 1:
                task_ids.append(parts[1])
    if args.dispatch:
        run_cmd([sys.executable, args.scheduler, "dispatch"], dry_run=args.dry_run)
    print({"run_id": run_id, "task_ids": task_ids, "suites": suites})


if __name__ == "__main__":
    main()
