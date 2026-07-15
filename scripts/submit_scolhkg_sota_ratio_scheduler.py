#!/usr/bin/env python3
"""Submit D/N SOTA frontier baselines to node001-node006."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scheduler_node_policy import allowed_node_flags, parse_cpu_nodes


DEFAULT_SCHEDULER = Path.home() / ".claude/skills/scheduler/scheduler.py"
# Scheduler staging starts from the local deploy mirror; node routing maps it
# to the shared remote path after submission.
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"
BOTORCH_OVERLAY = "/home/zhengliang01/scheduleurm_work/python_pkgs/botorch_overlay_py310"


def parse_csv(value, cast=str):
    out = []
    for part in str(value or "").split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def run_cmd(cmd, dry_run=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    if dry_run:
        return ""
    return subprocess.check_output([str(c) for c in cmd], text=True)


def run_cmd_input(cmd, payload, dry_run=False):
    print("+", " ".join(map(str, cmd)), f"< {len(payload.splitlines())} jsonl tasks", flush=True)
    if dry_run:
        print(payload, end="")
        return ""
    return subprocess.check_output([str(c) for c in cmd], input=payload, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", default=str(DEFAULT_SCHEDULER))
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--nodes", default="node001,node002,node003,node004,node005,node006")
    parser.add_argument("--problems", default="HighDimStatePolicyRZDT1")
    parser.add_argument("--dims", default="1000,10000")
    parser.add_argument("--budgets", default="40,80,160")
    parser.add_argument(
        "--baselines",
        default=(
            "sobol,random,rembo_lite,baxus_lite,turbo_lite,scbo_lite,"
            "botorch_turbo,botorch_scbo,botorch_saasbo"
        ),
    )
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--baseline-batch-candidates", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--embedding-dim-max", type=int, default=32)
    parser.add_argument("--botorch-fallback", choices=("lite", "error"), default="error")
    parser.add_argument("--botorch-raw-samples", type=int, default=1024)
    parser.add_argument("--botorch-num-restarts", type=int, default=10)
    parser.add_argument("--botorch-maxiter", type=int, default=100)
    parser.add_argument("--botorch-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--botorch-max-candidate-failures", type=int, default=1)
    parser.add_argument("--botorch-ts-candidates", type=int, default=0)
    parser.add_argument("--saas-warmup-steps", type=int, default=256)
    parser.add_argument("--saas-num-samples", type=int, default=128)
    parser.add_argument("--saas-thinning", type=int, default=16)
    parser.add_argument("--saas-max-tree-depth", type=int, default=6)
    parser.add_argument("--saas-mc-samples", type=int, default=256)
    parser.add_argument("--disable-recommendation-calibration", action="store_true")
    parser.add_argument("--disable-recommendation-axis-oracle", action="store_true")
    parser.add_argument("--disable-problem-initial-samples", action="store_true")
    parser.add_argument("--disable-boundary-initial-samples", action="store_true")
    parser.add_argument("--disable-recommendation-refinement", action="store_true")
    parser.add_argument("--out-dir", default="results/sota_ratio")
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("sota_ratio_%Y%m%d_%H%M%S")
    cwd = args.deploy / "SC-OLH-KG"
    nodes = parse_cpu_nodes(args.nodes)
    specs = []
    task_ids = []
    submit_index = 0
    for problem in parse_csv(args.problems):
        for dim in parse_csv(args.dims, int):
            for budget in parse_csv(args.budgets, int):
                for baseline in parse_csv(args.baselines):
                    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.n_seeds)):
                        node = nodes[submit_index % len(nodes)]
                        out_prefix = (
                            f"{run_id}_{problem.lower()}_d{dim}_n{budget}_"
                            f"{baseline}_seed{seed:03d}"
                        )
                        checkpoint_root = cwd / "checkpoints" / run_id
                        checkpoint_dir = (
                            cwd / "checkpoints" / run_id / problem
                            / f"d{dim}" / f"N{budget}" / baseline
                            / f"seed{seed:03d}"
                        )
                        command = (
                            f"{PYTHON} performance/benchmark_sota.py "
                            f"--problem {problem} --d {dim} --N {budget} --n0 {args.n0} "
                            f"--baselines {baseline} --seeds {seed} --jobs {args.jobs} "
                            f"--exclude_olhkg --exclude_sc "
                            f"--baseline_batch_candidates {args.baseline_batch_candidates} "
                            f"--embedding_dim {args.embedding_dim} "
                            f"--embedding_dim_max {args.embedding_dim_max} "
                            f"--botorch_fallback {args.botorch_fallback} "
                            f"--botorch_raw_samples {args.botorch_raw_samples} "
                            f"--botorch_num_restarts {args.botorch_num_restarts} "
                            f"--botorch_maxiter {args.botorch_maxiter} "
                            f"--botorch_timeout_sec {args.botorch_timeout_sec} "
                            f"--botorch_max_candidate_failures {args.botorch_max_candidate_failures} "
                            f"--botorch_ts_candidates {args.botorch_ts_candidates} "
                            f"--saas_warmup_steps {args.saas_warmup_steps} "
                            f"--saas_num_samples {args.saas_num_samples} "
                            f"--saas_thinning {args.saas_thinning} "
                            f"--saas_max_tree_depth {args.saas_max_tree_depth} "
                            f"--saas_mc_samples {args.saas_mc_samples} "
                            f"--botorch_checkpoint_dir {checkpoint_root} "
                            f"--botorch_checkpoint_resume "
                            f"{'--disable_recommendation_calibration ' if args.disable_recommendation_calibration else ''}"
                            f"{'--disable_recommendation_axis_oracle ' if args.disable_recommendation_axis_oracle else ''}"
                            f"{'--disable_problem_initial_samples ' if args.disable_problem_initial_samples else ''}"
                            f"{'--disable_boundary_initial_samples ' if args.disable_boundary_initial_samples else ''}"
                            f"{'--disable_recommendation_refinement ' if args.disable_recommendation_refinement else ''}"
                            f"--disable_saas_failure_fallback "
                            f"--out_dir {args.out_dir} --out_prefix {out_prefix}"
                        )
                        cmd = "; ".join([
                            "export LC_ALL=C LANG=C",
                            f"export OMP_NUM_THREADS={int(args.cpu)} MKL_NUM_THREADS={int(args.cpu)} OPENBLAS_NUM_THREADS={int(args.cpu)}",
                            f"export PYTHONPATH={BOTORCH_OVERLAY}:$PYTHONPATH",
                            f"{command} && echo DONE",
                        ])
                        specs.append({
                            "description": (
                                f"SOTA ratio {baseline} {problem} d={dim} N={budget} "
                                f"seed={seed} {run_id}"
                            ),
                            "cmd": cmd,
                            "cwd": str(cwd),
                            "signature": (
                                f"KG_op/sota_ratio/{run_id}/{problem}/d{dim}/"
                                f"N{budget}/{baseline}/seed{seed:03d}"
                            ),
                            "project": "KG-SYNTH",
                            "vram": 0,
                            "cpu": int(args.cpu),
                            "ram_mb": int(args.ram_mb),
                            "require_node": node,
                            "allowed_nodes": list(nodes),
                            "ckpt_dir": str(checkpoint_dir),
                            "ckpt_glob": "**/*.pkl",
                            "allow_initial_resume_scan_error": True,
                            "allow_no_resume": True,
                            "allow_duplicate": True,
                        })
                        submit_index += 1

    payload = "".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs)
    out = run_cmd_input([
        sys.executable,
        args.scheduler,
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        f"sota-ratio-submit-{run_id}",
    ], payload, dry_run=args.dry_run)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
        summary = json.loads(out)
        task_ids.extend(item["id"] for item in summary.get("submitted", []))

    if args.dispatch and task_ids:
        dispatch_cmd = [
            sys.executable,
            args.scheduler,
            "dispatch",
            "--bulk-window",
            "--intent-label",
            f"sota-ratio-{run_id}",
        ]
        for task_id in task_ids:
            dispatch_cmd.extend(["--task-id", task_id])
        run_cmd(dispatch_cmd, dry_run=args.dry_run)
    print({"run_id": run_id, "task_count": len(task_ids), "task_ids": task_ids})


if __name__ == "__main__":
    main()
