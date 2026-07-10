#!/usr/bin/env python3
"""Submit LF-OS ratio sweeps to node001-node006."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scheduler_node_policy import allowed_node_flags, parse_cpu_nodes


DEFAULT_SCHEDULER = Path.home() / ".claude/skills/scheduler/scheduler.py"
DEFAULT_DEPLOY = Path("/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy")
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"
BOTORCH_OVERLAY = "/home/zhengliang01/scheduleurm_work/python_pkgs/botorch_overlay_py310"


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
    parser.add_argument("--problems", default="HighDimStatePolicyRZDT1,FactorShockStatePolicyRZDT1")
    parser.add_argument("--dims", default="1000,10000")
    parser.add_argument("--budgets", default="30,40,80")
    parser.add_argument("--encoder-kinds", default="synthetic,lf_os")
    parser.add_argument("--lf-os-low-frequency-components-grid", default="5,8,12")
    parser.add_argument("--lf-os-max-active-grid", default="5,8")
    parser.add_argument("--lf-os-residual-floor-scale-grid", default="0.0,0.02,0.05")
    parser.add_argument("--encoder-latent-dim", type=int, default=8)
    parser.add_argument("--encoder-fit-pool-size", type=int, default=512)
    parser.add_argument("--lf-os-max-library-size", type=int, default=30)
    parser.add_argument("--lf-os-graph-neighbors", type=int, default=12)
    parser.add_argument("--disable-lf-os-problem-state-anchor", action="store_true")
    parser.add_argument("--N", type=int, default=40)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--K1", type=int, default=12)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--posterior-pool-size", type=int, default=180)
    parser.add_argument("--posterior-keep", type=int, default=10)
    parser.add_argument("--axis-candidate-count", type=int, default=-1)
    parser.add_argument("--structured-candidate-count", type=int, default=0)
    parser.add_argument("--state-candidate-count", type=int, default=16)
    parser.add_argument("--state-inverse-pool-size", type=int, default=400)
    parser.add_argument("--state-inverse-neighbors", type=int, default=2)
    parser.add_argument("--eval-pool-size", type=int, default=300)
    parser.add_argument("--state-basis-mode", default="manifold")
    parser.add_argument("--acquisition-modes", default="additive")
    parser.add_argument("--disable-recommendation-calibration", action="store_true")
    parser.add_argument("--enable-certification-calibration", action="store_true")
    parser.add_argument("--disable-recommendation-axis-oracle", action="store_true")
    parser.add_argument("--disable-problem-initial-samples", action="store_true")
    parser.add_argument("--disable-boundary-initial-samples", action="store_true")
    parser.add_argument("--disable-recommendation-refinement", action="store_true")
    parser.add_argument("--disable-state-basis", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--split-seeds",
        action="store_true",
        help="Submit each seed as separate scheduler tasks instead of serializing seeds inside a shard.",
    )
    parser.add_argument("--jobs-per-shard", type=int, default=1)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=-1, help="Exclusive shard end; -1 means --shards.")
    parser.add_argument("--exact-kg-jobs", type=int, default=1)
    parser.add_argument("--checkpoint-keep-last", type=int, default=2)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--no-bulk-submit",
        action="store_true",
        help="Fallback to one scheduler.py submit process per shard.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("lfos_ratio_%Y%m%d_%H%M%S")
    cwd = args.deploy / "SC-OLH-KG"
    ckpt_root = args.deploy / "SC-OLH-KG" / "checkpoints" / run_id
    nodes = parse_cpu_nodes(args.nodes)
    task_ids = []
    bulk_specs = []
    seed_values = (
        list(range(int(args.seed_start), int(args.seed_start) + int(args.n_seeds)))
        if args.split_seeds
        else [None]
    )
    shard_end = int(args.shards) if int(args.shard_end) < 0 else min(int(args.shard_end), int(args.shards))
    submit_index = 0
    for seed in seed_values:
        for shard in range(int(args.shard_start), shard_end):
            node = nodes[submit_index % len(nodes)]
            seed_suffix = "" if seed is None else f"_seed{int(seed)}"
            out_prefix = f"{run_id}{seed_suffix}"
            seed_args = (
                f"--seed_start {int(seed)} --n_seeds 1 "
                if seed is not None
                else f"--seed_start {int(args.seed_start)} --n_seeds {args.n_seeds} "
            )
            command = (
            f"{PYTHON} performance/benchmark_lfos_ratio_sweep.py "
            f"--problems {args.problems} --dims {args.dims} "
            f"--budgets {args.budgets} "
            f"--encoder_kinds {args.encoder_kinds} "
            f"--lf_os_low_frequency_components_grid {args.lf_os_low_frequency_components_grid} "
            f"--lf_os_max_active_grid {args.lf_os_max_active_grid} "
            f"--lf_os_residual_floor_scale_grid {args.lf_os_residual_floor_scale_grid} "
            f"--encoder_latent_dim {args.encoder_latent_dim} "
            f"--encoder_fit_pool_size {args.encoder_fit_pool_size} "
            f"--lf_os_max_library_size {args.lf_os_max_library_size} "
            f"--lf_os_graph_neighbors {args.lf_os_graph_neighbors} "
            f"{'--disable_lf_os_problem_state_anchor ' if args.disable_lf_os_problem_state_anchor else ''}"
            f"--n0 {args.n0} --K1 {args.K1} --K2 {args.K2} "
            f"--posterior_pool_size {args.posterior_pool_size} "
            f"--posterior_keep {args.posterior_keep} "
            f"--axis_candidate_count {args.axis_candidate_count} "
            f"--structured_candidate_count {args.structured_candidate_count} "
            f"--state_candidate_count {args.state_candidate_count} "
            f"--state_inverse_pool_size {args.state_inverse_pool_size} "
            f"--state_inverse_neighbors {args.state_inverse_neighbors} "
            f"--eval_pool_size {args.eval_pool_size} "
            f"--state_basis_mode {args.state_basis_mode} "
            f"{'--disable_recommendation_calibration ' if args.disable_recommendation_calibration else ''}"
            f"{'--enable_certification_calibration ' if args.enable_certification_calibration else ''}"
            f"{'--disable_recommendation_axis_oracle ' if args.disable_recommendation_axis_oracle else ''}"
            f"{'--disable_problem_initial_samples ' if args.disable_problem_initial_samples else ''}"
            f"{'--disable_boundary_initial_samples ' if args.disable_boundary_initial_samples else ''}"
            f"{'--disable_recommendation_refinement ' if args.disable_recommendation_refinement else ''}"
            f"{'--disable_state_basis ' if args.disable_state_basis else ''}"
            f"--acquisition_modes {args.acquisition_modes} "
            f"{seed_args}--jobs {args.jobs_per_shard} "
            f"--exact_kg_jobs {args.exact_kg_jobs} "
            f"--checkpoint_dir {ckpt_root} --checkpoint_resume "
            f"--checkpoint_interval 1 --checkpoint_keep_last {args.checkpoint_keep_last} "
            f"--shards {args.shards} --shard_index {shard} "
            f"--out_prefix {out_prefix}"
        )
            cmd = "; ".join([
                "export LC_ALL=C LANG=C",
                "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                f"export PYTHONPATH={BOTORCH_OVERLAY}:$PYTHONPATH",
                f"{command} && echo DONE",
            ])
            desc_seed = "" if seed is None else f" seed {int(seed)}"
            signature_seed = "" if seed is None else f"/seed{int(seed):03d}"
            spec = {
                "description": (
                    f"LF-OS ratio sweep{desc_seed} shard {shard}/{args.shards} {run_id}"
                ),
                "cmd": cmd,
                "cwd": str(cwd),
                "signature": (
                    f"KG_op/lfos_ratio/{run_id}{signature_seed}/shard{shard:02d}"
                ),
                "project": "KG-SYNTH",
                "vram": 0,
                "cpu": int(args.cpu),
                "ram_mb": int(args.ram_mb),
                "require_node": node,
                "allowed_nodes": list(nodes),
                "allow_no_ckpt": True,
                "allow_no_resume": True,
                "allow_duplicate": True,
            }
            if args.no_bulk_submit:
                out = run_cmd([
                sys.executable, args.scheduler, "submit",
                "--description", spec["description"],
                "--cmd", spec["cmd"],
                "--cwd", spec["cwd"],
                "--signature", spec["signature"],
                "--project", spec["project"],
                "--vram", str(spec["vram"]),
                "--cpu", str(spec["cpu"]),
                "--ram-mb", str(spec["ram_mb"]),
                "--require-node", node,
                *allowed_node_flags(nodes),
                "--allow-no-ckpt",
                "--allow-no-resume",
                "--allow-duplicate",
            ], dry_run=args.dry_run)
            else:
                bulk_specs.append(spec)
                out = ""
            submit_index += 1
            if out:
                print(out, end="")
                parts = out.split()
                if len(parts) > 1:
                    task_ids.append(parts[1])
    if (not args.no_bulk_submit) and bulk_specs:
        payload = "".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in bulk_specs)
        out = run_cmd_input([
            sys.executable,
            args.scheduler,
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            f"lfos-ratio-submit-{run_id}",
        ], payload, dry_run=args.dry_run)
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
            summary = json.loads(out)
            task_ids.extend(item["id"] for item in summary.get("submitted", []))
    if args.dispatch:
        dispatch_cmd = [
            sys.executable,
            args.scheduler,
            "dispatch",
            "--bulk-window",
            "--intent-label",
            f"lfos-ratio-{run_id}",
        ]
        for task_id in task_ids:
            dispatch_cmd.extend(["--task-id", task_id])
        run_cmd(dispatch_cmd, dry_run=args.dry_run)
    print({"run_id": run_id, "task_ids": task_ids})


if __name__ == "__main__":
    main()
