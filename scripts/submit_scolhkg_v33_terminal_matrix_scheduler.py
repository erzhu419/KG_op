#!/usr/bin/env python3
"""Submit the preregistered V33 terminal-policy matrix as one bulk intent."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from submit_scolhkg_manifest_gate_scheduler import (  # noqa: E402
    CPU_NODES,
    DEFAULT_DEPLOY,
    DEFAULT_PYTHON,
    DEFAULT_SCHEDULER,
    command_for,
    parse_csv,
)


VARIANTS = (
    ("v32", "legacy", "legacy"),
    ("posterior_only", "legacy", "off"),
    (
        "commit_before_switch",
        "commit_before_switch",
        "certified_only",
    ),
    ("terminal_kg_1step", "terminal_kg_1step", "certified_only"),
    ("terminal_kg_depth3", "terminal_kg_depth3", "certified_only"),
)


def parse_task_indices(value, total):
    value = str(value or "").strip()
    if not value:
        return list(range(int(total)))
    selected = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(token))
    if not selected or min(selected) < 0 or max(selected) >= int(total):
        raise ValueError(
            f"task indices must be within [0, {int(total) - 1}]")
    return sorted(selected)


def _promoted_config(path):
    payload = json.loads(Path(path).read_text())
    promoted = payload["promoted_candidate"]
    config = dict(promoted["config"])
    config["line"] = str(promoted.get("line", "lodo_teacher"))
    return config


def apply_promoted_v32(args):
    config = _promoted_config(args.baseline)
    mapping = {
        "line": "line",
        "ordered_cumulative_exposure": "ordered_cumulative_exposure",
        "ordered_max_frequency": "ordered_max_frequency",
        "ordered_active_dim": "ordered_active_dim",
        "ordered_frequency_penalty": "ordered_frequency_penalty",
        "ordered_basis_mode": "ordered_basis_mode",
        "ordered_adaptive_sparsity": "ordered_adaptive_sparsity",
        "ordered_replace_local_kernel": "ordered_replace_local_kernel",
        "ordered_semiparametric_residual": (
            "ordered_semiparametric_residual"),
        "ordered_latent_structure_selection": (
            "ordered_latent_structure_selection"),
        "ordered_group_shared_shrinkage": (
            "ordered_group_shared_shrinkage"),
        "ordered_group_ridge_learning": "ordered_group_ridge_learning",
        "task_posterior_safe_generalized": (
            "task_posterior_safe_generalized"),
        "task_posterior_mandatory_universal_count": (
            "task_posterior_mandatory_universal_count"),
        "task_latent_inference_mode": "task_latent_inference_mode",
        "task_latent_calibration_mode": "task_latent_calibration_mode",
        "observable_mean_coordinate": "observable_mean_coordinate",
        "source_observation_mode": "source_observation_mode",
        "source_observation_replicates": "source_observation_replicates",
        "source_design_mode": "source_design_mode",
        "source_universal_fraction": "source_universal_fraction",
        "source_consensus_template_count": (
            "source_consensus_template_count"),
        "finalist_replication_budget": "finalist_replication_budget",
        "finalist_replication_count": "finalist_replication_count",
        "finalist_observed_safety_count": (
            "finalist_observed_safety_count"),
        "finalist_replication_min_replicates": (
            "finalist_replication_min_replicates"),
        "finalist_replication_delta": "finalist_replication_delta",
        "finalist_replication_variance_prior_df": (
            "finalist_replication_variance_prior_df"),
        "finalist_replication_expert_stratified": (
            "finalist_replication_expert_stratified"),
        "finalist_replication_adaptive_race": (
            "finalist_replication_adaptive_race"),
        "finalist_replication_fixed_universe": (
            "finalist_replication_fixed_universe"),
        "finalist_frontier_policy": "finalist_frontier_policy",
        "exact_kg_mc_samples": "exact_mc_samples",
        "exact_kg_jobs": "exact_jobs",
        "exact_kg_parallel_backend": "parallel_backend",
        "exact_kg_sampling_mode": "exact_sampling_mode",
        "exact_kg_terminal_mode": "exact_terminal_mode",
        "decision_contract_mode": "decision_contract_mode",
    }
    for source, destination in mapping.items():
        if source in config:
            setattr(args, destination, config[source])
    return args


def build_specs(args):
    args = apply_promoted_v32(copy.copy(args))
    heldouts = parse_csv(args.heldouts)
    nodes = parse_csv(args.nodes)
    if len(heldouts) != 3:
        raise ValueError("V33 preregistration requires exactly three domains")
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    if int(args.n_seeds) != 7:
        raise ValueError("V33 preregistration requires exactly seven seeds")

    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    profile_root = deploy_project / "profiles" / str(args.run_id)
    checkpoint_root = deploy_project / "checkpoints" / str(args.run_id)
    specs = []
    task_index = 0
    for variant, policy, empirical_override in VARIANTS:
        variant_args = copy.copy(args)
        variant_args.finalist_replication_policy = policy
        variant_args.finalist_empirical_override = empirical_override
        for heldout in heldouts:
            for offset in range(int(args.n_seeds)):
                seed = int(args.seed_start) + offset
                node = nodes[task_index % len(nodes)]
                task_index += 1
                result_dir = (
                    profile_root / variant / heldout / f"seed{seed}")
                checkpoint_dir = (
                    checkpoint_root / variant / heldout / f"seed{seed}")
                result_file = result_dir / "result.json"
                specs.append({
                    "description": (
                        f"SC-OLH-KG V33 {variant} {heldout} seed={seed}"
                    ),
                    "cmd": command_for(
                        variant_args,
                        heldout,
                        seed,
                        result_file,
                        checkpoint_dir,
                    ),
                    "cwd": str(deploy_project),
                    "signature": (
                        f"KG_op/scolhkg_v33/{args.run_id}/{variant}/"
                        f"{heldout}/seed{seed}"
                    ),
                    "project": "KG-SYNTH",
                    "vram": 0,
                    "cpu": int(args.cpu),
                    "ram_mb": int(args.ram_mb),
                    "require_node": node,
                    "allowed_nodes": list(nodes),
                    "ckpt_dir": str(checkpoint_dir),
                    "ckpt_glob": "checkpoint_latest.pkl",
                    "allow_initial_resume_scan_error": True,
                    "allow_no_resume": True,
                    "result_dir": str(result_dir),
                    "local_result_dir": str(result_dir),
                    "stage_excludes": ["checkpoints", "profiles", "results"],
                    "allow_duplicate": bool(args.allow_duplicate),
                })
    expected = len(VARIANTS) * 3 * 7
    if len(specs) != expected:
        raise AssertionError(
            f"V33 matrix must contain {expected} tasks, got {len(specs)}")
    indices = parse_task_indices(
        getattr(args, "task_indices", ""), expected)
    return [specs[index] for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            DEFAULT_DEPLOY
            / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "SC-OLH-KG/performance/baselines/lodo_current.json"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=f"v33_terminal_matrix_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--heldouts",
        default=(
            "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
            "QueueResourceControl"
        ),
    )
    parser.add_argument("--line", default="lodo_teacher")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=7)
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--cpu", type=int, default=33)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--finalist-terminal-max-arms", type=int, default=4)
    parser.add_argument("--finalist-terminal-mc-samples", type=int, default=2)
    parser.add_argument(
        "--task-indices",
        default="",
        help="Stable zero-based indices/ranges, e.g. 0-28,63,70-75.",
    )
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    # Fields consumed by the shared command builder and frozen by V32.
    args.ordered_cumulative_exposure = True
    args.ordered_max_frequency = 8
    args.ordered_active_dim = 2
    args.ordered_frequency_penalty = 0.1
    args.ordered_basis_mode = "diagonal_quadratic"
    args.ordered_adaptive_sparsity = True
    args.ordered_replace_local_kernel = False
    args.ordered_semiparametric_residual = False
    args.ordered_latent_structure_selection = True
    args.ordered_group_shared_shrinkage = False
    args.ordered_group_ridge_learning = True
    args.task_posterior_safe_generalized = True
    args.task_posterior_safe_boundary_weight = 1.0
    args.task_posterior_safe_pairwise_weight = 1.0
    args.task_posterior_safe_pairwise_history = 16
    args.task_posterior_safe_pairwise_floor = 1e-6
    args.task_latent_inference_mode = "shadow"
    args.task_latent_calibration_mode = "source_profiles"
    args.exact_sampling_mode = "iid"
    args.exact_mc_samples = 2
    args.exact_jobs = 32
    args.parallel_backend = "process_fork"
    args.finalist_replication_budget = 3
    args.finalist_replication_count = 2
    args.finalist_replication_min_replicates = 2
    args.finalist_replication_delta = 0.05
    args.finalist_replication_variance_prior_df = 2.0
    args.finalist_replication_expert_stratified = True
    args.finalist_replication_adaptive_race = True
    args.finalist_replication_fixed_universe = True

    specs = build_specs(args)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    command = [
        sys.executable,
        str(args.scheduler),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        f"scolhkg-v33-{args.run_id}",
        "--intent-ttl",
        "120",
    ]
    output = subprocess.check_output(
        command,
        input=json.dumps(specs),
        text=True,
    )
    print(output, end="" if output.endswith("\n") else "\n")
    submitted = json.loads(output).get("submitted", [])
    task_ids = [item["id"] for item in submitted if item.get("id")]
    if args.dispatch and task_ids:
        subprocess.check_call([
            sys.executable, str(args.scheduler), "dispatch",
        ])
    print(json.dumps({
        "run_id": args.run_id,
        "task_ids": task_ids,
        "n_tasks": len(task_ids),
        "variants": [item[0] for item in VARIANTS],
    }))


if __name__ == "__main__":
    main()
