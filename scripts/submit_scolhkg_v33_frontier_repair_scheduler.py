#!/usr/bin/env python3
"""Submit the V33 coherent-contract/frontier repair sentinel matrix."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(SCRIPT_ROOT))

from submit_scolhkg_manifest_gate_scheduler import (  # noqa: E402
    CPU_NODES,
    DEFAULT_DEPLOY,
    DEFAULT_PYTHON,
    DEFAULT_SCHEDULER,
    command_for,
    parse_csv,
)
from submit_scolhkg_v33_terminal_matrix_scheduler import (  # noqa: E402
    apply_promoted_v32,
    parse_task_indices,
)


# name, replication policy, empirical override, frontier policy, max arms,
# coherent decision contract, finalist terminal value, main exact-KG value.
VARIANTS = (
    (
        "v32", "legacy", "legacy", "legacy", 4, "legacy",
        "model_default", "bayes_risk",
    ),
    (
        "v33_legacy_4", "terminal_kg_1step", "certified_only", "legacy",
        4, "legacy", "model_default", "bayes_risk",
    ),
    (
        "v33_coherent_coverage_4", "terminal_kg_1step", "off",
        "coverage_reserved", 4, "certified_lexicographic",
        "certified_lexicographic", "tcb_certified_lexicographic",
    ),
    (
        "v33_coherent_coverage_8", "terminal_kg_1step", "off",
        "coverage_reserved", 8, "certified_lexicographic",
        "certified_lexicographic", "tcb_certified_lexicographic",
    ),
)


def build_specs(args):
    args = apply_promoted_v32(copy.copy(args))
    heldouts = parse_csv(args.heldouts)
    nodes = parse_csv(args.nodes)
    if len(heldouts) != 3:
        raise ValueError("V33 repair gate requires exactly three domains")
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a subset of node001-node006")
    if int(args.n_seeds) != 7:
        raise ValueError("V33 repair sentinel is fixed at seven seeds")

    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    profile_root = deploy_project / "profiles" / str(args.run_id)
    checkpoint_root = deploy_project / "checkpoints" / str(args.run_id)
    local_root = (
        PROJECT_ROOT / "SC-OLH-KG/profiles" / str(args.run_id))
    specs = []
    task_index = 0
    for (
        variant,
        policy,
        empirical_override,
        frontier_policy,
        max_arms,
        decision_contract,
        terminal_value_mode,
        exact_terminal_mode,
    ) in VARIANTS:
        variant_args = copy.copy(args)
        variant_args.finalist_replication_policy = policy
        variant_args.finalist_empirical_override = empirical_override
        variant_args.finalist_frontier_policy = frontier_policy
        variant_args.finalist_terminal_max_arms = int(max_arms)
        variant_args.decision_contract_mode = decision_contract
        variant_args.finalist_terminal_value_mode = terminal_value_mode
        variant_args.exact_terminal_mode = exact_terminal_mode
        variant_args.tcb_v2_mode = "off"
        for heldout in heldouts:
            for offset in range(int(args.n_seeds)):
                seed = int(args.seed_start) + offset
                node = nodes[task_index % len(nodes)]
                task_index += 1
                relative = Path(variant) / heldout / f"seed{seed}"
                result_dir = profile_root / relative
                checkpoint_dir = checkpoint_root / relative
                specs.append({
                    "description": (
                        f"SC-OLH-KG V33 repair {variant} "
                        f"{heldout} seed={seed}"
                    ),
                    "cmd": command_for(
                        variant_args,
                        heldout,
                        seed,
                        result_dir / "result.json",
                        checkpoint_dir,
                    ),
                    "cwd": str(deploy_project),
                    "signature": (
                        f"KG_op/scolhkg_v33_repair/{args.run_id}/"
                        f"{variant}/{heldout}/seed{seed}"
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
                    "local_result_dir": str(local_root / relative),
                    "stage_excludes": ["checkpoints", "profiles", "results"],
                    "allow_duplicate": bool(args.allow_duplicate),
                })
    expected = len(VARIANTS) * 3 * 7
    if len(specs) != expected:
        raise AssertionError(
            f"V33 repair matrix requires {expected} tasks, got {len(specs)}")
    indices = parse_task_indices(args.task_indices, expected)
    return [specs[index] for index in indices]


def _configure_base_args(args):
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
    args.exact_jobs = 11
    args.parallel_backend = "process_fork"
    args.finalist_replication_budget = 3
    args.finalist_replication_count = 2
    args.finalist_replication_min_replicates = 2
    args.finalist_replication_delta = 0.05
    args.finalist_replication_variance_prior_df = 2.0
    args.finalist_replication_expert_stratified = True
    args.finalist_replication_adaptive_race = True
    args.finalist_replication_fixed_universe = True
    args.finalist_terminal_mc_samples = 2
    args.tcb_v2_frontier_count = 0
    return args


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
            PROJECT_ROOT
            / "SC-OLH-KG/performance/baselines/lodo_current.json"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=f"v33_frontier_repair_{time.strftime('%Y%m%d_%H%M%S')}",
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
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--task-indices", default="")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = _configure_base_args(parser.parse_args())

    specs = build_specs(args)
    if args.dry_run:
        print(json.dumps({
            "run_id": args.run_id,
            "n_tasks": len(specs),
            "variants": [row[0] for row in VARIANTS],
            "first_spec": specs[0] if specs else None,
        }, indent=2))
        return
    command = [
        sys.executable,
        str(args.scheduler),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        f"scolhkg-v33-repair-{args.run_id}",
        "--intent-ttl",
        "120",
    ]
    output = subprocess.check_output(
        command, input=json.dumps(specs), text=True)
    print(output, end="" if output.endswith("\n") else "\n")
    submitted = json.loads(output).get("submitted", [])
    task_ids = [item["id"] for item in submitted if item.get("id")]
    if args.dispatch and task_ids:
        subprocess.check_call([
            sys.executable, str(args.scheduler), "dispatch",
        ])
    print(json.dumps({
        "run_id": args.run_id,
        "n_tasks": len(task_ids),
        "task_ids": task_ids,
        "variants": [row[0] for row in VARIANTS],
    }))


if __name__ == "__main__":
    main()
