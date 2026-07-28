#!/usr/bin/env python3
"""Submit the V33/TCB-V2 joint sentinel after the source gate passes."""

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


VARIANTS = (
    (
        "v32", "legacy", "legacy", "legacy", 4, "legacy",
        "off", "model_default", "bayes_risk",
    ),
    (
        "v33_coherent_coverage_8", "terminal_kg_1step", "off",
        "coverage_reserved", 8, "certified_lexicographic",
        "off", "certified_lexicographic",
        "tcb_certified_lexicographic",
    ),
    (
        "tcb_frontier_only", "terminal_kg_1step", "off",
        "coverage_reserved", 8, "certified_lexicographic",
        "frontier", "certified_lexicographic",
        "tcb_certified_lexicographic",
    ),
    (
        "tcb_full_three_layer_4", "terminal_kg_1step", "off",
        "coverage_reserved", 4, "certified_lexicographic",
        "certified", "certified_lexicographic",
        "tcb_certified_lexicographic",
    ),
    (
        "tcb_full_three_layer_8", "terminal_kg_1step", "off",
        "coverage_reserved", 8, "certified_lexicographic",
        "certified", "certified_lexicographic",
        "tcb_certified_lexicographic",
    ),
)


def load_promoted_tcb(path):
    payload = json.loads(Path(path).read_text())
    if not payload.get("gate_pass"):
        raise RuntimeError(
            "TCB-V2 source gate did not pass; online V33 sentinel is blocked")
    promoted = payload.get("promoted_candidate")
    if (
        payload.get("selection_protocol")
        != "nested_source_lodo_per_outer_target"
        or payload.get("outer_truth_used_for_hyperparameter_selection")
        is not False
    ):
        raise ValueError(
            "online TCB-V2 requires nested source-only hyperparameter selection")
    if not promoted or "config_by_heldout" not in promoted:
        raise ValueError("gate summary has no held-out TCB-V2 configuration map")
    return {
        str(heldout): dict(config)
        for heldout, config in promoted["config_by_heldout"].items()
    }


def apply_tcb_config(args, config):
    args.tcb_v2_descriptor_mode = str(
        config.get("descriptor_mode", "learned_risk"))
    args.tcb_v2_coordinate = str(config["coordinate"])
    args.tcb_v2_geometry = str(config["geometry"])
    args.tcb_v2_rank = int(config["rank"])
    args.tcb_v2_ridge = float(config["ridge"])
    args.tcb_v2_domain_penalty = float(config["domain_penalty"])
    args.tcb_v2_adaptation_ridge = float(config["adaptation_ridge"])
    args.tcb_v2_effect_ridge = float(config["effect_ridge"])
    args.tcb_v2_rotation_mode = str(config.get("rotation_mode", "none"))
    args.tcb_v2_rotation_ridge = float(config.get("rotation_ridge", 5.0))
    args.tcb_v2_target_residual_rank = int(
        config.get("target_residual_rank", 0))
    args.tcb_v2_residual_ridge = float(config.get("residual_ridge", 5.0))
    args.tcb_v2_upper_alpha = float(config["upper_alpha"])
    args.tcb_v2_boundary_temperature = 1.0
    args.tcb_v2_calibration_prior_df = 2.0
    args.tcb_v2_hierarchy_iterations = 5
    args.tcb_v2_frontier_count = 2
    return args


def build_specs(args):
    args = apply_promoted_v32(copy.copy(args))
    tcb_configs = load_promoted_tcb(args.gate_summary)
    heldouts = parse_csv(args.heldouts)
    nodes = parse_csv(args.nodes)
    if len(heldouts) != 3:
        raise ValueError("joint sentinel requires exactly three held-out domains")
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a subset of node001-node006")
    if int(args.n_seeds) != 7:
        raise ValueError("joint sentinel is preregistered for seven seeds")

    remote_root = (
        Path(args.deploy) / "SC-OLH-KG/profiles" / str(args.run_id))
    remote_checkpoint_root = (
        Path(args.deploy) / "SC-OLH-KG/checkpoints" / str(args.run_id))
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
        tcb_mode,
        terminal_value_mode,
        exact_terminal_mode,
    ) in VARIANTS:
        variant_args = copy.copy(args)
        variant_args.finalist_replication_policy = policy
        variant_args.finalist_empirical_override = empirical_override
        variant_args.finalist_frontier_policy = frontier_policy
        variant_args.finalist_terminal_max_arms = int(max_arms)
        variant_args.decision_contract_mode = decision_contract
        variant_args.tcb_v2_mode = tcb_mode
        variant_args.finalist_terminal_value_mode = terminal_value_mode
        variant_args.exact_terminal_mode = exact_terminal_mode
        for heldout in heldouts:
            if heldout not in tcb_configs:
                raise ValueError(
                    f"gate summary has no source-selected config for {heldout}")
            heldout_args = apply_tcb_config(
                copy.copy(variant_args), tcb_configs[heldout])
            for seed_offset in range(int(args.n_seeds)):
                seed = int(args.seed_start) + seed_offset
                node = nodes[task_index % len(nodes)]
                task_index += 1
                relative = Path(variant) / heldout / f"seed{seed}"
                remote_result_dir = remote_root / relative
                local_result_dir = local_root / relative
                checkpoint_dir = remote_checkpoint_root / relative
                result_file = remote_result_dir / "result.json"
                specs.append({
                    "description": (
                        f"SC-OLH-KG TCB-V2/V33 {variant} "
                        f"{heldout} seed={seed}"
                    ),
                    "cmd": command_for(
                        heldout_args,
                        heldout,
                        seed,
                        result_file,
                        checkpoint_dir,
                    ),
                    "cwd": str(Path(args.deploy) / "SC-OLH-KG"),
                    "signature": (
                        f"KG_op/tcb_v2_v33/{args.run_id}/{variant}/"
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
                    "result_dir": str(remote_result_dir),
                    "local_result_dir": str(local_result_dir),
                    "stage_excludes": ["checkpoints", "profiles", "results"],
                    "allow_duplicate": bool(args.allow_duplicate),
                })
    expected = len(VARIANTS) * 3 * 7
    if len(specs) != expected:
        raise AssertionError(
            f"joint matrix requires {expected} tasks, got {len(specs)}")
    indices = parse_task_indices(args.task_indices, expected)
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
            PROJECT_ROOT
            / "SC-OLH-KG/performance/baselines/lodo_current.json"
        ),
    )
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument(
        "--run-id",
        default=f"tcb_v2_v33_{time.strftime('%Y%m%d_%H%M%S')}",
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
    parser.add_argument("--finalist-terminal-max-arms", type=int, default=4)
    parser.add_argument("--finalist-terminal-mc-samples", type=int, default=2)
    parser.add_argument("--task-indices", default="")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

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
    args.exact_terminal_mode = "hard_certified"
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

    specs = build_specs(args)
    if args.dry_run:
        print(json.dumps({
            "run_id": args.run_id,
            "n_tasks": len(specs),
            "variants": [variant[0] for variant in VARIANTS],
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
        f"scolhkg-tcb-v2-v33-{args.run_id}",
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
        "variants": [variant[0] for variant in VARIANTS],
    }))


if __name__ == "__main__":
    main()
