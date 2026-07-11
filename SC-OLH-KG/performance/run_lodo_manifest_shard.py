"""Run one reproducible LODO shard by overriding a frozen manifest.

This keeps causal challengers tied to the exact baseline configuration.  Only
the explicitly exposed fields below may differ from the source manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_lodo_meta_prior import run_one  # noqa: E402
from performance.benchmark_quality import json_safe  # noqa: E402


def load_config(path):
    payload = json.loads(Path(path).read_text())
    if "config" not in payload:
        raise ValueError("LODO manifest must contain a config object")
    config = dict(payload["config"])
    config.setdefault("exact_kg_clip_negative", True)
    config.setdefault("exact_kg_sampling_mode", "iid")
    config.setdefault("task_posterior_safe_generalized", False)
    config.setdefault("task_posterior_safe_boundary_score_weight", 1.0)
    config.setdefault("task_posterior_safe_pairwise_score_weight", 1.0)
    config.setdefault("task_posterior_safe_pairwise_max_history", 16)
    config.setdefault("task_posterior_safe_pairwise_probability_floor", 1e-6)
    config.setdefault("finalist_replication_budget", 0)
    config.setdefault("finalist_replication_count", 2)
    config.setdefault("finalist_replication_min_replicates", 2)
    config.setdefault("finalist_replication_delta", 0.05)
    config.setdefault("finalist_replication_variance_prior_df", 2.0)
    config.setdefault("finalist_replication_expert_stratified", False)
    config.setdefault("finalist_replication_adaptive_race", False)
    config.setdefault("finalist_replication_fixed_universe", False)
    return config


def atomic_write_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--line", default="lodo_teacher")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runtime-checkpoint-dir", required=True)
    parser.add_argument(
        "--ordered-cumulative-exposure",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ordered-max-frequency", type=int, default=8)
    parser.add_argument("--ordered-active-dim", type=int, default=2)
    parser.add_argument("--ordered-frequency-penalty", type=float, default=0.10)
    parser.add_argument(
        "--ordered-basis-mode",
        choices=["full_quadratic", "diagonal_quadratic"],
        default="full_quadratic",
    )
    parser.add_argument(
        "--ordered-adaptive-sparsity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ordered-replace-local-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ordered-semiparametric-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ordered-latent-structure-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ordered-group-shared-shrinkage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ordered-group-ridge-learning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--task-posterior-safe-generalized",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--task-posterior-safe-boundary-weight", type=float, default=1.0)
    parser.add_argument(
        "--task-posterior-safe-pairwise-weight", type=float, default=1.0)
    parser.add_argument(
        "--task-posterior-safe-pairwise-history", type=int, default=16)
    parser.add_argument(
        "--task-posterior-safe-pairwise-floor", type=float, default=1e-6)
    parser.add_argument("--exact-sampling-mode", default="iid")
    parser.add_argument("--exact-mc-samples", type=int, default=2)
    parser.add_argument("--exact-jobs", type=int, default=32)
    parser.add_argument("--parallel-backend", default="process_fork")
    parser.add_argument("--finalist-replication-budget", type=int, default=0)
    parser.add_argument("--finalist-replication-count", type=int, default=2)
    parser.add_argument(
        "--finalist-replication-min-replicates", type=int, default=2)
    parser.add_argument(
        "--finalist-replication-delta", type=float, default=0.05)
    parser.add_argument(
        "--finalist-replication-variance-prior-df", type=float, default=2.0)
    parser.add_argument(
        "--finalist-replication-expert-stratified",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--finalist-replication-adaptive-race",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--finalist-replication-fixed-universe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    config = load_config(args.manifest)
    config.update({
        "meta_ordered_cumulative_exposure": bool(
            args.ordered_cumulative_exposure),
        "meta_ordered_exposure_max_frequency": int(
            args.ordered_max_frequency),
        "meta_ordered_exposure_active_dim": int(args.ordered_active_dim),
        "meta_ordered_exposure_frequency_penalty": float(
            args.ordered_frequency_penalty),
        "meta_ordered_exposure_basis_mode": str(args.ordered_basis_mode),
        "meta_ordered_exposure_adaptive_sparsity": bool(
            args.ordered_adaptive_sparsity),
        "meta_ordered_exposure_replace_local_kernel": bool(
            args.ordered_replace_local_kernel),
        "meta_ordered_exposure_semiparametric_residual": bool(
            args.ordered_semiparametric_residual),
        "meta_ordered_exposure_latent_structure_selection": bool(
            args.ordered_latent_structure_selection),
        "meta_ordered_exposure_group_shared_shrinkage": bool(
            args.ordered_group_shared_shrinkage),
        "meta_ordered_exposure_group_ridge_learning": bool(
            args.ordered_group_ridge_learning),
        "task_posterior_safe_generalized": bool(
            args.task_posterior_safe_generalized),
        "task_posterior_safe_boundary_score_weight": float(
            args.task_posterior_safe_boundary_weight),
        "task_posterior_safe_pairwise_score_weight": float(
            args.task_posterior_safe_pairwise_weight),
        "task_posterior_safe_pairwise_max_history": int(
            args.task_posterior_safe_pairwise_history),
        "task_posterior_safe_pairwise_probability_floor": float(
            args.task_posterior_safe_pairwise_floor),
        "exact_kg_sampling_mode": str(args.exact_sampling_mode),
        "exact_kg_mc_samples": int(args.exact_mc_samples),
        "exact_kg_jobs": int(args.exact_jobs),
        "exact_kg_parallel_backend": str(args.parallel_backend),
        "finalist_replication_budget": int(
            args.finalist_replication_budget),
        "finalist_replication_count": int(
            args.finalist_replication_count),
        "finalist_replication_min_replicates": int(
            args.finalist_replication_min_replicates),
        "finalist_replication_delta": float(
            args.finalist_replication_delta),
        "finalist_replication_variance_prior_df": float(
            args.finalist_replication_variance_prior_df),
        "finalist_replication_expert_stratified": bool(
            args.finalist_replication_expert_stratified),
        "finalist_replication_adaptive_race": bool(
            args.finalist_replication_adaptive_race),
        "finalist_replication_fixed_universe": bool(
            args.finalist_replication_fixed_universe),
        "runtime_checkpoint_dir": str(args.runtime_checkpoint_dir),
        "runtime_checkpoint_resume": True,
        "runtime_checkpoint_interval": 1,
        "checkpoint_path": "",
        "resume_completed_from": "",
        "progress_logging": True,
        "progress_exact_updates": 10,
        "offline_only": True,
    })
    task = {
        "args": config,
        "heldout": str(args.heldout),
        "line": str(args.line),
        "seed": int(args.seed),
    }
    row = run_one(task)
    payload = {
        "schema_version": 1,
        "source_manifest": str(args.manifest),
        "causal_overrides": {
            "meta_ordered_cumulative_exposure": bool(
                args.ordered_cumulative_exposure),
            "meta_ordered_exposure_max_frequency": int(
                args.ordered_max_frequency),
            "meta_ordered_exposure_active_dim": int(args.ordered_active_dim),
            "meta_ordered_exposure_frequency_penalty": float(
                args.ordered_frequency_penalty),
            "meta_ordered_exposure_basis_mode": str(args.ordered_basis_mode),
            "meta_ordered_exposure_adaptive_sparsity": bool(
                args.ordered_adaptive_sparsity),
            "meta_ordered_exposure_replace_local_kernel": bool(
                args.ordered_replace_local_kernel),
            "meta_ordered_exposure_semiparametric_residual": bool(
                args.ordered_semiparametric_residual),
            "meta_ordered_exposure_latent_structure_selection": bool(
                args.ordered_latent_structure_selection),
            "meta_ordered_exposure_group_shared_shrinkage": bool(
                args.ordered_group_shared_shrinkage),
            "meta_ordered_exposure_group_ridge_learning": bool(
                args.ordered_group_ridge_learning),
            "task_posterior_safe_generalized": bool(
                args.task_posterior_safe_generalized),
            "task_posterior_safe_boundary_score_weight": float(
                args.task_posterior_safe_boundary_weight),
            "task_posterior_safe_pairwise_score_weight": float(
                args.task_posterior_safe_pairwise_weight),
            "task_posterior_safe_pairwise_max_history": int(
                args.task_posterior_safe_pairwise_history),
            "task_posterior_safe_pairwise_probability_floor": float(
                args.task_posterior_safe_pairwise_floor),
            "exact_kg_sampling_mode": str(args.exact_sampling_mode),
            "exact_kg_mc_samples": int(args.exact_mc_samples),
            "exact_kg_jobs": int(args.exact_jobs),
            "exact_kg_parallel_backend": str(args.parallel_backend),
            "finalist_replication_budget": int(
                args.finalist_replication_budget),
            "finalist_replication_count": int(
                args.finalist_replication_count),
            "finalist_replication_min_replicates": int(
                args.finalist_replication_min_replicates),
            "finalist_replication_delta": float(
                args.finalist_replication_delta),
            "finalist_replication_variance_prior_df": float(
                args.finalist_replication_variance_prior_df),
            "finalist_replication_expert_stratified": bool(
                args.finalist_replication_expert_stratified),
            "finalist_replication_adaptive_race": bool(
                args.finalist_replication_adaptive_race),
            "finalist_replication_fixed_universe": bool(
                args.finalist_replication_fixed_universe),
        },
        "config": config,
        "rows": [row],
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
