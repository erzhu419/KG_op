#!/usr/bin/env python3
"""Submit the controlled heteroscedastic optimization/certificate gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "scripts/sync_scolhkg_scheduler_deploy.sh"
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
REMOTE_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy")
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python")
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
SCENARIOS = (
    "homoscedastic",
    "smooth_boundary",
    "optimum_hotspot",
    "safe_interior_hotspot",
    "regime_step",
    "sparse_axis",
    "shared_factor",
    "hidden_periodic",
)
VARIANCE_MODES = ("pooled", "orthogonal", "factor", "oracle")
BACKENDS = ("sobol", "risk_ts", "constrained_ts", "joint_voi")


def _parse_csv(value):
    return tuple(
        item.strip() for item in str(value).split(",") if item.strip())


def _matrix_variants(args):
    scenarios = _parse_csv(args.scenarios)
    explicit = _parse_csv(getattr(args, "variant_specs", ""))
    if explicit:
        variants = []
        for value in explicit:
            parts = tuple(part.strip() for part in value.split(":"))
            if len(parts) != 3:
                raise ValueError(
                    "each variant spec must be backend:variance:selection")
            variants.append(parts)
        modes = tuple(mode for _, mode, _ in variants)
        backends = tuple(backend for backend, _, _ in variants)
        selections = tuple(selection for _, _, selection in variants)
    else:
        modes = _parse_csv(args.variance_modes)
        backends = _parse_csv(args.backends)
        selection = str(getattr(
            args, "terminal_safe_interior_selection", "diverse"))
        selections = (selection,)
        variants = [
            (backend, mode, selection)
            for backend in backends
            for mode in modes
            if backend != "joint_voi" or mode == "factor"
        ]
    unknown_scenarios = sorted(set(scenarios) - set(SCENARIOS))
    unknown_modes = sorted(set(modes) - set(VARIANCE_MODES))
    unknown_backends = sorted(set(backends) - set(BACKENDS))
    unknown_selections = sorted(
        set(selections) - {"diverse", "objective_ranked"})
    if (
        unknown_scenarios
        or unknown_modes
        or unknown_backends
        or unknown_selections
    ):
        raise ValueError({
            "unknown_scenarios": unknown_scenarios,
            "unknown_variance_modes": unknown_modes,
            "unknown_backends": unknown_backends,
            "unknown_terminal_selections": unknown_selections,
        })
    if len(variants) != len(set(variants)):
        raise ValueError("controlled heteroscedastic variants must be unique")
    return scenarios, tuple(variants)


def build_specs(args):
    scenarios, variants = _matrix_variants(args)
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    remote_project = REMOTE_ROOT / "SC-OLH-KG"
    specs = []
    for scenario in scenarios:
        for backend, variance_mode, support_selection in variants:
            exact = backend == "joint_voi"
            cpu = int(args.exact_cpu if exact else args.light_cpu)
            ram_mb = int(args.exact_ram_mb if exact else args.light_ram_mb)
            for seed in range(
                int(args.seed_start),
                int(args.seed_start) + int(args.n_seeds),
            ):
                remote_dir = (
                    remote_project / "profiles" / args.run_id
                    / scenario / variance_mode / backend
                    / support_selection
                    / f"seed{seed:04d}"
                )
                local_dir = (
                    deploy_project / "profiles" / args.run_id
                    / scenario / variance_mode / backend
                    / support_selection
                    / f"seed{seed:04d}"
                )
                output = remote_dir / "result.json"
                command = [
                    "env",
                    "LC_ALL=C",
                    "LANG=C",
                    "SCOLHKG_OFFLINE=1",
                    "PYTHONUNBUFFERED=1",
                    "PYTHONDONTWRITEBYTECODE=1",
                    f"OMP_NUM_THREADS={cpu}",
                    f"MKL_NUM_THREADS={cpu}",
                    f"OPENBLAS_NUM_THREADS={cpu}",
                    str(REMOTE_PYTHON),
                    "performance/benchmark_controlled_heteroscedastic_optimum.py",
                    "--scenario", scenario,
                    "--variance-mode", variance_mode,
                    "--backend", backend,
                    "--seed", str(seed),
                    "--d", str(args.d),
                    "--N", str(args.N),
                    "--n0", str(args.n0),
                    "--K1", str(args.K1),
                    "--posterior-pool-size", str(args.posterior_pool_size),
                    "--state-candidate-count", str(args.state_candidate_count),
                    "--state-inverse-pool-size",
                    str(args.state_inverse_pool_size),
                    "--exact-mc-samples", str(args.exact_mc_samples),
                    "--exact-jobs", str(cpu if exact else 1),
                    "--verification-primary-budget",
                    str(args.verification_primary_budget),
                    "--verification-support-budget",
                    str(args.verification_support_budget),
                    "--verification-delta",
                    str(args.verification_delta),
                    "--terminal-safe-interior-scope",
                    str(args.terminal_safe_interior_scope),
                    "--terminal-safe-interior-selection",
                    str(support_selection),
                    "--decision-terminal-rule",
                    str(getattr(
                        args, "decision_terminal_rule", "bayes_risk")),
                    "--decision-terminal-maximum-violation-probability",
                    str(getattr(
                        args,
                        "decision_terminal_maximum_violation_probability",
                        0.05,
                    )),
                    "--out", str(output),
                ]
                specs.append({
                    "description": (
                        f"controlled hetero {scenario} {variance_mode} "
                        f"{backend} {support_selection} seed={seed}"
                    ),
                    "cmd": f"{shlex.join(command)} && echo DONE",
                    "cwd": str(deploy_project),
                    "signature": (
                        f"KG_op/controlled_hetero/{args.run_id}/{scenario}/"
                        f"{variance_mode}/{backend}/{support_selection}/"
                        f"seed{seed:04d}"
                    ),
                    "project": "KG-SYNTH",
                    "vram": 0,
                    "cpu": cpu,
                    "ram_mb": ram_mb,
                    "allowed_nodes": list(CPU_NODES),
                    "result_dir": str(remote_dir),
                    "local_result_dir": str(local_dir),
                    "stage_excludes": [
                        "checkpoints", "profiles", "results",
                    ],
                    "allow_duplicate": True,
                })
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--run-id", default=(
        "controlled_hetero_optimum_gate_d1000_n20_s5_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument(
        "--variance-modes", default=",".join(VARIANCE_MODES))
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument(
        "--variant-specs",
        default="",
        help=(
            "Optional comma-separated backend:variance:selection triples; "
            "when set they replace the backend/variance Cartesian matrix."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--K1", type=int, default=24)
    parser.add_argument("--posterior-pool-size", type=int, default=128)
    parser.add_argument("--state-candidate-count", type=int, default=8)
    parser.add_argument("--state-inverse-pool-size", type=int, default=256)
    parser.add_argument("--exact-mc-samples", type=int, default=8)
    parser.add_argument("--verification-primary-budget", type=int, default=80)
    parser.add_argument("--verification-support-budget", type=int, default=96)
    parser.add_argument("--verification-delta", type=float, default=0.05)
    parser.add_argument(
        "--terminal-safe-interior-scope",
        choices=("initial", "observed"),
        default="initial",
    )
    parser.add_argument(
        "--terminal-safe-interior-selection",
        choices=("diverse", "objective_ranked"),
        default="diverse",
    )
    parser.add_argument(
        "--decision-terminal-rule",
        choices=("bayes_risk", "feasible_first"),
        default="bayes_risk",
    )
    parser.add_argument(
        "--decision-terminal-maximum-violation-probability",
        type=float,
        default=0.05,
    )
    parser.add_argument("--light-cpu", type=int, default=1)
    parser.add_argument("--light-ram-mb", type=int, default=4096)
    parser.add_argument("--exact-cpu", type=int, default=12)
    parser.add_argument("--exact-ram-mb", type=int, default=16384)
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios, variants = _matrix_variants(args)
    specs = build_specs(args)
    expected = len(scenarios) * len(variants) * int(args.n_seeds)
    if len(specs) != expected:
        raise RuntimeError(f"built {len(specs)} tasks, expected {expected}")
    signatures = [spec["signature"] for spec in specs]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("controlled heteroscedastic signatures are not unique")
    if args.d < 3 or args.n0 > args.N:
        raise ValueError("invalid d/n0/N contract")

    registration = {
        "schema_version": 1,
        "run_id": args.run_id,
        "submitted_at": None,
        "status": "dry_run" if args.dry_run else "submitted",
        "scenario_count": int(len(scenarios)),
        "variant_count_per_scenario": int(len(variants)),
        "seeds_per_variant": int(args.n_seeds),
        "task_count": int(len(specs)),
        "scenarios": list(scenarios),
        "variants": [
            {
                "backend": backend,
                "variance_mode": mode,
                "terminal_safe_interior_selection": selection,
            }
            for backend, mode, selection in variants
        ],
        "contract": {
            "raw_dimension": int(args.d),
            "search_budget": int(args.N),
            "initial_budget": int(args.n0),
            "initial_design": "common_sobol",
            "source_archive_used": False,
            "source_proposal_used": False,
            "problem_specific_refinement_used": False,
            "terminal_verification_primary_budget": int(
                args.verification_primary_budget),
            "terminal_verification_support_budget": int(
                args.verification_support_budget),
            "verification_updates_optimizer": False,
            "terminal_safe_interior_candidate_scope": str(
                args.terminal_safe_interior_scope),
            "terminal_safe_interior_selection_modes": sorted({
                selection for _, _, selection in variants
            }),
            "decision_terminal_rule": str(args.decision_terminal_rule),
            "decision_terminal_maximum_violation_probability": float(
                args.decision_terminal_maximum_violation_probability),
            "oracle_rows_are_diagnostic_only": True,
        },
        "allowed_nodes": list(CPU_NODES),
        "checkpoint_results_synced_locally": False,
    }
    if args.dry_run:
        registration["specs"] = specs
        print(json.dumps(registration, indent=2))
        return
    if args.sync_remote:
        subprocess.run([str(SYNC)], cwd=ROOT, check=True)
    output = subprocess.check_output(
        [
            sys.executable,
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            f"controlled-hetero-{args.run_id}",
        ],
        input=json.dumps(specs),
        text=True,
    )
    response = json.loads(output)
    submitted = response.get("submitted", [])
    task_ids = [row["id"] for row in submitted if row.get("id")]
    registration["submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    registration["task_ids"] = task_ids
    registration_path = (
        Path(args.deploy) / "SC-OLH-KG" / "profiles" / args.run_id
        / "submission_manifest.json"
    )
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.dispatch and task_ids:
        subprocess.run(
            [
                sys.executable,
                str(args.scheduler),
                "dispatch",
                *sum((["--task-id", task_id] for task_id in task_ids), []),
            ],
            check=True,
        )
    print(json.dumps(registration, indent=2))


if __name__ == "__main__":
    main()
