#!/usr/bin/env python3
"""Submit acquisition-agnostic SC-OLH causal ablation matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "scripts/sync_scolhkg_scheduler_deploy.sh"
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
REMOTE_ROOT = Path("/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy")
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"
)
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
HELDOUTS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
BACKENDS = (
    "n0_best",
    "random",
    "sobol",
    "risk_ts",
    "bayes_risk_ei",
    "constrained_ei",
    "transfer_utility",
    "additive",
    "exact_kg",
)
PRIOR_PROFILES = (
    "none",
    "low_frequency_only",
    "orthogonality_only",
    "sparsity_only",
    "additivity_only",
    "leave_out_low_frequency",
    "leave_out_orthogonality",
    "leave_out_sparsity",
    "leave_out_additivity",
    "full",
)
HVD_PROFILES = (
    "pooled",
    "class",
    "orthogonal_pointwise",
    "factor_pointwise",
    "factor_cumulative",
)
TRACKS = ("backends", "priors", "hvd", "discrepancy", "recheck", "penalty")
SUPPLEMENTAL_TRACKS = ("replication_dominance",)


def _parse_csv(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _base_flags(exact_jobs, exact_terminal_mode="bayes_risk"):
    return [
        "--ordered-max-frequency", "8",
        "--ordered-active-dim", "2",
        "--ordered-frequency-penalty", "0.1",
        "--ordered-basis-mode", "diagonal_quadratic",
        "--ordered-orthogonal-coordinates",
        "--ordered-cumulative-exposure",
        "--ordered-adaptive-sparsity",
        "--ordered-replace-local-kernel",
        "--ordered-latent-structure-selection",
        "--ordered-group-ridge-learning",
        "--exact-sampling-mode", "iid",
        "--exact-terminal-mode", str(exact_terminal_mode),
        "--exact-mc-samples", "2",
        "--exact-jobs", str(int(exact_jobs)),
        "--parallel-backend", "process_fork",
        "--source-observation-mode", "replicated",
        "--source-observation-replicates", "3",
        "--source-design-mode", "universal_mixture",
        "--source-universal-fraction", "1.0",
        "--source-consensus-template-count", "12",
        "--observable-mean-mode", "latent",
        "--observable-mean-latent-dim", "2",
        "--observable-mean-training-target", "constraint_mean",
        "--task-posterior-safe-generalized",
        "--task-posterior-safe-boundary-weight", "1.0",
        "--task-posterior-safe-pairwise-weight", "1.0",
        "--task-posterior-safe-pairwise-history", "16",
        "--task-posterior-safe-pairwise-floor", "1e-06",
        "--task-posterior-mandatory-universal-count", "10",
        "--task-latent-inference-mode", "shadow",
        "--task-latent-calibration-mode", "source_profiles",
        "--finalist-replication-budget", "0",
        "--finalist-empirical-override", "off",
        "--finalist-frontier-policy", "legacy",
        "--decision-contract-mode", "legacy",
        "--decision-recommend-observed-only",
        "--no-observable-mean-coordinate",
    ]


def experiment_variants(tracks):
    rows = []
    if "backends" in tracks:
        for initial_design in ("common_sobol", "source_informed"):
            for backend in BACKENDS:
                rows.append({
                    "track": "backends",
                    "name": f"{initial_design}_{backend}",
                    "initial_design": initial_design,
                    "backend": backend,
                    "prior": "full",
                    "hvd": "factor_cumulative",
                    "discrepancy": True,
                    "recheck_top_k": 0,
                    "risk_penalty": 5.0,
                    "utility_weight": 1.0,
                })
    if "priors" in tracks:
        for profile in PRIOR_PROFILES:
            rows.append({
                "track": "priors",
                "name": profile,
                "initial_design": "source_informed",
                "backend": "risk_ts",
                "prior": profile,
                "hvd": "factor_cumulative",
                "discrepancy": True,
                "recheck_top_k": 0,
                "risk_penalty": 5.0,
                "utility_weight": 1.0,
            })
    if "hvd" in tracks:
        for profile in HVD_PROFILES:
            rows.append({
                "track": "hvd",
                "name": profile,
                "initial_design": "source_informed",
                "backend": "risk_ts",
                "prior": "full",
                "hvd": profile,
                "discrepancy": True,
                "recheck_top_k": 0,
                "risk_penalty": 5.0,
                "utility_weight": 1.0,
            })
    if "discrepancy" in tracks:
        for enabled in (False, True):
            rows.append({
                "track": "discrepancy",
                "name": "adaptive" if enabled else "frozen",
                "initial_design": "source_informed",
                "backend": "risk_ts",
                "prior": "full",
                "hvd": "factor_cumulative",
                "discrepancy": enabled,
                "recheck_top_k": 0,
                "risk_penalty": 5.0,
                "utility_weight": 1.0,
            })
    if "recheck" in tracks:
        for top_k in (0, 1, 2):
            rows.append({
                "track": "recheck",
                "name": f"top{top_k}_rep3",
                "initial_design": "source_informed",
                "backend": "risk_ts",
                "prior": "full",
                "hvd": "factor_cumulative",
                "discrepancy": True,
                "recheck_top_k": top_k,
                "risk_penalty": 5.0,
                "utility_weight": 1.0,
            })
    if "penalty" in tracks:
        for backend in ("risk_ts", "bayes_risk_ei"):
            for penalty in (2.0, 5.0, 10.0, 20.0):
                rows.append({
                    "track": "penalty",
                    "name": f"{backend}_rho{penalty:g}",
                    "initial_design": "source_informed",
                    "backend": backend,
                    "prior": "full",
                    "hvd": "factor_cumulative",
                    "discrepancy": True,
                    "recheck_top_k": 0,
                    "risk_penalty": penalty,
                    "utility_weight": 1.0,
                })
        for weight in (0.25, 0.5, 1.0, 2.0):
            rows.append({
                "track": "penalty",
                "name": f"transfer_utility_w{weight:g}",
                "initial_design": "source_informed",
                "backend": "transfer_utility",
                "prior": "full",
                "hvd": "factor_cumulative",
                "discrepancy": True,
                "recheck_top_k": 0,
                "risk_penalty": 5.0,
                "utility_weight": weight,
            })
    if "replication_dominance" in tracks:
        for initial_design in ("common_sobol", "source_informed"):
            rows.append({
                "track": "replication_dominance",
                "name": f"{initial_design}_posterior_dominance",
                "initial_design": initial_design,
                "backend": "exact_kg",
                "prior": "full",
                "hvd": "factor_cumulative",
                "discrepancy": True,
                "recheck_top_k": 0,
                "risk_penalty": 5.0,
                "utility_weight": 1.0,
                "adaptive_replication_voi": True,
                "replication_candidate_count": 10,
                "posterior_dominance": True,
                "posterior_dominance_delta": 0.05,
                "exact_terminal_mode": "bayes_risk_dominance",
            })

    unique = []
    fingerprints = set()
    for row in rows:
        fingerprint = tuple(sorted(
            (key, str(value)) for key, value in row.items()
            if key not in {"track", "name"}
        ))
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        unique.append(row)
    return unique


def build_specs(args):
    nodes = _parse_csv(args.nodes)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    heldouts = _parse_csv(args.heldouts)
    if not heldouts or any(heldout not in HELDOUTS for heldout in heldouts):
        raise ValueError("heldouts must use the registered three-domain matrix")
    tracks = _parse_csv(args.tracks)
    allowed_tracks = (*TRACKS, *SUPPLEMENTAL_TRACKS)
    if not tracks or any(track not in allowed_tracks for track in tracks):
        raise ValueError(f"tracks must be selected from {allowed_tracks}")

    local_project = Path(args.deploy) / "SC-OLH-KG"
    remote_project = REMOTE_ROOT / "SC-OLH-KG"
    try:
        manifest_relative = Path(args.manifest).relative_to(local_project)
        remote_manifest = remote_project / manifest_relative
    except ValueError:
        remote_manifest = Path(args.manifest)
    variants = experiment_variants(tracks)
    specs = []
    for variant in variants:
        for heldout in heldouts:
            local_design = (
                local_project / "archives" / args.source_run_id / heldout
                / "source_initial_designs.json"
            )
            remote_design = (
                remote_project / "archives" / args.source_run_id / heldout
                / "source_initial_designs.json"
            )
            for offset in range(int(args.n_seeds)):
                seed = int(args.seed_start) + offset
                cell = f"{variant['track']}/{variant['name']}"
                remote_result_dir = (
                    remote_project / "profiles" / args.run_id / cell
                    / heldout / f"seed{seed}"
                )
                local_result_dir = (
                    local_project / "profiles" / args.run_id / cell
                    / heldout / f"seed{seed}"
                )
                checkpoint_dir = (
                    remote_project / "checkpoints" / args.run_id / cell
                    / heldout / f"seed{seed}"
                )
                command = [
                    "env", "LC_ALL=C", "LANG=C",
                    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                    "OPENBLAS_NUM_THREADS=1", "SCOLHKG_OFFLINE=1",
                    "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                    str(args.python),
                    "performance/run_lodo_manifest_shard.py",
                    "--manifest", str(remote_manifest),
                    "--heldout", heldout,
                    "--line", "lodo",
                    "--seed", str(seed),
                    "--experiment-variant", (
                        f"structural_backend/{cell}"
                    ),
                    "--out", str(remote_result_dir / "result.json"),
                    "--runtime-checkpoint-dir", str(checkpoint_dir),
                    "--d", str(args.d),
                    "--N", str(args.N),
                    "--n0", str(args.n0),
                    "--initial-design", str(variant["initial_design"]),
                    "--structural-prior-profile", str(variant["prior"]),
                    "--hvd-profile", str(variant["hvd"]),
                    "--decision-backend", str(variant["backend"]),
                    "--decision-risk-penalty", str(variant["risk_penalty"]),
                    "--decision-source-utility-weight",
                    str(variant["utility_weight"]),
                    "--certification-recheck-top-k",
                    str(variant["recheck_top_k"]),
                    "--certification-recheck-min-replicates", "3",
                    *_base_flags(
                        args.exact_jobs,
                        variant.get("exact_terminal_mode", "bayes_risk"),
                    ),
                ]
                if variant["track"] == "replication_dominance":
                    command.extend([
                        "--replication-candidate-count",
                        str(variant["replication_candidate_count"]),
                        "--replication-max-per-solution", "5",
                        "--posterior-dominance-delta",
                        str(variant["posterior_dominance_delta"]),
                    ])
                    command.append(
                        "--adaptive-replication-voi"
                        if variant["adaptive_replication_voi"]
                        else "--no-adaptive-replication-voi"
                    )
                    command.append(
                        "--posterior-dominance-enabled"
                        if variant["posterior_dominance"]
                        else "--no-posterior-dominance-enabled"
                    )
                command.append(
                    "--source-discrepancy-update"
                    if variant["discrepancy"]
                    else "--no-source-discrepancy-update"
                )
                wait_for_files = []
                if variant["initial_design"] == "source_informed":
                    command.extend([
                        "--initial-design-file", str(remote_design),
                    ])
                    wait_for_files.append(str(local_design))
                specs.append({
                    "description": (
                        f"SC-OLH {args.run_id} {cell} {heldout} seed={seed}"
                    ),
                    "cmd": f"{shlex.join(command)} && echo DONE",
                    "cwd": str(local_project),
                    "signature": (
                        f"KG_op/scolhkg_structural_backend/{args.run_id}/"
                        f"{cell}/{heldout}/seed{seed}"
                    ),
                    "project": "KG-SYNTH",
                    "vram": 0,
                    "cpu": int(args.cpu),
                    "ram_mb": int(args.ram_mb),
                    "allowed_nodes": list(nodes),
                    "wait_for_files": wait_for_files,
                    "ckpt_dir": str(checkpoint_dir),
                    "ckpt_glob": "checkpoint_latest.pkl",
                    "result_dir": str(remote_result_dir),
                    "local_result_dir": str(local_result_dir),
                    "stage_excludes": ["checkpoints", "profiles", "results"],
                    "allow_duplicate": True,
                })
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--python", type=Path, default=REMOTE_PYTHON)
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--heldouts", default=",".join(HELDOUTS))
    parser.add_argument("--tracks", default=",".join(TRACKS))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            DEFAULT_DEPLOY
            / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"
        ),
    )
    parser.add_argument(
        "--source-run-id",
        default="transfer_source_informed_official_n20_s20_20260716",
    )
    parser.add_argument(
        "--run-id",
        default="scolh_structural_backend_gate_n20_s10_20260716",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--exact-jobs", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs = build_specs(args)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    if not args.no_sync:
        subprocess.run([str(SYNC)], check=True, cwd=ROOT)
    payload = "\n".join(json.dumps(spec) for spec in specs) + "\n"
    subprocess.run(
        [
            sys.executable,
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            str(args.run_id),
        ],
        input=payload,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
