#!/usr/bin/env python3
"""Submit the source-aligned chance-boundary coordinate causal gate."""

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
DEFAULT_SOURCE_RUN_ID = (
    "scolh_lowfreq_support_dimholdout_d1000_s5_20260716/"
    "proposals/risk_objective_atlas/low_frequency_only"
)
VARIANTS = {
    "latent_control": {
        "observable_mean_mode": "latent",
        "training_target": "constraint_mean",
        "boundary_candidate_count": 0,
    },
    "phi_mean_only": {
        "observable_mean_mode": "boundary_aligned",
        "training_target": "chance_margin",
        "boundary_candidate_count": 0,
    },
    "phi_mean_proposal": {
        "observable_mean_mode": "boundary_aligned",
        "training_target": "chance_margin",
        "boundary_candidate_count": 12,
    },
}
SCENARIOS = (
    ("FactorShockStatePolicyRZDT1", 0.0),
    ("FactorShockStatePolicyRZDT1", 4.0),
    ("InventorySupplyChain", 1.0),
    ("QueueResourceControl", 1.0),
)


def _parse_csv(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _common_flags(variant, args):
    profile = VARIANTS[variant]
    return [
        "--ordered-max-frequency", "8",
        "--ordered-active-dim", "2",
        "--ordered-frequency-penalty", "0.1",
        "--ordered-basis-mode", "diagonal_quadratic",
        "--ordered-orthogonal-coordinates",
        "--ordered-cumulative-exposure",
        "--source-observation-mode", "replicated",
        "--source-observation-replicates", "3",
        "--source-design-mode", "universal_mixture",
        "--source-universal-fraction", "1.0",
        "--source-consensus-template-count", "12",
        "--observable-mean-coordinate",
        "--observable-mean-mode", profile["observable_mean_mode"],
        "--observable-mean-latent-dim", "2",
        "--observable-mean-training-target", profile["training_target"],
        "--source-constraint-mean-coefficient-prior",
        "--source-constraint-mean-adaptation-mode", "evidence_mixture",
        "--source-discrepancy-update",
        "--task-posterior-safe-generalized",
        "--task-posterior-safe-boundary-weight", "1.0",
        "--task-posterior-safe-pairwise-weight", "1.0",
        "--task-posterior-safe-pairwise-history", "16",
        "--task-posterior-safe-pairwise-floor", "1e-06",
        "--task-posterior-mandatory-universal-count", "10",
        "--task-posterior-robust-certificate-mode", "joint_tangent",
        "--task-latent-inference-mode", "shadow",
        "--task-latent-calibration-mode", "source_profiles",
        "--hvd-source-task-weight-mode", "constraint_mean",
        "--hvd-cumulative-target-evidence-mode", "prequential_upper",
        "--decision-backend", "sobol_new",
        "--exact-terminal-mode", "bayes_risk",
        "--replication-candidate-count", "0",
        "--no-adaptive-replication-voi",
        "--finalist-replication-budget", "0",
        "--finalist-empirical-override", "off",
        "--finalist-frontier-policy", "legacy",
        "--certification-recheck-top-k", "0",
        "--no-posterior-dominance-enabled",
        "--decision-recommend-observed-only",
        "--boundary-coordinate-candidate-count",
        str(profile["boundary_candidate_count"]),
        "--boundary-coordinate-pool-size",
        str(args.boundary_coordinate_pool_size),
        "--boundary-coordinate-safe-fraction", "0.30",
        "--boundary-coordinate-boundary-fraction", "0.40",
        "--boundary-coordinate-coverage-fraction", "0.30",
        "--truth-pool-diagnostics",
        "--variance-audit-size", "128",
    ]


def build_specs(args):
    nodes = _parse_csv(args.nodes)
    variants = _parse_csv(args.variants)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    if not variants or any(variant not in VARIANTS for variant in variants):
        raise ValueError("unknown boundary-coordinate gate variant")
    if int(args.n0) > int(args.N):
        raise ValueError("n0 must not exceed N")

    local_project = Path(args.deploy) / "SC-OLH-KG"
    remote_project = REMOTE_ROOT / "SC-OLH-KG"
    try:
        relative_manifest = Path(args.manifest).relative_to(local_project)
        remote_manifest = remote_project / relative_manifest
    except ValueError:
        remote_manifest = Path(args.manifest)

    specs = []
    for variant in variants:
        for heldout, shock_scale in SCENARIOS:
            local_design = (
                local_project / "archives" / args.source_run_id / heldout
                / "source_initial_designs.json"
            )
            remote_design = (
                remote_project / "archives" / args.source_run_id / heldout
                / "source_initial_designs.json"
            )
            shock_label = f"{shock_scale:g}".replace(".", "p")
            for offset in range(int(args.n_seeds)):
                seed = int(args.seed_start) + offset
                cell = f"{variant}/shock{shock_label}"
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
                    "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                    "OPENBLAS_NUM_THREADS=1", "PYTHONUNBUFFERED=1",
                    "PYTHONDONTWRITEBYTECODE=1", str(args.python),
                    "performance/run_lodo_manifest_shard.py",
                    "--manifest", str(remote_manifest),
                    "--heldout", heldout,
                    "--line", "lodo",
                    "--seed", str(seed),
                    "--experiment-variant",
                    f"boundary_coordinate_gate/{cell}",
                    "--out", str(remote_result_dir / "result.json"),
                    "--runtime-checkpoint-dir", str(checkpoint_dir),
                    "--runtime-checkpoint-interval", "0",
                    "--evaluate-interval", str(args.evaluate_interval),
                    "--d", str(args.d),
                    "--meta-source-d", str(args.source_d),
                    "--N", str(args.N),
                    "--n0", str(args.n0),
                    "--initial-design", "source_informed",
                    "--initial-design-file", str(remote_design),
                    "--structural-prior-profile", "none",
                    "--hvd-profile", "factor_hierarchical",
                    "--target-shared-shock-scale", str(shock_scale),
                    *_common_flags(variant, args),
                ]
                specs.append({
                    "description": (
                        f"Boundary coordinate gate {variant} {heldout} "
                        f"shock={shock_scale:g} seed={seed}"
                    ),
                    "cmd": f"{shlex.join(command)} && echo DONE",
                    "cwd": str(local_project),
                    "signature": (
                        f"KG_op/boundary_coordinate_gate/{args.run_id}/"
                        f"{cell}/{heldout}/seed{seed}"
                    ),
                    "project": "KG-SYNTH",
                    "vram": 0,
                    "cpu": int(args.cpu),
                    "ram_mb": int(args.ram_mb),
                    "allowed_nodes": list(nodes),
                    "wait_for_files": [str(local_design)],
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
    parser.add_argument("--manifest", type=Path, default=(
        DEFAULT_DEPLOY
        / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"))
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--run-id", default=(
        f"scolh_boundary_coordinate_gate_s5_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--boundary-coordinate-pool-size", type=int, default=512)
    parser.add_argument("--evaluate-interval", type=int, default=20)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=4096)
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
            "submit-jsonl", "--stdin", "--trusted", "--json",
            "--intent-label", str(args.run_id),
        ],
        input=payload,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
