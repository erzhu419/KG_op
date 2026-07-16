#!/usr/bin/env python3
"""Submit matched-initial-design SC-OLH-KG transfer matrices."""

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


def _parse_csv(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _algorithm_flags(exact_jobs):
    return [
        "--ordered-max-frequency", "8",
        "--spectral-orthogonalization", "symmetric",
        "--ordered-active-dim", "2",
        "--ordered-frequency-penalty", "0.1",
        "--ordered-basis-mode", "diagonal_quadratic",
        "--ordered-orthogonal-coordinates",
        "--exact-sampling-mode", "iid",
        "--exact-terminal-mode", "hard_certified",
        "--exact-mc-samples", "2",
        "--exact-jobs", str(int(exact_jobs)),
        "--parallel-backend", "process_fork",
        "--tcb-v2-mode", "off",
        "--tcb-v2-frontier-count", "1",
        "--tcb-v2-descriptor-mode", "learned_risk",
        "--tcb-v2-coordinate", "boundary_latent",
        "--tcb-v2-geometry", "low_rank_psd",
        "--tcb-v2-rank", "2",
        "--tcb-v2-ridge", "0.001",
        "--tcb-v2-domain-penalty", "0.5",
        "--tcb-v2-boundary-temperature", "1.0",
        "--tcb-v2-adaptation-ridge", "5.0",
        "--tcb-v2-upper-alpha", "0.01",
        "--tcb-v2-calibration-prior-df", "2.0",
        "--tcb-v2-hierarchy-iterations", "5",
        "--tcb-v2-effect-ridge", "1.0",
        "--tcb-v2-rotation-mode", "none",
        "--tcb-v2-rotation-ridge", "5.0",
        "--tcb-v2-target-residual-rank", "0",
        "--tcb-v2-residual-ridge", "5.0",
        "--source-observation-mode", "replicated",
        "--source-observation-replicates", "3",
        "--source-design-mode", "universal_mixture",
        "--source-universal-fraction", "1.0",
        "--source-consensus-template-count", "12",
        "--observable-mean-mode", "latent",
        "--observable-mean-latent-dim", "2",
        "--observable-mean-training-target", "constraint_mean",
        "--finalist-terminal-value-mode", "model_default",
        "--finalist-replication-budget", "3",
        "--finalist-replication-count", "3",
        "--finalist-observed-safety-count", "2",
        "--finalist-replication-min-replicates", "2",
        "--finalist-replication-delta", "0.05",
        "--finalist-replication-variance-prior-df", "2.0",
        "--finalist-replication-policy", "commit_before_switch",
        "--finalist-empirical-override", "legacy",
        "--finalist-frontier-policy", "observed_safety_reserved",
        "--finalist-terminal-max-arms", "4",
        "--finalist-terminal-mc-samples", "2",
        "--decision-contract-mode", "legacy",
        "--task-posterior-safe-boundary-weight", "1.0",
        "--task-posterior-safe-pairwise-weight", "1.0",
        "--task-posterior-safe-pairwise-history", "16",
        "--task-posterior-safe-pairwise-floor", "1e-06",
        "--task-latent-inference-mode", "shadow",
        "--task-latent-calibration-mode", "source_profiles",
        "--finalist-replication-expert-stratified",
        "--finalist-replication-adaptive-race",
        "--finalist-replication-fixed-universe",
        "--ordered-cumulative-exposure",
        "--ordered-adaptive-sparsity",
        "--ordered-replace-local-kernel",
        "--ordered-latent-structure-selection",
        "--ordered-group-ridge-learning",
        "--task-posterior-safe-generalized",
        "--no-observable-mean-coordinate",
        "--task-posterior-mandatory-universal-count", "10",
    ]


def build_specs(args):
    nodes = _parse_csv(args.nodes)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    heldouts = _parse_csv(args.heldouts)
    if not heldouts or any(heldout not in HELDOUTS for heldout in heldouts):
        raise ValueError("heldouts must use the registered three-domain matrix")

    local_project = Path(args.deploy) / "SC-OLH-KG"
    remote_project = REMOTE_ROOT / "SC-OLH-KG"
    try:
        manifest_relative = Path(args.manifest).relative_to(local_project)
        remote_manifest = remote_project / manifest_relative
    except ValueError:
        remote_manifest = Path(args.manifest)
    specs = []
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
            remote_result_dir = (
                remote_project / "profiles" / args.run_id / heldout
                / f"seed{seed}"
            )
            local_result_dir = (
                local_project / "profiles" / args.run_id / heldout
                / f"seed{seed}"
            )
            checkpoint_dir = (
                remote_project / "checkpoints" / args.run_id / heldout
                / f"seed{seed}"
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
                    "oracle_free_source_informed_matched"
                    if args.initial_design == "source_informed"
                    else "oracle_free_common_sobol_matched"
                ),
                "--out", str(remote_result_dir / "result.json"),
                "--runtime-checkpoint-dir", str(checkpoint_dir),
                "--d", str(args.d),
                "--N", str(args.N),
                "--n0", str(args.n0),
                "--initial-design", str(args.initial_design),
                *_algorithm_flags(args.exact_jobs),
            ]
            if args.initial_design == "source_informed":
                command.extend([
                    "--initial-design-file", str(remote_design),
                ])
            specs.append({
                "description": (
                    f"SC-OLH-KG {args.run_id} {heldout} seed={seed}"
                ),
                "cmd": f"{shlex.join(command)} && echo DONE",
                "cwd": str(local_project),
                "signature": (
                    f"KG_op/scolhkg_manifest/{args.run_id}/{heldout}/seed{seed}"
                ),
                "project": "KG-SYNTH",
                "vram": 0,
                "cpu": int(args.cpu),
                "ram_mb": int(args.ram_mb),
                "allowed_nodes": list(nodes),
                "wait_for_files": (
                    [str(local_design)]
                    if args.initial_design == "source_informed" else []
                ),
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
    parser.add_argument(
        "--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--python", type=Path, default=REMOTE_PYTHON)
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--heldouts", default=",".join(HELDOUTS))
    parser.add_argument(
        "--manifest", type=Path,
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
        default="lodo_oracle_free_source_informed_n20_s20_c12_20260716",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument(
        "--initial-design",
        choices=("common_sobol", "source_informed"),
        default="source_informed",
    )
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
