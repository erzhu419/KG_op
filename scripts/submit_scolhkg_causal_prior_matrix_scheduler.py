#!/usr/bin/env python3
"""Submit the corrected proposal/posterior structural-prior matrix."""

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
PROFILES = (
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
CAUSAL_MODES = ("proposal_only", "posterior_only", "joint")


def _parse_csv(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _base_flags():
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
        "--source-discrepancy-update",
    ]


def _archive_paths(local_project, remote_project, source_run_id, heldout):
    relative = Path("archives") / source_run_id / heldout / f"heldout_{heldout}.json"
    return local_project / relative, remote_project / relative


def _design_paths(
    local_project,
    remote_project,
    run_id,
    proposal_mode,
    profile,
    heldout,
):
    relative = (
        Path("archives") / run_id / "proposals" / proposal_mode / profile
        / heldout / "source_initial_designs.json"
    )
    return local_project / relative, remote_project / relative


def build_specs(args):
    nodes = _parse_csv(args.nodes)
    heldouts = _parse_csv(args.heldouts)
    profiles = _parse_csv(args.profiles)
    modes = _parse_csv(args.causal_modes)
    proposal_modes = _parse_csv(args.proposal_modes)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    if not heldouts or any(item not in HELDOUTS for item in heldouts):
        raise ValueError("unknown heldout domain")
    if not profiles or any(item not in PROFILES for item in profiles):
        raise ValueError("unknown structural-prior profile")
    if not modes or any(item not in CAUSAL_MODES for item in modes):
        raise ValueError("unknown causal mode")
    if not proposal_modes or any(
        item not in {"rank_spanning", "risk_coordinate_atlas"}
        for item in proposal_modes
    ):
        raise ValueError("unknown proposal mode")

    local_project = Path(args.deploy) / "SC-OLH-KG"
    remote_project = REMOTE_ROOT / "SC-OLH-KG"
    try:
        manifest_relative = Path(args.manifest).relative_to(local_project)
        remote_manifest = remote_project / manifest_relative
    except ValueError:
        remote_manifest = Path(args.manifest)
    source_run_id = f"{args.run_id}_source_d{int(args.source_d)}"
    specs = []

    for heldout in heldouts:
        local_archive, remote_archive = _archive_paths(
            local_project, remote_project, source_run_id, heldout)
        archive_command = [
            "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
            "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
            f"OMP_NUM_THREADS={int(args.cpu)}",
            f"MKL_NUM_THREADS={int(args.cpu)}",
            str(args.python),
            "performance/materialize_transfer_archives.py",
            "--manifest", str(remote_manifest),
            "--heldouts", heldout,
            "--out-dir", str(remote_archive.parent),
            "--d", str(args.source_d),
        ]
        specs.append({
            "description": f"causal-prior source archive {heldout}",
            "cmd": f"{shlex.join(archive_command)} && echo DONE",
            "cwd": str(local_project),
            "signature": f"KG_op/causal_prior_archive/{args.run_id}/{heldout}",
            "project": "KG-SYNTH",
            "vram": 0,
            "cpu": int(args.cpu),
            "ram_mb": int(args.ram_mb),
            "allowed_nodes": list(nodes),
            "result_dir": str(remote_archive.parent),
            "local_result_dir": str(local_archive.parent),
            "stage_excludes": ["checkpoints", "profiles", "results"],
            "allow_duplicate": True,
        })

        for proposal_mode in proposal_modes:
            for profile in profiles:
                local_design, remote_design = _design_paths(
                    local_project,
                    remote_project,
                    args.run_id,
                    proposal_mode,
                    profile,
                    heldout,
                )
                design_command = [
                    "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                    "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                    f"OMP_NUM_THREADS={int(args.cpu)}",
                    f"MKL_NUM_THREADS={int(args.cpu)}",
                    str(args.python),
                    "performance/materialize_source_initial_designs.py",
                    "--manifest", str(remote_manifest),
                    "--heldout", heldout,
                    "--archive", str(remote_archive),
                    "--out", str(remote_design),
                    "--source-d", str(args.source_d),
                    "--d", str(args.d),
                    "--n0", str(args.n0),
                    "--seed-start", str(args.seed_start),
                    "--n-seeds", str(args.n_seeds),
                    "--structural-prior-profile", profile,
                    "--proposal-mode", proposal_mode,
                ]
                specs.append({
                    "description": (
                        f"causal-prior design {proposal_mode} {profile} {heldout}"
                    ),
                    "cmd": f"{shlex.join(design_command)} && echo DONE",
                    "cwd": str(local_project),
                    "signature": (
                        f"KG_op/causal_prior_design/{args.run_id}/"
                        f"{proposal_mode}/{profile}/{heldout}"
                    ),
                    "project": "KG-SYNTH",
                    "vram": 0,
                    "cpu": int(args.cpu),
                    "ram_mb": int(args.ram_mb),
                    "allowed_nodes": list(nodes),
                    "wait_for_files": [str(local_archive)],
                    "result_dir": str(remote_design.parent),
                    "local_result_dir": str(local_design.parent),
                    "stage_excludes": ["checkpoints", "profiles", "results"],
                    "allow_duplicate": True,
                })

    for mode in modes:
        for proposal_mode in proposal_modes:
            for profile in profiles:
                posterior_profile = profile if mode != "proposal_only" else "none"
                initial_design = (
                    "common_sobol" if mode == "posterior_only"
                    else "source_informed"
                )
                for heldout in heldouts:
                    local_design, remote_design = _design_paths(
                        local_project,
                        remote_project,
                        args.run_id,
                        proposal_mode,
                        profile,
                        heldout,
                    )
                    for offset in range(int(args.n_seeds)):
                        seed = int(args.seed_start) + offset
                        cell = f"{mode}/{proposal_mode}/{profile}"
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
                            "--experiment-variant", f"causal_prior_v2/{cell}",
                            "--out", str(remote_result_dir / "result.json"),
                            "--runtime-checkpoint-dir", str(checkpoint_dir),
                            "--d", str(args.d),
                            "--meta-source-d", str(args.source_d),
                            "--N", str(args.N),
                            "--n0", str(args.n0),
                            "--initial-design", initial_design,
                            "--structural-prior-profile", posterior_profile,
                            "--hvd-profile", "pooled",
                            "--decision-backend", "sobol",
                            *_base_flags(),
                        ]
                        wait_for_files = []
                        if initial_design == "source_informed":
                            command.extend([
                                "--initial-design-file", str(remote_design),
                            ])
                            wait_for_files.append(str(local_design))
                        specs.append({
                            "description": (
                                f"causal-prior {cell} {heldout} seed={seed}"
                            ),
                            "cmd": f"{shlex.join(command)} && echo DONE",
                            "cwd": str(local_project),
                            "signature": (
                                f"KG_op/causal_prior_v2/{args.run_id}/{cell}/"
                                f"{heldout}/seed{seed}"
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
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--causal-modes", default=",".join(CAUSAL_MODES))
    parser.add_argument("--proposal-modes", default="risk_coordinate_atlas")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            DEFAULT_DEPLOY
            / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=f"scolh_causal_prior_v2_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=12)
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
