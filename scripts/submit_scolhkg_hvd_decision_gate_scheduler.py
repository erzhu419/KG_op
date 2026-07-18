#!/usr/bin/env python3
"""Submit the frozen-proposal cumulative-HVD causal closure gate."""

from __future__ import annotations

import argparse
import itertools
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
HVD_PROFILES = ("pooled", "factor_cumulative", "factor_hierarchical")
DISCREPANCY_PROFILES = ("frozen", "adaptive")
ACTION_PROFILES = (
    "new_only", "hvd_voi", "joint_voi",
    "certificate_depth_new", "certificate_depth_search",
)
DEFAULT_ACTION_PROFILES = ("new_only", "hvd_voi")
CERTIFICATE_MODES = ("separable", "joint_tangent")
HVD_SOURCE_TASK_WEIGHT_MODES = ("independent", "constraint_mean")
HVD_TARGET_EVIDENCE_MODES = ("replication_only", "prequential_upper")
MEAN_PROFILES = (
    "legacy",
    "eta_empirical",
    "eta_source_prior",
    "eta_source_adaptive",
    "eta_source_sequential",
)
OBSERVABLE_MEAN_MODES = (
    "latent", "consensus", "source_affine", "source_rank")
DEFAULT_SOURCE_RUN_ID = (
    "scolh_lowfreq_support_dimholdout_d1000_s5_20260716/"
    "proposals/risk_objective_atlas/low_frequency_only"
)


def _parse_csv(value, cast=str):
    return tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())


def _base_flags(observable_mean_mode="latent"):
    observable_mean_mode = str(observable_mean_mode).strip().lower()
    if observable_mean_mode not in OBSERVABLE_MEAN_MODES:
        raise ValueError(
            "observable mean mode must be latent, consensus, source_affine, "
            "or source_rank")
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
        "--observable-mean-mode", observable_mean_mode,
        "--observable-mean-latent-dim", (
            "0"
            if observable_mean_mode in {"consensus", "source_rank"}
            else "2"
        ),
        "--observable-mean-training-target", "constraint_mean",
        "--task-posterior-safe-generalized",
        "--task-posterior-safe-boundary-weight", "1.0",
        "--task-posterior-safe-pairwise-weight", "1.0",
        "--task-posterior-safe-pairwise-history", "16",
        "--task-posterior-safe-pairwise-floor", "1e-06",
        "--task-posterior-mandatory-universal-count", "10",
        "--task-latent-inference-mode", "shadow",
        "--task-latent-calibration-mode", "source_profiles",
        "--exact-terminal-mode", "bayes_risk",
        "--finalist-replication-budget", "0",
        "--finalist-empirical-override", "off",
        "--finalist-frontier-policy", "legacy",
        "--certification-recheck-top-k", "0",
        "--no-posterior-dominance-enabled",
        "--decision-recommend-observed-only",
        "--variance-audit-size", "128",
    ]


def build_specs(args):
    nodes = _parse_csv(args.nodes)
    heldouts = _parse_csv(args.heldouts)
    hvd_profiles = _parse_csv(args.hvd_profiles)
    discrepancy_profiles = _parse_csv(args.discrepancy_profiles)
    action_profiles = _parse_csv(args.action_profiles)
    certificate_modes = _parse_csv(
        getattr(args, "certificate_modes", "separable"))
    mean_profiles = _parse_csv(getattr(args, "mean_profiles", "legacy"))
    observable_mean_mode = str(getattr(
        args, "observable_mean_mode", "latent")).strip().lower()
    hvd_source_task_weight_modes = _parse_csv(getattr(
        args, "hvd_source_task_weight_modes", "independent"))
    hvd_target_evidence_modes = _parse_csv(getattr(
        args, "hvd_target_evidence_modes", "replication_only"))
    shock_scales = _parse_csv(args.shock_scales, float)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a nonempty subset of node001-node006")
    if not heldouts or any(item not in HELDOUTS for item in heldouts):
        raise ValueError("unknown heldout domain")
    if not hvd_profiles or any(item not in HVD_PROFILES for item in hvd_profiles):
        raise ValueError("unknown HVD profile")
    if not discrepancy_profiles or any(
        item not in DISCREPANCY_PROFILES for item in discrepancy_profiles
    ):
        raise ValueError("unknown discrepancy profile")
    if not action_profiles or any(
        item not in ACTION_PROFILES for item in action_profiles
    ):
        raise ValueError("unknown action profile")
    if not certificate_modes or any(
        item not in CERTIFICATE_MODES for item in certificate_modes
    ):
        raise ValueError("unknown task-posterior certificate mode")
    if not mean_profiles or any(item not in MEAN_PROFILES for item in mean_profiles):
        raise ValueError("unknown constraint-mean profile")
    if observable_mean_mode not in OBSERVABLE_MEAN_MODES:
        raise ValueError("unknown observable constraint-mean mode")
    if not hvd_source_task_weight_modes or any(
        item not in HVD_SOURCE_TASK_WEIGHT_MODES
        for item in hvd_source_task_weight_modes
    ):
        raise ValueError("unknown HVD source-task weight mode")
    if not hvd_target_evidence_modes or any(
        item not in HVD_TARGET_EVIDENCE_MODES
        for item in hvd_target_evidence_modes
    ):
        raise ValueError("unknown HVD target-evidence mode")
    if (
        "constraint_mean" in hvd_source_task_weight_modes
        and any(profile not in {
            "eta_source_adaptive", "eta_source_sequential",
        } for profile in mean_profiles)
    ):
        raise ValueError(
            "constraint-mean HVD weighting requires adaptive or sequential "
            "source-mean profiles"
        )
    if not shock_scales or any(value < 0.0 for value in shock_scales):
        raise ValueError("shock scales must be nonnegative")

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
        domain_shocks = (
            shock_scales
            if heldout == "FactorShockStatePolicyRZDT1"
            else (1.0,)
        )
        for shock_scale in domain_shocks:
            shock_label = f"{shock_scale:g}".replace(".", "p")
            for hvd_profile in hvd_profiles:
                for discrepancy in discrepancy_profiles:
                    discrepancy_enabled = discrepancy == "adaptive"
                    for action_profile in action_profiles:
                        adaptive_replication = action_profile in {
                            "hvd_voi", "joint_voi",
                        }
                        backend = {
                            "new_only": "sobol_new",
                            "hvd_voi": "sobol_hvd_voi",
                            "joint_voi": "sobol_joint_voi",
                            "certificate_depth_new": "certificate_depth_new",
                            "certificate_depth_search": "certificate_depth_new",
                        }[action_profile]
                        safe_interior_count = (
                            int(args.safe_interior_candidate_count)
                            if action_profile == "certificate_depth_search"
                            else 0
                        )
                        for (
                            mean_profile,
                            certificate_mode,
                            source_task_weight_mode,
                            target_evidence_mode,
                            offset,
                        ) in itertools.product(
                            mean_profiles,
                            certificate_modes,
                            hvd_source_task_weight_modes,
                            hvd_target_evidence_modes,
                            range(int(args.n_seeds)),
                        ):
                            seed = int(args.seed_start) + offset
                            base_cell = (
                                f"{hvd_profile}/{discrepancy}/{action_profile}/"
                                f"{certificate_mode}/shock{shock_label}"
                            )
                            cell = (
                                base_cell
                                if mean_profiles == ("legacy",)
                                else f"mean_{mean_profile}/{base_cell}"
                            )
                            if hvd_source_task_weight_modes != ("independent",):
                                cell = (
                                    f"hvd_task_{source_task_weight_mode}/{cell}"
                                )
                            if hvd_target_evidence_modes != ("replication_only",):
                                cell = (
                                    f"hvd_evidence_{target_evidence_mode}/{cell}"
                                )
                            if mean_profile != "legacy":
                                cell = (
                                    f"mean_coord_{observable_mean_mode}/{cell}"
                                )
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
                                "--experiment-variant", (
                                    f"hvd_decision_gate/{cell}"
                                ),
                                "--out", str(remote_result_dir / "result.json"),
                                "--runtime-checkpoint-dir", str(checkpoint_dir),
                                "--runtime-checkpoint-interval",
                                str(getattr(
                                    args, "runtime_checkpoint_interval", 0)),
                                "--evaluate-interval",
                                str(getattr(args, "evaluate_interval", 20)),
                                "--d", str(args.d),
                                "--meta-source-d", str(args.source_d),
                                "--N", str(args.N),
                                "--n0", str(args.n0),
                                "--initial-design", "source_informed",
                                "--initial-design-file", str(remote_design),
                                "--structural-prior-profile", "none",
                                "--hvd-profile", hvd_profile,
                                "--decision-backend", backend,
                                "--task-posterior-robust-certificate-mode",
                                certificate_mode,
                                "--target-shared-shock-scale", str(shock_scale),
                                "--hvd-source-task-weight-mode",
                                source_task_weight_mode,
                                "--hvd-cumulative-target-evidence-mode",
                                target_evidence_mode,
                                "--replication-candidate-count", (
                                    str(args.replication_candidate_count)
                                    if adaptive_replication else "0"
                                ),
                                "--replication-max-per-solution",
                                str(args.replication_max_per_solution),
                                "--safe-interior-candidate-count",
                                str(safe_interior_count),
                                "--safe-interior-pool-size",
                                str(args.safe_interior_pool_size),
                                "--safe-interior-margin",
                                str(args.safe_interior_margin),
                                *_base_flags(observable_mean_mode),
                            ]
                            if mean_profile == "legacy":
                                command.extend([
                                    "--no-observable-mean-coordinate",
                                    "--no-source-constraint-mean-coefficient-prior",
                                    "--source-constraint-mean-adaptation-mode",
                                    "frozen",
                                ])
                            elif mean_profile == "eta_empirical":
                                command.extend([
                                    "--observable-mean-coordinate",
                                    "--no-source-constraint-mean-coefficient-prior",
                                    "--source-constraint-mean-adaptation-mode",
                                    "frozen",
                                ])
                            elif mean_profile == "eta_source_prior":
                                command.extend([
                                    "--observable-mean-coordinate",
                                    "--source-constraint-mean-coefficient-prior",
                                    "--source-constraint-mean-adaptation-mode",
                                    "frozen",
                                ])
                            elif mean_profile == "eta_source_adaptive":
                                command.extend([
                                    "--observable-mean-coordinate",
                                    "--source-constraint-mean-coefficient-prior",
                                    "--source-constraint-mean-adaptation-mode",
                                    "evidence_mixture",
                                ])
                            elif mean_profile == "eta_source_sequential":
                                command.extend([
                                    "--observable-mean-coordinate",
                                    "--source-constraint-mean-coefficient-prior",
                                    "--source-constraint-mean-adaptation-mode",
                                    "sequential_evidence_mixture",
                                ])
                            else:
                                raise ValueError(
                                    f"unknown source mean profile {mean_profile!r}"
                                )
                            command.append(
                                "--adaptive-replication-voi"
                                if adaptive_replication
                                else "--no-adaptive-replication-voi"
                            )
                            command.append(
                                "--source-discrepancy-update"
                                if discrepancy_enabled
                                else "--no-source-discrepancy-update"
                            )
                            specs.append({
                                "description": (
                                    f"HVD decision gate {cell} {heldout} seed={seed}"
                                ),
                                "cmd": f"{shlex.join(command)} && echo DONE",
                                "cwd": str(local_project),
                                "signature": (
                                    f"KG_op/hvd_decision_gate/{args.run_id}/{cell}/"
                                    f"{heldout}/seed{seed}"
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
                                "stage_excludes": [
                                    "checkpoints", "profiles", "results"
                                ],
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
    parser.add_argument("--hvd-profiles", default=",".join(HVD_PROFILES))
    parser.add_argument(
        "--discrepancy-profiles", default=",".join(DISCREPANCY_PROFILES))
    parser.add_argument(
        "--action-profiles", default=",".join(DEFAULT_ACTION_PROFILES))
    parser.add_argument("--certificate-modes", default="separable")
    parser.add_argument("--mean-profiles", default="legacy")
    parser.add_argument(
        "--observable-mean-mode",
        choices=OBSERVABLE_MEAN_MODES,
        default="latent",
    )
    parser.add_argument(
        "--hvd-source-task-weight-modes", default="independent")
    parser.add_argument(
        "--hvd-target-evidence-modes", default="replication_only")
    parser.add_argument("--shock-scales", default="0,0.25,1,4")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            DEFAULT_DEPLOY
            / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"
        ),
    )
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument(
        "--run-id",
        default=f"scolh_hvd_decision_gate_s5_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--replication-candidate-count", type=int, default=10)
    parser.add_argument("--replication-max-per-solution", type=int, default=8)
    parser.add_argument("--safe-interior-candidate-count", type=int, default=12)
    parser.add_argument("--safe-interior-pool-size", type=int, default=512)
    parser.add_argument("--safe-interior-margin", type=float, default=0.0)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument(
        "--runtime-checkpoint-interval",
        type=int,
        default=0,
        help="Checkpoint every N stages; 0 disables runtime checkpoints.",
    )
    parser.add_argument("--evaluate-interval", type=int, default=20)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n0 > args.N:
        raise ValueError("n0 must not exceed N")
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
