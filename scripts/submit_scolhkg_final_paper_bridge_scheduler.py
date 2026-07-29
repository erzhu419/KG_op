#!/usr/bin/env python3
"""Submit the final frozen-front-end bridge to the legacy RZDT families."""

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
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python")
SAAS_PYTHON = Path(
    "/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python")
BOTORCH_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/botorch_overlay_py310")
CPU_NODES = tuple(f"node{i:03d}" for i in range(1, 7))
GPU_NODES = ("jtl110gpu", "jtl110gpu2", "node007")
TARGETS = ("PaperRZDT1", "PaperRZDT2", "PaperRZDT5_RR")
SNAPSHOT_MARKER = ".scolhkg_execution_snapshot.json"
METHOD_CONTRACT_ID = "or_transfer_frontend_saas_v1"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _execution_snapshot(args):
    code_root = getattr(args, "code_root", None)
    required = bool(getattr(args, "require_frozen_snapshot", False))
    if code_root in (None, ""):
        if required:
            raise ValueError(
                "--require-frozen-snapshot needs --code-root")
        return None
    marker = Path(code_root) / SNAPSHOT_MARKER
    if not marker.is_file():
        raise FileNotFoundError(
            f"frozen code snapshot marker is missing: {marker}")
    snapshot = _read_json(marker)
    required_fields = (
        "repository_commit",
        "scolhkg_tree",
        "proof_tree",
        "scripts_tree",
        "theory_contract_id",
        "snapshot_root",
    )
    missing = [
        field for field in required_fields if not snapshot.get(field)
    ]
    if missing:
        raise ValueError(
            f"frozen code snapshot is incomplete: {missing}")
    if Path(snapshot["snapshot_root"]).resolve() != Path(code_root).resolve():
        raise ValueError("snapshot marker/root mismatch")
    if snapshot.get("status") != "frozen":
        raise ValueError("code snapshot is not frozen")
    return snapshot


def _execution_env(snapshot):
    if snapshot is None:
        return []
    values = {
        "SCOLHKG_EXECUTION_PROVENANCE_REQUIRED": "1",
        "SCOLHKG_EXECUTION_COMMIT": snapshot["repository_commit"],
        "SCOLHKG_SCOLHKG_TREE": snapshot["scolhkg_tree"],
        "SCOLHKG_PROOF_TREE": snapshot["proof_tree"],
        "SCOLHKG_SCRIPTS_TREE": snapshot["scripts_tree"],
        "SCOLHKG_METHOD_CONTRACT_ID": METHOD_CONTRACT_ID,
        "SCOLHKG_THEORY_CONTRACT_ID": snapshot["theory_contract_id"],
        "SCOLHKG_CODE_SNAPSHOT_ROOT": snapshot["snapshot_root"],
    }
    return [f"{key}={value}" for key, value in values.items()]


def _terminal_flags():
    return [
        "--terminal-verification",
        "--terminal-verification-primary-budget", "80",
        "--terminal-verification-support-budget", "128",
        "--terminal-verification-candidate-budgets", "80,128,128",
        "--terminal-verification-delta", "0.05",
        "--terminal-verification-method", "normal_quantile_tolerance",
        "--terminal-verification-shortlist-mode",
        "posterior_objective_challenger_then_safe",
        "--terminal-verification-shortlist-size", "3",
        "--terminal-objective-challenger-max-violation-probability", "0.5",
        "--terminal-objective-incumbent-guard",
        "--terminal-objective-comparison-budget", "8",
        "--terminal-objective-comparison-delta", str(0.05 / 3.0),
        "--terminal-safe-interior-probability-slack", "0.05",
        "--no-terminal-safe-interior-require-provider",
    ]


def build_specs(args):
    deploy = Path(args.deploy)
    deploy_project = deploy / "SC-OLH-KG"
    code_project = (
        Path(args.code_root) / "SC-OLH-KG"
        if getattr(args, "code_root", None)
        else deploy_project
    )
    execution_snapshot = _execution_snapshot(args)
    manifest = (
        code_project / "performance/manifests/v18b_exactkg_mcdiag.json")
    archive = (
        deploy_project / "archives" / args.archive_run_id
        / "QueueResourceControl"
        / "heldout_QueueResourceControl.json"
    )
    design_root = (
        deploy_project / "archives" / args.run_id / "paper_bridge")
    design_cmd = [
        "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
        "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
        f"OMP_NUM_THREADS={int(args.cpu)}",
        f"MKL_NUM_THREADS={int(args.cpu)}",
        f"OPENBLAS_NUM_THREADS={int(args.cpu)}",
        *_execution_env(execution_snapshot),
        str(REMOTE_PYTHON),
        str(
            code_project
            / "performance/materialize_external_paper_bridge_designs.py"),
        "--manifest", str(manifest),
        "--archive", str(archive),
        "--out-dir", str(design_root),
        "--source-d", str(args.source_d),
        "--target-d", str(args.d),
        "--n0", str(args.n0),
        "--seed-start", str(args.seed_start),
        "--n-seeds", str(args.n_seeds),
    ]
    specs = [{
        "description": "paper final legacy RZDT bridge proposals",
        "cmd": f"{shlex.join(design_cmd)} && echo DONE",
        "cwd": str(code_project),
        "signature": f"KG_op/final_paper_bridge/{args.run_id}/design",
        "project": "KG-SYNTH",
        "vram": 0,
        "cpu": int(args.cpu),
        "ram_mb": int(args.ram_mb),
        "allowed_nodes": list(CPU_NODES),
        "wait_for_files": [str(archive)],
        "result_dir": str(design_root),
        "local_result_dir": str(design_root),
        "stage_excludes": ["checkpoints", "profiles", "results"],
        "allow_duplicate": True,
    }]
    for target in TARGETS:
        design = design_root / target / "source_initial_designs.json"
        for seed in range(
            int(args.seed_start),
            int(args.seed_start) + int(args.n_seeds),
        ):
            result_dir = (
                deploy_project / "profiles" / args.run_id / target
                / f"seed{seed:04d}"
            )
            checkpoint_dir = (
                deploy_project / "checkpoints" / args.run_id / target
                / f"seed{seed:04d}"
            )
            command = [
                "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                f"OMP_NUM_THREADS={int(args.gpu_cpu)}",
                f"MKL_NUM_THREADS={int(args.gpu_cpu)}",
                f"OPENBLAS_NUM_THREADS={int(args.gpu_cpu)}",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "SCOLHKG_TORCH_DETERMINISTIC=1",
                "CUBLAS_WORKSPACE_CONFIG=:4096:8",
                *_execution_env(execution_snapshot),
                f"PYTHONPATH={BOTORCH_OVERLAY}",
                str(SAAS_PYTHON),
                str(
                    code_project
                    / "performance/benchmark_sota_fairness.py"),
                "--protocol", "shared_archive_n13",
                "--method", "botorch_saasbo",
                "--heldout", target,
                "--seed", str(seed),
                "--manifest", str(manifest),
                "--out", str(result_dir / "result.json"),
                "--checkpoint-dir", str(checkpoint_dir),
                "--initial-design-file", str(design),
                "--offline-source-calls-override", "384",
                "--target-budget", str(args.N),
                "--d", str(args.d),
                "--n0", str(args.n0),
                "--candidate-timeout-sec", "3600",
                "--torch-device", "cuda",
                "--torch-deterministic",
                "--saas-refit-schedule", "every_iteration",
                *_terminal_flags(),
            ]
            specs.append({
                "description": (
                    f"paper final legacy bridge {target} seed={seed}"),
                "cmd": f"{shlex.join(command)} && echo DONE",
                "cwd": str(code_project),
                "signature": (
                    f"KG_op/final_paper_bridge/{args.run_id}/"
                    f"{target}/seed{seed:04d}"
                ),
                "project": "KG-SYNTH",
                "vram": int(args.vram_mb),
                "cpu": int(args.gpu_cpu),
                "ram_mb": int(args.gpu_ram_mb),
                "allowed_nodes": list(GPU_NODES),
                "wait_for_files": [str(design)],
                "result_dir": str(result_dir),
                "local_result_dir": str(result_dir),
                "stage_excludes": ["checkpoints", "profiles", "results"],
                "allow_duplicate": True,
                "vram_resource_family": (
                    f"KG-SYNTH/final-paper-bridge/{target}"),
            })
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument(
        "--require-frozen-snapshot",
        action="store_true",
    )
    parser.add_argument("--run-id", default=(
        "paper_final_legacy_bridge_d5_n13_s80_99_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument(
        "--archive-run-id",
        default="transfer_source_informed_official_n20_s20_20260716",
    )
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=16384)
    parser.add_argument("--gpu-cpu", type=int, default=12)
    parser.add_argument("--gpu-ram-mb", type=int, default=24576)
    parser.add_argument("--vram-mb", type=int, default=2048)
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = build_specs(args)
    expected = 1 + len(TARGETS) * int(args.n_seeds)
    if len(specs) != expected:
        raise RuntimeError(f"built {len(specs)} tasks, expected {expected}")
    if len({spec["signature"] for spec in specs}) != len(specs):
        raise RuntimeError("paper bridge signatures are not unique")
    if args.dry_run:
        print(json.dumps({
            "run_id": args.run_id,
            "task_count": len(specs),
            "specs": specs,
        }, indent=2))
        return
    if args.sync_remote:
        if args.require_frozen_snapshot and args.code_root is None:
            raise ValueError(
                "--require-frozen-snapshot needs --code-root")
        if args.code_root is None:
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
            f"paper-final-legacy-bridge-{args.run_id}",
        ],
        input=json.dumps(specs),
        text=True,
    )
    response = json.loads(output)
    task_ids = [
        row["id"] for row in response.get("submitted", [])
        if row.get("id")
    ]
    registration = {
        "schema_version": 1,
        "run_id": args.run_id,
        "task_count": len(specs),
        "task_ids": task_ids,
        "contract": {
            "source_calls": 384,
            "source_domains": list((
                "FactorShockStatePolicyRZDT1",
                "InventorySupplyChain",
            )),
            "target_domains": list(TARGETS),
            "target_search_calls": int(args.N),
            "backend": "canonical_saasbo_every_iteration",
            "verifier": "v69_no_provider_legacy_bridge",
            "cumulative_risk_provider_required": False,
            "target_oracle_used_for_selection": False,
            "metric_bridge": (
                "new scalarized regret; legacy HV/IGD/CVR context only"),
            "direct_metric_equality_claim_allowed": False,
            "execution_snapshot": (
                _execution_snapshot(args)
                if args.code_root is not None
                else {"status": "unregistered"}
            ),
        },
        "checkpoint_results_synced_locally": False,
    }
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
