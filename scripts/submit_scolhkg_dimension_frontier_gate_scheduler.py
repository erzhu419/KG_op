#!/usr/bin/env python3
"""Submit d=200/d=10000 frontier gates for the frozen proposal and SAAS."""

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
GPU_NODES = ("jtl110gpu", "jtl110gpu2", "jtl311linux", "node007")
DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
BACKENDS = ("proposal_only", "saasbo")
SNAPSHOT_MARKER = ".scolhkg_execution_snapshot.json"
METHOD_CONTRACT_ID = "or_transfer_frontend_saas_v1"


def _parse_csv(value):
    return tuple(
        item.strip() for item in str(value).split(",") if item.strip())


def _target_budgets(args):
    configured = getattr(args, "budgets", "")
    values = (
        tuple(int(value) for value in _parse_csv(configured))
        if str(configured).strip()
        else (int(args.N),)
    )
    if not values or any(value <= int(args.n0) for value in values):
        raise ValueError("every target budget must exceed n0")
    if len(values) != len(set(values)):
        raise ValueError("target budgets must be unique")
    return values


def _saas_route(args, dimension):
    mode = str(getattr(args, "saas_device", "auto")).strip().lower()
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError("saas-device must be auto, cpu, or cuda")
    if mode == "auto":
        mode = (
            "cpu"
            if int(dimension) <= int(getattr(args, "saas_cpu_max_d", 10000))
            else "cuda"
        )
    if mode == "cpu":
        return {
            "device": "cpu",
            "python": REMOTE_PYTHON,
            "cpu": int(args.cpu),
            "ram_mb": int(args.ram_mb),
            "vram_mb": 0,
            "nodes": CPU_NODES,
        }
    return {
        "device": "cuda",
        "python": SAAS_PYTHON,
        "cpu": int(args.gpu_cpu),
        "ram_mb": int(args.gpu_ram_mb),
        "vram_mb": int(args.vram_mb),
        "nodes": GPU_NODES,
    }


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
    ]


def validate_archives(args):
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    audit = {"domains": {}}
    for heldout in _parse_csv(args.heldouts):
        archive_path = (
            deploy_project / "archives" / args.archive_run_id / heldout
            / f"heldout_{heldout}.json"
        )
        archive = _read_json(archive_path)
        dimensions = {
            len(task["X"][0]) for task in archive["tasks"]
        }
        if dimensions != {int(args.source_d)}:
            raise ValueError(f"{heldout} archive dimension changed")
        calls = sum(
            sum(len(row) for row in task["Y_replicates"])
            for task in archive["tasks"]
        )
        if int(calls) != int(args.offline_source_calls):
            raise ValueError(f"{heldout} source cost changed")
        audit["domains"][heldout] = {
            "archive_fingerprint": str(archive["fingerprint"]),
            "source_dimension": int(args.source_d),
            "source_calls": int(calls),
        }
    return audit


def build_specs(args):
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    code_project = (
        Path(args.code_root) / "SC-OLH-KG"
        if getattr(args, "code_root", None)
        else deploy_project
    )
    execution_snapshot = _execution_snapshot(args)
    manifest = (
        code_project / "performance/manifests/v18b_exactkg_mcdiag.json")
    dimensions = tuple(int(value) for value in _parse_csv(args.dimensions))
    target_budgets = _target_budgets(args)
    backends = _parse_csv(args.backends)
    if sorted(set(backends) - set(BACKENDS)):
        raise ValueError("unknown frontier backend")
    specs = []
    for dimension in dimensions:
        for heldout in _parse_csv(args.heldouts):
            archive = (
                deploy_project / "archives" / args.archive_run_id / heldout
                / f"heldout_{heldout}.json"
            )
            design = (
                deploy_project / "archives" / args.run_id
                / f"d{dimension}" / heldout / "source_initial_designs.json"
            )
            design_command = [
                "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                f"OMP_NUM_THREADS={int(args.cpu)}",
                f"MKL_NUM_THREADS={int(args.cpu)}",
                f"OPENBLAS_NUM_THREADS={int(args.cpu)}",
                *_execution_env(execution_snapshot),
                str(REMOTE_PYTHON),
                str(
                    code_project
                    / "performance/materialize_source_initial_designs.py"),
                "--manifest", str(manifest),
                "--heldout", heldout,
                "--archive", str(archive),
                "--out", str(design),
                "--d", str(dimension),
                "--source-d", str(args.source_d),
                "--n0", str(args.n0),
                "--seed-start", str(args.seed_start),
                "--n-seeds", str(args.n_seeds),
                "--structural-prior-profile", "low_frequency_only",
                "--proposal-mode", "risk_objective_atlas",
                "--source-design-mode", "universal_mixture",
            ]
            specs.append({
                "description": (
                    f"frontier design d={dimension} {heldout}"),
                "cmd": f"{shlex.join(design_command)} && echo DONE",
                "cwd": str(code_project),
                "signature": (
                    f"KG_op/dimension_frontier/{args.run_id}/design/"
                    f"d{dimension}/{heldout}"
                ),
                "project": "KG-SYNTH",
                "vram": 0,
                "cpu": int(args.cpu),
                "ram_mb": int(args.ram_mb),
                "allowed_nodes": list(CPU_NODES),
                "wait_for_files": [str(archive)],
                "result_dir": str(design.parent),
                "local_result_dir": str(design.parent),
                "stage_excludes": [
                    "checkpoints", "profiles", "results"],
                "allow_duplicate": True,
            })
            for seed in range(
                int(args.seed_start),
                int(args.seed_start) + int(args.n_seeds),
            ):
                if "proposal_only" in backends:
                    result_dir = (
                        deploy_project / "profiles" / args.run_id
                        / f"d{dimension}" / "N10" / "proposal_only"
                        / heldout / f"seed{seed:04d}"
                    )
                    command = [
                        "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                        "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                        "OPENBLAS_NUM_THREADS=1",
                        *_execution_env(execution_snapshot),
                        str(REMOTE_PYTHON),
                        str(
                            code_project
                            / "performance/benchmark_frozen_proposal_only.py"),
                        "--heldout", heldout,
                        "--seed", str(seed),
                        "--initial-design", "source_informed",
                        "--initial-design-file", str(design),
                        "--out", str(result_dir / "result.json"),
                        "--source-d", str(args.source_d),
                        "--d", str(dimension),
                        "--n0", str(args.n0),
                        "--offline-source-calls",
                        str(args.offline_source_calls),
                        *_terminal_flags(),
                    ]
                    specs.append({
                        "description": (
                            f"frontier proposal d={dimension} {heldout} "
                            f"seed={seed}"
                        ),
                        "cmd": f"{shlex.join(command)} && echo DONE",
                        "cwd": str(code_project),
                        "signature": (
                            f"KG_op/dimension_frontier/{args.run_id}/"
                            f"d{dimension}/proposal/{heldout}/seed{seed:04d}"
                        ),
                        "project": "KG-SYNTH",
                        "vram": 0,
                        "cpu": 1,
                        "ram_mb": 4096,
                        "allowed_nodes": list(CPU_NODES),
                        "wait_for_files": [str(design)],
                        "result_dir": str(result_dir),
                        "local_result_dir": str(result_dir),
                        "stage_excludes": [
                            "checkpoints", "profiles", "results"],
                        "allow_duplicate": True,
                    })
                if "saasbo" in backends:
                    saas_route = _saas_route(args, dimension)
                    for target_budget in target_budgets:
                        result_dir = (
                            deploy_project / "profiles" / args.run_id
                            / f"d{dimension}" / f"N{target_budget}"
                            / "saasbo" / heldout / f"seed{seed:04d}"
                        )
                        checkpoint_dir = (
                            deploy_project / "checkpoints" / args.run_id
                            / f"d{dimension}" / f"N{target_budget}"
                            / "saasbo" / heldout / f"seed{seed:04d}"
                        )
                        command = [
                            "env", "LC_ALL=C", "LANG=C",
                            "SCOLHKG_OFFLINE=1",
                            "PYTHONUNBUFFERED=1",
                            "PYTHONDONTWRITEBYTECODE=1",
                            f"OMP_NUM_THREADS={saas_route['cpu']}",
                            f"MKL_NUM_THREADS={saas_route['cpu']}",
                            f"OPENBLAS_NUM_THREADS={saas_route['cpu']}",
                            "SCOLHKG_TORCH_DETERMINISTIC=1",
                            *_execution_env(execution_snapshot),
                            *(
                                [
                                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                                    "CUBLAS_WORKSPACE_CONFIG=:4096:8",
                                ]
                                if saas_route["device"] == "cuda"
                                else []
                            ),
                            f"PYTHONPATH={BOTORCH_OVERLAY}",
                            str(saas_route["python"]),
                            str(
                                code_project
                                / "performance/benchmark_sota_fairness.py"),
                            "--protocol", "shared_archive_n13",
                            "--method", "botorch_saasbo",
                            "--heldout", heldout,
                            "--seed", str(seed),
                            "--manifest", str(manifest),
                            "--out", str(result_dir / "result.json"),
                            "--checkpoint-dir", str(checkpoint_dir),
                            "--initial-design-file", str(design),
                            "--target-budget", str(target_budget),
                            "--d", str(dimension),
                            "--n0", str(args.n0),
                            "--candidate-timeout-sec", "3600",
                            "--torch-device", saas_route["device"],
                            "--torch-deterministic",
                            "--saas-refit-schedule", "every_iteration",
                            "--terminal-verification",
                            *_terminal_flags(),
                        ]
                        spec = {
                            "description": (
                                f"frontier SAAS d={dimension} "
                                f"N={target_budget} {heldout} seed={seed}"
                            ),
                            "cmd": f"{shlex.join(command)} && echo DONE",
                            "cwd": str(code_project),
                            "signature": (
                                f"KG_op/dimension_frontier/{args.run_id}/"
                                f"d{dimension}/N{target_budget}/saas/"
                                f"{heldout}/seed{seed:04d}"
                            ),
                            "project": "KG-SYNTH",
                            "vram": saas_route["vram_mb"],
                            "cpu": saas_route["cpu"],
                            "ram_mb": saas_route["ram_mb"],
                            "allowed_nodes": list(saas_route["nodes"]),
                            "wait_for_files": [str(design)],
                            "result_dir": str(result_dir),
                            "local_result_dir": str(result_dir),
                            "stage_excludes": [
                                "checkpoints", "profiles", "results"],
                            "allow_duplicate": True,
                            "vram_resource_family": (
                                f"KG-SYNTH/frontier-saas/d{dimension}/"
                                f"{heldout}"
                                if saas_route["device"] == "cuda"
                                else None
                            ),
                        }
                        if saas_route["device"] == "cpu":
                            spec.update({
                                "allow_cpu_training": True,
                                "cpu_training_justification": (
                                    "Canonical SAAS through d=10000 has "
                                    "higher "
                                    "aggregate throughput on node001-node006 "
                                    "without changing NUTS or refit settings."
                                ),
                            })
                        specs.append(spec)
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
        "paper_dimension_frontier_gate_d200_d10000_n13_s80_84_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument(
        "--archive-run-id",
        default="transfer_source_informed_official_n20_s20_20260716",
    )
    parser.add_argument("--heldouts", default=",".join(DOMAINS))
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--dimensions", default="200,10000")
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument(
        "--budgets",
        default="",
        help="Comma-separated target budgets; overrides --N when nonempty.",
    )
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--offline-source-calls", type=int, default=384)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=16384)
    parser.add_argument("--gpu-cpu", type=int, default=12)
    parser.add_argument("--gpu-ram-mb", type=int, default=24576)
    parser.add_argument("--vram-mb", type=int, default=2048)
    parser.add_argument(
        "--saas-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--saas-cpu-max-d", type=int, default=10000)
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    audit = validate_archives(args)
    specs = build_specs(args)
    expected = (
        len(_parse_csv(args.dimensions))
        * len(_parse_csv(args.heldouts))
        * (
            1
            + int(args.n_seeds)
            * (
                int("proposal_only" in _parse_csv(args.backends))
                + int("saasbo" in _parse_csv(args.backends))
                * len(_target_budgets(args))
            )
        )
    )
    if len(specs) != expected:
        raise RuntimeError(f"built {len(specs)} tasks, expected {expected}")
    signatures = [spec["signature"] for spec in specs]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("frontier signatures are not unique")
    if args.dry_run:
        print(json.dumps({
            "audit": audit,
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
            f"dimension-frontier-gate-{args.run_id}",
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
        "run_id": str(args.run_id),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_count": int(len(specs)),
        "task_ids": task_ids,
        "audit": audit,
        "contract": {
            "source_dimension": int(args.source_d),
            "target_dimensions": [
                int(value) for value in _parse_csv(args.dimensions)
            ],
            "source_calls": int(args.offline_source_calls),
            "n0": int(args.n0),
            "saas_search_budgets": list(_target_budgets(args)),
            "proposal": "frozen_risk_objective_atlas",
            "backend": "canonical_saasbo_every_iteration",
            "verifier": "v69_independent_three_policy_objective_guard",
            "target_oracle_used": False,
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
