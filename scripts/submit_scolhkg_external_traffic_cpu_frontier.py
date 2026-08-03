#!/usr/bin/env python3
"""Submit the strict no-history SUMO CPU budget frontier.

This is an external-validity experiment, not the immutable canonical-SAAS
track.  It keeps the frozen front end and fresh-seed verifier fixed while
using only non-SAAS BoTorch backends on node001--node006.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = ROOT / "SC-OLH-KG"
sys.path.insert(0, str(PROJECT_SOURCE))

from performance.task_descriptor_retrieval import (  # noqa: E402
    DESCRIPTOR_NEAREST,
    DOMAIN_BLIND_CONTROL,
    SOURCE_SELECTION_MODES,
    source_selection_contract,
)


SYNC = ROOT / "scripts/sync_scolhkg_scheduler_deploy.sh"
SYNC_TRAFFIC_ASSETS = ROOT / "scripts/sync_scolhkg_traffic_problem_assets.sh"
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python")
BOTORCH_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/botorch_overlay_py310")
SUMO_PKG = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/eclipse_sumo_1_25")
CPU_NODES = tuple(f"node{i:03d}" for i in range(1, 7))
CPU_BACKENDS = ("botorch_scbo", "botorch_turbo")
SNAPSHOT_MARKER = ".scolhkg_execution_snapshot.json"
METHOD_CONTRACT_ID = "external_cpu_frontend_budget_frontier_v1"


def parse_csv(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def parse_int_csv(value):
    rows = tuple(int(item) for item in parse_csv(value))
    if not rows or any(item < 10 for item in rows):
        raise ValueError("traffic budgets must be a nonempty list with N >= 10")
    return rows


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _execution_snapshot(args):
    if args.code_root is None:
        if args.require_frozen_snapshot:
            raise ValueError("--require-frozen-snapshot needs --code-root")
        return None
    marker = Path(args.code_root) / SNAPSHOT_MARKER
    if not marker.is_file():
        raise FileNotFoundError(f"frozen traffic snapshot is missing: {marker}")
    snapshot = _read_json(marker)
    required = (
        "repository_commit",
        "scolhkg_tree",
        "proof_tree",
        "scripts_tree",
        "legacy_traffic_tree",
        "traffic_decision_space_blob",
        "traffic_baseline_blob",
        "theory_contract_id",
        "snapshot_root",
    )
    missing = [key for key in required if not snapshot.get(key)]
    if missing:
        raise ValueError(f"frozen traffic snapshot is incomplete: {missing}")
    if snapshot.get("status") != "frozen":
        raise ValueError("traffic snapshot is not frozen")
    if Path(snapshot["snapshot_root"]).resolve() != Path(args.code_root).resolve():
        raise ValueError("traffic snapshot marker/root mismatch")
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
        "SCOLHKG_LEGACY_TRAFFIC_TREE": snapshot["legacy_traffic_tree"],
        "SCOLHKG_TRAFFIC_DECISION_SPACE_BLOB": (
            snapshot["traffic_decision_space_blob"]),
        "SCOLHKG_TRAFFIC_BASELINE_BLOB": snapshot["traffic_baseline_blob"],
        "SCOLHKG_METHOD_CONTRACT_ID": METHOD_CONTRACT_ID,
        "SCOLHKG_THEORY_CONTRACT_ID": snapshot["theory_contract_id"],
        "SCOLHKG_CODE_SNAPSHOT_ROOT": snapshot["snapshot_root"],
    }
    return [f"{key}={value}" for key, value in values.items()]


def _sumo_env(cpu):
    return [
        "env",
        "LC_ALL=C",
        "LANG=C",
        "PYTHONUNBUFFERED=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "SCOLHKG_OFFLINE=1",
        f"OMP_NUM_THREADS={int(cpu)}",
        f"MKL_NUM_THREADS={int(cpu)}",
        f"OPENBLAS_NUM_THREADS={int(cpu)}",
        f"SUMO_PKG={SUMO_PKG}",
        f"SUMO_HOME={SUMO_PKG / 'sumo'}",
        (
            f"PATH={SUMO_PKG / 'sumo/bin'}:/usr/local/sbin:"
            "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        (
            f"LD_LIBRARY_PATH={SUMO_PKG / 'libsumo.libs'}:"
            f"{SUMO_PKG / 'eclipse_sumo.libs'}"
        ),
        (
            f"PYTHONPATH={BOTORCH_OVERLAY}:{SUMO_PKG}:"
            f"{SUMO_PKG / 'sumo/tools'}"
        ),
    ]


def _method_label(source_mode, backend, budget):
    source = (
        "Descriptor" if source_mode == DESCRIPTOR_NEAREST else "DomainBlind")
    name = {
        "botorch_scbo": "SCBO",
        "botorch_turbo": "TuRBO",
    }[backend]
    return f"ExternalTraffic-{source}Proposal-{name}-N{int(budget)}"


def build_specs(args):
    backend = str(args.backend)
    if backend not in CPU_BACKENDS:
        raise ValueError(f"backend must be one of {CPU_BACKENDS}")
    budgets = parse_int_csv(args.budgets)
    source_modes = parse_csv(args.source_selection_modes)
    if not source_modes or not set(source_modes).issubset(SOURCE_SELECTION_MODES):
        raise ValueError(
            f"source modes must be drawn from {SOURCE_SELECTION_MODES}")

    snapshot = _execution_snapshot(args)
    deploy = Path(args.deploy)
    deploy_project = deploy / "SC-OLH-KG"
    deploy_gpr_code = deploy / "Final_Submission" / "GPR_KG_Code"
    code_root = Path(args.code_root) if args.code_root else deploy
    code_project = code_root / "SC-OLH-KG"
    manifest = code_project / "performance/manifests/v18b_exactkg_mcdiag.json"
    specs = []

    for source_mode in source_modes:
        selection = source_selection_contract(source_mode)
        archive = (
            deploy_project / "archives" / args.archive_run_id
            / selection.source_split_heldout
            / f"heldout_{selection.source_split_heldout}.json"
        )
        design = (
            deploy_project / "archives" / args.run_id / source_mode
            / "source_initial_designs.json"
        )
        design_cmd = [
            *_sumo_env(args.cpu),
            *_execution_env(snapshot),
            str(REMOTE_PYTHON),
            str(code_project / "performance/materialize_external_traffic_design.py"),
            "--manifest", str(manifest),
            "--archive", str(archive),
            "--out", str(design),
            "--source-d", str(args.source_d),
            "--n0", str(args.n0),
            "--seed-start", str(args.seed_start),
            "--n-seeds", str(args.n_seeds),
            "--source-selection-mode", source_mode,
        ]
        specs.append({
            "description": f"external traffic CPU proposal {source_mode}",
            "cmd": f"{shlex.join(design_cmd)} && echo DONE",
            "cwd": str(code_project),
            "signature": (
                f"KG_op/external_traffic_cpu/{args.run_id}/"
                f"{source_mode}/design"
            ),
            "project": "KG-SUMO",
            "vram": 0,
            "cpu": int(args.cpu),
            "ram_mb": int(args.ram_mb),
            "allowed_nodes": list(CPU_NODES),
            "wait_for_files": [str(archive)],
            "result_dir": str(design.parent),
            "local_result_dir": str(design.parent),
            "stage_excludes": ["checkpoints", "profiles", "results"],
            "allow_duplicate": True,
        })

        for budget in budgets:
            oos_paths = []
            method_label = _method_label(source_mode, backend, budget)
            for seed in range(
                int(args.seed_start),
                int(args.seed_start) + int(args.n_seeds),
            ):
                partition = (
                    f"{args.run_id}_{source_mode}_{backend}_N{budget}_"
                    f"seed{seed:04d}"
                )
                run_dir = (
                    deploy_gpr_code / "results" / "ingolstadt21"
                    / f"ExternalCPU_{partition}"
                )
                checkpoint_dir = (
                    deploy_project / "checkpoints" / args.run_id
                    / source_mode / backend / f"N{budget}" / f"seed{seed:04d}"
                )
                search_cmd = [
                    *_sumo_env(args.cpu),
                    *_execution_env(snapshot),
                    "SCOLHKG_TORCH_DETERMINISTIC=1",
                    str(REMOTE_PYTHON),
                    str(code_project / "performance/benchmark_traffic_final_contract.py"),
                    "--initial-design-file", str(design),
                    "--output-dir", str(run_dir),
                    "--checkpoint-dir", str(checkpoint_dir),
                    "--seed", str(seed),
                    "--method-label", method_label,
                    "--partition-method", partition,
                    "--backend", backend,
                    "--N", str(budget),
                    "--n0", str(args.n0),
                    "--raw-samples", str(args.raw_samples),
                    "--num-restarts", str(args.num_restarts),
                    "--maxiter", str(args.maxiter),
                    "--ts-candidates", str(args.ts_candidates),
                    "--candidate-timeout-sec", str(args.candidate_timeout_sec),
                    "--torch-device", "cpu",
                    "--torch-deterministic",
                    "--resume",
                ]
                specs.append({
                    "description": (
                        f"external traffic CPU {source_mode} {backend} "
                        f"N={budget} seed={seed}"
                    ),
                    "cmd": f"{shlex.join(search_cmd)} && echo DONE",
                    "cwd": str(code_project),
                    "signature": (
                        f"KG_op/external_traffic_cpu/{args.run_id}/"
                        f"{source_mode}/{backend}/N{budget}/search/seed{seed:04d}"
                    ),
                    "project": "KG-SUMO",
                    "vram": 0,
                    "cpu": int(args.cpu),
                    "ram_mb": int(args.ram_mb),
                    "allowed_nodes": list(CPU_NODES),
                    "allow_cpu_training": True,
                    "cpu_training_justification": (
                        "Registered non-SAAS strict no-history SUMO frontier; "
                        "all GPU nodes are excluded by contract."
                    ),
                    "wait_for_files": [str(design)],
                    "result_dir": str(run_dir),
                    "local_result_dir": str(run_dir),
                    "stage_excludes": ["checkpoints", "profiles", "results"],
                    "allow_duplicate": True,
                })

                oos_path = (
                    deploy_project / "profiles" / args.run_id / source_mode
                    / backend / f"N{budget}" / f"seed{seed:04d}"
                    / f"oos_R{int(args.R)}.json"
                )
                oos_paths.append(oos_path)
                oos_cmd = [
                    *_sumo_env(args.cpu),
                    *_execution_env(snapshot),
                    str(REMOTE_PYTHON),
                    str(code_project / "performance/run_traffic_oos_explicit.py"),
                    "--results-root",
                    str(deploy_gpr_code / "results" / "ingolstadt21"),
                    "--method", method_label,
                    "--partition", partition,
                    "--R", str(args.R),
                    "--seed-start", str(
                        int(args.verification_seed_start)
                        + 100000 * int(budget)
                        + 1000 * int(seed)
                    ),
                    "--seed-mode", "common",
                    "--source-indexes", "0,1,2",
                    "--max-points", "3",
                    "--dedupe", "none",
                    "--jobs", str(args.cpu),
                    "--backend", "libsumo",
                    "--progress-every", str(max(1, int(args.R) // 2)),
                    "--out", str(oos_path),
                    "--resume",
                ]
                specs.append({
                    "description": (
                        f"external traffic OOS {source_mode} {backend} "
                        f"N={budget} seed={seed} R={args.R}"
                    ),
                    "cmd": f"{shlex.join(oos_cmd)} && echo DONE",
                    "cwd": str(code_project),
                    "signature": (
                        f"KG_op/external_traffic_cpu/{args.run_id}/"
                        f"{source_mode}/{backend}/N{budget}/oos/seed{seed:04d}"
                    ),
                    "project": "KG-SUMO",
                    "vram": 0,
                    "cpu": int(args.cpu),
                    "ram_mb": int(args.ram_mb),
                    "allowed_nodes": list(CPU_NODES),
                    "wait_for_files": [str(run_dir / "summary.json")],
                    "result_dir": str(oos_path.parent),
                    "local_result_dir": str(oos_path.parent),
                    "stage_excludes": ["checkpoints", "profiles", "results"],
                    "allow_duplicate": True,
                })

            audit_path = (
                deploy_project / "profiles" / args.run_id / source_mode
                / backend / f"N{budget}" / "external_traffic_audit.json"
            )
            analyze_cmd = [
                "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                *_execution_env(snapshot),
                str(REMOTE_PYTHON),
                str(code_project / "performance/analyze_traffic_final_contract.py"),
                *map(str, oos_paths),
                "--out", str(audit_path),
                "--target-probability", "0.95",
                "--familywise-delta", "0.05",
                "--source-domains", ",".join(selection.source_domains),
                "--excluded-nearest-source-analogue",
                selection.source_split_heldout,
                "--information-track", selection.track,
                "--source-selection-mode", selection.mode,
                "--source-split-heldout", selection.source_split_heldout,
                "--target-domain", "Ingolstadt21Traffic",
            ]
            if selection.heldout_task_family_identifier_used:
                analyze_cmd.append("--heldout-task-family-identifier-used")
            specs.append({
                "description": (
                    f"external traffic audit {source_mode} {backend} N={budget}"
                ),
                "cmd": f"{shlex.join(analyze_cmd)} && echo DONE",
                "cwd": str(code_project),
                "signature": (
                    f"KG_op/external_traffic_cpu/{args.run_id}/"
                    f"{source_mode}/{backend}/N{budget}/audit"
                ),
                "project": "KG-SYNTH",
                "vram": 0,
                "cpu": 1,
                "ram_mb": 4096,
                "allowed_nodes": list(CPU_NODES),
                "wait_for_files": [str(path) for path in oos_paths],
                "result_dir": str(audit_path.parent),
                "local_result_dir": str(audit_path.parent),
                "stage_excludes": ["checkpoints", "profiles", "results"],
                "allow_duplicate": True,
            })
    return specs, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--require-frozen-snapshot", action="store_true")
    parser.add_argument(
        "--run-id",
        default=(
            "external_traffic_cpu_frontier_s80_84_R100_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    parser.add_argument(
        "--archive-run-id",
        default="transfer_source_informed_official_n20_s20_20260716",
    )
    parser.add_argument(
        "--source-selection-modes",
        default=f"{DESCRIPTOR_NEAREST},{DOMAIN_BLIND_CONTROL}",
    )
    parser.add_argument("--backend", choices=CPU_BACKENDS, default="botorch_scbo")
    parser.add_argument("--budgets", default="13,40,80")
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--verification-seed-start", type=int, default=900000)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--ts-candidates", type=int, default=2000)
    parser.add_argument("--candidate-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=24576)
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs, snapshot = build_specs(args)
    modes = parse_csv(args.source_selection_modes)
    budgets = parse_int_csv(args.budgets)
    expected = len(modes) * (1 + len(budgets) * (2 * args.n_seeds + 1))
    if len(specs) != expected:
        raise RuntimeError(f"built {len(specs)} tasks, expected {expected}")
    signatures = [spec["signature"] for spec in specs]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("external CPU traffic signatures are not unique")
    if any(spec["vram"] != 0 for spec in specs):
        raise RuntimeError("CPU traffic frontier requested GPU memory")
    if any(set(spec["allowed_nodes"]) - set(CPU_NODES) for spec in specs):
        raise RuntimeError("CPU traffic frontier escaped node001--node006")

    if args.dry_run:
        print(json.dumps({
            "run_id": args.run_id,
            "task_count": len(specs),
            "specs": specs,
        }, indent=2))
        return

    if args.sync_remote:
        if args.require_frozen_snapshot and args.code_root is None:
            raise ValueError("--require-frozen-snapshot needs --code-root")
        if args.code_root is None:
            subprocess.run([str(SYNC)], cwd=ROOT, check=True)
        subprocess.run([str(SYNC_TRAFFIC_ASSETS)], cwd=ROOT, check=True)

    output = subprocess.check_output(
        [
            sys.executable,
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            f"external-traffic-cpu-{args.run_id}",
        ],
        input=json.dumps(specs),
        text=True,
    )
    response = json.loads(output)
    task_ids = [
        row["id"] for row in response.get("submitted", []) if row.get("id")
    ]
    registration = {
        "schema_version": 1,
        "run_id": args.run_id,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_count": len(specs),
        "task_ids": task_ids,
        "contract": {
            "contract_id": METHOD_CONTRACT_ID,
            "source_selection_modes": list(modes),
            "source_calls": 384,
            "target_search_budgets": list(budgets),
            "n0": int(args.n0),
            "fresh_seed_replications_per_shortlist_policy": int(args.R),
            "backend": args.backend,
            "saas_used": False,
            "gpu_used": False,
            "allowed_nodes": list(CPU_NODES),
            "historical_traffic_anchor_used": False,
            "target_labels_used_to_fit_proposal": False,
            "target_oracle_used": False,
            "verifier": "fresh_seed_familywise_exact_binomial_shortlist_v1",
            "execution_snapshot": (
                snapshot if snapshot is not None else {"status": "unregistered"}
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
                *sum(([
                    "--task-id", task_id
                ] for task_id in task_ids), []),
            ],
            check=True,
        )
    print(json.dumps(registration, indent=2))


if __name__ == "__main__":
    main()
