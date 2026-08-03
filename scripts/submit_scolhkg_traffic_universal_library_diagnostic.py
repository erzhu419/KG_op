#!/usr/bin/env python3
"""Submit the fixed universal-library SUMO certifiability diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python")
BOTORCH_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/botorch_overlay_py310")
SUMO_PKG = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/eclipse_sumo_1_25")
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
METHOD = "Universal-Library-Posthoc"
PARTITION = "universal_shape_v1"
SNAPSHOT_MARKER = ".scolhkg_execution_snapshot.json"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _execution_snapshot(code_root):
    marker = Path(code_root) / SNAPSHOT_MARKER
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
        "method_contract_id",
        "theory_contract_id",
        "snapshot_root",
    )
    missing = [key for key in required if not snapshot.get(key)]
    if missing:
        raise ValueError(f"frozen traffic snapshot is incomplete: {missing}")
    if snapshot.get("status") != "frozen":
        raise ValueError("traffic snapshot is not frozen")
    if Path(snapshot["snapshot_root"]).resolve() != Path(code_root).resolve():
        raise ValueError("traffic snapshot marker/root mismatch")
    return snapshot


def _execution_env(snapshot):
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
        "SCOLHKG_METHOD_CONTRACT_ID": snapshot["method_contract_id"],
        "SCOLHKG_THEORY_CONTRACT_ID": snapshot["theory_contract_id"],
        "SCOLHKG_CODE_SNAPSHOT_ROOT": snapshot["snapshot_root"],
    }
    return [f"{key}={value}" for key, value in values.items()]


def _sumo_env(cpu):
    return [
        "env",
        "LC_ALL=C",
        "LANG=C",
        "SCOLHKG_OFFLINE=1",
        "PYTHONUNBUFFERED=1",
        "PYTHONDONTWRITEBYTECODE=1",
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


def build_specs(args):
    snapshot = _execution_snapshot(args.code_root)
    code_root = Path(args.code_root)
    code_project = code_root / "SC-OLH-KG"
    deploy = Path(args.deploy)
    deploy_project = deploy / "SC-OLH-KG"
    deploy_gpr = deploy / "Final_Submission" / "GPR_KG_Code"
    archive = (
        deploy_project / "archives" / args.archive_run_id
        / args.source_split_heldout
        / f"heldout_{args.source_split_heldout}.json"
    )
    manifest = code_project / "performance/manifests/v18b_exactkg_mcdiag.json"
    library_dir = (
        deploy_gpr / "results" / "ingolstadt21"
        / f"PosthocUniversalLibrary_{args.run_id}"
    )
    summary_path = library_dir / "summary.json"
    profile_root = deploy_project / "profiles" / args.run_id
    static_input = str(
        code_root / "Final_Submission" / "GPR_KG_Code" / "results"
        / "ingolstadt21"
    )
    stage_excludes = ["checkpoints", "profiles"]

    design_command = [
        *_sumo_env(args.cpu),
        *_execution_env(snapshot),
        str(REMOTE_PYTHON),
        str(code_project / (
            "performance/materialize_traffic_universal_library_diagnostic.py"
        )),
        "--manifest", str(manifest),
        "--archive", str(archive),
        "--output-dir", str(library_dir),
        "--source-d", str(args.source_d),
        "--source-selection-mode", "descriptor_nearest",
    ]
    specs = [{
        "description": "posthoc traffic universal-library materialization",
        "cmd": f"{shlex.join(design_command)} && echo DONE",
        "cwd": str(code_root),
        "signature": f"KG_op/traffic_universal_posthoc/{args.run_id}/design",
        "project": "KG-SYNTH",
        "vram": 0,
        "cpu": int(args.cpu),
        "ram_mb": int(args.ram_mb),
        "allowed_nodes": list(CPU_NODES),
        "wait_for_files": [str(archive)],
        "stage_input_paths": [static_input],
        "result_dir": str(library_dir),
        "stage_excludes": list(stage_excludes),
        "allow_duplicate": True,
    }]

    shard_paths = []
    for shard_index in range(int(args.num_shards)):
        out = (
            profile_root / "oos_shards"
            / f"shard{shard_index:03d}.json"
        )
        shard_paths.append(out)
        command = [
            *_sumo_env(args.cpu),
            *_execution_env(snapshot),
            str(REMOTE_PYTHON),
            str(code_project / "performance/run_traffic_oos_explicit.py"),
            "--results-root",
            str(deploy_gpr / "results" / "ingolstadt21"),
            "--method", METHOD,
            "--partition", PARTITION,
            "--R", str(args.R),
            "--seed-start", str(args.verification_seed_start),
            "--seed-mode", "common",
            "--num-shards", str(args.num_shards),
            "--shard-index", str(shard_index),
            "--dedupe", "by_x",
            "--jobs", str(args.cpu),
            "--backend", "libsumo",
            "--progress-every", str(max(1, int(args.R) // 2)),
            "--out", str(out),
            "--resume",
        ]
        specs.append({
            "description": (
                "posthoc traffic universal-library OOS "
                f"shard={shard_index}/{args.num_shards}"
            ),
            "cmd": f"{shlex.join(command)} && echo DONE",
            "cwd": str(code_root),
            "signature": (
                f"KG_op/traffic_universal_posthoc/{args.run_id}/"
                f"oos/shard{shard_index:03d}"
            ),
            "project": "KG-SUMO",
            "vram": 0,
            "cpu": int(args.cpu),
            "ram_mb": int(args.ram_mb),
            "allowed_nodes": list(CPU_NODES),
            "wait_for_files": [str(summary_path)],
            "stage_input_paths": [static_input],
            "result_dir": str(out.parent),
            "stage_excludes": list(stage_excludes),
            "allow_duplicate": True,
        })

    audit_path = profile_root / "universal_library_diagnostic.json"
    audit_command = [
        "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
        "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
        *_execution_env(snapshot),
        str(REMOTE_PYTHON),
        str(code_project / (
            "performance/analyze_traffic_universal_library_diagnostic.py"
        )),
        *map(str, shard_paths),
        "--out", str(audit_path),
        "--expected-library-size", str(args.expected_library_size),
        "--target-probability", "0.95",
        "--familywise-delta", "0.05",
        "--redact-policy-vectors",
    ]
    specs.append({
        "description": "posthoc traffic universal-library compact audit",
        "cmd": f"{shlex.join(audit_command)} && echo DONE",
        "cwd": str(code_root),
        "signature": f"KG_op/traffic_universal_posthoc/{args.run_id}/audit",
        "project": "KG-SYNTH",
        "vram": 0,
        "cpu": 1,
        "ram_mb": 4096,
        "allowed_nodes": list(CPU_NODES),
        "wait_for_files": [str(path) for path in shard_paths],
        "stage_input_paths": [static_input],
        "result_dir": str(audit_path.parent),
        "local_result_dir": str(audit_path.parent),
        "stage_excludes": list(stage_excludes),
        "allow_duplicate": True,
    })
    return specs, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--require-frozen-snapshot", action="store_true")
    parser.add_argument(
        "--run-id",
        default=(
            "traffic_universal_library_posthoc_R200_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    parser.add_argument(
        "--archive-run-id",
        default="transfer_source_informed_official_n20_s20_20260716",
    )
    parser.add_argument(
        "--source-split-heldout",
        default="FactorShockStatePolicyRZDT1",
    )
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--expected-library-size", type=int, default=111)
    parser.add_argument("--R", type=int, default=200)
    parser.add_argument("--verification-seed-start", type=int, default=1200000)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=24576)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs, snapshot = build_specs(args)
    expected_tasks = int(args.num_shards) + 2
    if len(specs) != expected_tasks:
        raise RuntimeError(f"built {len(specs)} tasks, expected {expected_tasks}")
    if any(spec["vram"] != 0 for spec in specs):
        raise RuntimeError("posthoc SUMO diagnostic requested GPU memory")
    if any(set(spec["allowed_nodes"]) - set(CPU_NODES) for spec in specs):
        raise RuntimeError("posthoc SUMO diagnostic escaped CPU nodes")
    if len({spec["signature"] for spec in specs}) != len(specs):
        raise RuntimeError("posthoc SUMO diagnostic signatures are not unique")
    if args.dry_run:
        print(json.dumps({
            "run_id": args.run_id,
            "task_count": len(specs),
            "specs": specs,
        }, indent=2, sort_keys=True))
        return

    output = subprocess.check_output(
        [
            sys.executable,
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            f"traffic-universal-posthoc-{args.run_id}",
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
            "evidence_phase": "posthoc_certifiability_diagnostic",
            "admissible_for_method_selection": False,
            "admissible_for_confirmatory_claim": False,
            "library_frozen_before_target_outcomes": True,
            "expected_library_size": int(args.expected_library_size),
            "fresh_seed_replications_per_candidate": int(args.R),
            "verification_seed_start": int(args.verification_seed_start),
            "saas_used": False,
            "gpu_used": False,
            "allowed_nodes": list(CPU_NODES),
            "raw_oos_or_checkpoints_synced_locally": False,
            "execution_snapshot": snapshot,
        },
    }
    path = (
        Path(args.deploy) / "SC-OLH-KG" / "profiles" / args.run_id
        / "submission_manifest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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
    print(json.dumps(registration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
