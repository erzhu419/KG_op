#!/usr/bin/env python3
"""Submit the fixed-budget lower-envelope paired gate to CPU nodes only."""

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
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
FRONTENDS = ("v1", "lower_envelope_v2")
BACKENDS = ("proposal_only", "botorch_scbo")
SNAPSHOT_MARKER = ".scolhkg_execution_snapshot.json"


def _parse_csv(value):
    return tuple(
        item.strip() for item in str(value).split(",") if item.strip())


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _execution_snapshot(args):
    marker = Path(args.code_root) / SNAPSHOT_MARKER
    if not marker.is_file():
        if args.require_frozen_snapshot:
            raise FileNotFoundError(
                f"frozen code snapshot marker is missing: {marker}")
        return None
    snapshot = _read_json(marker)
    required = (
        "repository_commit",
        "scolhkg_tree",
        "proof_tree",
        "scripts_tree",
        "method_contract_id",
        "theory_contract_id",
        "snapshot_root",
    )
    missing = [field for field in required if not snapshot.get(field)]
    if missing:
        raise ValueError(f"frozen snapshot is incomplete: {missing}")
    if (
        Path(snapshot["snapshot_root"]).resolve()
        != Path(args.code_root).resolve()
    ):
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
        "SCOLHKG_METHOD_CONTRACT_ID": snapshot["method_contract_id"],
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
    project = Path(args.deploy) / "SC-OLH-KG"
    audit = {}
    for heldout in _parse_csv(args.heldouts):
        path = (
            project / "archives" / args.archive_run_id / heldout
            / f"heldout_{heldout}.json"
        )
        archive = _read_json(path)
        dimensions = {len(task["X"][0]) for task in archive["tasks"]}
        calls = sum(
            sum(len(row) for row in task["Y_replicates"])
            for task in archive["tasks"]
        )
        if dimensions != {int(args.source_d)}:
            raise ValueError(f"{heldout} source dimension changed")
        if int(calls) != int(args.offline_source_calls):
            raise ValueError(f"{heldout} source cost changed")
        audit[heldout] = {
            "archive_fingerprint": str(archive["fingerprint"]),
            "source_dimension": int(args.source_d),
            "source_calls": int(calls),
        }
    return audit


def build_specs(args):
    project = Path(args.deploy) / "SC-OLH-KG"
    code_project = Path(args.code_root) / "SC-OLH-KG"
    snapshot = _execution_snapshot(args)
    manifest = code_project / "performance/manifests/v18b_exactkg_mcdiag.json"
    heldouts = _parse_csv(args.heldouts)
    frontends = _parse_csv(args.frontends)
    backends = _parse_csv(args.backends)
    if not heldouts or not set(heldouts).issubset(DOMAINS):
        raise ValueError("heldouts must use the registered synthetic domains")
    if not frontends or not set(frontends).issubset(FRONTENDS):
        raise ValueError(f"frontends must be drawn from {FRONTENDS}")
    if not backends or not set(backends).issubset(BACKENDS):
        raise ValueError(f"backends must be drawn from {BACKENDS}")

    specs = []
    for frontend in frontends:
        for heldout in heldouts:
            archive = (
                project / "archives" / args.archive_run_id / heldout
                / f"heldout_{heldout}.json"
            )
            design = (
                project / "archives" / args.run_id / frontend / heldout
                / "source_initial_designs.json"
            )
            design_command = [
                "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                f"OMP_NUM_THREADS={int(args.cpu)}",
                f"MKL_NUM_THREADS={int(args.cpu)}",
                f"OPENBLAS_NUM_THREADS={int(args.cpu)}",
                *_execution_env(snapshot),
                str(REMOTE_PYTHON),
                str(code_project / "performance/materialize_source_initial_designs.py"),
                "--manifest", str(manifest),
                "--heldout", heldout,
                "--archive", str(archive),
                "--out", str(design),
                "--d", str(args.d),
                "--source-d", str(args.source_d),
                "--n0", str(args.n0),
                "--seed-start", str(args.seed_start),
                "--n-seeds", str(args.n_seeds),
                "--structural-prior-profile", "low_frequency_only",
                "--proposal-mode", "risk_objective_atlas",
                "--source-design-mode", "universal_mixture",
            ]
            if frontend == "lower_envelope_v2":
                design_command.append("--protect-lower-envelope-sentinel")
            specs.append({
                "description": f"lower-envelope gate design {frontend} {heldout}",
                "cmd": f"{shlex.join(design_command)} && echo DONE",
                "cwd": str(code_project),
                "signature": (
                    f"KG_op/lower_envelope_gate/{args.run_id}/design/"
                    f"{frontend}/{heldout}"
                ),
                "project": "KG-SYNTH",
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
            for seed in range(
                int(args.seed_start),
                int(args.seed_start) + int(args.n_seeds),
            ):
                if "proposal_only" in backends:
                    result_dir = (
                        project / "profiles" / args.run_id / frontend
                        / "proposal_only" / heldout / f"seed{seed:04d}"
                    )
                    command = [
                        "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                        "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                        "OPENBLAS_NUM_THREADS=1",
                        *_execution_env(snapshot),
                        str(REMOTE_PYTHON),
                        str(code_project / "performance/benchmark_frozen_proposal_only.py"),
                        "--heldout", heldout,
                        "--seed", str(seed),
                        "--initial-design", "source_informed",
                        "--initial-design-file", str(design),
                        "--out", str(result_dir / "result.json"),
                        "--source-d", str(args.source_d),
                        "--d", str(args.d),
                        "--n0", str(args.n0),
                        "--offline-source-calls", str(args.offline_source_calls),
                        *_terminal_flags(),
                    ]
                    specs.append({
                        "description": (
                            f"lower-envelope gate proposal {frontend} "
                            f"{heldout} seed={seed}"
                        ),
                        "cmd": f"{shlex.join(command)} && echo DONE",
                        "cwd": str(code_project),
                        "signature": (
                            f"KG_op/lower_envelope_gate/{args.run_id}/"
                            f"{frontend}/proposal/{heldout}/seed{seed:04d}"
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
                if "botorch_scbo" in backends:
                    result_dir = (
                        project / "profiles" / args.run_id / frontend
                        / "botorch_scbo" / heldout / f"seed{seed:04d}"
                    )
                    checkpoint_dir = (
                        project / "checkpoints" / args.run_id / frontend
                        / "botorch_scbo" / heldout / f"seed{seed:04d}"
                    )
                    command = [
                        "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                        "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                        f"OMP_NUM_THREADS={int(args.cpu)}",
                        f"MKL_NUM_THREADS={int(args.cpu)}",
                        f"OPENBLAS_NUM_THREADS={int(args.cpu)}",
                        "SCOLHKG_TORCH_DETERMINISTIC=1",
                        *_execution_env(snapshot),
                        f"PYTHONPATH={BOTORCH_OVERLAY}",
                        str(REMOTE_PYTHON),
                        str(code_project / "performance/benchmark_sota_fairness.py"),
                        "--protocol", "shared_archive_hvd_n13",
                        "--method", "botorch_scbo",
                        "--heldout", heldout,
                        "--seed", str(seed),
                        "--manifest", str(manifest),
                        "--out", str(result_dir / "result.json"),
                        "--checkpoint-dir", str(checkpoint_dir),
                        "--initial-design-file", str(design),
                        "--source-archive-file", str(archive),
                        "--aleatoric-head-mode", "pooled",
                        "--target-budget", str(args.N),
                        "--d", str(args.d),
                        "--n0", str(args.n0),
                        "--raw-samples", str(args.raw_samples),
                        "--num-restarts", str(args.num_restarts),
                        "--maxiter", str(args.maxiter),
                        "--ts-candidates", str(args.ts_candidates),
                        "--candidate-timeout-sec", str(args.candidate_timeout_sec),
                        "--torch-device", "cpu",
                        "--torch-deterministic",
                        "--terminal-verification",
                        *_terminal_flags(),
                    ]
                    specs.append({
                        "description": (
                            f"lower-envelope gate SCBO {frontend} {heldout} "
                            f"seed={seed}"
                        ),
                        "cmd": f"{shlex.join(command)} && echo DONE",
                        "cwd": str(code_project),
                        "signature": (
                            f"KG_op/lower_envelope_gate/{args.run_id}/"
                            f"{frontend}/scbo/{heldout}/seed{seed:04d}"
                        ),
                        "project": "KG-SYNTH",
                        "vram": 0,
                        "cpu": int(args.cpu),
                        "ram_mb": int(args.ram_mb),
                        "allowed_nodes": list(CPU_NODES),
                        "allow_cpu_training": True,
                        "cpu_training_justification": (
                            "Non-SAAS paired proposal gate on node001-node006; "
                            "GPU resources are excluded by experiment contract."
                        ),
                        "wait_for_files": [str(archive), str(design)],
                        "result_dir": str(result_dir),
                        "local_result_dir": str(result_dir),
                        "stage_excludes": [
                            "checkpoints", "profiles", "results"],
                        "allow_duplicate": True,
                    })
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--require-frozen-snapshot", action="store_true")
    parser.add_argument("--run-id", default=(
        "lower_envelope_synthetic_gate_d50_d1000_n13_s80_84_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument(
        "--archive-run-id",
        default="transfer_source_informed_official_n20_s20_20260716",
    )
    parser.add_argument("--heldouts", default=",".join(DOMAINS))
    parser.add_argument("--frontends", default=",".join(FRONTENDS))
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--offline-source-calls", type=int, default=384)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--ts-candidates", type=int, default=2000)
    parser.add_argument("--candidate-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=24576)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    audit = validate_archives(args)
    specs = build_specs(args)
    expected = (
        len(_parse_csv(args.frontends))
        * len(_parse_csv(args.heldouts))
        * (1 + int(args.n_seeds) * len(_parse_csv(args.backends)))
    )
    if len(specs) != expected:
        raise RuntimeError(f"built {len(specs)} tasks, expected {expected}")
    if len({spec["signature"] for spec in specs}) != len(specs):
        raise RuntimeError("lower-envelope gate signatures are not unique")
    if args.dry_run:
        print(json.dumps({
            "task_count": len(specs),
            "archive_audit": audit,
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
            f"lower-envelope-synthetic-gate-{args.run_id}",
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
        "run_id": str(args.run_id),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_count": len(specs),
        "task_ids": task_ids,
        "archive_audit": audit,
        "contract": {
            "intervention": (
                "reserve one fixed n0 slot for a target-label-free universal "
                "lower-envelope sentinel; displace one source template"),
            "same_n0": int(args.n0),
            "same_target_budget": int(args.N),
            "same_source_archive": True,
            "same_backends": list(_parse_csv(args.backends)),
            "same_independent_verifier": True,
            "frontends": list(_parse_csv(args.frontends)),
            "saas_used": False,
            "gpu_used": False,
            "allowed_nodes": list(CPU_NODES),
            "target_oracle_used_during_search": False,
            "checkpoint_results_synced_locally": False,
            "execution_snapshot": _execution_snapshot(args),
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
