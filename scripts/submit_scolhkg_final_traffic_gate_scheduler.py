#!/usr/bin/env python3
"""Submit the strict no-history external SUMO gate for the final method."""

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
    SOURCE_SELECTION_MODES,
    source_selection_contract,
    traffic_method_label,
)

SYNC = ROOT / "scripts/sync_scolhkg_scheduler_deploy.sh"
SYNC_TRAFFIC_ASSETS = (
    ROOT / "scripts/sync_scolhkg_traffic_problem_assets.sh")
DEFAULT_SCHEDULER = Path.home() / "mine_code/scheduleurm/skill/scheduler.py"
DEFAULT_DEPLOY = Path.home() / "mine_code/KG_op_scheduler_deploy"
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python")
BOTORCH_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/botorch_overlay_py310")
SUMO_PKG = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/eclipse_sumo_1_25")
CPU_NODES = tuple(f"node{i:03d}" for i in range(1, 7))
GPU_NODES = ("jtl110gpu", "jtl110gpu2", "node007")


def _sumo_env(cpu):
    return [
        "env",
        "LC_ALL=C",
        "LANG=C",
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
    source_selection_mode = str(getattr(
        args, "source_selection_mode", DESCRIPTOR_NEAREST))
    selection = source_selection_contract(source_selection_mode)
    method_label = traffic_method_label(source_selection_mode)
    gpu_nodes = [
        node.strip()
        for node in str(getattr(
            args, "gpu_nodes", ",".join(GPU_NODES))).split(",")
        if node.strip()
    ]
    if not gpu_nodes or any(node not in GPU_NODES for node in gpu_nodes):
        raise ValueError(
            "gpu_nodes must be a nonempty subset of "
            + ",".join(GPU_NODES)
        )
    gpu_cpu = int(getattr(args, "gpu_cpu", 12))
    gpu_ram_mb = int(getattr(args, "gpu_ram_mb", 32768))
    gpu_vram_mb = int(getattr(args, "gpu_vram_mb", 2048))
    deploy = Path(args.deploy)
    project = deploy / "SC-OLH-KG"
    gpr_code = deploy / "Final_Submission" / "GPR_KG_Code"
    manifest = (
        project / "performance/manifests/v18b_exactkg_mcdiag.json")
    archive = (
        project / "archives" / args.archive_run_id
        / selection.source_split_heldout
        / f"heldout_{selection.source_split_heldout}.json"
    )
    gpu_runner = project / "runners/run_traffic_gpu_python.sh"
    design = (
        project / "archives" / args.run_id / "traffic"
        / "source_initial_designs.json"
    )
    design_cmd = [
        *_sumo_env(args.cpu),
        str(REMOTE_PYTHON),
        "performance/materialize_external_traffic_design.py",
        "--manifest", str(manifest),
        "--archive", str(archive),
        "--out", str(design),
        "--source-d", str(args.source_d),
        "--n0", str(args.n0),
        "--seed-start", str(args.seed_start),
        "--n-seeds", str(args.n_seeds),
        "--source-selection-mode", selection.mode,
    ]
    specs = [{
        "description": "paper final external traffic source proposal",
        "cmd": f"{shlex.join(design_cmd)} && echo DONE",
        "cwd": str(project),
        "signature": f"KG_op/final_traffic/{args.run_id}/design",
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
    }]
    oos_paths = []
    for seed in range(
        int(args.seed_start),
        int(args.seed_start) + int(args.n_seeds),
    ):
        partition = f"{args.run_id}_seed{seed:04d}"
        run_dir = (
            gpr_code / "results" / "ingolstadt21"
            / f"PaperFinal_SourceProposal_SAAS_{partition}_seed{seed}"
        )
        checkpoint_dir = (
            project / "checkpoints" / args.run_id / "traffic"
            / f"seed{seed:04d}"
        )
        search_cmd = [
            *_sumo_env(gpu_cpu),
            "SCOLHKG_OFFLINE=1",
            "SCOLHKG_TORCH_DETERMINISTIC=1",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            str(gpu_runner),
            "performance/benchmark_traffic_final_contract.py",
            "--initial-design-file", str(design),
            "--output-dir", str(run_dir),
            "--checkpoint-dir", str(checkpoint_dir),
            "--seed", str(seed),
            "--method-label", method_label,
            "--partition-method", partition,
            "--N", str(args.N),
            "--n0", str(args.n0),
            "--torch-device", "cuda",
            "--torch-deterministic",
            "--saas-parallel-threads-per-model",
            str(max(1, gpu_cpu // 2)),
            "--resume",
        ]
        specs.append({
            "description": f"paper final traffic SAAS seed={seed}",
            "cmd": f"{shlex.join(search_cmd)} && echo DONE",
            "cwd": str(project),
            "signature": (
                f"KG_op/final_traffic/{args.run_id}/search/seed{seed:04d}"
            ),
            "project": "KG-SYNTH",
            "vram": gpu_vram_mb,
            "cpu": gpu_cpu,
            "ram_mb": gpu_ram_mb,
            "allowed_nodes": gpu_nodes,
            "wait_for_files": [str(design)],
            "result_dir": str(run_dir),
            "local_result_dir": str(run_dir),
            "stage_excludes": ["checkpoints", "profiles", "results"],
            "allow_duplicate": True,
        })
        oos_path = (
            project / "profiles" / args.run_id / "traffic"
            / f"seed{seed:04d}" / f"oos_R{int(args.R)}.json"
        )
        oos_paths.append(oos_path)
        oos_cmd = [
            *_sumo_env(args.cpu),
            str(REMOTE_PYTHON),
            "-m", "experiments.ingolstadt21.validate_oos_feasibility",
            "--method", method_label,
            "--partition", partition,
            "--R", str(args.R),
            "--seed-start", str(
                int(args.verification_seed_start) + 1000 * seed),
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
                f"paper final traffic fresh OOS R={args.R} seed={seed}"),
            "cmd": f"{shlex.join(oos_cmd)} && echo DONE",
            "cwd": str(gpr_code),
            "signature": (
                f"KG_op/final_traffic/{args.run_id}/oos/seed{seed:04d}"
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
        project / "profiles" / args.run_id / "traffic"
        / "external_traffic_audit.json"
    )
    analyze_cmd = [
        "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
        "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
        str(REMOTE_PYTHON),
        "performance/analyze_traffic_final_contract.py",
        *map(str, oos_paths),
        "--out", str(audit_path),
        "--target-probability", "0.95",
        "--familywise-delta", "0.05",
        "--source-domains",
        ",".join(selection.source_domains),
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
        "description": "paper final external traffic aggregate audit",
        "cmd": f"{shlex.join(analyze_cmd)} && echo DONE",
        "cwd": str(project),
        "signature": f"KG_op/final_traffic/{args.run_id}/audit",
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
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--run-id", default=(
        "paper_final_external_traffic_s80_84_R100_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument(
        "--archive-run-id",
        default="transfer_source_informed_official_n20_s20_20260716",
    )
    parser.add_argument(
        "--source-selection-mode",
        choices=SOURCE_SELECTION_MODES,
        default=DESCRIPTOR_NEAREST,
    )
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--verification-seed-start", type=int, default=900000)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=24576)
    parser.add_argument("--gpu-nodes", default=",".join(GPU_NODES))
    parser.add_argument("--gpu-cpu", type=int, default=12)
    parser.add_argument("--gpu-ram-mb", type=int, default=32768)
    parser.add_argument("--gpu-vram-mb", type=int, default=2048)
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = build_specs(args)
    expected = 2 * int(args.n_seeds) + 2
    if len(specs) != expected:
        raise RuntimeError(f"built {len(specs)} tasks, expected {expected}")
    signatures = [spec["signature"] for spec in specs]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("traffic gate signatures are not unique")
    if args.dry_run:
        print(json.dumps({
            "run_id": args.run_id,
            "task_count": len(specs),
            "specs": specs,
        }, indent=2))
        return
    if args.sync_remote:
        subprocess.run([str(SYNC)], cwd=ROOT, check=True)
        subprocess.run(
            [str(SYNC_TRAFFIC_ASSETS)], cwd=ROOT, check=True)
    output = subprocess.check_output(
        [
            sys.executable,
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            f"paper-final-traffic-{args.run_id}",
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
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_count": len(specs),
        "task_ids": task_ids,
        "contract": {
            "source_selection": source_selection_contract(
                args.source_selection_mode).as_dict(),
            "source_calls": 384,
            "target_domain": "Ingolstadt21Traffic",
            "target_search_calls": int(args.N),
            "target_verification_calls": int(3 * args.R),
            "historical_traffic_anchor_used": False,
            "target_labels_used_to_fit_proposal": False,
            "target_oracle_used": False,
            "backend": "canonical_saasbo_every_iteration",
            "verifier": (
                "fresh_seed_familywise_exact_binomial_shortlist_v1"),
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
