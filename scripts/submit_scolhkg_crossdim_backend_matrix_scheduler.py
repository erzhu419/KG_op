#!/usr/bin/env python3
"""Submit the frozen cross-dimension proposal backend-causality matrix."""

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
SAAS_PYTHON = Path(
    "/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python")
BOTORCH_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/botorch_overlay_py310")
TRANSFER_TORCH_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/transfer_torch_py310")
TRANSFERGPBO_OVERLAY = Path(
    "/home/zhengliang01/scheduleurm_work/python_pkgs/transfergpbo_py310")
EXTERNAL_REPOS = REMOTE_ROOT / "external_repos"
CPU_NODES = tuple(f"node{i:03d}" for i in range(1, 7))
GPU_NODES = ("jtl110gpu", "jtl110gpu2", "node007")
DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
BACKENDS = ("proposal_only", "stacked_gp", "saasbo")
INITIAL_DESIGNS = ("source_informed", "common_sobol")


def _parse_csv(value):
    return tuple(
        item.strip() for item in str(value).split(",") if item.strip())


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _terminal_verification_flags(args):
    profile = str(
        getattr(args, "terminal_profile", "v7")
    ).strip().lower()
    if profile in {"v9", "v69"}:
        flags = [
            "--terminal-verification-primary-budget", "80",
            "--terminal-verification-support-budget", "128",
            "--terminal-verification-candidate-budgets", "80,128,128",
            "--terminal-verification-delta", "0.05",
            "--terminal-verification-method",
            "normal_quantile_tolerance",
            "--terminal-verification-shortlist-mode",
            "posterior_objective_challenger_then_safe",
            "--terminal-verification-shortlist-size", "3",
            "--terminal-objective-challenger-max-violation-probability",
            "0.5",
            "--terminal-safe-interior-probability-slack", "0.05",
        ]
        if profile == "v69":
            flags.extend([
                "--terminal-objective-incumbent-guard",
                "--terminal-objective-comparison-budget", "8",
                "--terminal-objective-comparison-delta",
                str(0.05 / 3.0),
            ])
        return flags
    if profile == "v7":
        return [
            "--terminal-verification-primary-budget", "80",
            "--terminal-verification-support-budget", "96",
            "--terminal-verification-delta", "0.05",
            "--terminal-verification-method",
            "normal_quantile_tolerance",
        ]
    raise ValueError("terminal profile must be v7, v9, or v69")


def validate_contract(args):
    """Audit source inputs and any reused SC rows before adding challengers."""

    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    domains = _parse_csv(args.heldouts)
    initial_design_mode = str(getattr(
        args, "initial_design_mode", "source_informed"
    )).strip().lower()
    if initial_design_mode not in INITIAL_DESIGNS:
        raise ValueError(
            f"initial design must be one of {INITIAL_DESIGNS}")
    required_seeds = set(range(
        int(args.seed_start),
        int(args.seed_start) + int(args.n_seeds),
    ))
    audit = {
        "initial_design_mode": initial_design_mode,
        "domains": {},
        "v64_reused_result_count": 0,
        "all_initial_designs_byte_identical_by_seed": True,
    }
    for heldout in domains:
        archive_path = (
            deploy_project / "archives" / args.archive_run_id / heldout
            / f"heldout_{heldout}.json"
        )
        design_path = (
            deploy_project / "archives" / args.design_run_id / heldout
            / "source_initial_designs.json"
        )
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"missing cross-dimension source archive for {heldout}")
        archive = _read_json(archive_path)
        archive_dimensions = {
            len(task["X"][0]) for task in archive["tasks"]
        }
        if archive_dimensions != {int(args.source_d)}:
            raise ValueError(f"{heldout} archive is not source-dimension data")
        if initial_design_mode == "common_sobol":
            audit["domains"][heldout] = {
                "archive_fingerprint": archive["fingerprint"],
                "source_dimension": int(args.source_d),
                "target_dimension": int(args.d),
                "n0": int(args.n0),
                "seed_count": int(len(required_seeds)),
                "v64_rows_reused": 0,
                "common_sobol_generator": (
                    "core.designs.common_sobol_integer_design"
                ),
            }
            continue
        if not design_path.is_file():
            raise FileNotFoundError(
                f"missing frozen cross-dimension design for {heldout}")
        design = _read_json(design_path)
        if archive["fingerprint"] != design["source_archive_fingerprint"]:
            raise ValueError(f"{heldout} archive/design fingerprint mismatch")
        if int(design["source_dimension"]) != int(args.source_d):
            raise ValueError(f"{heldout} design source dimension mismatch")
        if int(design["dimension"]) != int(args.d):
            raise ValueError(f"{heldout} design target dimension mismatch")
        if int(design["n0"]) != int(args.n0):
            raise ValueError(f"{heldout} design n0 mismatch")
        available = {int(seed) for seed in design["designs"]}
        if not required_seeds.issubset(available):
            raise ValueError(f"{heldout} frozen design is missing seeds")

        v64_root = (
            deploy_project / "profiles" / args.v64_run_id
            / "v64_powered_safe_interior_verification"
        )
        v64_paths = list(v64_root.rglob(f"{heldout}/seed*/result.json"))
        rows_by_seed = {}
        for path in v64_paths:
            payload = _read_json(path)
            if len(payload.get("rows", ())) != 1:
                continue
            row = payload["rows"][0]
            seed = int(row["seed"])
            if seed in required_seeds:
                rows_by_seed[seed] = row
        if set(rows_by_seed) != required_seeds:
            missing = sorted(required_seeds - set(rows_by_seed))
            raise ValueError(
                f"{heldout} V64 reuse set is incomplete: {missing}")
        for seed, row in rows_by_seed.items():
            expected = design["designs"][str(seed)]["fingerprint"]
            consumed = row["task_initial_design"]["fingerprint"]
            if consumed != expected:
                raise ValueError(
                    f"{heldout} seed {seed} V64 consumed another n0")
            if int(row["n0"]) != int(args.n0):
                raise ValueError(f"{heldout} seed {seed} V64 n0 changed")
            if int(row["n_search_simulations"]) != int(args.N):
                raise ValueError(
                    f"{heldout} seed {seed} V64 search budget changed")
            if row["initial_design_archive_contract"]["matches"] is not True:
                raise ValueError(
                    f"{heldout} seed {seed} V64 archive contract failed")
        audit["domains"][heldout] = {
            "archive_fingerprint": archive["fingerprint"],
            "source_dimension": int(args.source_d),
            "target_dimension": int(args.d),
            "n0": int(args.n0),
            "seed_count": int(len(required_seeds)),
            "v64_rows_reused": int(len(rows_by_seed)),
        }
        audit["v64_reused_result_count"] += len(rows_by_seed)
    return audit


def _base_spec(args, *, backend, heldout, seed, command, cpu, ram_mb,
               vram, allowed_nodes):
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    local_result = (
        deploy_project / "profiles" / args.run_id / backend / heldout
        / f"seed{seed:04d}"
    )
    return {
        "description": (
            f"crossdim {getattr(args, 'initial_design_mode', 'source_informed')} "
            f"backend {backend} {heldout} "
            f"seed={seed}"
        ),
        "cmd": f"{shlex.join(command)} && echo DONE",
        "cwd": str(deploy_project),
        "signature": (
            f"KG_op/crossdim_backend_matrix/{args.run_id}/{backend}/"
            f"{heldout}/seed{seed:04d}"
        ),
        "project": "KG-SYNTH",
        "vram": int(vram),
        "cpu": int(cpu),
        "ram_mb": int(ram_mb),
        "allowed_nodes": list(allowed_nodes),
        # Scheduler rewrites paths rooted at the local deployment for compute
        # nodes while leaving them unchanged on GPU jump hosts. Hard-coding
        # the compute-node root makes the same task non-portable to jtl hosts.
        "result_dir": str(local_result),
        "local_result_dir": str(local_result),
        "stage_excludes": ["checkpoints", "profiles", "results"],
        "allow_duplicate": True,
    }


def build_specs(args):
    deploy_project = Path(args.deploy) / "SC-OLH-KG"
    manifest = (
        deploy_project / "performance/manifests/v18b_exactkg_mcdiag.json")
    domains = _parse_csv(args.heldouts)
    backends = _parse_csv(args.backends)
    initial_design_mode = str(getattr(
        args, "initial_design_mode", "source_informed"
    )).strip().lower()
    if initial_design_mode not in INITIAL_DESIGNS:
        raise ValueError(
            f"initial design must be one of {INITIAL_DESIGNS}")
    unknown = sorted(set(backends) - set(BACKENDS))
    if unknown:
        raise ValueError(f"unknown backend rows: {unknown}")
    specs = []
    terminal_flags = _terminal_verification_flags(args)
    for heldout in domains:
        local_archive = (
            deploy_project / "archives" / args.archive_run_id / heldout
            / f"heldout_{heldout}.json"
        )
        local_design = (
            deploy_project / "archives" / args.design_run_id / heldout
            / "source_initial_designs.json"
        )
        for seed in range(
            int(args.seed_start),
            int(args.seed_start) + int(args.n_seeds),
        ):
            if "proposal_only" in backends:
                execution_result = (
                    deploy_project / "profiles" / args.run_id
                    / "proposal_only" / heldout / f"seed{seed:04d}"
                    / "result.json"
                )
                command = [
                    "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                    "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                    "OPENBLAS_NUM_THREADS=1",
                    str(REMOTE_PYTHON),
                    "performance/benchmark_frozen_proposal_only.py",
                    "--heldout", heldout,
                    "--seed", str(seed),
                    "--initial-design", initial_design_mode,
                    "--out", str(execution_result),
                    "--source-d", str(args.source_d),
                    "--d", str(args.d),
                    "--n0", str(args.n0),
                    "--offline-source-calls", str(
                        args.offline_source_calls
                        if initial_design_mode == "source_informed"
                        else 0
                    ),
                    *terminal_flags,
                ]
                if initial_design_mode == "source_informed":
                    command.extend([
                        "--initial-design-file", str(local_design),
                    ])
                spec = _base_spec(
                    args,
                    backend="proposal_only",
                    heldout=heldout,
                    seed=seed,
                    command=command,
                    cpu=1,
                    ram_mb=4096,
                    vram=0,
                    allowed_nodes=CPU_NODES,
                )
                spec["wait_for_files"] = (
                    [str(local_design)]
                    if initial_design_mode == "source_informed"
                    else []
                )
                specs.append(spec)

            if "stacked_gp" in backends:
                execution_result = (
                    deploy_project / "profiles" / args.run_id
                    / "stacked_gp" / heldout / f"seed{seed:04d}"
                    / "result.json"
                )
                checkpoint = (
                    deploy_project / "checkpoints" / args.run_id
                    / "stacked_gp" / heldout / f"seed{seed:04d}"
                )
                command = [
                    "env", "LC_ALL=C", "LANG=C", "SCOLHKG_OFFLINE=1",
                    "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
                    f"OMP_NUM_THREADS={int(args.cpu)}",
                    f"MKL_NUM_THREADS={int(args.cpu)}",
                    f"OPENBLAS_NUM_THREADS={int(args.cpu)}",
                    f"PYTHONPATH={TRANSFER_TORCH_OVERLAY}:"
                    f"{TRANSFERGPBO_OVERLAY}",
                    f"SCOLHKG_EXTERNAL_REPO_ROOT={EXTERNAL_REPOS}",
                    f"SCOLHKG_TRANSFERGPBO_OVERLAY={TRANSFERGPBO_OVERLAY}",
                    str(REMOTE_PYTHON),
                    "performance/benchmark_transfer_fairness.py",
                    "--method", "stacked_transfer_gp_cbo",
                    "--implementation", "official",
                    "--heldout", heldout,
                    "--archive", str(local_archive),
                    "--initial-design", initial_design_mode,
                    "--out", str(execution_result),
                    "--checkpoint-dir", str(checkpoint),
                    "--seed", str(seed),
                    "--d", str(args.d),
                    "--N", str(args.N),
                    "--n0", str(args.n0),
                    "--source-dimension-adapter",
                    "ordered_dct_quadratic",
                    "--source-coordinate-max-frequency", "8",
                    "--source-coordinate-frequency-penalty", "0.10",
                    "--terminal-verification",
                    *terminal_flags,
                ]
                if initial_design_mode == "source_informed":
                    command.extend([
                        "--initial-design-file", str(local_design),
                    ])
                spec = _base_spec(
                    args,
                    backend="stacked_gp",
                    heldout=heldout,
                    seed=seed,
                    command=command,
                    cpu=args.cpu,
                    ram_mb=args.ram_mb,
                    vram=0,
                    allowed_nodes=CPU_NODES,
                )
                spec["wait_for_files"] = [str(local_archive)]
                if initial_design_mode == "source_informed":
                    spec["wait_for_files"].append(str(local_design))
                specs.append(spec)

            if "saasbo" in backends:
                execution_result = (
                    deploy_project / "profiles" / args.run_id
                    / "saasbo" / heldout / f"seed{seed:04d}"
                    / "result.json"
                )
                checkpoint = (
                    deploy_project / "checkpoints" / args.run_id
                    / "saasbo" / heldout / f"seed{seed:04d}"
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
                    f"PYTHONPATH={BOTORCH_OVERLAY}",
                    str(SAAS_PYTHON),
                    "performance/benchmark_sota_fairness.py",
                    "--protocol", (
                        "shared_archive_n13"
                        if initial_design_mode == "source_informed"
                        else "target_n13"
                    ),
                    "--method", "botorch_saasbo",
                    "--heldout", heldout,
                    "--seed", str(seed),
                    "--manifest", str(manifest),
                    "--out", str(execution_result),
                    "--checkpoint-dir", str(checkpoint),
                    "--target-budget", str(args.N),
                    "--d", str(args.d),
                    "--n0", str(args.n0),
                    "--candidate-timeout-sec", "3600",
                    "--torch-device", "cuda",
                    "--torch-deterministic",
                    "--saas-refit-schedule", "every_iteration",
                    "--terminal-verification",
                    *terminal_flags,
                ]
                if initial_design_mode == "source_informed":
                    command.extend([
                        "--initial-design-file", str(local_design),
                    ])
                else:
                    command.append("--common-sobol-initial-design")
                spec = _base_spec(
                    args,
                    backend="saasbo",
                    heldout=heldout,
                    seed=seed,
                    command=command,
                    cpu=args.gpu_cpu,
                    ram_mb=args.gpu_ram_mb,
                    vram=args.gpu_vram_mb,
                    allowed_nodes=GPU_NODES,
                )
                spec["wait_for_files"] = (
                    [str(local_design)]
                    if initial_design_mode == "source_informed"
                    else []
                )
                spec["vram_resource_family"] = (
                    f"KG-SYNTH/crossdim-backend/saasbo/{heldout}")
                specs.append(spec)
    if bool(getattr(args, "skip_existing_success", False)):
        filtered = []
        for spec in specs:
            result_path = Path(spec["local_result_dir"]) / "result.json"
            if result_path.is_file():
                try:
                    if _read_json(result_path).get("status") == "ok":
                        continue
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            filtered.append(spec)
        specs = filtered
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--run-id", default=(
        "paper_certified_crossdim_backend_matrix_d50_d1000_n13_"
        f"s80_99_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument(
        "--archive-run-id",
        default="transfer_source_informed_official_n20_s20_20260716",
    )
    parser.add_argument(
        "--design-run-id",
        default="paper_certified_scolh_v64_cross_d50_d1000_s80_99_v1",
    )
    parser.add_argument(
        "--v64-run-id",
        default=(
            "paper_certified_scolh_v64_cross_d50_d1000_n13_s80_99_v1"),
    )
    parser.add_argument("--heldouts", default=",".join(DOMAINS))
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument(
        "--initial-design-mode",
        choices=INITIAL_DESIGNS,
        default="source_informed",
    )
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--offline-source-calls", type=int, default=384)
    parser.add_argument(
        "--terminal-profile",
        choices=("v7", "v9", "v69"),
        default="v69",
        help=(
            "Shared terminal protocol for every backend. v9 freezes an "
            "objective challenger, strict primary, and safe support with "
            "80/128/128 independent verification calls. v69 adds an "
            "independent paired objective-incumbent comparison."
        ),
    )
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=16384)
    parser.add_argument("--gpu-cpu", type=int, default=12)
    parser.add_argument("--gpu-ram-mb", type=int, default=24576)
    parser.add_argument("--gpu-vram-mb", type=int, default=2048)
    parser.add_argument(
        "--sync-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-existing-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Submit only logical cells whose local compact result.json is "
            "missing or does not have status=ok."
        ),
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    audit = validate_contract(args)
    specs = build_specs(args)
    expected = (
        len(_parse_csv(args.heldouts))
        * int(args.n_seeds)
        * len(_parse_csv(args.backends))
    )
    if not args.skip_existing_success and len(specs) != expected:
        raise RuntimeError(
            f"built {len(specs)} tasks, expected {expected}")
    if args.skip_existing_success and len(specs) > expected:
        raise RuntimeError(
            f"built {len(specs)} recovery tasks, expected at most {expected}")
    signatures = [spec["signature"] for spec in specs]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("crossdim backend signatures are not unique")
    if args.dry_run:
        print(json.dumps({
            "audit": audit,
            "task_count": len(specs),
            "specs": specs,
        }, indent=2))
        return
    if args.sync_remote:
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
            f"crossdim-backend-matrix-{args.run_id}",
        ],
        input=json.dumps(specs),
        text=True,
    )
    response = json.loads(output)
    submitted = response.get("submitted", [])
    task_ids = [row["id"] for row in submitted if row.get("id")]
    registration = {
        "schema_version": 1,
        "run_id": args.run_id,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "audit": audit,
        "new_task_count": int(len(specs)),
        "skip_existing_success": bool(args.skip_existing_success),
        "candidate_task_count_before_success_filter": int(expected),
        "reused_v64_result_count": int(
            audit["v64_reused_result_count"]),
        "matrix_cell_count": int(
            len(specs) + audit["v64_reused_result_count"]),
        "terminal_profile": str(args.terminal_profile),
        "initial_design_mode": str(args.initial_design_mode),
        "terminal_verification_contract": {
            "shortlist_mode": (
                "posterior_objective_challenger_then_safe"
                if args.terminal_profile in {"v9", "v69"}
                else "posterior_primary_safe_interior"
            ),
            "candidate_budgets": (
                [80, 128, 128]
                if args.terminal_profile in {"v9", "v69"}
                else [80, 96]
            ),
            "objective_incumbent_guard": bool(
                args.terminal_profile == "v69"),
            "objective_comparison_budget_per_policy": (
                8 if args.terminal_profile == "v69" else 0),
            "familywise_delta": 0.05,
            "verification_updates_optimizer": False,
            "target_oracle_used": False,
        },
        "task_ids": task_ids,
        "checkpoint_results_synced_locally": False,
    }
    manifest_name = (
        f"recovery_submission_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
        if args.skip_existing_success
        else "submission_manifest.json"
    )
    registration_path = (
        Path(args.deploy) / "SC-OLH-KG" / "profiles" / args.run_id
        / manifest_name
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
