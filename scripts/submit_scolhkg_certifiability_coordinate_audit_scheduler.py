#!/usr/bin/env python3
"""Submit one offline certifiability/coordinate audit per domain and seed."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import submit_scolhkg_tcb_v3_family_gate_scheduler as v3  # noqa: E402


DEFAULT_DEPLOY = (
    v3.shared.DEFAULT_DEPLOY.parent / "KG_op_scheduler_deploy_cert_audit")


def configuration_grid(args):
    del args
    return [{
        "coordinate": "boundary_latent",
        "descriptor_mode": "raw+learned_risk",
        "geometry": "low_rank_psd",
        "rank": 2,
        "adaptation_ridge": 1.0,
        "effect_ridge": 1.0,
        "rotation_mode": "none",
        "rotation_ridge": 1.0,
        "target_residual_rank": 0,
        "residual_ridge": 1.0,
    }]


def command_for(args, config, heldout, seed, result_file):
    command = v3.shared.command_for(
        args, config, heldout, seed, result_file)
    command = command.replace(
        "performance/benchmark_tcb_v2_source_gate.py",
        "performance/benchmark_certifiability_coordinate_audit.py",
        1,
    )
    for variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        command = command.replace(
            f"{variable}=1", f"{variable}={int(args.cpu)}")
    command += " " + shlex.join([
        "--coefficient-ridge", str(float(args.coefficient_ridge)),
        "--coefficient-prior-strength",
        str(float(args.coefficient_prior_strength)),
        "--coefficient-floor", str(float(args.coefficient_floor)),
        "--oracle-candidate-pool", str(int(args.oracle_candidate_pool)),
        "--hook-pool-per-source", str(int(args.hook_pool_per_source)),
        "--train-sizes", str(args.train_sizes),
        "--training-policies", str(args.training_policies),
        "--regressors", str(args.regressors),
        "--replicate-budgets", str(args.replicate_budgets),
        "--aliasing-neighbors", str(int(args.aliasing_neighbors)),
    ])
    return command


def build_specs(args):
    return v3.shared.build_specs(
        args,
        grid_builder=configuration_grid,
        command_builder=command_for,
        model_tag="certifiability_coordinate_audit",
    )


def build_parser():
    parser = v3.build_parser()
    parser.description = (
        "Offline noise-certifiability and coordinate-sufficiency audit")
    parser.set_defaults(
        deploy=DEFAULT_DEPLOY,
        run_id=f"certifiability_coordinate_{time.strftime('%Y%m%d_%H%M%S')}",
        n_seeds=10,
        cpu=12,
        ram_mb=12288,
    )
    parser.add_argument("--coefficient-ridge", type=float, default=0.1)
    parser.add_argument(
        "--coefficient-prior-strength", type=float, default=0.5)
    parser.add_argument("--coefficient-floor", type=float, default=0.0)
    parser.add_argument("--oracle-candidate-pool", type=int, default=512)
    parser.add_argument("--hook-pool-per-source", type=int, default=512)
    parser.add_argument("--train-sizes", default="10,20,40,80")
    parser.add_argument(
        "--training-policies", default="random,oracle_boundary_stratified")
    parser.add_argument("--regressors", default="ridge_linear,rbf_kernel_ridge")
    parser.add_argument(
        "--replicate-budgets", default="1,3,5,10,20,50,100")
    parser.add_argument("--aliasing-neighbors", type=int, default=5)
    return parser


def main():
    args = build_parser().parse_args()
    if not args.dry_run and not args.skip_prepare_deploy:
        target = v3.prepare_local_deploy(args.deploy)
        required = target / (
            "performance/benchmark_certifiability_coordinate_audit.py")
        if not required.is_file():
            raise RuntimeError(f"prepared audit deploy is missing {required}")
    specs, grid, n_domains = build_specs(args)
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "configuration_grid": grid,
        "n_configurations": len(grid),
        "n_domains": n_domains,
        "n_seeds": int(args.n_seeds),
        "n_tasks": len(specs),
        "nodes": v3.shared.parse_csv(args.nodes),
        "cpu_per_task": int(args.cpu),
        "offline_only": True,
        "one_domain_seed_per_task": True,
        "target_oracle_diagnostic": True,
        "promotion_eligible": False,
        "purpose": (
            "separate noise-limited certifiability from coordinate sufficiency"
        ),
    }
    manifest_path = (
        ROOT / "SC-OLH-KG/profiles" / str(args.run_id) / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps({
            **manifest,
            "first_spec": specs[0] if specs else None,
            "manifest": str(manifest_path),
        }, indent=2))
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
            f"scolhkg-cert-coordinate-{args.run_id}",
            "--intent-ttl",
            "120",
        ],
        input=json.dumps(specs),
        text=True,
    )
    print(output, end="" if output.endswith("\n") else "\n")
    submitted = json.loads(output).get("submitted", [])
    task_ids = [item["id"] for item in submitted if item.get("id")]
    if args.dispatch and task_ids:
        subprocess.check_call([
            sys.executable, str(args.scheduler), "dispatch",
        ])
    print(json.dumps({
        "run_id": args.run_id,
        "n_tasks": len(task_ids),
        "task_ids": task_ids,
        "manifest": str(manifest_path),
    }))


if __name__ == "__main__":
    main()
