#!/usr/bin/env python3
"""Submit the strict TCB-V4 continuous family-synthesis gate."""

from __future__ import annotations

import argparse
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
    v3.shared.DEFAULT_DEPLOY.parent / "KG_op_scheduler_deploy_tcbv4")


def configuration_grid(args):
    del args
    common = {
        "coordinate": "boundary_latent",
        "adaptation_ridge": 1.0,
        "effect_ridge": 1.0,
        "rotation_mode": "none",
        "rotation_ridge": 1.0,
        "target_residual_rank": 0,
        "residual_ridge": 1.0,
        "coefficient_floor": 0.0,
    }
    return [
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 1,
            "coefficient_ridge": 0.01,
            "coefficient_prior_strength": 0.25,
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 1,
            "coefficient_ridge": 0.1,
            "coefficient_prior_strength": 0.5,
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 1,
            "coefficient_ridge": 1.0,
            "coefficient_prior_strength": 1.0,
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 2,
            "coefficient_ridge": 0.1,
            "coefficient_prior_strength": 0.5,
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "low_rank_psd",
            "rank": 2,
            "coefficient_ridge": 0.1,
            "coefficient_prior_strength": 0.5,
        },
        {
            **common,
            "descriptor_mode": "raw+learned_risk",
            "geometry": "low_rank_psd",
            "rank": 2,
            "coefficient_ridge": 0.1,
            "coefficient_prior_strength": 0.5,
        },
    ]


def command_for(args, config, heldout, seed, result_file):
    command = v3.shared.command_for(
        args, config, heldout, seed, result_file)
    command = command.replace(
        "performance/benchmark_tcb_v2_source_gate.py",
        "performance/benchmark_tcb_v4_synthesis_gate.py",
        1,
    )
    command += " " + shlex.join([
        "--coefficient-ridge", str(float(config["coefficient_ridge"])),
        "--coefficient-prior-strength",
        str(float(config["coefficient_prior_strength"])),
        "--coefficient-floor", str(float(config["coefficient_floor"])),
    ])
    return command


def build_specs(args):
    return v3.shared.build_specs(
        args,
        grid_builder=configuration_grid,
        command_builder=command_for,
        model_tag="tcb_v4_synthesis",
    )


def build_parser():
    parser = v3.build_parser()
    parser.description = "TCB-V4 continuous boundary-family synthesis gate"
    parser.set_defaults(
        deploy=DEFAULT_DEPLOY,
        run_id=f"tcb_v4_synthesis_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if not args.dry_run and not args.skip_prepare_deploy:
        target = v3.prepare_local_deploy(args.deploy)
        required = target / "performance/benchmark_tcb_v4_synthesis_gate.py"
        if not required.is_file():
            raise RuntimeError(f"prepared V4 deploy is missing {required}")
    specs, grid, n_domains = build_specs(args)
    manifest = {
        "schema_version": 4,
        "run_id": args.run_id,
        "configuration_grid": grid,
        "n_configurations": len(grid),
        "n_domains": n_domains,
        "n_seeds": int(args.n_seeds),
        "n_tasks": len(specs),
        "nodes": v3.shared.parse_csv(args.nodes),
        "offline_only": True,
        "model": "tcb_v4_boundary_family_synthesis",
        "selection_protocol": "nested_source_lodo_per_outer_target",
        "target_domain_label_used": False,
        "target_oracle_used_for_fit": False,
        "outer_truth_used_for_hyperparameter_selection": False,
        "online_kg_blocked_until_gate_passes": True,
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
            f"scolhkg-tcb-v4-{args.run_id}",
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
