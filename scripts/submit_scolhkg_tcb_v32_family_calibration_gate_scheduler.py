#!/usr/bin/env python3
"""Submit the TCB-V3.2 family-conditional calibration gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import submit_scolhkg_tcb_v3_family_gate_scheduler as v3  # noqa: E402


DEFAULT_DEPLOY = (
    v3.shared.DEFAULT_DEPLOY.parent / "KG_op_scheduler_deploy_tcbv32")


def configuration_grid(args):
    del args
    common = {
        "coordinate": "boundary_latent",
        "adaptation_ridge": 1.0,
        "effect_ridge": 1.0,
        "rotation_mode": "none",
        "rotation_ridge": 1.0,
        "family_delta": 0.025,
        "evidence_temperature": 1.0,
        "family_guard_scale": 0.0,
        "family_strategy": "source_domain_atoms",
    }
    return [
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 1,
            "target_residual_rank": 0,
            "residual_ridge": 1.0,
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 2,
            "target_residual_rank": 1,
            "residual_ridge": 0.1,
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 2,
            "target_residual_rank": 1,
            "residual_ridge": 1.0,
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 3,
            "target_residual_rank": 2,
            "residual_ridge": 1.0,
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "low_rank_psd",
            "rank": 2,
            "target_residual_rank": 1,
            "residual_ridge": 1.0,
        },
        {
            **common,
            "descriptor_mode": "raw+learned_risk",
            "geometry": "low_rank_psd",
            "rank": 2,
            "target_residual_rank": 1,
            "residual_ridge": 1.0,
        },
    ]


def build_specs(args):
    return v3.shared.build_specs(
        args,
        grid_builder=configuration_grid,
        command_builder=v3.command_for,
        model_tag="tcb_v32_family_calibration",
    )


def build_parser():
    parser = v3.build_parser()
    parser.description = (
        "TCB-V3.2 atomic-family orthogonal calibration gate")
    parser.set_defaults(
        deploy=DEFAULT_DEPLOY,
        run_id=f"tcb_v32_family_calibration_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if not args.dry_run and not args.skip_prepare_deploy:
        v3.prepare_local_deploy(args.deploy)
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
        "model": "tcb_v32_family_conditional_orthogonal_calibration",
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
            f"scolhkg-tcb-v32-{args.run_id}",
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
