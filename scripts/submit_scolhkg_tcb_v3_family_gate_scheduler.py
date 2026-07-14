#!/usr/bin/env python3
"""Submit the strict TCB-V3 source-only family gate to node001-node006."""

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

import submit_scolhkg_tcb_v2_source_gate_scheduler as shared  # noqa: E402


DEFAULT_V3_DEPLOY = (
    shared.DEFAULT_DEPLOY.parent / "KG_op_scheduler_deploy_tcbv31")


def prepare_local_deploy(deploy):
    """Build a fresh lightweight staging tree from the current workspace."""

    deploy = Path(deploy).resolve()
    source = (ROOT / "SC-OLH-KG").resolve()
    target = deploy / "SC-OLH-KG"
    if deploy == ROOT.resolve() or source == target:
        raise ValueError("V3 deploy must be separate from the source workspace")
    target.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "rsync", "-a", "--delete",
        "--exclude=profiles/",
        "--exclude=results/",
        "--exclude=checkpoints/",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        str(source) + "/",
        str(target) + "/",
    ])
    required = (
        target / "performance/benchmark_tcb_v3_family_gate.py",
        target / "representation/transferable_boundary.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"prepared V3 deploy is missing {missing}")
    return target


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
    }
    return [
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 1,
            "family_delta": 0.01,
            "evidence_temperature": 0.50,
            "family_guard_scale": 0.0,
            "family_strategy": "source_domain_atoms",
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 1,
            "family_delta": 0.025,
            "evidence_temperature": 1.00,
            "family_guard_scale": 0.0,
            "family_strategy": "source_domain_atoms",
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 1,
            "family_delta": 0.04,
            "evidence_temperature": 2.00,
            "family_guard_scale": 0.0,
            "family_strategy": "source_domain_atoms",
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "low_rank_psd",
            "rank": 2,
            "family_delta": 0.025,
            "evidence_temperature": 1.00,
            "family_guard_scale": 0.0,
            "family_strategy": "source_domain_atoms",
        },
        {
            **common,
            "descriptor_mode": "raw+learned_risk",
            "geometry": "low_rank_psd",
            "rank": 2,
            "family_delta": 0.025,
            "evidence_temperature": 1.00,
            "family_guard_scale": 0.0,
            "family_strategy": "source_domain_atoms",
        },
        {
            **common,
            "descriptor_mode": "raw",
            "geometry": "linear_monotone",
            "rank": 1,
            "family_delta": 0.025,
            "evidence_temperature": 1.00,
            "family_guard_scale": 0.0,
            "family_strategy": "pooled_plus_source_domain_atoms",
        },
    ]


def command_for(args, config, heldout, seed, result_file):
    base = shared.command_for(args, config, heldout, seed, result_file)
    command = base.replace(
        "performance/benchmark_tcb_v2_source_gate.py",
        "performance/benchmark_tcb_v3_family_gate.py",
        1,
    )
    command += " " + shlex.join([
        "--family-delta", str(float(config["family_delta"])),
        "--evidence-temperature", str(float(config["evidence_temperature"])),
        "--family-guard-scale", str(float(config["family_guard_scale"])),
        "--family-strategy", str(config["family_strategy"]),
    ])
    return command


def build_specs(args):
    return shared.build_specs(
        args,
        grid_builder=configuration_grid,
        command_builder=command_for,
        model_tag="tcb_v3_family",
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=shared.DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_V3_DEPLOY)
    parser.add_argument("--python", default=shared.DEFAULT_PYTHON)
    parser.add_argument(
        "--run-id",
        default=f"tcb_v3_family_gate_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--domains", default=",".join(shared.DEFAULT_DOMAINS))
    parser.add_argument("--meta-local-dim", type=int, default=3)
    parser.add_argument("--meta-shared-dim", type=int, default=2)
    parser.add_argument("--meta-soft-temperature", type=float, default=0.75)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--domain-penalty", type=float, default=0.5)
    parser.add_argument("--boundary-temperature", type=float, default=1.0)
    parser.add_argument("--upper-alpha", type=float, default=0.01)
    parser.add_argument("--calibration-prior-df", type=float, default=2.0)
    parser.add_argument("--hierarchy-iterations", type=int, default=5)
    parser.add_argument("--source-seed", type=int, default=7001)
    parser.add_argument("--source-records", type=int, default=96)
    parser.add_argument("--source-replicates", type=int, default=3)
    parser.add_argument("--pilot-pool", type=int, default=256)
    parser.add_argument("--evaluation-pool", type=int, default=512)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--target-replicates", type=int, default=3)
    parser.add_argument("--pilot-policies", default="random,source_boundary")
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--nodes", default=",".join(shared.CPU_NODES))
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--task-indices", default="")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--skip-prepare-deploy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.dry_run and not args.skip_prepare_deploy:
        prepare_local_deploy(args.deploy)
    specs, grid, n_domains = build_specs(args)
    manifest = {
        "schema_version": 3,
        "run_id": args.run_id,
        "configuration_grid": grid,
        "n_configurations": len(grid),
        "n_domains": n_domains,
        "n_seeds": int(args.n_seeds),
        "n_tasks": len(specs),
        "nodes": shared.parse_csv(args.nodes),
        "offline_only": True,
        "model": "tcb_v3_boundary_family_mixture",
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
    command = [
        sys.executable,
        str(args.scheduler),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        f"scolhkg-tcb-v3-family-{args.run_id}",
        "--intent-ttl",
        "120",
    ]
    output = subprocess.check_output(
        command, input=json.dumps(specs), text=True)
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
