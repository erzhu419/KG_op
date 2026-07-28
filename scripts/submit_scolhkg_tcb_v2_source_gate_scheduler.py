#!/usr/bin/env python3
"""Submit the source-only multi-family TCB-V2 gate as one bulk intent."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from submit_scolhkg_manifest_gate_scheduler import (  # noqa: E402
    CPU_NODES,
    DEFAULT_DEPLOY,
    DEFAULT_PYTHON,
    DEFAULT_SCHEDULER,
    parse_csv,
)


DEFAULT_DOMAINS = (
    "RZDT1",
    "StatePolicyRZDT1",
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)


def parse_task_indices(value, total):
    value = str(value or "").strip()
    if not value:
        return list(range(int(total)))
    selected = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(token))
    if not selected or min(selected) < 0 or max(selected) >= int(total):
        raise ValueError(f"task indices must lie in [0, {int(total) - 1}]")
    return sorted(selected)


def _float_csv(value):
    return [float(item) for item in parse_csv(value)]


def _int_csv(value):
    return [int(item) for item in parse_csv(value)]


def configuration_grid(args):
    if str(getattr(args, "preset", "focused_v2")) == "focused_v2":
        common = {
            "coordinate": "boundary_latent",
            "geometry": "low_rank_psd",
            "rank": 2,
            "adaptation_ridge": 1.0,
            "effect_ridge": 1.0,
            "rotation_ridge": 1.0,
            "residual_ridge": 1.0,
        }
        return [
            {
                **common,
                "descriptor_mode": "raw",
                "rotation_mode": "none",
                "target_residual_rank": 0,
            },
            {
                **common,
                "descriptor_mode": "raw",
                "rotation_mode": "planar",
                "target_residual_rank": 1,
            },
            {
                **common,
                "descriptor_mode": "learned_risk",
                "rotation_mode": "none",
                "target_residual_rank": 0,
            },
            {
                **common,
                "descriptor_mode": "learned_risk",
                "rotation_mode": "planar",
                "target_residual_rank": 1,
            },
            {
                **common,
                "descriptor_mode": "raw+learned_risk",
                "rotation_mode": "planar",
                "target_residual_rank": 1,
            },
        ]
    if str(args.preset) != "cartesian":
        raise ValueError(f"unknown TCB-V2 source-gate preset {args.preset!r}")
    return [
        {
            "descriptor_mode": descriptor_mode,
            "coordinate": coordinate,
            "geometry": geometry,
            "rank": rank,
            "adaptation_ridge": adaptation_ridge,
            "effect_ridge": effect_ridge,
            "rotation_mode": rotation_mode,
            "rotation_ridge": rotation_ridge,
            "target_residual_rank": target_residual_rank,
            "residual_ridge": residual_ridge,
        }
        for descriptor_mode, coordinate, geometry, rank, adaptation_ridge, effect_ridge, rotation_mode, rotation_ridge, target_residual_rank, residual_ridge
        in itertools.product(
            parse_csv(args.descriptor_modes),
            parse_csv(args.coordinates),
            parse_csv(args.geometries),
            _int_csv(args.ranks),
            _float_csv(args.adaptation_ridges),
            _float_csv(args.effect_ridges),
            parse_csv(args.rotation_modes),
            _float_csv(args.rotation_ridges),
            _int_csv(args.target_residual_ranks),
            _float_csv(args.residual_ridges),
        )
    ]


def command_for(args, config, heldout, seed, result_file):
    command = [
        "env",
        "LC_ALL=C",
        "LANG=C",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "SCOLHKG_OFFLINE=1",
        "PYTHONUNBUFFERED=1",
        "PYTHONDONTWRITEBYTECODE=1",
        str(args.python),
        "performance/benchmark_tcb_v2_source_gate.py",
        "--heldout", str(heldout),
        "--target-seed", str(int(seed)),
        "--source-seed", str(int(args.source_seed)),
        "--out", str(result_file),
        "--domains", str(args.domains),
        "--d", str(int(args.d)),
        "--L", str(int(args.L)),
        "--sigma", str(float(args.sigma)),
        "--alpha", str(float(args.alpha)),
        "--source-records", str(int(args.source_records)),
        "--source-replicates", str(int(args.source_replicates)),
        "--pilot-pool", str(int(args.pilot_pool)),
        "--evaluation-pool", str(int(args.evaluation_pool)),
        "--n0", str(int(args.n0)),
        "--target-replicates", str(int(args.target_replicates)),
        "--pilot-policies", str(args.pilot_policies),
        "--coordinate", str(config["coordinate"]),
        "--descriptor-mode", str(config["descriptor_mode"]),
        "--meta-local-dim", str(int(args.meta_local_dim)),
        "--meta-shared-dim", str(int(args.meta_shared_dim)),
        "--meta-soft-temperature", str(float(args.meta_soft_temperature)),
        "--geometry", str(config["geometry"]),
        "--rank", str(int(config["rank"])),
        "--ridge", str(float(args.ridge)),
        "--domain-penalty", str(float(args.domain_penalty)),
        "--boundary-temperature", str(float(args.boundary_temperature)),
        "--adaptation-ridge", str(float(config["adaptation_ridge"])),
        "--upper-alpha", str(float(args.upper_alpha)),
        "--calibration-prior-df", str(float(args.calibration_prior_df)),
        "--hierarchy-iterations", str(int(args.hierarchy_iterations)),
        "--effect-ridge", str(float(config["effect_ridge"])),
        "--rotation-mode", str(config["rotation_mode"]),
        "--rotation-ridge", str(float(config["rotation_ridge"])),
        "--target-residual-rank", str(int(config["target_residual_rank"])),
        "--residual-ridge", str(float(config["residual_ridge"])),
    ]
    return shlex.join(command)


def build_specs(
    args,
    *,
    grid_builder=None,
    command_builder=None,
    model_tag="tcb_v2",
):
    grid_builder = configuration_grid if grid_builder is None else grid_builder
    command_builder = command_for if command_builder is None else command_builder
    domains = parse_csv(args.domains)
    nodes = parse_csv(args.nodes)
    if len(domains) < 3:
        raise ValueError("TCB-V2 gate needs at least three domains")
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("nodes must be a subset of node001-node006")
    grid = grid_builder(args)
    remote_root = (
        Path(args.deploy) / "SC-OLH-KG/profiles" / str(args.run_id))
    local_root = ROOT / "SC-OLH-KG/profiles" / str(args.run_id)
    specs = []
    task_index = 0
    for config_index, config in enumerate(grid):
        for heldout in domains:
            for seed_offset in range(int(args.n_seeds)):
                seed = int(args.seed_start) + seed_offset
                node = nodes[task_index % len(nodes)]
                task_index += 1
                relative = Path(
                    f"config{config_index:03d}/{heldout}/seed{seed}")
                remote_result_dir = remote_root / relative
                local_result_dir = local_root / relative
                result_file = remote_result_dir / "result.json"
                specs.append({
                    "description": (
                        f"SC-OLH-KG {model_tag.upper()} gate c={config_index} "
                        f"{heldout} seed={seed}"
                    ),
                    "cmd": command_builder(
                        args, config, heldout, seed, result_file),
                    "cwd": str(Path(args.deploy) / "SC-OLH-KG"),
                    "signature": (
                        f"KG_op/{model_tag}_source/{args.run_id}/"
                        f"config{config_index:03d}/{heldout}/seed{seed}"
                    ),
                    "project": "KG-SYNTH",
                    "vram": 0,
                    "cpu": int(args.cpu),
                    "ram_mb": int(args.ram_mb),
                    "require_node": node,
                    "allowed_nodes": list(nodes),
                    "result_dir": str(remote_result_dir),
                    "local_result_dir": str(local_result_dir),
                    "stage_excludes": ["checkpoints", "profiles", "results"],
                    "allow_duplicate": bool(args.allow_duplicate),
                })
    indices = parse_task_indices(args.task_indices, len(specs))
    return [specs[index] for index in indices], grid, len(domains)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument(
        "--run-id",
        default=f"tcb_v2_source_gate_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--preset",
        choices=("focused_v2", "cartesian"),
        default="focused_v2",
        help=(
            "focused_v2 is the preregistered five-configuration gate; "
            "cartesian is an exploratory sweep"
        ),
    )
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument(
        "--descriptor-modes", default="learned_risk,raw+learned_risk")
    parser.add_argument("--meta-local-dim", type=int, default=3)
    parser.add_argument("--meta-shared-dim", type=int, default=2)
    parser.add_argument("--meta-soft-temperature", type=float, default=0.75)
    parser.add_argument("--coordinates", default="boundary_latent,hybrid_explicit_latent")
    parser.add_argument(
        "--geometries", default="linear_monotone,diagonal_psd,low_rank_psd")
    parser.add_argument("--ranks", default="1,2")
    parser.add_argument("--adaptation-ridges", default="1.0,5.0")
    parser.add_argument("--effect-ridges", default="0.25,1.0")
    parser.add_argument("--rotation-modes", default="none,planar")
    parser.add_argument("--rotation-ridges", default="1.0,5.0")
    parser.add_argument("--target-residual-ranks", default="0,1")
    parser.add_argument("--residual-ridges", default="1.0,5.0")
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
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=4096)
    parser.add_argument("--task-indices", default="")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()

    specs, grid, n_domains = build_specs(args)
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "configuration_grid": grid,
        "n_configurations": len(grid),
        "n_domains": n_domains,
        "n_seeds": int(args.n_seeds),
        "n_tasks": len(specs),
        "nodes": parse_csv(args.nodes),
        "offline_only": True,
        "target_oracle_used_for_fit": False,
        "selection_protocol": "nested_source_lodo_per_outer_target",
        "outer_truth_used_for_hyperparameter_selection": False,
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
        f"scolhkg-tcb-v2-{args.run_id}",
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
