#!/usr/bin/env python3
"""Submit the preregistered SC-OLH paper matrix after all freeze gates pass."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


closure = _load(
    "paper_promoted_v51_closure",
    ROOT / "scripts/submit_scolhkg_promoted_v51_closure_gate_scheduler.py",
)
designs = _load(
    "paper_source_designs",
    ROOT / "scripts/submit_scolhkg_mean_alignment_v51_s20_designs_scheduler.py",
)
CPU_NODES = closure.CPU_NODES
DEFAULT_REGISTRATION = (
    ROOT / "SC-OLH-KG/performance/manifests/paper_main_matrix_v1.json")
DEFAULT_SOURCE_MANIFEST = (
    closure._root_module().DEFAULT_DEPLOY
    / "SC-OLH-KG/performance/manifests/v18b_exactkg_mcdiag.json"
)
VARIANTS = (
    "frozen_proposal",
    "proposal_sobol",
    "promoted_joint_voi",
    "new_point_only",
    "pooled_variance",
    "frozen_source_discrepancy",
)
IMPLEMENTATION_CONTRACT_ID = "promoted_v51_observed_terminal_closure"
THEORY_CONTRACT_ID = "v51_statistical_closure_v2"


def _parse_csv(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def load_registration(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("paper registration schema_version must be 1")
    return payload


def validate_freeze(registration):
    if registration.get("status") != "frozen":
        raise RuntimeError(
            "paper matrix is not frozen; Gate B and Gate C must finish first")
    evidence = registration.get("freeze_evidence") or {}
    for field in ("replication_cap", "exact_mc_samples", "exact_shortlist_size"):
        value = evidence.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"frozen registration requires positive {field}")
    if evidence.get("exact_sampling_mode") != "antithetic_nested":
        raise ValueError("paper exact sampling mode must be antithetic_nested")
    contracts = dict(registration.get("contracts") or {})
    if contracts.get("implementation_contract_id") != IMPLEMENTATION_CONTRACT_ID:
        raise ValueError("paper implementation contract is missing or changed")
    if contracts.get("theory_contract_id") != THEORY_CONTRACT_ID:
        raise ValueError("paper theory contract is missing or changed")
    return evidence


def inspect_plan(registration):
    domains = tuple(registration["synthetic_domains"])
    variants = tuple(registration["main_variants"])
    rows = []
    total = 0
    for frontier in registration["frontier"]:
        count = (
            len(frontier["budgets"])
            * int(frontier["seeds"])
            * len(domains)
            * len(variants)
        )
        total += count
        rows.append({
            "d": int(frontier["d"]),
            "budgets": list(map(int, frontier["budgets"])),
            "seeds": int(frontier["seeds"]),
            "run_tasks": count,
            "requires_d1000_gate": bool(
                frontier.get("requires_d1000_gate", False)),
        })
    return {
        "status": registration.get("status"),
        "domains": list(domains),
        "variants": list(variants),
        "frontier": rows,
        "design_tasks": len(domains) * len(registration["frontier"]),
        "run_tasks": total,
        "total_tasks": total + len(domains) * len(registration["frontier"]),
    }


def _variant_profiles(registration):
    freeze = validate_freeze(registration)
    source = registration["source_contract"]
    base = {
        **closure.PROFILE,
        "source_records_per_domain": int(source["records_per_source_domain"]),
        "exact_mc_samples": int(freeze["exact_mc_samples"]),
        "exact_sampling_mode": str(freeze["exact_sampling_mode"]),
        "evaluate_or_replicate_new_action_count": int(
            freeze["exact_shortlist_size"]),
        "replication_max_per_solution": int(freeze["replication_cap"]),
        "hvd_profile": "factor_hierarchical",
        "source_discrepancy_update": True,
    }
    return {
        "frozen_proposal": {
            **base,
            "decision_backend": "n0_best",
            "exact_mc_samples": 0,
            "adaptive_replication_voi": False,
            "replication_candidate_count": 0,
        },
        "proposal_sobol": {
            **base,
            "decision_backend": "sobol_new",
            "exact_mc_samples": 0,
            "adaptive_replication_voi": False,
            "replication_candidate_count": 0,
        },
        "promoted_joint_voi": base,
        "new_point_only": {
            **base,
            "adaptive_replication_voi": False,
            "replication_candidate_count": 0,
        },
        "pooled_variance": {
            **base,
            "hvd_profile": "pooled",
        },
        "frozen_source_discrepancy": {
            **base,
            "source_discrepancy_update": False,
        },
    }


def _selected_frontier(registration, dims, budgets):
    requested_dims = set(map(int, dims)) if dims else None
    requested_budgets = set(map(int, budgets)) if budgets else None
    selected = []
    freeze = registration["freeze_evidence"]
    for row in registration["frontier"]:
        dimension = int(row["d"])
        if requested_dims is not None and dimension not in requested_dims:
            continue
        cell_budgets = [
            int(value) for value in row["budgets"]
            if requested_budgets is None or int(value) in requested_budgets
        ]
        if not cell_budgets:
            continue
        if (
            row.get("requires_d1000_gate")
            and freeze.get("d1000_frontier_passed") is not True
        ):
            raise RuntimeError(
                f"d={dimension} is blocked until d=1000 frontier passes")
        selected.append({**row, "budgets": cell_budgets})
    if not selected:
        raise ValueError("no registered dimension/budget cells selected")
    return selected


def _proposal_run_id(run_id, dimension):
    return (
        f"paper_main_v1/{run_id}/proposals/d{int(dimension)}/"
        "risk_objective_atlas/low_frequency_only"
    )


def _design_specs(args, registration, frontier):
    source = registration["source_contract"]
    output = []
    domains = tuple(registration["synthetic_domains"])
    for row in frontier:
        dimension = int(row["d"])
        child = copy.copy(args)
        child.manifest = args.source_manifest
        child.archive_run_id = args.archive_run_id
        child.output_source_run_id = _proposal_run_id(args.run_id, dimension)
        child.heldouts = ",".join(domains)
        child.source_d = int(source["source_dimension"])
        child.d = dimension
        child.n0 = int(args.n0)
        child.seed_start = 0
        child.n_seeds = int(row["seeds"])
        child.run_id = f"{args.run_id}_design_d{dimension}"
        output.extend(designs.build_specs(child))
    return output


def _run_specs(args, registration, frontier, selected_variants):
    profiles = _variant_profiles(registration)
    root = closure._root_module()
    source = registration["source_contract"]
    output = []
    scenarios = tuple(
        item for item in closure.promoted.v51.v50.v49.v27.MEAN_SCENARIOS
        if item[0] in set(registration["synthetic_domains"])
    )
    for row in frontier:
        dimension = int(row["d"])
        for budget in row["budgets"]:
            child = copy.copy(args)
            child.manifest = args.source_manifest
            child.source_run_id = _proposal_run_id(args.run_id, dimension)
            child.remote_design_only = False
            child.source_d = int(source["source_dimension"])
            child.source_records_per_domain = int(
                source["records_per_source_domain"])
            child.d = dimension
            child.N = int(budget)
            child.n0 = int(args.n0)
            child.seed_start = 0
            child.n_seeds = int(row["seeds"])
            child.implementation_contract_id = IMPLEMENTATION_CONTRACT_ID
            child.theory_contract_id = THEORY_CONTRACT_ID
            child.variant_profiles = {
                name: profiles[name] for name in selected_variants
            }
            child.variants = ",".join(selected_variants)
            child.scenarios = scenarios
            child.stage_family = "paper_main_v1"
            child.gate_label = "Paper main matrix V1"
            child.sequential = True
            child.run_id = (
                f"{args.run_id}_d{dimension}_n{int(budget)}")
            output.extend(root.build_specs(child))
    return output


def build_specs(args, registration):
    validate_freeze(registration)
    nodes = _parse_csv(args.nodes)
    if not nodes or any(node not in CPU_NODES for node in nodes):
        raise ValueError("paper jobs must use only node001-node006")
    selected_variants = _parse_csv(args.variants)
    if not selected_variants or any(name not in VARIANTS for name in selected_variants):
        raise ValueError(f"variants must be selected from {VARIANTS}")
    dims = tuple(map(int, _parse_csv(args.dims))) if args.dims else ()
    budgets = tuple(map(int, _parse_csv(args.budgets))) if args.budgets else ()
    frontier = _selected_frontier(registration, dims, budgets)
    design_specs = _design_specs(args, registration, frontier)
    run_specs = _run_specs(args, registration, frontier, selected_variants)
    signatures = [spec["signature"] for spec in design_specs + run_specs]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("paper matrix contains duplicate scheduler signatures")
    return design_specs + run_specs


def main():
    defaults = closure._root_module()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=Path, default=defaults.DEFAULT_SCHEDULER)
    parser.add_argument("--deploy", type=Path, default=defaults.DEFAULT_DEPLOY)
    parser.add_argument("--python", type=Path, default=defaults.REMOTE_PYTHON)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--source-manifest", type=Path,
                        default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--archive-run-id", default=designs.DEFAULT_ARCHIVE_RUN_ID)
    parser.add_argument("--run-id", default=(
        "scolh_paper_main_v1_" + time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--nodes", default=",".join(CPU_NODES))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--dims", default="")
    parser.add_argument("--budgets", default="")
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--variance-audit-size", type=int, default=512)
    parser.add_argument("--misspecification-prior-df", type=float, default=4.0)
    parser.add_argument("--misspecification-ridge", type=float, default=1.0)
    parser.add_argument("--misspecification-max-scale", type=float, default=100.0)
    parser.add_argument("--misspecification-delta", type=float, default=0.05)
    parser.add_argument("--contrast-scale", type=float, default=1.0)
    parser.add_argument("--null-geometry-ridge", type=float, default=1e-3)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--ram-mb", type=int, default=8192)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--inspect-plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registration = load_registration(args.registration)
    if args.inspect_plan:
        print(json.dumps(inspect_plan(registration), indent=2))
        return
    specs = build_specs(args, registration)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    if not args.no_sync:
        subprocess.run([str(defaults.SYNC)], check=True, cwd=ROOT)
    payload = "\n".join(json.dumps(spec) for spec in specs) + "\n"
    subprocess.run(
        [
            sys.executable, str(args.scheduler), "submit-jsonl", "--stdin",
            "--trusted", "--json", "--intent-label", str(args.run_id),
        ],
        input=payload,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
