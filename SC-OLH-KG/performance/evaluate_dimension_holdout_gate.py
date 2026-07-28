#!/usr/bin/env python3
"""Evaluate the pre-registered dimension-holdout proposal gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_completed_matrix import load_rows
from analyze_causal_prior_effects import (
    build_paired_effects,
    build_proposal_mode_effects,
    parse_causal_variant,
)


DEFAULT_DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)


def _csv(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _counts(items, key):
    values = [item.get(key) for item in items if item.get(key) is not None]
    return {
        "wins": sum(value > 0 for value in values),
        "losses": sum(value < 0 for value in values),
        "net": sum(values),
        "denominator": len(values),
    }


def _proposal_comparison(
        rows, profile, challenger, reference, domains, expected_pairs):
    pairs = [
        item for item in build_proposal_mode_effects(
            rows,
            challenger_mode=challenger,
            reference_mode=reference,
        )
        if item["causal_mode"] == "joint"
        and item["profile"] == profile
        and item["domain"] in domains
    ]
    domain_feasibility = {
        domain: _counts(
            [item for item in pairs if item["domain"] == domain],
            "final_feasible_delta",
        )
        for domain in domains
    }
    feasibility = _counts(pairs, "final_feasible_delta")
    regret = _counts([
        {"regret_win": -item["final_regret_delta"]}
        for item in pairs
        if item.get("final_regret_delta") is not None
    ], "regret_win")
    domain_noninferior = all(
        item["net"] >= 0 for item in domain_feasibility.values())
    lexicographic_pass = (
        len(pairs) == int(expected_pairs)
        and domain_noninferior
        and feasibility["net"] >= 0
        and (
            feasibility["net"] > 0
            or regret["wins"] >= regret["losses"]
        )
    )
    return {
        "reference": reference,
        "n_pairs": len(pairs),
        "domain_feasibility": domain_feasibility,
        "feasibility": feasibility,
        "conditional_regret": regret,
        "pass": bool(lexicographic_pass),
    }


def _structural_effect(
        rows, profile, challenger, domains, expected_pairs):
    component = profile.removesuffix("_only")
    pairs = [
        item for item in build_paired_effects(rows)
        if item["causal_mode"] == "joint"
        and item["proposal_mode"] == challenger
        and item["contrast"] == "single_only_vs_none"
        and item["component"] == component
        and item["domain"] in domains
    ]
    feasibility = _counts(pairs, "final_feasible_delta")
    regret = _counts([
        {"regret_win": -item["final_regret_delta"]}
        for item in pairs
        if item.get("final_regret_delta") is not None
    ], "regret_win")
    passed = (
        len(pairs) == int(expected_pairs)
        and (
            feasibility["net"] > 0
            or (
                feasibility["net"] == 0
                and regret["wins"] > regret["losses"]
            )
        )
    )
    return {
        "n_pairs": len(pairs),
        "feasibility": feasibility,
        "conditional_regret": regret,
        "pass": bool(passed),
    }


def evaluate_gate(
        rows,
        *,
        challenger,
        references,
        profiles,
        domains=DEFAULT_DOMAINS,
        seeds=range(5)):
    domains = tuple(domains)
    seeds = tuple(int(seed) for seed in seeds)
    index = {}
    for row in rows:
        variant = parse_causal_variant(row.get("variant"))
        if variant is None or row.get("status") != "ok":
            continue
        key = (
            variant["causal_mode"],
            variant["proposal_mode"],
            variant["profile"],
            row.get("domain"),
            int(row.get("seed")),
        )
        if key in index:
            raise ValueError(f"duplicate gate cell {key}")
        index[key] = row

    expected_modes = ("proposal_only", "joint")
    expected = {
        (mode, challenger, profile, domain, seed)
        for mode in expected_modes
        for profile in ("none", *profiles)
        for domain in domains
        for seed in seeds
    }
    missing = sorted(expected.difference(index))
    profile_results = {}
    for profile in profiles:
        joint = [
            index.get(("joint", challenger, profile, domain, seed))
            for domain in domains
            for seed in seeds
        ]
        joint = [item for item in joint if item is not None]
        per_domain_feasible = {
            domain: sum(
                index.get(("joint", challenger, profile, domain, seed), {}).get(
                    "true_feasible") is True
                for seed in seeds
            )
            for domain in domains
        }
        safety = {
            "overall_feasible": sum(
                item.get("true_feasible") is True for item in joint),
            "overall_denominator": len(joint),
            "per_domain_feasible": per_domain_feasible,
            "adaptive_loss_count": sum(
                item.get("adaptive_loss") is True for item in joint),
        }
        safety["pass"] = bool(
            safety["overall_feasible"] >= 12
            and all(value >= 3 for value in per_domain_feasible.values())
            and safety["adaptive_loss_count"] <= 1
        )
        comparisons = {
            reference: _proposal_comparison(
                rows,
                profile,
                challenger,
                reference,
                domains,
                len(domains) * len(seeds),
            )
            for reference in references
        }
        structural = _structural_effect(
            rows,
            profile,
            challenger,
            domains,
            len(domains) * len(seeds),
        )
        complete = all(
            (mode, challenger, profile, domain, seed) in index
            for mode in expected_modes
            for domain in domains
            for seed in seeds
        )
        profile_results[profile] = {
            "complete": complete,
            "safety": safety,
            "proposal_comparisons": comparisons,
            "structural_effect_vs_none": structural,
            "promote": bool(
                complete
                and safety["pass"]
                and all(item["pass"] for item in comparisons.values())
                and structural["pass"]
            ),
        }

    return {
        "schema_version": 1,
        "challenger": challenger,
        "references": list(references),
        "profiles": list(profiles),
        "domains": list(domains),
        "seeds": list(seeds),
        "expected_cell_count": len(expected),
        "missing_cell_count": len(missing),
        "missing_cells": [list(item) for item in missing],
        "profiles_result": profile_results,
        "promoted_profiles": [
            profile for profile, item in profile_results.items()
            if item["promote"]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--challenger", default="risk_objective_atlas")
    parser.add_argument(
        "--references", default="risk_coordinate_atlas,rank_spanning")
    parser.add_argument(
        "--profiles", default="additivity_only,orthogonality_only")
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    rows, errors = load_rows(args.roots)
    report = evaluate_gate(
        rows,
        challenger=args.challenger,
        references=_csv(args.references),
        profiles=_csv(args.profiles),
        domains=_csv(args.domains),
        seeds=range(args.seed_start, args.seed_start + args.n_seeds),
    )
    report["parse_errors"] = errors
    report["pass"] = bool(
        not errors
        and report["missing_cell_count"] == 0
        and len(report["promoted_profiles"]) > 0
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": report["pass"],
        "promoted_profiles": report["promoted_profiles"],
        "missing_cells": report["missing_cell_count"],
        "parse_errors": len(errors),
    }))


if __name__ == "__main__":
    main()
