#!/usr/bin/env python3
"""Evaluate the registered low-frequency support dimension gate.

The gate treats the frozen proposal and joint posterior paths as separate
causal claims.  It reads only ``result.json`` rows through the common matrix
aggregator and compares ``low_frequency_only`` against ``none`` on matched
domain/seed cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_completed_matrix import load_rows
from analyze_causal_prior_effects import build_paired_effects, parse_causal_variant


DEFAULT_DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
DEFAULT_CAUSAL_MODES = ("proposal_only", "joint")


def _csv(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _counts(items, key):
    values = [item.get(key) for item in items if item.get(key) is not None]
    wins = sum(value > 0 for value in values)
    losses = sum(value < 0 for value in values)
    return {
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
        "denominator": len(values),
    }


def evaluate_gate(
        rows,
        *,
        proposal_mode="risk_objective_atlas",
        challenger_profile="low_frequency_only",
        reference_profile="none",
        expected_source_calls=384,
        causal_modes=DEFAULT_CAUSAL_MODES,
        domains=DEFAULT_DOMAINS,
        seeds=range(5)):
    causal_modes = tuple(causal_modes)
    domains = tuple(domains)
    seeds = tuple(int(seed) for seed in seeds)
    profiles = (reference_profile, challenger_profile)

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

    expected = {
        (mode, proposal_mode, profile, domain, seed)
        for mode in causal_modes
        for profile in profiles
        for domain in domains
        for seed in seeds
    }
    missing = sorted(expected.difference(index))
    archive_fingerprints = {
        domain: sorted({
            str(index[key].get("source_archive_fingerprint"))
            for key in expected
            if key in index
            and key[3] == domain
            and index[key].get("source_archive_fingerprint")
        })
        for domain in domains
    }
    source_call_values = sorted({
        int(index[key]["source_calls"])
        for key in expected
        if key in index and index[key].get("source_calls") is not None
    })
    archive_contract_pass = bool(
        all(len(values) == 1 for values in archive_fingerprints.values())
        and source_call_values == [int(expected_source_calls)]
    )

    component = challenger_profile.removesuffix("_only")
    all_pairs = [
        pair for pair in build_paired_effects(rows)
        if pair["proposal_mode"] == proposal_mode
        and pair["contrast"] == "single_only_vs_none"
        and pair["component"] == component
        and pair["domain"] in domains
        and int(pair["seed"]) in seeds
    ]

    mode_results = {}
    expected_per_mode = len(domains) * len(seeds)
    for mode in causal_modes:
        challenger_rows = [
            index.get((mode, proposal_mode, challenger_profile, domain, seed))
            for domain in domains
            for seed in seeds
        ]
        challenger_rows = [row for row in challenger_rows if row is not None]
        pairs = [pair for pair in all_pairs if pair["causal_mode"] == mode]
        per_domain_feasible = {
            domain: sum(
                index.get(
                    (mode, proposal_mode, challenger_profile, domain, seed),
                    {},
                ).get("true_feasible") is True
                for seed in seeds
            )
            for domain in domains
        }
        safety = {
            "overall_feasible": sum(
                row.get("true_feasible") is True for row in challenger_rows),
            "overall_denominator": len(challenger_rows),
            "per_domain_feasible": per_domain_feasible,
            "adaptive_loss_count": sum(
                row.get("adaptive_loss") is True for row in challenger_rows),
        }
        safety["pass"] = bool(
            len(challenger_rows) == expected_per_mode
            and safety["overall_feasible"] >= 12
            and all(value >= 3 for value in per_domain_feasible.values())
            and safety["adaptive_loss_count"] <= 1
        )

        feasibility = _counts(pairs, "final_feasible_delta")
        regret_values = [
            {"regret_win": -pair["final_regret_delta"]}
            for pair in pairs
            if pair.get("final_regret_delta") is not None
        ]
        regret = _counts(regret_values, "regret_win")
        domain_feasibility = {
            domain: _counts(
                [pair for pair in pairs if pair["domain"] == domain],
                "final_feasible_delta",
            )
            for domain in domains
        }
        paired_effect = {
            "n_pairs": len(pairs),
            "archive_match_count": sum(pair["archive_match"] for pair in pairs),
            "initial_fingerprint_match_count": sum(
                pair["initial_fingerprint_match"] for pair in pairs),
            "domain_feasibility": domain_feasibility,
            "feasibility": feasibility,
            "conditional_regret": regret,
        }
        paired_effect["pass"] = bool(
            len(pairs) == expected_per_mode
            and paired_effect["archive_match_count"] == expected_per_mode
            and (
                feasibility["net"] > 0
                or (
                    feasibility["net"] == 0
                    and regret["wins"] > regret["losses"]
                )
            )
        )

        complete = all(
            (mode, proposal_mode, profile, domain, seed) in index
            for profile in profiles
            for domain in domains
            for seed in seeds
        )
        mode_results[mode] = {
            "complete": complete,
            "safety": safety,
            "paired_effect_vs_none": paired_effect,
            "promote": bool(
                complete
                and archive_contract_pass
                and safety["pass"]
                and paired_effect["pass"]
            ),
        }

    promoted_modes = [
        mode for mode, result in mode_results.items() if result["promote"]
    ]
    return {
        "schema_version": 1,
        "proposal_mode": proposal_mode,
        "challenger_profile": challenger_profile,
        "reference_profile": reference_profile,
        "causal_modes": list(causal_modes),
        "domains": list(domains),
        "seeds": list(seeds),
        "expected_cell_count": len(expected),
        "missing_cell_count": len(missing),
        "missing_cells": [list(item) for item in missing],
        "source_archive_fingerprints": archive_fingerprints,
        "expected_source_calls": int(expected_source_calls),
        "source_call_values": source_call_values,
        "archive_contract_pass": archive_contract_pass,
        "mode_results": mode_results,
        "promoted_modes": promoted_modes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--proposal-mode", default="risk_objective_atlas")
    parser.add_argument("--challenger-profile", default="low_frequency_only")
    parser.add_argument("--reference-profile", default="none")
    parser.add_argument("--expected-source-calls", type=int, default=384)
    parser.add_argument("--causal-modes", default=",".join(DEFAULT_CAUSAL_MODES))
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    rows, errors = load_rows(args.roots)
    report = evaluate_gate(
        rows,
        proposal_mode=args.proposal_mode,
        challenger_profile=args.challenger_profile,
        reference_profile=args.reference_profile,
        expected_source_calls=args.expected_source_calls,
        causal_modes=_csv(args.causal_modes),
        domains=_csv(args.domains),
        seeds=range(args.seed_start, args.seed_start + args.n_seeds),
    )
    report["parse_errors"] = errors
    report["pass"] = bool(
        not errors
        and report["missing_cell_count"] == 0
        and report["promoted_modes"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": report["pass"],
        "promoted_modes": report["promoted_modes"],
        "missing_cells": report["missing_cell_count"],
        "parse_errors": len(errors),
    }))


if __name__ == "__main__":
    main()
