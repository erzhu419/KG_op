#!/usr/bin/env python3
"""Audit the paired dimension-by-budget frontier before seed expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
PROPOSAL_TRACK = (
    "dimension_budget_frontier_gate_proposal_d200_d1000_n10")
SAAS_TRACKS = {
    20: "dimension_budget_frontier_gate_saas_d200_d1000_n20",
    40: "dimension_budget_frontier_gate_saas_d200_d1000_n40",
    80: "dimension_budget_frontier_gate_saas_d200_d1000_n80",
}
MATCH_FIELDS = (
    "source_archive_fingerprint",
    "initial_design_fingerprint",
    "problem_contract_fingerprint",
    "verifier_signature",
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_csv(value, cast=str):
    return tuple(
        cast(item.strip())
        for item in str(value).split(",")
        if item.strip()
    )


def _finite_regret(row):
    value = row.get("feasible_regret")
    if value is None or not np.isfinite(float(value)):
        return float("inf")
    return float(value)


def _frontier_row(record):
    track_id = str(record.get("track_id", ""))
    if track_id == PROPOSAL_TRACK:
        phase = "proposal"
        budget = 10
    else:
        matching = [
            budget for budget, expected_track in SAAS_TRACKS.items()
            if track_id == expected_track
        ]
        if not matching:
            return None
        phase = "saas"
        budget = matching[0]
    if record.get("status") != "ok":
        return {
            "phase": phase,
            "budget": budget,
            "dimension": int(record["target_dimension"]),
            "domain": str(record["domain"]),
            "seed": int(record["seed"]),
            "status": str(record.get("status")),
            "failure_type": record.get("failure_type"),
        }
    return {
        "phase": phase,
        "budget": budget,
        "dimension": int(record["target_dimension"]),
        "domain": str(record["domain"]),
        "seed": int(record["seed"]),
        "status": "ok",
        "true_feasible": bool(record["true_feasible"]),
        "terminal_certified": bool(
            record.get("terminal_certified", False)),
        "false_certificate": bool(
            record.get("false_certificate", False)),
        "feasible_regret": _finite_regret(record),
        **{
            field: record.get(field)
            for field in MATCH_FIELDS
        },
    }


def _summary(rows):
    finite_regrets = [
        row["feasible_regret"] for row in rows
        if row["status"] == "ok"
        and np.isfinite(row["feasible_regret"])
    ]
    return {
        "row_count": len(rows),
        "ok_count": sum(row["status"] == "ok" for row in rows),
        "true_feasible_count": sum(
            row.get("true_feasible", False) for row in rows),
        "certified_count": sum(
            row.get("terminal_certified", False) for row in rows),
        "false_certificate_count": sum(
            row.get("false_certificate", False) for row in rows),
        "median_feasible_regret": (
            None
            if not finite_regrets
            else float(np.median(finite_regrets))
        ),
    }


def analyze_rows(
    rows,
    *,
    dimensions,
    budgets,
    domains,
    seeds,
    evidence_source,
):
    rows = [dict(row) for row in rows]
    by_key = {}
    duplicates = []
    for row in rows:
        key = (
            row["dimension"],
            row["domain"],
            row["seed"],
            row["budget"],
        )
        if key in by_key:
            duplicates.append(key)
        by_key[key] = row

    expected = {
        (dimension, domain, seed, budget)
        for dimension in dimensions
        for domain in domains
        for seed in seeds
        for budget in (10, *budgets)
    }
    missing = sorted(expected - set(by_key))
    unexpected = sorted(set(by_key) - expected)
    pair_rows = []
    contract_mismatches = []
    summaries = {}
    gates = {}

    for dimension in dimensions:
        proposal_rows = [
            by_key[(dimension, domain, seed, 10)]
            for domain in domains
            for seed in seeds
            if (dimension, domain, seed, 10) in by_key
        ]
        summaries[f"d{dimension}_N10"] = _summary(proposal_rows)
        for budget in budgets:
            current_rows = [
                by_key[(dimension, domain, seed, budget)]
                for domain in domains
                for seed in seeds
                if (dimension, domain, seed, budget) in by_key
            ]
            summaries[f"d{dimension}_N{budget}"] = _summary(current_rows)
            adaptive_rescue = 0
            adaptive_loss = 0
            paired_regret_deltas = []
            complete_pairs = 0
            for domain in domains:
                for seed in seeds:
                    proposal = by_key.get(
                        (dimension, domain, seed, 10))
                    current = by_key.get(
                        (dimension, domain, seed, budget))
                    if proposal is None or current is None:
                        continue
                    complete_pairs += 1
                    mismatched = [
                        field for field in MATCH_FIELDS
                        if proposal.get(field) != current.get(field)
                    ]
                    if mismatched:
                        contract_mismatches.append({
                            "dimension": dimension,
                            "domain": domain,
                            "seed": seed,
                            "budget": budget,
                            "fields": mismatched,
                        })
                    if (
                        proposal["status"] == "ok"
                        and current["status"] == "ok"
                    ):
                        initial_feasible = proposal["true_feasible"]
                        final_feasible = current["true_feasible"]
                        adaptive_rescue += int(
                            not initial_feasible and final_feasible)
                        adaptive_loss += int(
                            initial_feasible and not final_feasible)
                        if initial_feasible and final_feasible:
                            paired_regret_deltas.append(
                                proposal["feasible_regret"]
                                - current["feasible_regret"])
                    pair_rows.append({
                        "dimension": dimension,
                        "domain": domain,
                        "seed": seed,
                        "budget": budget,
                        "adaptive_rescue": int(
                            proposal.get("true_feasible", False) is False
                            and current.get("true_feasible", False) is True
                        ),
                        "adaptive_loss": int(
                            proposal.get("true_feasible", False) is True
                            and current.get("true_feasible", False) is False
                        ),
                        "proposal_minus_final_regret": (
                            None
                            if not (
                                proposal.get("true_feasible", False)
                                and current.get("true_feasible", False)
                            )
                            else float(
                                proposal["feasible_regret"]
                                - current["feasible_regret"])
                        ),
                    })
            proposal_summary = summaries[f"d{dimension}_N10"]
            current_summary = summaries[f"d{dimension}_N{budget}"]
            median_delta = (
                None
                if not paired_regret_deltas
                else float(np.median(paired_regret_deltas))
            )
            gates[f"d{dimension}_N{budget}"] = {
                "dimension_over_search_budget": (
                    float(dimension) / float(budget)),
                "complete_pair_count": complete_pairs,
                "adaptive_rescue_count": adaptive_rescue,
                "adaptive_loss_count": adaptive_loss,
                "median_proposal_minus_final_regret": median_delta,
                "all_rows_ok": (
                    current_summary["ok_count"] == len(domains) * len(seeds)),
                "false_certification_free": (
                    current_summary["false_certificate_count"] == 0),
                "feasibility_not_harmed": (
                    current_summary["true_feasible_count"]
                    >= proposal_summary["true_feasible_count"]),
                "regret_not_harmed": (
                    median_delta is not None and median_delta >= -1e-12),
            }

    complete_contract = not (
        duplicates or missing or unexpected or contract_mismatches)
    expand = bool(
        complete_contract
        and all(
            gate["all_rows_ok"]
            and gate["false_certification_free"]
            and gate["adaptive_loss_count"] == 0
            and gate["feasibility_not_harmed"]
            and gate["regret_not_harmed"]
            for gate in gates.values()
        )
    )
    return {
        "schema_version": 1,
        "status": "complete" if complete_contract else "incomplete",
        "evidence_source": str(evidence_source),
        "expected": {
            "dimensions": list(dimensions),
            "budgets": [10, *budgets],
            "domains": list(domains),
            "seeds": list(seeds),
            "row_count": len(expected),
        },
        "observed_row_count": len(rows),
        "duplicate_keys": duplicates,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "contract_mismatches": contract_mismatches,
        "summaries": summaries,
        "pair_audits": pair_rows,
        "gates": gates,
        "expand_to_20_seeds": expand,
    }


def analyze_record_shards(
    paths,
    *,
    dimensions,
    budgets,
    domains,
    seeds,
):
    rows = []
    receipts = []
    relevant_tracks = {PROPOSAL_TRACK} | {
        SAAS_TRACKS[budget] for budget in budgets
    }
    for path in map(Path, paths):
        payload = _read_json(path)
        selected = [
            record for record in payload.get("records", ())
            if record.get("track_id") in relevant_tracks
        ]
        rows.extend(
            row for record in selected
            if (row := _frontier_row(record)) is not None
        )
        receipts.append({
            "path": str(path),
            "sha256": _sha256(path),
            "origin": payload.get("origin"),
            "selected_record_count": len(selected),
            "policy_vectors_exported": payload.get(
                "policy_vectors_exported"),
            "raw_checkpoints_or_model_weights_read": payload.get(
                "raw_checkpoints_or_model_weights_read"),
        })
    report = analyze_rows(
        rows,
        dimensions=dimensions,
        budgets=budgets,
        domains=domains,
        seeds=seeds,
        evidence_source="remote_compact_record_shards",
    )
    report["record_shard_receipts"] = receipts
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record-shard",
        action="append",
        required=True,
    )
    parser.add_argument("--dimensions", default="200,1000")
    parser.add_argument("--budgets", default="20,40,80")
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = analyze_record_shards(
        args.record_shard,
        dimensions=_parse_csv(args.dimensions, int),
        budgets=_parse_csv(args.budgets, int),
        domains=_parse_csv(args.domains),
        seeds=tuple(range(
            args.seed_start,
            args.seed_start + args.n_seeds,
        )),
    )
    _atomic_json(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "observed_row_count": report["observed_row_count"],
        "expand_to_20_seeds": report["expand_to_20_seeds"],
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
