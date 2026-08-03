#!/usr/bin/env python3
"""Freeze the final dimension/budget evidence with stratified seed roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
PAIR_FIELDS = (
    "source_archive_fingerprint",
    "initial_design_fingerprint",
    "problem_contract_fingerprint",
    "verifier_signature",
)
PROPOSAL = "frozen_crossdim_proposal_only"
SAAS = "canonical_saasbo_every_iteration"
SPECS = {
    200: {
        "evidence_role": "exploratory_low_dimension_sentinel",
        "seeds": tuple(range(80, 85)),
        "required_execution_commit": None,
        "cells": {
            10: ("dimension_frontier_d200_d10000_n13", PROPOSAL),
            13: ("dimension_frontier_d200_d10000_n13", SAAS),
        },
    },
    1000: {
        "evidence_role": "primary_statistical_dimension",
        "seeds": tuple(range(80, 100)),
        "required_execution_commit": (
            "27d55e0d5f265034f91ee7b3f7988dd8233881e5"
        ),
        "cells": {
            10: ("final_frozen_source_frontend_backend_d1000_n13", PROPOSAL),
            13: ("final_frozen_source_frontend_backend_d1000_n13", SAAS),
        },
    },
    10000: {
        "evidence_role": "extreme_dimension_stress_test",
        "seeds": tuple(range(80, 90)),
        "required_execution_commit": (
            "da01f40dfce175cd9bb723ba680ef6c719ec7ad6"
        ),
        "cells": {
            10: ("final_dimension_frontier_d10000_proposal_n10", PROPOSAL),
            13: ("final_dimension_frontier_d10000_saas_n13", SAAS),
            20: ("final_dimension_frontier_d10000_saas_n20", SAAS),
            40: ("final_dimension_frontier_d10000_saas_n40", SAAS),
        },
    },
}


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


def _finite_regrets(rows):
    return [
        float(row["feasible_regret"])
        for row in rows
        if row.get("true_feasible") and row.get("feasible_regret") is not None
    ]


def _cell_summary(rows, dimension, budget, spec):
    regrets = _finite_regrets(rows)
    verification = [int(row["target_verification_calls"]) for row in rows]
    source_calls = {int(row["source_calls"]) for row in rows}
    search_calls = {int(row["target_search_calls"]) for row in rows}
    execution_commits = {
        row.get("execution_repository_commit") for row in rows
    }
    required_commit = spec["required_execution_commit"]
    return {
        "dimension": int(dimension),
        "target_search_calls": int(budget),
        "dimension_over_target_search_calls": float(dimension) / float(budget),
        "result_count": len(rows),
        "all_rows_ok": all(row.get("status") == "ok" for row in rows),
        "true_feasible_count": sum(bool(row.get("true_feasible")) for row in rows),
        "certified_count": sum(bool(row.get("terminal_certified")) for row in rows),
        "false_certificate_count": sum(bool(row.get("false_certificate")) for row in rows),
        "median_feasible_regret": (
            None if not regrets else float(statistics.median(regrets))
        ),
        "mean_target_verification_calls": float(statistics.mean(verification)),
        "median_target_verification_calls": float(statistics.median(verification)),
        "source_calls": sorted(source_calls),
        "observed_target_search_calls": sorted(search_calls),
        "execution_repository_commits": sorted(
            "unregistered" if value is None else str(value)
            for value in execution_commits
        ),
        "frozen_execution_provenance_required": required_commit is not None,
        "frozen_execution_provenance_pass": (
            True
            if required_commit is None
            else execution_commits == {required_commit}
        ),
    }


def analyze(audit):
    if audit.get("status") != "pass":
        raise ValueError("dimension evidence requires a passed paper audit")
    records = list(audit.get("records", ()))
    selected = {}
    duplicates = []
    unexpected = []
    for dimension, spec in SPECS.items():
        for budget, (track, method) in spec["cells"].items():
            for row in records:
                if (
                    row.get("track_id") != track
                    or row.get("method_identity") != method
                    or int(row.get("target_dimension", -1)) != dimension
                ):
                    continue
                key = (dimension, budget, str(row["domain"]), int(row["seed"]))
                if key in selected:
                    duplicates.append(key)
                selected[key] = row
    expected = {
        (dimension, budget, domain, seed)
        for dimension, spec in SPECS.items()
        for budget in spec["cells"]
        for domain in DOMAINS
        for seed in spec["seeds"]
    }
    missing = sorted(expected - set(selected))
    unexpected.extend(sorted(set(selected) - expected))
    cells = {}
    pair_audits = {}
    contract_mismatches = []
    for dimension, spec in SPECS.items():
        proposal_budget = 10
        for budget in spec["cells"]:
            rows = [
                selected[(dimension, budget, domain, seed)]
                for domain in DOMAINS
                for seed in spec["seeds"]
                if (dimension, budget, domain, seed) in selected
            ]
            cells[f"d{dimension}_N{budget}"] = _cell_summary(
                rows, dimension, budget, spec
            )
            if budget == proposal_budget:
                continue
            rescue = 0
            loss = 0
            regret_deltas = []
            pair_count = 0
            for domain in DOMAINS:
                for seed in spec["seeds"]:
                    base = selected.get((dimension, proposal_budget, domain, seed))
                    current = selected.get((dimension, budget, domain, seed))
                    if base is None or current is None:
                        continue
                    pair_count += 1
                    mismatched = [
                        field for field in PAIR_FIELDS
                        if base.get(field) != current.get(field)
                    ]
                    if mismatched:
                        contract_mismatches.append({
                            "dimension": dimension,
                            "budget": budget,
                            "domain": domain,
                            "seed": seed,
                            "fields": mismatched,
                        })
                    before = bool(base.get("true_feasible"))
                    after = bool(current.get("true_feasible"))
                    rescue += int(not before and after)
                    loss += int(before and not after)
                    if before and after:
                        regret_deltas.append(
                            float(base["feasible_regret"])
                            - float(current["feasible_regret"])
                        )
            pair_audits[f"d{dimension}_N10_to_N{budget}"] = {
                "complete_pair_count": pair_count,
                "adaptive_rescue_count": rescue,
                "adaptive_loss_count": loss,
                "median_proposal_minus_final_regret": (
                    None
                    if not regret_deltas
                    else float(statistics.median(regret_deltas))
                ),
            }

    failures = []
    if duplicates:
        failures.append("duplicate result cells")
    if missing:
        failures.append("missing result cells")
    if unexpected:
        failures.append("unexpected result cells")
    if contract_mismatches:
        failures.append("proposal/backend pairing contract drifted")
    for name, cell in cells.items():
        if not cell["all_rows_ok"]:
            failures.append(f"{name} has failed rows")
        if cell["false_certificate_count"]:
            failures.append(f"{name} has false certificates")
        if cell["source_calls"] != [384]:
            failures.append(f"{name} source budget drifted")
        if cell["observed_target_search_calls"] != [
            cell["target_search_calls"]
        ]:
            failures.append(f"{name} target budget drifted")
        if not cell["frozen_execution_provenance_pass"]:
            failures.append(f"{name} execution provenance drifted")

    release_cells = {
        name: cell
        for name, cell in cells.items()
        if cell["dimension"] in {1000, 10000}
    }
    return {
        "schema_version": 1,
        "status": "complete" if not failures else "incomplete",
        "contract_id": "stratified_final_dimension_budget_evidence_v1",
        "audit_registry_id": audit.get("registry_id"),
        "evidence_roles": {
            str(dimension): spec["evidence_role"]
            for dimension, spec in SPECS.items()
        },
        "headline_dimensions": [1000, 10000],
        "exploratory_dimensions": [200],
        "headline_seed_counts": {"1000": 20, "10000": 10},
        "exploratory_seed_counts": {"200": 5},
        "cells": cells,
        "pair_audits": pair_audits,
        "release_cell_count": len(release_cells),
        "all_release_rows_ok": all(
            row["all_rows_ok"] for row in release_cells.values()
        ),
        "all_release_rows_false_certificate_free": all(
            row["false_certificate_count"] == 0
            for row in release_cells.values()
        ),
        "all_release_rows_frozen": all(
            row["frozen_execution_provenance_pass"]
            for row in release_cells.values()
        ),
        "d200_is_descriptive_not_confirmatory": True,
        "failures": failures,
        "duplicate_keys": duplicates,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "contract_mismatches": contract_mismatches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = analyze(_read_json(args.audit))
    report["input_audit"] = {
        "path": str(args.audit),
        "sha256": _sha256(args.audit),
    }
    _atomic_json(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "release_cell_count": report["release_cell_count"],
        "missing_cell_count": len(report["missing_keys"]),
        "out": str(args.out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
