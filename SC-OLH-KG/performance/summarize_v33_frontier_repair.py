"""Summarize the preregistered V33 coherent-frontier repair matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
VARIANT_CONFIGS = {
    ("legacy", "legacy", "legacy", 4, "legacy"): "v32",
    (
        "terminal_kg_1step", "certified_only", "legacy", 4, "legacy",
    ): "v33_legacy_4",
    (
        "terminal_kg_1step", "off", "coverage_reserved", 4,
        "certified_lexicographic",
    ): "v33_coherent_coverage_4",
    (
        "terminal_kg_1step", "off", "coverage_reserved", 8,
        "certified_lexicographic",
    ): "v33_coherent_coverage_8",
}
VARIANTS = tuple(VARIANT_CONFIGS.values())
EXPECTED_SEEDS = tuple(range(7))
PRIMARY_BASELINE = "v32"
PRIMARY_CHALLENGER = "v33_coherent_coverage_8"
MANDATORY_FRONTIER_LABELS = (
    "minimum_bayes_risk",
    "minimum_certificate_margin",
    "minimum_robust_expected_violation",
    "minimum_nominal_expected_violation",
)


def _median(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.median(values))


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _result_files(inputs):
    files = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("result.json")))
        elif path.is_file():
            files.append(path)
    return files


def _variant(row):
    key = (
        str(row.get("finalist_replication_policy", "legacy")),
        str(row.get("finalist_empirical_override", "legacy")),
        str(row.get("finalist_frontier_policy", "legacy")),
        int(row.get("finalist_terminal_max_arms", 4)),
        str(row.get("decision_contract_mode", "legacy")),
    )
    if key not in VARIANT_CONFIGS:
        raise ValueError(f"unregistered V33 repair configuration {key}")
    return VARIANT_CONFIGS[key]


def load_rows(inputs):
    rows = []
    seen = set()
    for path in _result_files(inputs):
        payload = json.loads(path.read_text())
        payload_rows = payload.get("rows") or []
        if len(payload_rows) != 1:
            raise ValueError(f"one-seed shard expected in {path}")
        row = dict(payload_rows[0])
        config = payload.get("config") or {}
        for field in (
            "finalist_replication_policy",
            "finalist_empirical_override",
            "finalist_frontier_policy",
            "finalist_terminal_max_arms",
            "decision_contract_mode",
        ):
            row.setdefault(field, config.get(field))
        variant = _variant(row)
        heldout = str(row["heldout"])
        seed = int(row["seed"])
        key = (variant, heldout, seed)
        if key in seen:
            raise ValueError(f"duplicate V33 repair shard {key}")
        if heldout not in DOMAINS:
            raise ValueError(f"unregistered V33 repair domain {heldout}")
        seen.add(key)
        row["_variant"] = variant
        row["_path"] = str(path)
        rows.append(row)
    return rows


def _terminal_rows(finalist):
    return [
        row for row in (finalist.get("terminal_kg_rows") or [])
        if isinstance(row, dict)
    ]


def _group_summary(variant, heldout, rows):
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    coherent = variant.startswith("v33_coherent_")
    frontier_labels = []
    terminal_rows = []
    for row in rows:
        finalist = row.get("finalist_replication") or {}
        frontier_labels.append(tuple(finalist.get("labels") or []))
        terminal_rows.extend(_terminal_rows(finalist))
    mandatory_coverage = [
        all(label in labels for label in MANDATORY_FRONTIER_LABELS)
        for labels in frontier_labels
    ]
    coherent_contract = [
        bool((row.get("finalist_replication") or {}).get(
            "coherent_three_layer_contract", False))
        for row in rows
    ]
    frontier_policy = [
        str((row.get("finalist_replication") or {}).get(
            "frontier_policy", "legacy"))
        for row in rows
    ]
    terminal_modes = [
        str(row.get("terminal_kg_value_mode", "model_default"))
        for row in terminal_rows
    ]
    return {
        "variant": variant,
        "heldout": heldout,
        "n_seeds": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "true_feasible_count": int(sum(
            bool(row.get("true_feasible", False)) for row in rows)),
        "posterior_certified_count": int(sum(
            bool(row.get("posterior_feasible", False)) for row in rows)),
        "false_feasible_count": int(sum(
            bool(row.get("false_feasible", False)) for row in rows)),
        "median_feasible_regret": _median(
            row.get("feasible_simple_regret") for row in rows),
        "mean_constraint_violation": _mean(
            row.get("constraint_violation") for row in rows),
        "median_true_chance_margin": _median(
            row.get("true_chance_margin") for row in rows),
        "median_algorithm_time_sec": _median(
            row.get("algorithm_time_sec") for row in rows),
        "median_wall_time_sec": _median(
            row.get("wall_time_sec") for row in rows),
        "mandatory_frontier_coverage_count": int(sum(mandatory_coverage)),
        "mandatory_frontier_coverage_rate": float(np.mean(
            mandatory_coverage)) if mandatory_coverage else 0.0,
        "coherent_contract_count": int(sum(coherent_contract)),
        "coverage_reserved_count": int(sum(
            policy == "coverage_reserved" for policy in frontier_policy)),
        "terminal_kg_evaluation_count": int(len(terminal_rows)),
        "terminal_lexicographic_count": int(sum(
            mode == "certified_lexicographic" for mode in terminal_modes)),
        "replicated_finalist_used_count": int(sum(
            bool(row.get("replicated_finalist_used", False)) for row in rows)),
        "target_oracle_used_count": int(sum(
            bool((row.get("finalist_replication") or {}).get(
                "target_oracle_used", False))
            for row in rows)),
        "coherent_contract_audit_pass": bool(
            not coherent or (
                len(rows) == 7
                and all(mandatory_coverage)
                and all(coherent_contract)
                and all(policy == "coverage_reserved" for policy in frontier_policy)
                and len(terminal_rows) == 21
                and all(
                    mode == "certified_lexicographic"
                    for mode in terminal_modes
                )
                and not any(bool(row.get(
                    "replicated_finalist_used", False)) for row in rows)
            )
        ),
    }


def _regret_nonworse(challenger, baseline, tolerance=1e-12):
    left = challenger.get("median_feasible_regret")
    right = baseline.get("median_feasible_regret")
    return bool(
        left is not None and right is not None
        and float(left) <= float(right) + tolerance
    )


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["_variant"], str(row["heldout"]))
        groups.setdefault(key, []).append(row)
    summaries = [
        _group_summary(variant, heldout, group)
        for (variant, heldout), group in sorted(groups.items())
    ]
    by_key = {
        (row["variant"], row["heldout"]): row for row in summaries
    }
    expected = {
        (variant, domain) for variant in VARIANTS for domain in DOMAINS
    }
    complete = {
        key for key, row in by_key.items()
        if tuple(row["seeds"]) == EXPECTED_SEEDS
    }
    domain_checks = []
    comparisons = []
    for domain in DOMAINS:
        baseline = by_key.get((PRIMARY_BASELINE, domain))
        challenger = by_key.get((PRIMARY_CHALLENGER, domain))
        available = baseline is not None and challenger is not None
        domain_checks.append({
            "heldout": domain,
            "complete": bool(
                available
                and tuple(baseline["seeds"]) == EXPECTED_SEEDS
                and tuple(challenger["seeds"]) == EXPECTED_SEEDS),
            "true_feasible_nonworse": bool(
                available and challenger["true_feasible_count"]
                >= baseline["true_feasible_count"]),
            "false_feasible_nonworse": bool(
                available and challenger["false_feasible_count"]
                <= baseline["false_feasible_count"]),
            "median_feasible_regret_nonworse": bool(
                available and _regret_nonworse(challenger, baseline)),
            "coherent_contract_audit": bool(
                available and challenger["coherent_contract_audit_pass"]),
        })
        comparisons.append({
            "heldout": domain,
            "true_feasible_count_delta": (
                None if not available else
                challenger["true_feasible_count"]
                - baseline["true_feasible_count"]),
            "false_feasible_count_delta": (
                None if not available else
                challenger["false_feasible_count"]
                - baseline["false_feasible_count"]),
            "median_feasible_regret_delta": (
                None if not available
                or baseline["median_feasible_regret"] is None
                or challenger["median_feasible_regret"] is None
                else challenger["median_feasible_regret"]
                - baseline["median_feasible_regret"]),
            "median_algorithm_time_ratio": (
                None if not available
                or not baseline["median_algorithm_time_sec"]
                or challenger["median_algorithm_time_sec"] is None
                else challenger["median_algorithm_time_sec"]
                / baseline["median_algorithm_time_sec"]),
        })

    challenger_groups = [
        by_key.get((PRIMARY_CHALLENGER, domain)) for domain in DOMAINS
    ]
    target_feasibility = bool(
        all(group is not None for group in challenger_groups)
        and challenger_groups[0]["true_feasible_count"] == 7
        and challenger_groups[1]["true_feasible_count"] >= 5
        and challenger_groups[2]["true_feasible_count"] >= 5
    )
    gate = {
        "baseline": PRIMARY_BASELINE,
        "challenger": PRIMARY_CHALLENGER,
        "matrix_complete": complete == expected,
        "target_feasibility_passed": target_feasibility,
        "per_domain_nonworse_passed": bool(all(
            all(value for key, value in check.items() if key != "heldout")
            for check in domain_checks
        )),
        "no_target_oracle_passed": bool(all(
            row["target_oracle_used_count"] == 0 for row in summaries)),
        "coherent_contract_passed": bool(
            all(
                group is not None and group["coherent_contract_audit_pass"]
                for group in challenger_groups
            )),
        "domain_checks": domain_checks,
    }
    gate["passed"] = bool(
        gate["matrix_complete"]
        and gate["target_feasibility_passed"]
        and gate["per_domain_nonworse_passed"]
        and gate["no_target_oracle_passed"]
        and gate["coherent_contract_passed"]
    )
    return {
        "schema_version": 1,
        "preregistered_primary": {
            "baseline": PRIMARY_BASELINE,
            "challenger": PRIMARY_CHALLENGER,
            "domains": list(DOMAINS),
            "seeds": list(EXPECTED_SEEDS),
            "mandatory_frontier_labels": list(MANDATORY_FRONTIER_LABELS),
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "primary_gate": gate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    payload = summarize(load_rows(args.inputs))
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
