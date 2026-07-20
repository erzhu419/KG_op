#!/usr/bin/env python3
"""Analyze the V44 terminal Bayes-risk penalty diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import analyze_mean_alignment_v43_gate as base
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v43_gate as base


VARIANTS = ("v41_rho5", "v44_rho20", "v44_rho80")
PENALTIES = {
    "v41_rho5": 5.0,
    "v44_rho20": 20.0,
    "v44_rho80": 80.0,
}
SENTINEL_SEEDS = (1, 3)


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("result.json")):
        payload = json.loads(path.read_text())
        experiment = str(payload.get("experiment_variant", ""))
        variant = next(
            (name for name in VARIANTS if f"/{name}/" in f"/{experiment}/"),
            None,
        )
        if variant is None:
            continue
        for raw in payload.get("rows", []):
            row = dict(raw)
            row["gate_variant"] = variant
            rows.append(row)
    return rows


def _variant_summary(rows, variant, expected_count):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    regrets = np.asarray([
        (
            np.nan if row.get("feasible_simple_regret") is None
            else float(row["feasible_simple_regret"])
        )
        for row in selected
    ], dtype=float)
    regrets = regrets[np.isfinite(regrets)]
    return {
        "count": len(selected),
        "complete": len(selected) == int(expected_count),
        "true_feasible": int(sum(bool(
            row.get("true_feasible", False)) for row in selected)),
        "adaptive_losses": int(sum(bool(
            row.get("adaptive_loss", False)) for row in selected)),
        "adaptive_improvements": int(sum(bool(
            row.get("adaptive_improves_initial_best", False))
            for row in selected)),
        "median_feasible_regret": (
            float(np.median(regrets)) if len(regrets) else None),
        "posterior_certificate_count": int(sum(
            int((row.get("certificate_outcome_audit") or {}).get(
                "certified_true_feasible_count", 0) or 0)
            for row in selected)),
        "false_certificate_count": int(sum(
            int((row.get("certificate_outcome_audit") or {}).get(
                "false_certificate_count", 0) or 0)
            for row in selected)),
    }


def _paired_contract(rows, seeds):
    for seed in seeds:
        group = [
            row for row in rows
            if base._scenario(row) == base.QUEUE_SCENARIO
            and int(row.get("seed", -1)) == int(seed)
            and row.get("gate_variant") in VARIANTS
        ]
        if len(group) != len(VARIANTS):
            return False
        for field in (
            "source_archive_fingerprint",
            "target_design_fingerprint",
            "online_action_sequence_fingerprint",
        ):
            values = {base._paired_fingerprint(row, field) for row in group}
            if len(values) != 1 or "missing" in values:
                return False
        reference = list(group[0].get("online_action_trace") or [])
        for row in group:
            trace = list(row.get("online_action_trace") or [])
            if len(trace) != len(reference):
                return False
            if any(
                left.get("x_fingerprint") != right.get("x_fingerprint")
                or left.get("observed_response") != right.get("observed_response")
                or right.get("candidate_source") != "sobol_continuation"
                for left, right in zip(reference, trace)
            ):
                return False
    return True


def _penalty_contract(rows, variant):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    expected = PENALTIES[variant]
    return bool(selected) and all(
        np.isclose(float(row.get("decision_risk_penalty", np.nan)), expected)
        and row.get("decision_backend") == "sobol_new"
        and not bool((row.get("adaptive_replication_voi") or {}).get(
            "enabled", True))
        for row in selected
    )


def summarize(rows, seeds=SENTINEL_SEEDS):
    seeds = tuple(int(seed) for seed in seeds)
    selected = [
        row for row in rows
        if base._scenario(row) == base.QUEUE_SCENARIO
        and int(row.get("seed", -1)) in seeds
    ]
    summaries = {
        variant: _variant_summary(selected, variant, len(seeds))
        for variant in VARIANTS
    }
    control = summaries["v41_rho5"]
    penalty_contract = {
        variant: _penalty_contract(selected, variant) for variant in VARIANTS
    }
    improvements = [
        variant for variant in VARIANTS[1:]
        if summaries[variant]["true_feasible"] > control["true_feasible"]
        or summaries[variant]["adaptive_losses"] < control["adaptive_losses"]
        or summaries[variant]["adaptive_improvements"]
        > control["adaptive_improvements"]
    ]
    clean = [
        variant for variant in improvements
        if summaries[variant]["true_feasible"] == len(seeds)
        and summaries[variant]["adaptive_losses"] == 0
        and summaries[variant]["false_certificate_count"] == 0
    ]
    return {
        "scope": "terminal_loss_mechanism_diagnostic",
        "seeds": list(seeds),
        "paired_archive_design_actions_and_responses": _paired_contract(
            selected, seeds),
        "decision_penalty_contract": penalty_contract,
        "variant_summaries": summaries,
        "higher_penalty_improves_control": improvements,
        "source_dual_learning_warranted": bool(clean),
        "clean_mechanism_signals": clean,
        "promotion_eligible": [],
        "promotion_note": (
            "Fixed target-sentinel penalty sweeps are mechanism diagnostics; "
            "only a source-learned frozen dual law may be promoted."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", default="1,3")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds = tuple(
        int(value.strip()) for value in args.seeds.split(",") if value.strip())
    result = summarize(load_rows(args.root), seeds)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
