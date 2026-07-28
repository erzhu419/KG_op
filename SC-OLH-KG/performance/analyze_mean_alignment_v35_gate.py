#!/usr/bin/env python3
"""Analyze the V35 predictive/confidence covariance authority gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import analyze_mean_alignment_v34_gate as base
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v34_gate as base


VARIANTS = (
    "v29_scale_control",
    "v34_joint_control",
    "v35_confidence",
    "v35_confidence_task",
)
CHALLENGERS = VARIANTS[2:]
MISSPECIFICATION_MODES = {
    "v29_scale_control": "predictive_scale",
    "v34_joint_control": "predictive_scale_sandwich_hc3",
    "v35_confidence": "predictive_scale_sandwich_hc3_confidence",
    "v35_confidence_task": (
        "predictive_scale_sandwich_hc3_task_confidence"),
}


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


def _present_variants(rows):
    present = {str(row.get("gate_variant", "")) for row in rows}
    return tuple(variant for variant in VARIANTS if variant in present)


def _paired_online_contract(rows, variants, expected_seeds):
    paired_design = True
    paired_responses = True
    randomness_contract = True
    canonical_sobol = True
    for scenario in base.MEAN_SCENARIOS:
        scenario_rows = [
            row for row in rows if base._scenario(row) == scenario
        ]
        control_rows = [
            row for row in scenario_rows
            if row.get("gate_variant") == "v29_scale_control"
        ]
        seeds = sorted(set(int(row.get("seed", -1)) for row in control_rows))
        if len(seeds) != int(expected_seeds):
            paired_design = False
        for seed in seeds:
            group = [
                row for row in scenario_rows
                if int(row.get("seed", -1)) == seed
                and row.get("gate_variant") in variants
            ]
            if len(group) != len(variants):
                paired_design = False
                paired_responses = False
                randomness_contract = False
                canonical_sobol = False
                continue
            design_fingerprints = {
                str(row.get("target_design_fingerprint", "missing"))
                for row in group
            }
            action_fingerprints = {
                str(row.get(
                    "online_action_sequence_fingerprint", "missing"))
                for row in group
            }
            paired_design &= bool(
                len(design_fingerprints) == 1
                and "missing" not in design_fingerprints
                and len(action_fingerprints) == 1
                and "missing" not in action_fingerprints
            )
            traces = [list(row.get("online_action_trace") or []) for row in group]
            expected_online = max(
                0,
                int(group[0].get("N", 0)) - int(group[0].get("n0", 0)),
            )
            paired_responses &= bool(
                traces
                and expected_online > 0
                and len(traces[0]) == expected_online
                and len({len(trace) for trace in traces}) == 1
                and all(
                    all(
                        left.get("x_fingerprint")
                        == right.get("x_fingerprint")
                        and left.get("observed_response")
                        == right.get("observed_response")
                        for left, right in zip(traces[0], trace)
                    )
                    for trace in traces[1:]
                )
            )
            for row, trace in zip(group, traces):
                simulation = dict(
                    row.get("simulation_randomness_contract") or {})
                proposal = dict(row.get("proposal_randomness_contract") or {})
                randomness_contract &= bool(
                    simulation.get("proposal_rng_independent", False)
                    and simulation.get(
                        "common_random_numbers_by_evaluation_index", False)
                    and not simulation.get("target_oracle_used", True)
                    and proposal.get("component_streams_independent", False)
                    and proposal.get("simulation_rng_independent", False)
                    and not proposal.get("target_oracle_used", True)
                    and not row.get(
                        "online_action_trace_target_oracle_used", True)
                )
                canonical_sobol &= bool(
                    row.get("decision_backend") == "sobol_new"
                    and trace
                    and all(
                        step.get("candidate_source") == "sobol_continuation"
                        for step in trace
                    )
                )
    return {
        "paired_complete_online_design": bool(paired_design),
        "paired_common_random_responses": bool(paired_responses),
        "independent_random_stream_contract": bool(randomness_contract),
        "canonical_sobol_continuation_contract": bool(canonical_sobol),
    }


def summarize(rows, expected_seeds=5, variants=None):
    selected = tuple(variants or _present_variants(rows))
    if "v29_scale_control" not in selected:
        raise ValueError("V35 analysis requires the v29_scale_control arm")
    challengers = tuple(
        variant for variant in selected if variant.startswith("v35_")
    )
    if not challengers:
        raise ValueError("V35 analysis requires at least one V35 challenger")
    summary = base.summarize(
        rows,
        expected_seeds,
        variants=selected,
        challengers=challengers,
        misspecification_modes={
            variant: MISSPECIFICATION_MODES[variant]
            for variant in selected
        },
        complexity_order={
            "v35_confidence": 0,
            "v35_confidence_task": 1,
        },
    )
    online_contract = _paired_online_contract(
        rows, selected, expected_seeds)
    for variant in challengers:
        summary["variant_checks"][variant].update(online_contract)
    summary["randomness_and_design_contract"] = online_contract
    summary["decision_expansion_eligible"] = [
        variant for variant in challengers
        if all(
            value for key, value in summary["variant_checks"][variant].items()
            if key != "nonvacuous_true_certificate_coverage"
        )
    ]
    summary["promotion_eligible"] = [
        variant for variant in challengers
        if all(summary["variant_checks"][variant].values())
    ]
    complexity = {
        "v35_confidence": 0,
        "v35_confidence_task": 1,
    }
    eligible = summary["promotion_eligible"]
    summary["post_gate_baseline_recommendation"] = (
        min(
            eligible,
            key=lambda variant: (
                -summary["totals"][variant][
                    "audit_pool_true_certificates"],
                complexity[variant],
            ),
        )
        if eligible else None
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument(
        "--variants",
        help=(
            "comma-separated submitted arms; defaults to the arms actually "
            "present under ROOT"
        ),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.root)
    variants = (
        tuple(value.strip() for value in args.variants.split(",") if value.strip())
        if args.variants
        else None
    )
    summary = summarize(rows, args.expected_seeds, variants=variants)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
