#!/usr/bin/env python3
"""Audit V54 action support and nested pair-difference fidelity.

The MC512 layer is an empirical numerical reference, not an exact integral.
Lean's V54 theorem remains conditional on the selected pair radius covering
the exact fallback-relative score error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
ACTION_SUPPORT_VARIANT = "v53_mc128"
LOW_VARIANT = "v54_mc128"
REFERENCE_VARIANT = "v54_mc512"
KNOWN_VARIANTS = (LOW_VARIANT, REFERENCE_VARIANT, ACTION_SUPPORT_VARIANT)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _variant(experiment):
    marker = f"/{str(experiment).strip('/')}/"
    return next(
        (name for name in KNOWN_VARIANTS if f"/{name}/" in marker),
        None,
    )


def load_rows(root):
    rows = []
    errors = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            variant = _variant(payload.get("experiment_variant", ""))
            if variant is None:
                continue
            for raw in payload["rows"]:
                row = dict(raw)
                row["gate_variant"] = variant
                row["result_path"] = str(path)
                rows.append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return rows, errors


def _key(row):
    return str(row.get("heldout")), int(row.get("seed", -1))


def _trace(row):
    traces = list(row.get("online_action_trace") or [])
    return None if not traces else dict(traces[0])


def _initial_design_matches(left, right):
    first = dict(left.get("task_initial_design") or {})
    second = dict(right.get("task_initial_design") or {})
    return bool(
        first.get("fingerprint") is not None
        and first.get("fingerprint") == second.get("fingerprint")
        and first.get("source_archive_fingerprint")
        == second.get("source_archive_fingerprint")
        and int(first.get("n_unique", -1)) == 10
        and int(second.get("n_unique", -1)) == 10
    )


def _action_data(trace, require_prefix=False):
    if trace is None:
        return None
    fingerprints = list(
        trace.get("exact_kg_active_action_fingerprints") or [])
    replicates = list(trace.get("exact_kg_active_action_is_replicate") or [])
    fields = {
        "risk_raw": "exact_kg_raw_scores_active",
        "risk": "exact_kg_policy_scores_active",
        "certificate": "certificate_deficit_policy_scores_active",
    }
    if require_prefix:
        fields.update({
            "risk_prefix": "pairwise_prefix_risk_policy_scores_active",
            "certificate_prefix": (
                "pairwise_prefix_certificate_policy_scores_active"),
        })
    if not fingerprints or len(set(fingerprints)) != len(fingerprints):
        return None
    if len(replicates) != len(fingerprints):
        return None
    labels = list(trace.get("exact_kg_active_action_labels") or [])
    if labels and len(labels) != len(fingerprints):
        return None
    if not labels:
        labels = ["unlabeled"] * len(fingerprints)
    arrays = {}
    for name, field in fields.items():
        values = list(trace.get(field) or [])
        if len(values) != len(fingerprints):
            return None
        finite = [_finite(value) for value in values]
        if any(value is None for value in finite):
            return None
        arrays[name] = {
            str(fingerprint): float(value)
            for fingerprint, value in zip(fingerprints, finite)
        }
    return {
        "order": [str(value) for value in fingerprints],
        "replicate": {
            str(fingerprint): bool(value)
            for fingerprint, value in zip(fingerprints, replicates)
        },
        "label": {
            str(fingerprint): str(value)
            for fingerprint, value in zip(fingerprints, labels)
        },
        **arrays,
    }


def _baseline(data, baseline_new_count=4):
    order = list(data["order"])
    new = [item for item in order if not data["replicate"][item]]
    replicate = [item for item in order if data["replicate"][item]]
    baseline = new[:int(baseline_new_count)] + replicate
    if not baseline:
        raise ValueError("empty literal V51 baseline action set")
    index = {fingerprint: position for position, fingerprint in enumerate(order)}
    fallback = max(
        baseline,
        key=lambda item: (data["risk_raw"][item], -index[item]),
    )
    supplemental = new[int(baseline_new_count):]
    return fallback, baseline, supplemental


def _contract(row, expected_mc, paired):
    contract = dict(row.get("decision_backend_contract") or {})
    expected_implementation = (
        "v54_paired_nested_difference_guard"
        if paired
        else "v53_constrained_certificate_deficit_bounded_gain"
    )
    expected_theory = (
        "v54_paired_difference_guard_v1"
        if paired
        else "v53_constrained_certificate_deficit_v3"
    )
    return bool(
        str(row.get("implementation_contract_id"))
        == expected_implementation
        and str(row.get("theory_contract_id")) == expected_theory
        and int(row.get("exact_kg_mc_samples", -1)) == int(expected_mc)
        and str(row.get("exact_kg_sampling_mode"))
        == "factorized_rqmc_nested"
        and str(contract.get("policy_improvement_score_transform"))
        == "bounded_current_gain"
        and str(contract.get(
            "policy_improvement_guard_mode", "uniform_score"))
        == ("paired_nested_difference" if paired else "uniform_score")
        and (
            not paired
            or bool(contract.get("exact_kg_joint_terminal_reuse", False))
        )
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def analyze_action_support(
    root,
    seeds=range(0, 5),
    variant=ACTION_SUPPORT_VARIANT,
    baseline_new_count=4,
):
    rows, errors = load_rows(root)
    seeds = tuple(map(int, seeds))
    expected = {(domain, seed) for domain in DOMAINS for seed in seeds}
    indexed = {
        _key(row): row for row in rows
        if row.get("gate_variant") == str(variant)
        and _key(row) in expected
    }
    audits = []
    contracts_ok = True
    action_sets_ok = True
    for key in sorted(expected & set(indexed)):
        row = indexed[key]
        contracts_ok &= _contract(row, 128, paired=False)
        data = _action_data(_trace(row), require_prefix=False)
        if data is None:
            action_sets_ok = False
            continue
        fallback, baseline, supplemental = _baseline(
            data, baseline_new_count=baseline_new_count)
        risk0 = data["risk"][fallback]
        certificate0 = data["certificate"][fallback]
        joint = [
            item for item in supplemental
            if data["risk"][item] > risk0
            and data["certificate"][item] > certificate0
        ]
        all_joint = [
            item for item in data["order"]
            if item != fallback
            and data["risk"][item] > risk0
            and data["certificate"][item] > certificate0
        ]
        audits.append({
            "heldout": key[0],
            "seed": key[1],
            "action_count": len(data["order"]),
            "baseline_action_count": len(baseline),
            "supplemental_action_count": len(supplemental),
            "fallback_fingerprint": fallback,
            "supplemental_joint_dominator_count": len(joint),
            "supplemental_joint_dominator_fingerprints": joint,
            "supplemental_joint_dominator_labels": [
                data["label"][item] for item in joint
            ],
            "supplemental_action_audit": [
                {
                    "fingerprint": item,
                    "label": data["label"][item],
                    "risk_advantage": data["risk"][item] - risk0,
                    "certificate_advantage": (
                        data["certificate"][item] - certificate0),
                    "joint_dominator": bool(item in joint),
                }
                for item in supplemental
            ],
            "all_joint_dominator_count": len(all_joint),
            "max_supplemental_risk_advantage": (
                None if not supplemental else max(
                    data["risk"][item] - risk0 for item in supplemental)
            ),
            "max_supplemental_certificate_advantage": (
                None if not supplemental else max(
                    data["certificate"][item] - certificate0
                    for item in supplemental)
            ),
        })
    domains_with_joint = sorted({
        audit["heldout"] for audit in audits
        if audit["supplemental_joint_dominator_count"] > 0
    })
    complete = bool(
        len(indexed) == len(expected)
        and not errors
        and contracts_ok
        and action_sets_ok
        and len(audits) == len(expected)
    )
    return {
        "scope": "v54_action_support_mc128",
        "variant": str(variant),
        "seeds": list(seeds),
        "expected_count": len(expected),
        "row_count": len(indexed),
        "errors": errors,
        "contracts_ok": contracts_ok,
        "action_sets_ok": action_sets_ok,
        "supplemental_joint_dominator_cells": sum(
            audit["supplemental_joint_dominator_count"] > 0
            for audit in audits),
        "domains_with_supplemental_joint_dominator": domains_with_joint,
        "action_support_gate_complete": complete,
        "action_support_gate_pass": bool(
            complete and len(domains_with_joint) == len(DOMAINS)),
        "formal_status": (
            "diagnostic_only_not_v54_selector_evidence"),
        "audits": audits,
    }


def analyze_triple_fidelity(
    root,
    seeds=range(0, 5),
    multiplier=1.25,
    baseline_new_count=4,
):
    rows, errors = load_rows(root)
    seeds = tuple(map(int, seeds))
    expected = {(domain, seed) for domain in DOMAINS for seed in seeds}
    indexed = {
        variant: {
            _key(row): row for row in rows
            if row.get("gate_variant") == variant
            and _key(row) in expected
        }
        for variant in (LOW_VARIANT, REFERENCE_VARIANT)
    }
    paired = expected & set(indexed[LOW_VARIANT]) & set(
        indexed[REFERENCE_VARIANT])
    contracts_ok = True
    initial_designs_ok = True
    action_sets_ok = True
    fallback_stable = True
    prefix_identity_ok = True
    audits = []
    all_coverage = []
    selected_coverage = []
    selected_reference_regressions = 0
    switch_count = 0
    for key in sorted(paired):
        low_row = indexed[LOW_VARIANT][key]
        reference_row = indexed[REFERENCE_VARIANT][key]
        contracts_ok &= (
            _contract(low_row, 128, paired=True)
            and _contract(reference_row, 512, paired=True)
        )
        initial_designs_ok &= _initial_design_matches(low_row, reference_row)
        low_trace = _trace(low_row)
        reference_trace = _trace(reference_row)
        low = _action_data(low_trace, require_prefix=True)
        reference = _action_data(reference_trace, require_prefix=True)
        if low is None or reference is None:
            action_sets_ok = False
            continue
        same_actions = bool(
            set(low["order"]) == set(reference["order"])
            and low["replicate"] == reference["replicate"]
        )
        action_sets_ok &= same_actions
        if not same_actions:
            continue
        fallback, _, supplemental = _baseline(
            low, baseline_new_count=baseline_new_count)
        reference_fallback, _, _ = _baseline(
            reference, baseline_new_count=baseline_new_count)
        fallback_stable &= fallback == reference_fallback
        prefix_risk_error = max(
            abs(low["risk_prefix"][item]
                - reference["risk_prefix"][item])
            for item in low["order"])
        prefix_certificate_error = max(
            abs(low["certificate_prefix"][item]
                - reference["certificate_prefix"][item])
            for item in low["order"])
        prefix_identity_ok &= bool(
            prefix_risk_error <= 1e-12
            and prefix_certificate_error <= 1e-12)
        risk_radius = {}
        certificate_radius = {}
        risk_delta_128 = {}
        certificate_delta_128 = {}
        risk_delta_512 = {}
        certificate_delta_512 = {}
        action_coverage = {}
        for item in low["order"]:
            d32_risk = (
                low["risk_prefix"][item]
                - low["risk_prefix"][fallback])
            d128_risk = low["risk"][item] - low["risk"][fallback]
            d512_risk = (
                reference["risk"][item]
                - reference["risk"][fallback])
            d32_certificate = (
                low["certificate_prefix"][item]
                - low["certificate_prefix"][fallback])
            d128_certificate = (
                low["certificate"][item] - low["certificate"][fallback])
            d512_certificate = (
                reference["certificate"][item]
                - reference["certificate"][fallback])
            risk_delta_128[item] = d128_risk
            certificate_delta_128[item] = d128_certificate
            risk_delta_512[item] = d512_risk
            certificate_delta_512[item] = d512_certificate
            risk_radius[item] = float(
                multiplier * abs(d128_risk - d32_risk))
            certificate_radius[item] = float(
                multiplier * abs(d128_certificate - d32_certificate))
            covered = bool(
                abs(d512_risk - d128_risk) <= risk_radius[item] + 1e-15
                and abs(d512_certificate - d128_certificate)
                <= certificate_radius[item] + 1e-15
            )
            action_coverage[item] = covered
            all_coverage.append(covered)
        risk_admissible = [
            item for item in low["order"]
            if item == fallback
            or risk_delta_128[item] > risk_radius[item]
        ]
        order_index = {
            item: position for position, item in enumerate(low["order"])}
        challenger = max(
            risk_admissible,
            key=lambda item: (
                low["certificate"][item], -order_index[item]),
        )
        switched = bool(
            challenger != fallback
            and certificate_delta_128[challenger]
            > certificate_radius[challenger]
        )
        selected = challenger if switched else fallback
        switch_count += int(switched)
        selected_is_covered = action_coverage[selected]
        selected_coverage.append(selected_is_covered)
        reference_joint_improvement = bool(
            risk_delta_512[selected] > 0.0
            and certificate_delta_512[selected] > 0.0
        )
        if switched and not reference_joint_improvement:
            selected_reference_regressions += 1
        supplemental_joint = [
            item for item in supplemental
            if risk_delta_512[item] > 0.0
            and certificate_delta_512[item] > 0.0
        ]
        audits.append({
            "heldout": key[0],
            "seed": key[1],
            "action_count": len(low["order"]),
            "fallback_fingerprint": fallback,
            "reference_fallback_fingerprint": reference_fallback,
            "prefix_risk_max_abs_difference": prefix_risk_error,
            "prefix_certificate_max_abs_difference": (
                prefix_certificate_error),
            "selected_fingerprint": selected,
            "switched": switched,
            "selected_pair_radius_covered_by_mc512": selected_is_covered,
            "selected_mc512_joint_improvement": (
                reference_joint_improvement),
            "selected_risk_delta_mc128": risk_delta_128[selected],
            "selected_risk_delta_mc512": risk_delta_512[selected],
            "selected_certificate_delta_mc128": (
                certificate_delta_128[selected]),
            "selected_certificate_delta_mc512": (
                certificate_delta_512[selected]),
            "selected_risk_radius": risk_radius[selected],
            "selected_certificate_radius": certificate_radius[selected],
            "all_action_pair_coverage_rate": statistics.fmean(
                map(float, action_coverage.values())),
            "supplemental_mc512_joint_dominator_count": len(
                supplemental_joint),
        })
    complete = bool(
        len(paired) == len(expected)
        and not errors
        and contracts_ok
        and initial_designs_ok
        and action_sets_ok
        and fallback_stable
        and prefix_identity_ok
        and len(audits) == len(expected)
    )
    selected_coverage_rate = (
        None if not selected_coverage
        else statistics.fmean(map(float, selected_coverage))
    )
    all_coverage_rate = (
        None if not all_coverage
        else statistics.fmean(map(float, all_coverage))
    )
    passes = bool(
        complete
        and switch_count > 0
        and selected_coverage_rate == 1.0
        and selected_reference_regressions == 0
    )
    return {
        "scope": "v54_nested_pair_difference_mc32_mc128_mc512",
        "low_variant": LOW_VARIANT,
        "reference_variant": REFERENCE_VARIANT,
        "prefix_mc_samples": 32,
        "low_mc_samples": 128,
        "reference_mc_samples": 512,
        "safety_multiplier": float(multiplier),
        "seeds": list(seeds),
        "expected_pair_count": len(expected),
        "paired_count": len(paired),
        "errors": errors,
        "contracts_ok": contracts_ok,
        "paired_initial_designs_ok": initial_designs_ok,
        "identical_active_action_sets": action_sets_ok,
        "fallback_fingerprint_stable": fallback_stable,
        "nested_prefix_identity_ok": prefix_identity_ok,
        "switch_count": switch_count,
        "selected_pair_coverage_rate_against_mc512": (
            selected_coverage_rate),
        "all_action_pair_coverage_rate_against_mc512": all_coverage_rate,
        "selected_reference_regression_count": (
            selected_reference_regressions),
        "triple_fidelity_gate_complete": complete,
        "triple_fidelity_gate_pass": passes,
        "bound_status": (
            "mc512_is_empirical_reference_not_exact_integration"),
        "formal_contract": (
            "Lean theorem remains conditional on exact pair-error coverage"),
        "audits": audits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("action_support", "triple_fidelity"),
        required=True,
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--multiplier", type=float, default=1.25)
    parser.add_argument("--baseline-new-count", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    if args.mode == "action_support":
        result = analyze_action_support(
            args.root,
            seeds,
            baseline_new_count=args.baseline_new_count,
        )
    else:
        result = analyze_triple_fidelity(
            args.root,
            seeds,
            multiplier=args.multiplier,
            baseline_new_count=args.baseline_new_count,
        )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
