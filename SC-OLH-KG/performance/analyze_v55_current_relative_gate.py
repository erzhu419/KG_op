#!/usr/bin/env python3
"""Audit V55 current-relative joint-improvement gates.

MC512 is an empirical numerical reference, not an exact integral.  The Lean
result remains conditional on the nested common-random-number radii covering
the exact score errors of the selected action.
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
LOW_VARIANT = "v55_mc128"
REFERENCE_VARIANT = "v55_mc512"
KNOWN_VARIANTS = (LOW_VARIANT, REFERENCE_VARIANT)


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


def _contract(row, expected_mc):
    contract = dict(row.get("decision_backend_contract") or {})
    return bool(
        str(row.get("implementation_contract_id"))
        == "v55_current_relative_joint_guard"
        and str(row.get("theory_contract_id"))
        == "v55_current_relative_joint_improvement_v1"
        and int(row.get("exact_kg_mc_samples", -1)) == int(expected_mc)
        and str(row.get("exact_kg_sampling_mode"))
        == "factorized_rqmc_nested"
        and str(contract.get("policy_improvement_score_transform"))
        == "bounded_current_gain"
        and str(contract.get("policy_improvement_guard_mode"))
        == "paired_nested_absolute"
        and bool(contract.get("exact_kg_joint_terminal_reuse", False))
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def _action_data(trace, require_prefix=True):
    if trace is None:
        return None
    fingerprints = list(
        trace.get("exact_kg_active_action_fingerprints") or [])
    replicates = list(
        trace.get("exact_kg_active_action_is_replicate") or [])
    labels = list(trace.get("exact_kg_active_action_labels") or [])
    if not fingerprints or len(set(fingerprints)) != len(fingerprints):
        return None
    if len(replicates) != len(fingerprints):
        return None
    if labels and len(labels) != len(fingerprints):
        return None
    if not labels:
        labels = ["unlabeled"] * len(fingerprints)
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
    risk_scale = _finite(
        trace.get("policy_improvement_risk_score_scale")) or 1.0
    certificate_scale = _finite(
        trace.get("policy_improvement_certificate_score_scale")) or 1.0
    if risk_scale <= 0.0 or certificate_scale <= 0.0:
        return None
    for field in ("risk", "risk_prefix"):
        if field in arrays:
            arrays[field] = {
                key: value / risk_scale
                for key, value in arrays[field].items()
            }
    for field in ("certificate", "certificate_prefix"):
        if field in arrays:
            arrays[field] = {
                key: value / certificate_scale
                for key, value in arrays[field].items()
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
        "selected": str(trace.get("x_fingerprint")),
        "risk_scale": risk_scale,
        "certificate_scale": certificate_scale,
        **arrays,
    }


def _baseline(data, baseline_new_count=4):
    order = list(data["order"])
    new = [item for item in order if not data["replicate"][item]]
    replicate = [item for item in order if data["replicate"][item]]
    baseline = new[:int(baseline_new_count)] + replicate
    if not baseline:
        raise ValueError("empty literal V51 baseline action set")
    index = {item: position for position, item in enumerate(order)}
    fallback = max(
        baseline,
        key=lambda item: (data["risk_raw"][item], -index[item]),
    )
    return fallback


def audit_trace(trace, multiplier=1.25, baseline_new_count=4):
    data = _action_data(trace, require_prefix=True)
    if data is None:
        return None
    order_index = {
        item: position for position, item in enumerate(data["order"])}
    fallback = _baseline(data, baseline_new_count=baseline_new_count)
    risk_radius = {}
    certificate_radius = {}
    risk_lcb = {}
    certificate_lcb = {}
    joint_lcb = {}
    for item in data["order"]:
        risk_radius[item] = float(multiplier * abs(
            data["risk"][item] - data["risk_prefix"][item]))
        certificate_radius[item] = float(multiplier * abs(
            data["certificate"][item]
            - data["certificate_prefix"][item]))
        risk_lcb[item] = data["risk"][item] - risk_radius[item]
        certificate_lcb[item] = (
            data["certificate"][item] - certificate_radius[item])
        joint_lcb[item] = min(
            risk_lcb[item], certificate_lcb[item])
    admissible = [
        item for item in data["order"] if joint_lcb[item] > 0.0]
    expected = (
        max(
            admissible,
            key=lambda item: (
                joint_lcb[item],
                risk_lcb[item] + certificate_lcb[item],
                certificate_lcb[item],
                -order_index[item],
            ),
        )
        if admissible else fallback
    )
    selected = data["selected"]
    return {
        "selected_fingerprint": selected,
        "expected_selected_fingerprint": expected,
        "fallback_fingerprint": fallback,
        "selector_matches_trace": selected == expected,
        "selected_is_joint_admissible": selected in admissible,
        "selected_is_fallback": selected == fallback,
        "admissible_fingerprints": admissible,
        "admissible_count": len(admissible),
        "selected_label": data["label"].get(selected),
        "selected_risk_lcb": risk_lcb.get(selected),
        "selected_certificate_lcb": certificate_lcb.get(selected),
        "selected_joint_lcb": joint_lcb.get(selected),
        "selected_risk_radius": risk_radius.get(selected),
        "selected_certificate_radius": certificate_radius.get(selected),
        "risk_lcb": risk_lcb,
        "certificate_lcb": certificate_lcb,
        "joint_lcb": joint_lcb,
        "risk_radius": risk_radius,
        "certificate_radius": certificate_radius,
        "data": data,
    }


def analyze_activation(
    root,
    seeds=range(0, 5),
    multiplier=1.25,
    minimum_selected_per_domain=3,
    baseline_new_count=4,
):
    rows, errors = load_rows(root)
    seeds = tuple(map(int, seeds))
    expected = {(domain, seed) for domain in DOMAINS for seed in seeds}
    indexed = {
        _key(row): row for row in rows
        if row.get("gate_variant") == LOW_VARIANT
        and _key(row) in expected
    }
    contracts_ok = True
    action_sets_ok = True
    selector_ok = True
    audits = []
    for key in sorted(expected & set(indexed)):
        row = indexed[key]
        contracts_ok &= _contract(row, 128)
        audit = audit_trace(
            _trace(row),
            multiplier=multiplier,
            baseline_new_count=baseline_new_count,
        )
        if audit is None:
            action_sets_ok = False
            continue
        selector_ok &= bool(audit["selector_matches_trace"])
        audit.pop("data", None)
        audits.append({"heldout": key[0], "seed": key[1], **audit})
    selected_by_domain = {
        domain: sum(
            audit["selected_is_joint_admissible"]
            for audit in audits if audit["heldout"] == domain
        )
        for domain in DOMAINS
    }
    complete = bool(
        len(indexed) == len(expected)
        and not errors
        and contracts_ok
        and action_sets_ok
        and selector_ok
        and len(audits) == len(expected)
    )
    passes = bool(
        complete
        and all(
            selected_by_domain[domain]
            >= int(minimum_selected_per_domain)
            for domain in DOMAINS
        )
    )
    return {
        "scope": "v55_current_relative_mc32_mc128_activation",
        "variant": LOW_VARIANT,
        "prefix_mc_samples": 32,
        "high_mc_samples": 128,
        "safety_multiplier": float(multiplier),
        "seeds": list(seeds),
        "expected_count": len(expected),
        "row_count": len(indexed),
        "errors": errors,
        "contracts_ok": contracts_ok,
        "action_sets_ok": action_sets_ok,
        "selector_reconstruction_ok": selector_ok,
        "minimum_selected_joint_admissible_per_domain": int(
            minimum_selected_per_domain),
        "selected_joint_admissible_by_domain": selected_by_domain,
        "selected_joint_admissible_total": sum(
            selected_by_domain.values()),
        "activation_gate_complete": complete,
        "activation_gate_pass": passes,
        "formal_contract": (
            "positive nested-CRN LCBs imply current-relative reduction of "
            "both terminal costs when the exact errors are covered"),
        "audits": audits,
    }


def _load_index(root, variant, expected):
    rows, errors = load_rows(root)
    return ({
        _key(row): row for row in rows
        if row.get("gate_variant") == variant
        and _key(row) in expected
    }, errors)


def analyze_triple_fidelity(
    low_root,
    reference_root,
    seeds=range(0, 5),
    multiplier=1.25,
    baseline_new_count=4,
):
    seeds = tuple(map(int, seeds))
    expected = {(domain, seed) for domain in DOMAINS for seed in seeds}
    low_rows, low_errors = _load_index(low_root, LOW_VARIANT, expected)
    reference_rows, reference_errors = _load_index(
        reference_root, REFERENCE_VARIANT, expected)
    paired = expected & set(low_rows) & set(reference_rows)
    contracts_ok = True
    initial_designs_ok = True
    action_sets_ok = True
    selector_ok = True
    prefix_identity_ok = True
    selected_coverage = []
    all_coverage = []
    selected_reference_regressions = 0
    selected_joint_actions = 0
    audits = []
    for key in sorted(paired):
        low_row = low_rows[key]
        reference_row = reference_rows[key]
        contracts_ok &= (
            _contract(low_row, 128)
            and _contract(reference_row, 512)
        )
        initial_designs_ok &= _initial_design_matches(
            low_row, reference_row)
        low_audit = audit_trace(
            _trace(low_row),
            multiplier=multiplier,
            baseline_new_count=baseline_new_count,
        )
        reference = _action_data(_trace(reference_row), require_prefix=True)
        if low_audit is None or reference is None:
            action_sets_ok = False
            continue
        low = low_audit["data"]
        same_actions = bool(
            low["order"] == reference["order"]
            and low["replicate"] == reference["replicate"]
        )
        action_sets_ok &= same_actions
        if not same_actions:
            continue
        selector_ok &= bool(low_audit["selector_matches_trace"])
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
        coverage = {}
        for item in low["order"]:
            covered = bool(
                abs(reference["risk"][item] - low["risk"][item])
                <= low_audit["risk_radius"][item] + 1e-15
                and abs(
                    reference["certificate"][item]
                    - low["certificate"][item])
                <= low_audit["certificate_radius"][item] + 1e-15
            )
            coverage[item] = covered
            all_coverage.append(covered)
        selected = low_audit["selected_fingerprint"]
        selected_is_covered = bool(coverage[selected])
        selected_coverage.append(selected_is_covered)
        selected_reference_joint = bool(
            reference["risk"][selected] > 0.0
            and reference["certificate"][selected] > 0.0)
        if low_audit["selected_is_joint_admissible"]:
            selected_joint_actions += 1
            if not selected_reference_joint:
                selected_reference_regressions += 1
        audits.append({
            "heldout": key[0],
            "seed": key[1],
            "selected_fingerprint": selected,
            "selected_label": low_audit["selected_label"],
            "selected_is_joint_admissible_mc128": (
                low_audit["selected_is_joint_admissible"]),
            "selected_pair_radius_covered_by_mc512": selected_is_covered,
            "selected_mc512_joint_improvement": selected_reference_joint,
            "selected_risk_score_mc128": low["risk"][selected],
            "selected_risk_score_mc512": reference["risk"][selected],
            "selected_certificate_score_mc128": (
                low["certificate"][selected]),
            "selected_certificate_score_mc512": (
                reference["certificate"][selected]),
            "selected_risk_radius": low_audit["risk_radius"][selected],
            "selected_certificate_radius": (
                low_audit["certificate_radius"][selected]),
            "all_action_joint_coverage_rate": statistics.fmean(
                map(float, coverage.values())),
            "prefix_risk_max_abs_difference": prefix_risk_error,
            "prefix_certificate_max_abs_difference": (
                prefix_certificate_error),
        })
    complete = bool(
        len(paired) == len(expected)
        and not low_errors
        and not reference_errors
        and contracts_ok
        and initial_designs_ok
        and action_sets_ok
        and selector_ok
        and prefix_identity_ok
        and len(audits) == len(expected)
    )
    selected_coverage_rate = (
        None if not selected_coverage
        else statistics.fmean(map(float, selected_coverage)))
    all_coverage_rate = (
        None if not all_coverage
        else statistics.fmean(map(float, all_coverage)))
    passes = bool(
        complete
        and selected_joint_actions > 0
        and selected_coverage_rate == 1.0
        and all_coverage_rate is not None
        and all_coverage_rate >= 0.95
        and selected_reference_regressions == 0
    )
    return {
        "scope": "v55_current_relative_mc32_mc128_mc512",
        "low_variant": LOW_VARIANT,
        "reference_variant": REFERENCE_VARIANT,
        "prefix_mc_samples": 32,
        "low_mc_samples": 128,
        "reference_mc_samples": 512,
        "safety_multiplier": float(multiplier),
        "seeds": list(seeds),
        "expected_pair_count": len(expected),
        "paired_count": len(paired),
        "low_errors": low_errors,
        "reference_errors": reference_errors,
        "contracts_ok": contracts_ok,
        "paired_initial_designs_ok": initial_designs_ok,
        "identical_active_action_sets": action_sets_ok,
        "selector_reconstruction_ok": selector_ok,
        "nested_prefix_identity_ok": prefix_identity_ok,
        "selected_joint_action_count": selected_joint_actions,
        "selected_pair_coverage_rate_against_mc512": selected_coverage_rate,
        "all_action_joint_coverage_rate_against_mc512": all_coverage_rate,
        "selected_reference_regression_count": (
            selected_reference_regressions),
        "triple_fidelity_gate_complete": complete,
        "triple_fidelity_gate_pass": passes,
        "bound_status": (
            "mc512_is_empirical_reference_not_exact_integration"),
        "formal_contract": (
            "Lean theorem remains conditional on exact per-action error "
            "coverage"),
        "audits": audits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("activation", "triple_fidelity"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--low-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--multiplier", type=float, default=1.25)
    parser.add_argument("--minimum-selected-per-domain", type=int, default=3)
    parser.add_argument("--baseline-new-count", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    if args.mode == "activation":
        if args.root is None:
            parser.error("--root is required for activation")
        result = analyze_activation(
            args.root,
            seeds,
            multiplier=args.multiplier,
            minimum_selected_per_domain=args.minimum_selected_per_domain,
            baseline_new_count=args.baseline_new_count,
        )
    else:
        if args.low_root is None or args.reference_root is None:
            parser.error(
                "--low-root and --reference-root are required for "
                "triple_fidelity")
        result = analyze_triple_fidelity(
            args.low_root,
            args.reference_root,
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
