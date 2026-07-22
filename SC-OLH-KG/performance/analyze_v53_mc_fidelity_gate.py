#!/usr/bin/env python3
"""Calibrate V53 nested-MC score errors from paired MC8/MC32 runs."""

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
LOW = "v53_mc8"
HIGH = "v53_mc32"
MC_VARIANTS = ("v53_mc8", "v53_mc32", "v53_mc128")
VARIANTS = (LOW, HIGH)
RQMC_MODES = {
    "factorized_rqmc_nested",
    "rqmc_expert_nested",
    "nested_rqmc_expert",
}


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _variant(experiment):
    marker = f"/{str(experiment).strip('/')}/"
    return next((
        variant for variant in MC_VARIANTS
        if f"/{variant}/" in marker
    ), None)


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


def _score_map(trace, field):
    fingerprints = list(trace.get("exact_kg_active_action_fingerprints") or [])
    values = list(trace.get(field) or [])
    if len(fingerprints) != len(values) or not fingerprints:
        return None
    mapped = {}
    for fingerprint, value in zip(fingerprints, values):
        finite = _finite(value)
        if finite is None or fingerprint in mapped:
            return None
        mapped[str(fingerprint)] = finite
    return mapped


def _top_agrees(left, right):
    if not left or set(left) != set(right):
        return False
    return max(left, key=lambda key: (left[key], key)) == max(
        right, key=lambda key: (right[key], key))


def _pairwise_agreement(left, right):
    if not left or set(left) != set(right):
        return None
    keys = sorted(left)
    agree = total = 0
    for i, first in enumerate(keys):
        for second in keys[i + 1:]:
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            if abs(left_delta) <= 1e-15 or abs(right_delta) <= 1e-15:
                continue
            total += 1
            agree += int((left_delta > 0.0) == (right_delta > 0.0))
    return 1.0 if total == 0 else float(agree / total)


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


def _contract(
    row,
    expected_mc,
    expected_sampling_mode,
    score_normalization,
):
    contract = dict(row.get("decision_backend_contract") or {})
    normalized = str(score_normalization) == "current_terminal"
    expected_implementation = (
        "v53_constrained_certificate_deficit_normalized"
        if normalized
        else "v53_constrained_certificate_deficit"
    )
    expected_theory = (
        "v53_constrained_certificate_deficit_v2"
        if normalized
        else "v53_constrained_certificate_deficit_v1"
    )
    actual_normalization = str(
        contract.get("policy_improvement_score_normalization", "none"))
    return bool(
        str(row.get("implementation_contract_id"))
        == expected_implementation
        and str(row.get("theory_contract_id")) == expected_theory
        and int(row.get("exact_kg_mc_samples", -1)) == int(expected_mc)
        and str(row.get("exact_kg_sampling_mode"))
        == str(expected_sampling_mode)
        and str(contract.get("policy_improvement_contract"))
        == expected_theory
        and actual_normalization == str(score_normalization)
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def _terminal_scale(trace, field):
    value = trace.get(field)
    if value is None:
        return None
    pending = [value]
    finite = []
    while pending:
        item = pending.pop()
        if isinstance(item, (list, tuple)):
            pending.extend(item)
            continue
        number = _finite(item)
        if number is not None:
            finite.append(abs(number))
    return None if not finite else max(1.0, max(finite))


def _scale_map(values, scale):
    return {
        key: float(value) / float(scale)
        for key, value in values.items()
    }


def analyze(
    root,
    seeds=range(0, 10),
    multiplier=1.25,
    sampling_mode="antithetic_nested",
    score_normalization="none",
    low_variant=LOW,
    high_variant=HIGH,
    low_mc=8,
    high_mc=32,
):
    rows, errors = load_rows(root)
    seeds = tuple(int(seed) for seed in seeds)
    expected = {(domain, seed) for domain in DOMAINS for seed in seeds}
    variants = (str(low_variant), str(high_variant))
    if len(set(variants)) != 2:
        raise ValueError("low and high fidelity variants must differ")
    indexed = {
        variant: {
            _key(row): row for row in rows
            if row.get("gate_variant") == variant
            and _key(row) in expected
        }
        for variant in variants
    }

    risk_errors = []
    certificate_errors = []
    risk_top = []
    certificate_top = []
    risk_pairwise = []
    certificate_pairwise = []
    low_selector_l1 = []
    high_selector_l1 = []
    pair_audits = []
    contracts_ok = True
    action_sets_ok = True
    initial_designs_ok = True
    selector_plans_ok = True
    normalization_scales_ok = True
    risk_scales = []
    certificate_scales = []
    paired = (
        expected
        & set(indexed[low_variant])
        & set(indexed[high_variant])
    )
    for key in sorted(paired):
        low = indexed[low_variant][key]
        high = indexed[high_variant][key]
        contracts_ok &= (
            _contract(low, low_mc, sampling_mode, score_normalization)
            and _contract(high, high_mc, sampling_mode, score_normalization)
        )
        initial_designs_ok &= _initial_design_matches(low, high)
        low_trace = _trace(low)
        high_trace = _trace(high)
        if low_trace is None or high_trace is None:
            action_sets_ok = False
            continue
        low_plan = dict(low_trace.get("exact_kg_selector_plan") or {})
        high_plan = dict(high_trace.get("exact_kg_selector_plan") or {})
        if str(sampling_mode) in RQMC_MODES:
            low_l1 = _finite(low_plan.get("selector_l1_error"))
            high_l1 = _finite(high_plan.get("selector_l1_error"))
            valid_plans = bool(
                low_plan.get("mode") == "factorized_rqmc_nested"
                and high_plan.get("mode") == "factorized_rqmc_nested"
                and int(low_plan.get("sample_count", -1)) == int(low_mc)
                and int(high_plan.get("sample_count", -1)) == int(high_mc)
                and bool(low_plan.get("factorized_selector"))
                and bool(high_plan.get("factorized_selector"))
                and int(low_plan.get("finite_expert_count", 0)) > 0
                and int(high_plan.get("finite_expert_count", 0)) > 0
                and low_l1 is not None
                and high_l1 is not None
            )
            selector_plans_ok &= valid_plans
            if not valid_plans:
                continue
            low_selector_l1.append(low_l1)
            high_selector_l1.append(high_l1)
        low_risk = _score_map(low_trace, "exact_kg_raw_scores_active")
        high_risk = _score_map(high_trace, "exact_kg_raw_scores_active")
        low_certificate = _score_map(
            low_trace, "certificate_deficit_raw_scores_active")
        high_certificate = _score_map(
            high_trace, "certificate_deficit_raw_scores_active")
        maps = (low_risk, high_risk, low_certificate, high_certificate)
        if any(mapping is None for mapping in maps):
            action_sets_ok = False
            continue
        if str(score_normalization) == "current_terminal":
            low_risk_scale = _terminal_scale(
                low_trace, "exact_kg_current_terminal_value")
            high_risk_scale = _terminal_scale(
                high_trace, "exact_kg_current_terminal_value")
            low_certificate_scale = _terminal_scale(
                low_trace, "certificate_deficit_current_value")
            high_certificate_scale = _terminal_scale(
                high_trace, "certificate_deficit_current_value")
            scales = (
                low_risk_scale,
                high_risk_scale,
                low_certificate_scale,
                high_certificate_scale,
            )
            trace_modes = {
                str(low_trace.get(
                    "policy_improvement_score_normalization", "none")),
                str(high_trace.get(
                    "policy_improvement_score_normalization", "none")),
            }
            scale_pair_ok = bool(
                all(scale is not None for scale in scales)
                and math.isclose(
                    low_risk_scale, high_risk_scale,
                    rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(
                    low_certificate_scale, high_certificate_scale,
                    rel_tol=1e-12, abs_tol=1e-12)
                and trace_modes == {"current_terminal"}
            )
            normalization_scales_ok &= scale_pair_ok
            if not scale_pair_ok:
                continue
            risk_scales.append(float(low_risk_scale))
            certificate_scales.append(float(low_certificate_scale))
            low_risk = _scale_map(low_risk, low_risk_scale)
            high_risk = _scale_map(high_risk, high_risk_scale)
            low_certificate = _scale_map(
                low_certificate, low_certificate_scale)
            high_certificate = _scale_map(
                high_certificate, high_certificate_scale)
        elif str(score_normalization) != "none":
            raise ValueError(
                f"unknown score normalization {score_normalization!r}")
        same = (
            set(low_risk) == set(high_risk)
            and set(low_certificate) == set(high_certificate)
            and set(low_risk) == set(low_certificate)
        )
        action_sets_ok &= same
        if not same:
            continue
        pair_risk_errors = [
            abs(low_risk[action] - high_risk[action])
            for action in low_risk
        ]
        pair_certificate_errors = [
            abs(low_certificate[action] - high_certificate[action])
            for action in low_certificate
        ]
        risk_errors.extend(pair_risk_errors)
        certificate_errors.extend(pair_certificate_errors)
        risk_top.append(_top_agrees(low_risk, high_risk))
        certificate_top.append(_top_agrees(
            low_certificate, high_certificate))
        risk_pairwise.append(_pairwise_agreement(low_risk, high_risk))
        certificate_pairwise.append(_pairwise_agreement(
            low_certificate, high_certificate))
        pair_audits.append({
            "heldout": key[0],
            "seed": key[1],
            "action_count": len(low_risk),
            "risk_max_abs_mc8_mc32": max(pair_risk_errors),
            "certificate_max_abs_mc8_mc32": max(pair_certificate_errors),
            "risk_top1_agrees": risk_top[-1],
            "certificate_top1_agrees": certificate_top[-1],
            "risk_pairwise_agreement": risk_pairwise[-1],
            "certificate_pairwise_agreement": certificate_pairwise[-1],
            "risk_score_scale": (
                None if str(score_normalization) == "none"
                else risk_scales[-1]
            ),
            "certificate_score_scale": (
                None if str(score_normalization) == "none"
                else certificate_scales[-1]
            ),
            "mc8_selector_l1_error": (
                None
                if str(sampling_mode) not in RQMC_MODES
                else low_selector_l1[-1]
            ),
            "mc32_selector_l1_error": (
                None
                if str(sampling_mode) not in RQMC_MODES
                else high_selector_l1[-1]
            ),
        })

    risk_max = max(risk_errors, default=None)
    certificate_max = max(certificate_errors, default=None)
    risk_eta = (
        None if risk_max is None
        else max(1e-12, float(multiplier) * risk_max)
    )
    certificate_eta = (
        None if certificate_max is None
        else max(1e-12, float(multiplier) * certificate_max)
    )
    risk_top_rate = (
        None if not risk_top else statistics.fmean(map(float, risk_top)))
    certificate_top_rate = (
        None if not certificate_top
        else statistics.fmean(map(float, certificate_top)))
    risk_pairwise_rate = (
        None if not risk_pairwise else statistics.fmean(risk_pairwise))
    certificate_pairwise_rate = (
        None if not certificate_pairwise
        else statistics.fmean(certificate_pairwise))
    stable = bool(
        risk_top_rate is not None
        and certificate_top_rate is not None
        and risk_pairwise_rate is not None
        and certificate_pairwise_rate is not None
        and risk_top_rate >= 0.8
        and certificate_top_rate >= 0.8
        and risk_pairwise_rate >= 0.9
        and certificate_pairwise_rate >= 0.9
    )
    complete = bool(
        len(paired) == len(expected)
        and not errors
        and contracts_ok
        and action_sets_ok
        and initial_designs_ok
        and selector_plans_ok
        and normalization_scales_ok
        and risk_eta is not None
        and certificate_eta is not None
    )
    return {
        "scope": "v53_nested_mc_fidelity",
        "reference": (
            f"MC{int(high_mc)} nested extension of MC{int(low_mc)}"),
        "low_variant": str(low_variant),
        "high_variant": str(high_variant),
        "low_mc_samples": int(low_mc),
        "high_mc_samples": int(high_mc),
        "sampling_mode": str(sampling_mode),
        "score_normalization": str(score_normalization),
        "normalization_scales_ok": normalization_scales_ok,
        "risk_score_scale_min": (
            None if not risk_scales else min(risk_scales)),
        "risk_score_scale_max": (
            None if not risk_scales else max(risk_scales)),
        "certificate_score_scale_min": (
            None if not certificate_scales else min(certificate_scales)),
        "certificate_score_scale_max": (
            None if not certificate_scales else max(certificate_scales)),
        "finite_expert_marginalization": bool(
            str(sampling_mode) in {
                "stratified_expert_nested",
                "nested_stratified_expert",
                "rao_blackwellized_nested",
            }
        ),
        "seeds": list(seeds),
        "expected_pair_count": len(expected),
        "paired_count": len(paired),
        "errors": errors,
        "contracts_ok": contracts_ok,
        "paired_initial_designs_ok": initial_designs_ok,
        "identical_active_action_sets": action_sets_ok,
        "selector_plans_ok": selector_plans_ok,
        "mc8_selector_l1_mean": (
            None if not low_selector_l1
            else statistics.fmean(low_selector_l1)
        ),
        "mc8_selector_l1_max": (
            None if not low_selector_l1 else max(low_selector_l1)
        ),
        "mc32_selector_l1_mean": (
            None if not high_selector_l1
            else statistics.fmean(high_selector_l1)
        ),
        "mc32_selector_l1_max": (
            None if not high_selector_l1 else max(high_selector_l1)
        ),
        "risk_max_abs_mc8_mc32": risk_max,
        "certificate_max_abs_mc8_mc32": certificate_max,
        "risk_top1_agreement": risk_top_rate,
        "certificate_top1_agreement": certificate_top_rate,
        "risk_pairwise_agreement": risk_pairwise_rate,
        "certificate_pairwise_agreement": certificate_pairwise_rate,
        "safety_multiplier": float(multiplier),
        "recommended_risk_eta": risk_eta,
        "recommended_certificate_eta": certificate_eta,
        "recommended_mc_samples": (
            int(low_mc) if complete and stable else int(high_mc)),
        "fidelity_gate_complete": complete,
        "mc8_stable_enough_for_sentinel": bool(complete and stable),
        "bound_status": (
            "empirical_nested_mc_calibration_not_an_exact_uniform_bound"
        ),
        "formal_contract": (
            "Lean theorem remains conditional on true uniform error events"
        ),
        "pair_audits": pair_audits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--multiplier", type=float, default=1.25)
    parser.add_argument("--low-variant", default=LOW)
    parser.add_argument("--high-variant", default=HIGH)
    parser.add_argument("--low-mc", type=int, default=8)
    parser.add_argument("--high-mc", type=int, default=32)
    parser.add_argument(
        "--sampling-mode", default="antithetic_nested")
    parser.add_argument(
        "--score-normalization",
        choices=("none", "current_terminal"),
        default="none",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.root,
        range(args.seed_start, args.seed_start + args.n_seeds),
        multiplier=args.multiplier,
        sampling_mode=args.sampling_mode,
        score_normalization=args.score_normalization,
        low_variant=args.low_variant,
        high_variant=args.high_variant,
        low_mc=args.low_mc,
        high_mc=args.high_mc,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
