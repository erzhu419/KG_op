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
VARIANTS = (LOW, HIGH)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _variant(experiment):
    marker = f"/{str(experiment).strip('/')}/"
    return next((
        variant for variant in VARIANTS
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


def _contract(row, expected_mc, expected_sampling_mode):
    contract = dict(row.get("decision_backend_contract") or {})
    return bool(
        str(row.get("implementation_contract_id"))
        == "v53_constrained_certificate_deficit"
        and str(row.get("theory_contract_id"))
        == "v53_constrained_certificate_deficit_v1"
        and int(row.get("exact_kg_mc_samples", -1)) == int(expected_mc)
        and str(row.get("exact_kg_sampling_mode"))
        == str(expected_sampling_mode)
        and str(contract.get("policy_improvement_contract"))
        == "v53_constrained_certificate_deficit_v1"
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def analyze(
    root,
    seeds=range(0, 10),
    multiplier=1.25,
    sampling_mode="antithetic_nested",
):
    rows, errors = load_rows(root)
    seeds = tuple(int(seed) for seed in seeds)
    expected = {(domain, seed) for domain in DOMAINS for seed in seeds}
    indexed = {
        variant: {
            _key(row): row for row in rows
            if row.get("gate_variant") == variant
            and _key(row) in expected
        }
        for variant in VARIANTS
    }

    risk_errors = []
    certificate_errors = []
    risk_top = []
    certificate_top = []
    risk_pairwise = []
    certificate_pairwise = []
    pair_audits = []
    contracts_ok = True
    action_sets_ok = True
    initial_designs_ok = True
    paired = expected & set(indexed[LOW]) & set(indexed[HIGH])
    for key in sorted(paired):
        low = indexed[LOW][key]
        high = indexed[HIGH][key]
        contracts_ok &= (
            _contract(low, 8, sampling_mode)
            and _contract(high, 32, sampling_mode)
        )
        initial_designs_ok &= _initial_design_matches(low, high)
        low_trace = _trace(low)
        high_trace = _trace(high)
        if low_trace is None or high_trace is None:
            action_sets_ok = False
            continue
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
        and risk_eta is not None
        and certificate_eta is not None
    )
    return {
        "scope": "v53_nested_mc_fidelity",
        "reference": "MC32 nested extension of MC8",
        "sampling_mode": str(sampling_mode),
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
        "risk_max_abs_mc8_mc32": risk_max,
        "certificate_max_abs_mc8_mc32": certificate_max,
        "risk_top1_agreement": risk_top_rate,
        "certificate_top1_agreement": certificate_top_rate,
        "risk_pairwise_agreement": risk_pairwise_rate,
        "certificate_pairwise_agreement": certificate_pairwise_rate,
        "safety_multiplier": float(multiplier),
        "recommended_risk_eta": risk_eta,
        "recommended_certificate_eta": certificate_eta,
        "recommended_mc_samples": 8 if complete and stable else 32,
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
    parser.add_argument(
        "--sampling-mode", default="antithetic_nested")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.root,
        range(args.seed_start, args.seed_start + args.n_seeds),
        multiplier=args.multiplier,
        sampling_mode=args.sampling_mode,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
