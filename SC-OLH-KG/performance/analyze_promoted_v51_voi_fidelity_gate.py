#!/usr/bin/env python3
"""Estimate V51 shortlist and Monte Carlo VOI errors on common posteriors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
VARIANT_PATTERN = re.compile(r"^mc(?P<mc>\d+)_k(?P<k>\d+)$")


def _median(values):
    finite = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite.append(value)
    return None if not finite else float(statistics.median(finite))


def _variant(experiment):
    for part in str(experiment).strip("/").split("/"):
        if VARIANT_PATTERN.match(part):
            return part
    raise ValueError(f"missing MC/shortlist variant in {experiment!r}")


def _source_contract(row):
    training = dict((row.get("meta_prior") or {}).get("training") or {})
    transfer = dict(row.get("source_target_adaptation_contract") or {})
    return bool(
        int(training.get("source_archive_simulator_calls", -1)) == 384
        and int(transfer.get("source_simulator_calls", -1)) == 384
        and not bool(training.get("target_seed_used_for_source_training", True))
        and not bool(training.get("source_episode_target_oracle_used", True))
        and not bool(transfer.get("source_oracle_aided", True))
        and not bool(transfer.get("target_oracle_used_for_adaptation", True))
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def _closure_contract(row):
    contract = dict(row.get("decision_backend_contract") or {})
    diagnostic = dict(row.get("exact_kg_diagnostics") or {})
    return bool(
        "observed_actions" in str(contract.get("terminal_value_contract", ""))
        and bool(contract.get("coherent", False))
        and int(contract.get("forced_sampling_override_count", -1)) == 0
        and bool(diagnostic.get("nested_common_random_numbers", False))
        and str(diagnostic.get("sampling_mode")) == "antithetic_nested"
    )


def _initial_contract(rows):
    groups = {}
    for row in rows:
        design = dict(row.get("task_initial_design") or {})
        if (
            int(design.get("n_unique", -1)) != 10
            or bool(design.get("target_labels_used", True))
            or bool(design.get("target_oracle_used", True))
        ):
            return False
        key = str(row.get("heldout")), int(row.get("seed", -1))
        groups.setdefault(key, set()).add((
            design.get("fingerprint"),
            design.get("source_archive_fingerprint"),
        ))
    return bool(groups and all(
        len(values) == 1 and None not in next(iter(values))
        for values in groups.values()
    ))


def load_rows(root):
    rows = []
    errors = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            experiment = payload.get("experiment_variant")
            for raw in payload["rows"]:
                row = dict(raw)
                row["gate_variant"] = _variant(
                    row.get("experiment_variant") or experiment)
                row["result_path"] = str(path)
                trace = list(row.get("online_action_trace") or [])
                if len(trace) != 1:
                    raise ValueError(
                        "fidelity gate requires exactly one online decision")
                row["fidelity_action"] = dict(trace[0])
                rows.append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return rows, errors


def _score_map(row):
    action = row["fidelity_action"]
    fingerprints = list(action.get("exact_kg_active_action_fingerprints") or [])
    scores = list(action.get("exact_kg_raw_scores_active") or [])
    if len(fingerprints) != len(scores) or not fingerprints:
        return {}
    return {str(name): float(value) for name, value in zip(fingerprints, scores)}


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left, right):
    if len(left) < 2:
        return 1.0
    first = _rankdata(left)
    second = _rankdata(right)
    if np.std(first) <= 1e-15 or np.std(second) <= 1e-15:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _key(row):
    return str(row.get("heldout")), int(row.get("seed", -1))


def _profile(variant):
    match = VARIANT_PATTERN.match(variant)
    if match is None:
        raise ValueError(variant)
    return int(match.group("mc")), int(match.group("k"))


def analyze(root, expected_count=None):
    rows, errors = load_rows(root)
    variants = sorted(
        {row["gate_variant"] for row in rows},
        key=lambda value: (_profile(value)[0], _profile(value)[1]),
    )
    if not variants:
        reference = None
    else:
        reference = max(
            variants,
            key=lambda value: (_profile(value)[0], _profile(value)[1]),
        )
    by_variant = {
        variant: {_key(row): row for row in rows
                  if row["gate_variant"] == variant}
        for variant in variants
    }
    reference_rows = by_variant.get(reference, {})
    metrics = {}
    for variant in variants:
        mc, shortlist = _profile(variant)
        selected_agreement = []
        rank_correlations = []
        normalized_mc_errors = []
        normalized_shortlist_gaps = []
        nested_actions = []
        same_k_reference = f"mc{_profile(reference)[0]}_k{shortlist}"
        for key, row in by_variant[variant].items():
            ref = reference_rows.get(key)
            if ref is None:
                continue
            current_map = _score_map(row)
            ref_map = _score_map(ref)
            common = sorted(set(current_map) & set(ref_map))
            nested_actions.append(bool(current_map and set(current_map) <= set(ref_map)))
            selected_agreement.append(
                row["fidelity_action"].get("x_fingerprint")
                == ref["fidelity_action"].get("x_fingerprint"))
            if common:
                rank_correlations.append(_spearman(
                    [current_map[name] for name in common],
                    [ref_map[name] for name in common],
                ))
                full_values = np.asarray(list(ref_map.values()), dtype=float)
                scale = max(
                    float(np.ptp(full_values)),
                    float(np.max(np.abs(full_values))),
                    1e-12,
                )
                shortlist_best = max(ref_map[name] for name in common)
                normalized_shortlist_gaps.append(
                    max(0.0, max(ref_map.values()) - shortlist_best) / scale)
            mc_reference = by_variant.get(same_k_reference, {}).get(key)
            if mc_reference is not None:
                mc_map = _score_map(mc_reference)
                shared = sorted(set(current_map) & set(mc_map))
                if shared:
                    values = np.asarray([mc_map[name] for name in shared])
                    scale = max(
                        float(np.ptp(values)),
                        float(np.max(np.abs(values))),
                        1e-12,
                    )
                    normalized_mc_errors.append(max(
                        abs(current_map[name] - mc_map[name])
                        for name in shared
                    ) / scale)
        selected = list(by_variant[variant].values())
        metrics[variant] = {
            "mc_samples": mc,
            "new_action_shortlist_size": shortlist,
            "run_count": len(selected),
            "selected_arm_agreement": (
                float(np.mean(selected_agreement))
                if selected_agreement else None),
            "median_rank_correlation": _median(rank_correlations),
            "median_normalized_eta_mc_proxy": _median(normalized_mc_errors),
            "median_normalized_epsilon_shortlist_proxy": _median(
                normalized_shortlist_gaps),
            "nested_action_set_fraction": (
                float(np.mean(nested_actions)) if nested_actions else None),
            "median_wall_time_sec": _median(
                row.get("total_time_sec") for row in selected),
            "true_feasible_count": sum(bool(row.get("true_feasible", False))
                                       for row in selected),
            "false_certificate_count": sum(int((
                row.get("certificate_outcome_audit") or {}
            ).get("false_certificate_count", 0) or 0) for row in selected),
        }
    expected_per_variant = len(reference_rows)
    survivors = []
    checks = {}
    for variant in variants:
        value = metrics[variant]
        eta = value["median_normalized_eta_mc_proxy"]
        epsilon = value["median_normalized_epsilon_shortlist_proxy"]
        rank = value["median_rank_correlation"]
        agreement = value["selected_arm_agreement"]
        item = {
            "complete": value["run_count"] == expected_per_variant,
            "nested_action_sets": value["nested_action_set_fraction"] == 1.0,
            "selected_arm_agreement_at_least_80pct": (
                agreement is not None and agreement >= 0.8),
            "median_rank_correlation_at_least_0p9": (
                rank is not None and rank >= 0.9),
            "normalized_eta_mc_at_most_0p1": eta is not None and eta <= 0.1,
            "normalized_epsilon_shortlist_at_most_0p1": (
                epsilon is not None and epsilon <= 0.1),
            "zero_false_certificates": value["false_certificate_count"] == 0,
        }
        checks[variant] = item
        if all(item.values()):
            survivors.append(variant)
    survivors.sort(key=lambda value: (
        _profile(value)[0] * _profile(value)[1],
        _profile(value)[0],
        _profile(value)[1],
    ))
    complete = expected_count is None or len(rows) == int(expected_count)
    source_ok = bool(rows and all(_source_contract(row) for row in rows))
    closure_ok = bool(rows and all(_closure_contract(row) for row in rows))
    initial_ok = _initial_contract(rows)
    return {
        "scope": "promoted_v51_voi_fidelity_gate",
        "root": str(root),
        "parsed_count": len(rows),
        "expected_count": expected_count,
        "errors": errors,
        "reference_variant": reference,
        "source_contract": source_ok,
        "closure_contract": closure_ok,
        "initial_pairing_contract": initial_ok,
        "metrics": metrics,
        "checks": checks,
        "gate": {
            "complete_expected_matrix": bool(complete and not errors),
            "survivors_by_compute_cost": survivors,
            "selected_configuration": survivors[0] if survivors else None,
            "passes": bool(
                complete and not errors and source_ok and closure_ok
                and initial_ok and survivors),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args.root, expected_count=args.expected_count)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
