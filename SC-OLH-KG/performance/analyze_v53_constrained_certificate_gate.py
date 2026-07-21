#!/usr/bin/env python3
"""Strict paired V51/V52/V53 constrained-certificate sentinel analysis."""

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
CONTROL = "v51_control"
V52 = "v52_action_superset"
V53 = "v53_certificate_constrained"
VARIANTS = (CONTROL, V52, V53)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _median(values):
    values = [value for value in map(_finite, values) if value is not None]
    return None if not values else float(statistics.median(values))


def _mean(values):
    values = [value for value in map(_finite, values) if value is not None]
    return None if not values else float(statistics.fmean(values))


def load_rows(root):
    rows = []
    errors = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            experiment = f"/{str(payload.get('experiment_variant', '')).strip('/')}/"
            variant = next((
                name for name in VARIANTS if f"/{name}/" in experiment
            ), None)
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


def _outcome(row):
    feasible = bool(row.get("true_feasible", False))
    regret = _finite(row.get("feasible_simple_regret"))
    return feasible, regret if feasible and regret is not None else math.inf


def _design(row):
    variance = dict(row.get("variance_diagnostics") or row.get("variance") or {})
    designs = dict(variance.get("cumulative_statistical_design") or {})
    return dict(designs.get("1") or {})


def _calibration_rmse(row):
    audit = dict(row.get("variance_calibration_audit") or {})
    return _finite(audit.get("certified_log_variance_rmse"))


def _summary(rows):
    audits = [dict(row.get("certificate_outcome_audit") or {}) for row in rows]
    truth = [dict(row.get("truth_pool_diagnostics") or {}) for row in rows]
    designs = [_design(row) for row in rows]
    policies = [
        dict((row.get("decision_backend_diagnostics") or {}).get(
            "policy_improvement") or {})
        for row in rows
    ]
    return {
        "run_count": len(rows),
        "true_feasible": sum(bool(row.get("true_feasible", False))
                             for row in rows),
        "adaptive_improvements": sum(bool(row.get(
            "adaptive_improves_initial_best", False)) for row in rows),
        "adaptive_losses": sum(bool(row.get("adaptive_loss", False))
                               for row in rows),
        "median_feasible_regret": _median(
            row.get("feasible_simple_regret")
            for row in rows if row.get("true_feasible", False)),
        "posterior_certified_count": sum(int(audit.get(
            "posterior_certified_count", 0) or 0) for audit in audits),
        "false_certificate_count": sum(int(audit.get(
            "false_certificate_count", 0) or 0) for audit in audits),
        "vacuous_run_count": sum(bool(audit.get(
            "posterior_certificate_vacuous", False)) for audit in audits),
        "mean_pool_safe_good_support": _mean(
            item.get("pool_has_true_safe_good_rate") for item in truth),
        "mean_pool_min_posterior_margin": _mean(
            item.get("mean_pool_min_posterior_margin") for item in truth),
        "median_normalized_excitation_kappa": _median(
            item.get("lean_normalized_excitation_kappa") for item in designs),
        "median_certified_log_variance_rmse": _median(
            _calibration_rmse(row) for row in rows),
        "policy_switch_count": sum(int(item.get(
            "one_step_switch_count", 0) or 0) for item in policies),
        "rollout_switch_count": sum(int(item.get(
            "rollout_switch_count", 0) or 0) for item in policies),
        "by_domain": {
            domain: {
                "run_count": sum(row.get("heldout") == domain for row in rows),
                "true_feasible": sum(
                    row.get("heldout") == domain
                    and bool(row.get("true_feasible", False))
                    for row in rows),
                "adaptive_losses": sum(
                    row.get("heldout") == domain
                    and bool(row.get("adaptive_loss", False))
                    for row in rows),
                "mean_pool_safe_good_support": _mean(
                    (row.get("truth_pool_diagnostics") or {}).get(
                        "pool_has_true_safe_good_rate")
                    for row in rows if row.get("heldout") == domain),
                "mean_pool_min_posterior_margin": _mean(
                    (row.get("truth_pool_diagnostics") or {}).get(
                        "mean_pool_min_posterior_margin")
                    for row in rows if row.get("heldout") == domain),
                "median_normalized_excitation_kappa": _median(
                    _design(row).get("lean_normalized_excitation_kappa")
                    for row in rows if row.get("heldout") == domain),
                "median_certified_log_variance_rmse": _median(
                    _calibration_rmse(row)
                    for row in rows if row.get("heldout") == domain),
            }
            for domain in DOMAINS
        },
    }


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
        and not bool(second.get("target_labels_used", True))
        and not bool(second.get("target_oracle_used", True))
    )


def _contract(row, variant):
    contract = dict(row.get("decision_backend_contract") or {})
    expected = {
        CONTROL: (
            "promoted_v51_observed_terminal_closure",
            "v51_statistical_closure_v2",
            "disabled_v51_compatible",
        ),
        V52: (
            "v52_safeguarded_policy_improvement",
            "v52_safeguarded_closure_v1",
            "v52_safeguarded_policy_improvement_v1",
        ),
        V53: (
            "v53_constrained_certificate_deficit",
            "v53_constrained_certificate_deficit_v1",
            "v53_constrained_certificate_deficit_v1",
        ),
    }[variant]
    eta_ok = True
    if variant == V53:
        eta_ok = bool(
            float(contract.get("policy_improvement_mc_error_bound", 0.0)) > 0.0
            and float(contract.get(
                "policy_improvement_certificate_mc_error_bound", 0.0)) > 0.0
        )
    return bool(
        str(row.get("implementation_contract_id")) == expected[0]
        and str(row.get("theory_contract_id")) == expected[1]
        and str(contract.get("policy_improvement_contract")) == expected[2]
        and row.get("decision_backend") == "sobol_exact_joint_voi"
        and str(row.get("exact_kg_sampling_mode")) == "antithetic_nested"
        and int(contract.get("forced_sampling_override_count", -1)) == 0
        and not bool(row.get("online_action_trace_target_oracle_used", True))
        and eta_ok
    )


def _relative_noninferior(challenger, control, tolerance=0.05):
    if challenger is None or control is None:
        return False
    return challenger <= control * (1.0 + tolerance) + 1e-12


def analyze(root, seeds=range(5)):
    rows, errors = load_rows(root)
    seeds = tuple(int(seed) for seed in seeds)
    expected = {(domain, seed) for domain in DOMAINS for seed in seeds}
    indexed = {
        variant: {
            _key(row): row for row in rows
            if row.get("gate_variant") == variant and _key(row) in expected
        }
        for variant in VARIANTS
    }
    summaries = {
        variant: _summary(list(indexed[variant].values()))
        for variant in VARIANTS
    }
    comparisons = {}
    for variant in (V52, V53):
        paired = expected & set(indexed[CONTROL]) & set(indexed[variant])
        wins = losses = ties = 0
        initial_ok = True
        for key in paired:
            base = indexed[CONTROL][key]
            challenger = indexed[variant][key]
            initial_ok &= _initial_design_matches(base, challenger)
            base_feasible, base_regret = _outcome(base)
            challenger_feasible, challenger_regret = _outcome(challenger)
            if challenger_feasible and not base_feasible:
                wins += 1
            elif base_feasible and not challenger_feasible:
                losses += 1
            elif challenger_regret < base_regret - 1e-12:
                wins += 1
            elif base_regret < challenger_regret - 1e-12:
                losses += 1
            else:
                ties += 1
        contracts_ok = bool(
            len(indexed[variant]) == len(expected)
            and all(_contract(row, variant)
                    for row in indexed[variant].values())
        )
        comparisons[variant] = {
            "paired_count": len(paired),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "paired_initial_designs_ok": initial_ok,
            "contracts_ok": contracts_ok,
        }

    control = summaries[CONTROL]
    challenger = summaries[V53]
    calibration_noninferior = all(_relative_noninferior(
        challenger["by_domain"][domain][
            "median_certified_log_variance_rmse"],
        control["by_domain"][domain][
            "median_certified_log_variance_rmse"],
    ) for domain in DOMAINS)
    factor = DOMAINS[0]
    factor_pool_gain = bool(
        _finite(challenger["by_domain"][factor][
            "mean_pool_safe_good_support"]) is not None
        and _finite(control["by_domain"][factor][
            "mean_pool_safe_good_support"]) is not None
        and challenger["by_domain"][factor]["mean_pool_safe_good_support"]
        >= control["by_domain"][factor]["mean_pool_safe_good_support"]
        + 1e-12
    )
    factor_excitation_gain = bool(
        _finite(challenger["by_domain"][factor][
            "median_normalized_excitation_kappa"]) is not None
        and _finite(control["by_domain"][factor][
            "median_normalized_excitation_kappa"]) is not None
        and challenger["by_domain"][factor][
            "median_normalized_excitation_kappa"]
        >= control["by_domain"][factor][
            "median_normalized_excitation_kappa"] + 1e-12
    )
    certificate_progress = bool(
        challenger["posterior_certified_count"]
        > control["posterior_certified_count"]
        or (
            _finite(challenger["mean_pool_min_posterior_margin"]) is not None
            and _finite(control["mean_pool_min_posterior_margin"]) is not None
            and challenger["mean_pool_min_posterior_margin"]
            < control["mean_pool_min_posterior_margin"] - 1e-12
        )
    )
    median_regret_noninferior = bool(
        challenger["median_feasible_regret"] is not None
        and control["median_feasible_regret"] is not None
        and challenger["median_feasible_regret"]
        <= control["median_feasible_regret"] + 1e-12
    )
    v53_comparison = comparisons[V53]
    checks = {
        "complete_15_pairs": (
            v53_comparison["paired_count"] == len(expected)
            and not errors),
        "contracts_and_initial_design": (
            v53_comparison["contracts_ok"]
            and v53_comparison["paired_initial_designs_ok"]),
        "all_true_feasible": challenger["true_feasible"] == len(expected),
        "zero_adaptive_losses": challenger["adaptive_losses"] == 0,
        "zero_false_certificates": challenger["false_certificate_count"] == 0,
        "zero_paired_losses": v53_comparison["losses"] == 0,
        "median_regret_noninferior": median_regret_noninferior,
        "domainwise_variance_calibration_noninferior_5pct": (
            calibration_noninferior),
        "factor_pool_safe_good_gain": factor_pool_gain,
        "factor_excitation_gain": factor_excitation_gain,
        "certificate_depth_or_coverage_gain": certificate_progress,
        "rollout_removed": challenger["rollout_switch_count"] == 0,
    }
    sentinel_pass = all(checks.values())
    return {
        "scope": "v53_constrained_certificate_deficit_sentinel",
        "seeds": list(seeds),
        "expected_per_variant": len(expected),
        "errors": errors,
        "summaries": summaries,
        "comparisons": comparisons,
        "v53_checks": checks,
        "sentinel_eligible": [V53] if sentinel_pass else [],
        "promotion_eligible": [],
        "promotion_blocker": (
            "N=80, 20-seed paper gate remains required"
            if sentinel_pass else
            "one or more preregistered V53 sentinel checks failed"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.root,
        range(args.seed_start, args.seed_start + args.n_seeds),
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
