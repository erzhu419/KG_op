#!/usr/bin/env python3
"""Analyze the paired V51/V52 safeguarded statistical-closure gate."""

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
VARIANTS = (CONTROL, "action_superset", "guarded_rollout", "joint")


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _median(values):
    rows = [value for value in map(_finite, values) if value is not None]
    return None if not rows else float(statistics.median(rows))


def _mean(values):
    rows = [value for value in map(_finite, values) if value is not None]
    return None if not rows else float(statistics.fmean(rows))


def load_rows(root):
    rows = []
    errors = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            experiment = str(payload.get("experiment_variant", ""))
            variant = next((
                name for name in VARIANTS
                if f"/{name}/" in f"/{experiment}/"
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


def _summary(rows):
    audits = [dict(row.get("certificate_outcome_audit") or {}) for row in rows]
    truth = [dict(row.get("truth_pool_diagnostics") or {}) for row in rows]
    designs = [_design(row) for row in rows]
    policy = [
        dict((row.get("decision_backend_diagnostics") or {}).get(
            "policy_improvement") or {})
        for row in rows
    ]
    regrets = [
        row.get("feasible_simple_regret") for row in rows
        if row.get("true_feasible", False)
    ]
    return {
        "run_count": len(rows),
        "true_feasible": sum(bool(row.get("true_feasible", False))
                             for row in rows),
        "adaptive_improvements": sum(bool(row.get(
            "adaptive_improves_initial_best", False)) for row in rows),
        "adaptive_losses": sum(bool(row.get("adaptive_loss", False))
                               for row in rows),
        "median_feasible_regret": _median(regrets),
        "posterior_certified_count": sum(int(audit.get(
            "posterior_certified_count", 0) or 0) for audit in audits),
        "false_certificate_count": sum(int(audit.get(
            "false_certificate_count", 0) or 0) for audit in audits),
        "vacuous_run_count": sum(bool(audit.get(
            "posterior_certificate_vacuous", False)) for audit in audits),
        "mean_pool_safe_good_support": _mean(
            item.get("pool_has_true_safe_good_rate") for item in truth),
        "mean_pool_feasible_support": _mean(
            item.get("pool_has_true_feasible_rate") for item in truth),
        "mean_pool_min_posterior_margin": _mean(
            item.get("mean_pool_min_posterior_margin") for item in truth),
        "median_normalized_excitation_kappa": _median(
            item.get("lean_normalized_excitation_kappa") for item in designs),
        "median_normalized_condition_number": _median(
            (item.get("normalized_active_geometry") or {}).get(
                "condition_number_positive_spectrum")
            for item in designs),
        "one_step_switch_count": sum(int(item.get(
            "one_step_switch_count", 0) or 0) for item in policy),
        "rollout_switch_count": sum(int(item.get(
            "rollout_switch_count", 0) or 0) for item in policy),
        "by_domain": {
            domain: {
                "run_count": sum(row.get("heldout") == domain for row in rows),
                "true_feasible": sum(
                    row.get("heldout") == domain
                    and bool(row.get("true_feasible", False))
                    for row in rows
                ),
                "adaptive_losses": sum(
                    row.get("heldout") == domain
                    and bool(row.get("adaptive_loss", False))
                    for row in rows
                ),
                "mean_pool_safe_good_support": _mean(
                    (row.get("truth_pool_diagnostics") or {}).get(
                        "pool_has_true_safe_good_rate")
                    for row in rows if row.get("heldout") == domain
                ),
            }
            for domain in DOMAINS
        },
    }


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


def _decision_contract(row, variant):
    contract = dict(row.get("decision_backend_contract") or {})
    expected_impl = (
        "promoted_v51_observed_terminal_closure"
        if variant == CONTROL
        else "v52_safeguarded_policy_improvement"
    )
    expected_theory = (
        "v51_statistical_closure_v2"
        if variant == CONTROL
        else "v52_safeguarded_closure_v1"
    )
    return bool(
        str(row.get("implementation_contract_id")) == expected_impl
        and str(row.get("theory_contract_id")) == expected_theory
        and row.get("decision_backend") == "sobol_exact_joint_voi"
        and bool(contract.get("coherent", False))
        and int(contract.get("forced_sampling_override_count", -1)) == 0
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def _paired_initial(control, challenger):
    left = dict(control.get("task_initial_design") or {})
    right = dict(challenger.get("task_initial_design") or {})
    return bool(
        left.get("fingerprint") is not None
        and left.get("fingerprint") == right.get("fingerprint")
        and left.get("source_archive_fingerprint")
        == right.get("source_archive_fingerprint")
        and int(left.get("n_unique", -1)) == 10
        and int(right.get("n_unique", -1)) == 10
        and not bool(right.get("target_labels_used", True))
        and not bool(right.get("target_oracle_used", True))
    )


def analyze(root, seeds=range(5)):
    rows, errors = load_rows(root)
    seeds = tuple(int(seed) for seed in seeds)
    expected = {(domain, seed) for domain in DOMAINS for seed in seeds}
    indexed = {
        variant: {
            _key(row): row for row in rows
            if row.get("gate_variant") == variant
            and row.get("heldout") in DOMAINS
            and int(row.get("seed", -1)) in seeds
        }
        for variant in VARIANTS
    }
    summaries = {
        variant: _summary(list(indexed[variant].values()))
        for variant in VARIANTS
    }
    control = summaries[CONTROL]
    comparisons = {}
    promotion = []
    for variant in VARIANTS[1:]:
        paired = expected & set(indexed[CONTROL]) & set(indexed[variant])
        wins = losses = ties = 0
        initial_ok = True
        for key in paired:
            base = indexed[CONTROL][key]
            challenger = indexed[variant][key]
            initial_ok &= _paired_initial(base, challenger)
            c_feasible, c_regret = _outcome(challenger)
            b_feasible, b_regret = _outcome(base)
            if c_feasible and not b_feasible:
                wins += 1
            elif b_feasible and not c_feasible:
                losses += 1
            elif c_regret < b_regret - 1e-12:
                wins += 1
            elif b_regret < c_regret - 1e-12:
                losses += 1
            else:
                ties += 1
        challenger = summaries[variant]
        domainwise_safe = all(
            challenger["by_domain"][domain]["true_feasible"]
            >= control["by_domain"][domain]["true_feasible"]
            and challenger["by_domain"][domain]["adaptive_losses"]
            <= control["by_domain"][domain]["adaptive_losses"]
            for domain in DOMAINS
        )
        factor_gain = (
            _finite(challenger["by_domain"][DOMAINS[0]][
                "mean_pool_safe_good_support"]),
            _finite(control["by_domain"][DOMAINS[0]][
                "mean_pool_safe_good_support"]),
        )
        excitation_gain = (
            _finite(challenger["median_normalized_excitation_kappa"]),
            _finite(control["median_normalized_excitation_kappa"]),
        )
        closure_gain = bool(
            challenger["posterior_certified_count"]
            > control["posterior_certified_count"]
            or (
                factor_gain[0] is not None and factor_gain[1] is not None
                and factor_gain[0] > factor_gain[1] + 1e-12
            )
            or (
                excitation_gain[0] is not None
                and excitation_gain[1] is not None
                and excitation_gain[0] > excitation_gain[1] + 1e-12
            )
        )
        rows_for_variant = list(indexed[variant].values())
        contracts = bool(
            len(rows_for_variant) == len(expected)
            and all(_source_contract(row) for row in rows_for_variant)
            and all(_decision_contract(row, variant)
                    for row in rows_for_variant)
            and initial_ok
        )
        regret_noninferior = bool(
            challenger["median_feasible_regret"] is not None
            and control["median_feasible_regret"] is not None
            and challenger["median_feasible_regret"]
            <= control["median_feasible_regret"] + 1e-12
        )
        sentinel_pass = bool(
            len(paired) == len(expected)
            and contracts
            and domainwise_safe
            and challenger["true_feasible"] == len(expected)
            and challenger["adaptive_losses"] == 0
            and challenger["false_certificate_count"] == 0
            and regret_noninferior
            and losses <= wins
            and closure_gain
        )
        comparisons[variant] = {
            "paired_count": len(paired),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "paired_initial_contract": bool(initial_ok),
            "source_and_decision_contracts": contracts,
            "domainwise_safety_noninferior": domainwise_safe,
            "median_regret_noninferior": regret_noninferior,
            "closure_gain_observed": closure_gain,
            "sentinel_passes": sentinel_pass,
            "numerical_uniform_error_bound_requires_separate_fidelity_gate": (
                True
            ),
        }
        if sentinel_pass:
            promotion.append(variant)
    return {
        "scope": "v52_safeguarded_statistical_closure_gate",
        "seeds": list(seeds),
        "expected_per_variant": len(expected),
        "parsed_rows": len(rows),
        "errors": errors,
        "summaries": summaries,
        "comparisons": comparisons,
        "sentinel_eligible": promotion,
        "promotion_eligible": [],
        "promotion_blocker": (
            "separate MC uniform-error fidelity gate and N=80 gate required"
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
