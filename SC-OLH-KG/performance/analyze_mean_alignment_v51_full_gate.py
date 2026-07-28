#!/usr/bin/env python3
"""Analyze paired V51 balanced-four comparison gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = ("promoted_v27", "exact_canonical", "balanced4")
SCENARIOS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
SEEDS = tuple(range(5))


def load_rows(roots):
    if isinstance(roots, (str, Path)):
        roots = [roots]
    rows = []
    for root in roots:
        for path in sorted(Path(root).rglob("result.json")):
            payload = json.loads(path.read_text())
            experiment = str(payload.get("experiment_variant", ""))
            variant = next(
                (name for name in VARIANTS
                 if f"/{name}/" in f"/{experiment}/"),
                None,
            )
            if variant is None:
                continue
            for raw in payload.get("rows", []):
                row = dict(raw)
                row["gate_variant"] = variant
                rows.append(row)
    return rows


def _in_scope(row, seeds):
    return bool(
        str(row.get("heldout", "")) in SCENARIOS
        and int(row.get("seed", -1)) in seeds
    )


def _source_contract(row):
    training = dict((row.get("meta_prior") or {}).get("training") or {})
    contract = dict(row.get("source_target_adaptation_contract") or {})
    return bool(
        int(training.get("source_base_domain_count", -1)) == 2
        and int(training.get("source_episode_count_per_base_domain", -1)) == 1
        and int(training.get("source_archive_simulator_calls", -1)) == 384
        and not bool(training.get("target_seed_used_for_source_training", True))
        and not bool(training.get("source_episode_target_oracle_used", True))
        and int(contract.get("source_simulator_calls", -1)) == 384
        and not bool(contract.get("source_oracle_aided", True))
        and not bool(contract.get("target_oracle_used_for_adaptation", True))
        and not bool(row.get("online_action_trace_target_oracle_used", True))
    )


def _mode_contract(row, variant):
    if variant == "promoted_v27":
        return bool(
            row.get("source_constraint_mean_adaptation_mode")
            == "sequential_aggregate_hyperlaw"
            and not bool((row.get("audit") or {}).get(
                "uses_true_constraint", True))
        )
    decision = dict(row.get("decision_backend_contract") or {})
    expected_count = 1 if variant == "exact_canonical" else 4
    expected_policy = (
        "canonical_sobol"
        if variant == "exact_canonical"
        else "canonical_plus_posterior_risk"
    )
    return bool(
        row.get("decision_backend") == "sobol_exact_joint_voi"
        and row.get("decision_aleatoric_mode") == "posterior_central"
        and row.get("decision_ambiguity_mode") == "posterior_nominal"
        and row.get("decision_violation_loss_mode") == "positive_part"
        and decision.get("evaluate_or_replicate_new_action_count")
        == expected_count
        and decision.get("evaluate_or_replicate_new_action_policy")
        == expected_policy
        and bool(row.get("adaptive_replication_voi_enabled", False))
    )


def _initial_pairing_contract(rows, seeds):
    for scenario in SCENARIOS:
        for seed in seeds:
            group = {
                row.get("gate_variant"): row for row in rows
                if row.get("heldout") == scenario
                and int(row.get("seed", -1)) == seed
            }
            if set(group) != set(VARIANTS):
                return False
            designs = [dict(row.get("task_initial_design") or {})
                       for row in group.values()]
            fingerprints = {design.get("fingerprint") for design in designs}
            archives = {design.get("source_archive_fingerprint")
                        for design in designs}
            if (
                len(fingerprints) != 1
                or None in fingerprints
                or len(archives) != 1
                or None in archives
                or any(int(design.get("n_unique", -1)) != 10
                       for design in designs)
                or any(not bool(design.get("source_only", False))
                       for design in designs)
                or any(bool(design.get("target_labels_used", True))
                       for design in designs)
                or any(bool(design.get("target_oracle_used", True))
                       for design in designs)
            ):
                return False
            initial_regrets = [row.get("initial_best_feasible_regret")
                               for row in group.values()]
            finite = [float(value) for value in initial_regrets
                      if value is not None and np.isfinite(float(value))]
            if finite and not np.allclose(finite, finite[0]):
                return False
    return True


def _outcome(row):
    feasible = bool(row.get("true_feasible", False))
    regret = row.get("feasible_simple_regret")
    return feasible, (float(regret) if feasible and regret is not None else np.inf)


def _paired_comparison(rows, challenger, control, seeds):
    wins = losses = ties = 0
    by_domain = {}
    for scenario in SCENARIOS:
        domain_counts = [0, 0, 0]
        for seed in seeds:
            lookup = {
                row.get("gate_variant"): row for row in rows
                if row.get("heldout") == scenario
                and int(row.get("seed", -1)) == seed
                and row.get("gate_variant") in {challenger, control}
            }
            if set(lookup) != {challenger, control}:
                continue
            c_feasible, c_regret = _outcome(lookup[challenger])
            b_feasible, b_regret = _outcome(lookup[control])
            if c_feasible and not b_feasible:
                index = 0
            elif b_feasible and not c_feasible:
                index = 1
            elif c_regret < b_regret - 1e-12:
                index = 0
            elif b_regret < c_regret - 1e-12:
                index = 1
            else:
                index = 2
            domain_counts[index] += 1
        by_domain[scenario] = {
            "wins": domain_counts[0],
            "losses": domain_counts[1],
            "ties": domain_counts[2],
        }
        wins += domain_counts[0]
        losses += domain_counts[1]
        ties += domain_counts[2]
    return {"wins": wins, "losses": losses, "ties": ties,
            "by_domain": by_domain}


def _summary(rows, variant, seeds):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    regrets = np.asarray([
        float(row["feasible_simple_regret"])
        for row in selected
        if row.get("true_feasible", False)
        and row.get("feasible_simple_regret") is not None
    ], dtype=float)
    return {
        "count": len(selected),
        "complete": len(selected) == len(SCENARIOS) * len(seeds),
        "true_feasible": sum(bool(row.get("true_feasible", False))
                             for row in selected),
        "adaptive_losses": sum(bool(row.get("adaptive_loss", False))
                               for row in selected),
        "adaptive_improvements": sum(bool(row.get(
            "adaptive_improves_initial_best", False)) for row in selected),
        "false_certificates": sum(int((
            row.get("certificate_outcome_audit") or {}
        ).get("false_certificate_count", 0) or 0) for row in selected),
        "selected_replications": sum(int(row.get(
            "adaptive_replication_selected_count", 0) or 0)
            for row in selected),
        "selected_new_points": sum(int(row.get(
            "adaptive_new_point_selected_count", 0) or 0)
            for row in selected),
        "median_feasible_regret": (
            float(np.median(regrets)) if len(regrets) else None),
        "by_domain": {
            scenario: {
                "count": len(domain_rows),
                "true_feasible": sum(bool(row.get("true_feasible", False))
                                     for row in domain_rows),
                "adaptive_losses": sum(bool(row.get("adaptive_loss", False))
                                       for row in domain_rows),
                "adaptive_improvements": sum(bool(row.get(
                    "adaptive_improves_initial_best", False))
                    for row in domain_rows),
                "median_feasible_regret": (
                    float(np.median(domain_regrets))
                    if len(domain_regrets) else None),
            }
            for scenario in SCENARIOS
            for domain_rows in [[row for row in selected
                                 if row.get("heldout") == scenario]]
            for domain_regrets in [np.asarray([
                float(row["feasible_simple_regret"])
                for row in domain_rows
                if row.get("true_feasible", False)
                and row.get("feasible_simple_regret") is not None
            ], dtype=float)]
        },
    }


def summarize(rows, seeds=SEEDS):
    seeds = tuple(int(seed) for seed in seeds)
    rows = [row for row in rows if _in_scope(row, seeds)]
    summaries = {
        variant: _summary(rows, variant, seeds) for variant in VARIANTS
    }
    source_contract = {
        variant: summaries[variant]["complete"] and all(
            _source_contract(row) for row in rows
            if row.get("gate_variant") == variant)
        for variant in VARIANTS
    }
    mode_contract = {
        variant: summaries[variant]["complete"] and all(
            _mode_contract(row, variant) for row in rows
            if row.get("gate_variant") == variant)
        for variant in VARIANTS
    }
    comparisons = {
        control: _paired_comparison(rows, "balanced4", control, seeds)
        for control in ("promoted_v27", "exact_canonical")
    }
    challenger = summaries["balanced4"]
    controls = [summaries[name]
                for name in ("promoted_v27", "exact_canonical")]
    domainwise_safety = all(
        challenger["by_domain"][scenario]["true_feasible"]
        >= control["by_domain"][scenario]["true_feasible"]
        and challenger["by_domain"][scenario]["adaptive_losses"]
        <= control["by_domain"][scenario]["adaptive_losses"]
        for scenario in SCENARIOS for control in controls
    )
    strict_paired_gain = all(
        comparison["wins"] > comparison["losses"]
        for comparison in comparisons.values()
    )
    domainwise_regret = all(
        challenger["by_domain"][scenario]["median_feasible_regret"]
        <= control["by_domain"][scenario]["median_feasible_regret"] + 1e-12
        for scenario in SCENARIOS for control in controls
    )
    domainwise_paired = all(
        counts["wins"] >= counts["losses"]
        for comparison in comparisons.values()
        for counts in comparison["by_domain"].values()
    )
    gate_passes = bool(all((
        all(summary["complete"] for summary in summaries.values()),
        all(source_contract.values()),
        all(mode_contract.values()),
        _initial_pairing_contract(rows, seeds),
        challenger["false_certificates"] == 0,
        challenger["true_feasible"]
        >= max(control["true_feasible"] for control in controls),
        challenger["adaptive_losses"]
        <= min(control["adaptive_losses"] for control in controls),
        challenger["adaptive_improvements"]
        > max(control["adaptive_improvements"] for control in controls),
        domainwise_safety,
        domainwise_regret,
        domainwise_paired,
        strict_paired_gain,
    )))
    return {
        "scope": f"v51_balanced4_{len(seeds)}_seed_three_domain_gate",
        "seeds": list(seeds),
        "fixed_source_call_budget_per_variant": 384,
        "target_budget": 20,
        "initial_budget": 10,
        "source_contract": source_contract,
        "mode_contract": mode_contract,
        "initial_pairing_contract": _initial_pairing_contract(rows, seeds),
        "variant_summaries": summaries,
        "paired_balanced4_comparisons": comparisons,
        "domainwise_safety_noninferior": bool(domainwise_safety),
        "domainwise_regret_noninferior": bool(domainwise_regret),
        "domainwise_paired_noninferior": bool(domainwise_paired),
        "strict_paired_gain": bool(strict_paired_gain),
        "gate_passes": bool(gate_passes),
        "s20_warranted": bool(gate_passes and len(seeds) < 20),
        "promotion_eligible": (
            ["balanced4"] if gate_passes and len(seeds) >= 20 else []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seeds = tuple(range(args.seed_start, args.seed_start + args.n_seeds))
    result = summarize(load_rows(args.roots), seeds=seeds)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
