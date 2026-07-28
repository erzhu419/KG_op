#!/usr/bin/env python3
"""Analyze the V48 certified-only incumbent-preservation Queue gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


try:
    from . import analyze_mean_alignment_v47_gate as base
except ImportError:  # Direct script execution.
    import analyze_mean_alignment_v47_gate as base


VARIANTS = (
    "v27_promoted_no_preservation",
    "v27_certified_only",
    "v47_risk_preservation",
    "v47_no_preservation",
    "v48_certified_only",
)
SENTINEL_SEEDS = (1, 3)
QUEUE_SCENARIO = base.QUEUE_SCENARIO
V47_VARIANTS = set(VARIANTS[2:])
CERTIFIED_ONLY_VARIANTS = {
    "v27_certified_only", "v48_certified_only",
}
NO_PRESERVATION_VARIANTS = {
    "v27_promoted_no_preservation", "v47_no_preservation",
}
SUMMARY_ANALYZER = base.V43_ANALYZER
V42_ANALYZER = base.V42_ANALYZER


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


def _point_fingerprint(point):
    if point is None:
        return None
    values = ",".join(str(int(value)) for value in point).encode()
    return hashlib.sha256(values).hexdigest()[:16]


def _training(row):
    return dict((row.get("meta_prior") or {}).get("training") or {})


def _source_contract(rows, variant):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    if not selected:
        return False
    if variant in V47_VARIANTS:
        remapped = [
            dict(row, gate_variant="v47_deconvolved_task_bayes")
            for row in selected
        ]
        return bool(
            base._episode_contract(
                remapped, "v47_deconvolved_task_bayes")
            and base._hyperlaw_contract(
                remapped, "v47_deconvolved_task_bayes")
        )
    for row in selected:
        training = _training(row)
        contract = dict(row.get("source_target_adaptation_contract") or {})
        adaptation = str(contract.get(
            "source_constraint_mean_adaptation_mode",
            (row.get("config") or {}).get(
                "source_constraint_mean_adaptation_mode", ""),
        ))
        if adaptation != "sequential_aggregate_hyperlaw":
            return False
        if int(training.get("source_base_domain_count", -1)) != 2:
            return False
        if int(training.get("source_episode_count_per_base_domain", -1)) != 1:
            return False
        if int(training.get("source_archive_simulator_calls", -1)) != 384:
            return False
        if bool(training.get("target_seed_used_for_source_training", True)):
            return False
        if bool(training.get("source_episode_target_oracle_used", True)):
            return False
        if int(contract.get("source_simulator_calls", -1)) != 384:
            return False
        if bool(contract.get("source_oracle_aided", True)):
            return False
    return True


def _paired_target_contract(rows, seeds):
    for seed in seeds:
        group = [
            row for row in rows
            if base.base._scenario(row) == QUEUE_SCENARIO
            and int(row.get("seed", -1)) == int(seed)
            and row.get("gate_variant") in VARIANTS
        ]
        if len(group) != len(VARIANTS):
            return False
        if any(row.get("decision_backend") != "sobol_new" for row in group):
            return False
        proposal_archives = {
            str((row.get("initial_design_archive_contract") or {}).get(
                "proposal_archive_fingerprint", ""))
            for row in group
        }
        if len(proposal_archives) != 1 or "" in proposal_archives:
            return False
        for field in (
            "target_design_fingerprint",
            "online_action_sequence_fingerprint",
        ):
            values = {
                SUMMARY_ANALYZER._paired_fingerprint(row, field)
                for row in group
            }
            if len(values) != 1 or "missing" in values:
                return False
        traces = [list(row.get("online_action_trace") or []) for row in group]
        reference = traces[0] if traces else []
        if not reference:
            return False
        for trace in traces:
            if len(trace) != len(reference):
                return False
            if any(
                left.get("x_fingerprint") != right.get("x_fingerprint")
                or left.get("observed_response") != right.get("observed_response")
                or right.get("candidate_source") != "sobol_continuation"
                for left, right in zip(reference, trace)
            ):
                return False
        if any(bool(row.get(
            "online_action_trace_target_oracle_used", True
        )) for row in group):
            return False
    return True


def _terminal_audit_contract(row, selected_must_match):
    audit = dict(row.get("terminal_bayes_pool_audit") or {})
    if audit.get("status") != "ranked":
        return False
    if int(audit.get("pool_size", 0)) < 1:
        return False
    if bool(audit.get("target_oracle_used_for_ranking", True)):
        return False
    if bool(audit.get("target_oracle_used_for_decision", True)):
        return False
    if bool(audit.get("truth_admissible_decision_input", True)):
        return False
    if audit.get("truth_join_timing") != "post_terminal_rank":
        return False
    if selected_must_match and not bool(
        audit.get("selected_matches_counterfactual_bayes", False)
    ):
        return False
    return True


def _preservation_contract(rows, variant):
    selected = [row for row in rows if row.get("gate_variant") == variant]
    if not selected:
        return False
    for row in selected:
        dominance = dict(row.get("posterior_dominance") or {})
        history = list(dominance.get("history") or [])
        terminal_used = bool(row.get(
            "posterior_dominance_terminal_used", False))
        terminal_rule = str((row.get(
            "decision_backend_contract") or {}).get("terminal_rule", ""))
        backend_used = bool(row.get(
            "decision_backend_terminal_used",
            terminal_rule == "posterior_bayes_risk",
        ))
        audit_present = bool(row.get("terminal_bayes_pool_audit"))
        if variant == "v47_risk_preservation":
            if not dominance.get("enabled", False) or not terminal_used:
                return False
            if backend_used:
                return False
            if dominance.get("incumbent") is None or not history:
                return False
            if history[0].get("posterior_dominance_initialization") != "risk":
                return False
            if audit_present and not _terminal_audit_contract(
                row, selected_must_match=False
            ):
                return False
        elif variant in CERTIFIED_ONLY_VARIANTS:
            if not dominance.get("enabled", False) or terminal_used:
                return False
            if not backend_used or dominance.get("incumbent") is not None:
                return False
            if not history or any(
                item.get("status") != "uninitialized_no_certificate"
                or item.get("posterior_dominance_initialization")
                != "certified_only"
                or int(item.get("initial_certified_count", -1)) != 0
                or not item.get("terminal_fallback_required", False)
                or item.get("target_oracle_used", True)
                for item in history
            ):
                return False
            if audit_present and not _terminal_audit_contract(
                row, selected_must_match=True
            ):
                return False
        elif variant in NO_PRESERVATION_VARIANTS:
            if dominance.get("enabled", True) or terminal_used or not backend_used:
                return False
            if audit_present and not _terminal_audit_contract(
                row, selected_must_match=True
            ):
                return False
        else:
            return False
    return True


def _paired_outcome(rows, left, right, seeds):
    for seed in seeds:
        pair = {
            row.get("gate_variant"): row for row in rows
            if base.base._scenario(row) == QUEUE_SCENARIO
            and int(row.get("seed", -1)) == int(seed)
            and row.get("gate_variant") in {left, right}
        }
        if set(pair) != {left, right}:
            return False
        first, second = pair[left], pair[right]
        if _point_fingerprint(first.get("x_recommended")) != _point_fingerprint(
            second.get("x_recommended")
        ):
            return False
        for field in (
            "true_feasible", "adaptive_loss",
            "adaptive_improves_initial_best", "feasible_simple_regret",
        ):
            if first.get(field) != second.get(field):
                return False
    return True


def summarize(rows, seeds=SENTINEL_SEEDS):
    seeds = tuple(int(seed) for seed in seeds)
    selected = [
        row for row in rows
        if base.base._scenario(row) == QUEUE_SCENARIO
        and int(row.get("seed", -1)) in seeds
    ]
    summaries = {
        variant: SUMMARY_ANALYZER._variant_summary(
            selected, variant, len(seeds))
        for variant in VARIANTS
    }
    source_contracts = {
        variant: _source_contract(selected, variant) for variant in VARIANTS
    }
    preservation_contracts = {
        variant: _preservation_contract(selected, variant)
        for variant in VARIANTS
    }
    paired = _paired_target_contract(selected, seeds)
    v27_equivalent = _paired_outcome(
        selected,
        "v27_promoted_no_preservation",
        "v27_certified_only",
        seeds,
    )
    v47_equivalent = _paired_outcome(
        selected, "v47_no_preservation", "v48_certified_only", seeds)
    primary = summaries["v48_certified_only"]
    risk_control = summaries["v47_risk_preservation"]
    promoted = summaries["v27_promoted_no_preservation"]
    false_certificates = (
        primary["audit_pool_false_certificates"]
        + primary["evaluated_false_certificates"])
    strict_over_risk = bool(
        primary["true_feasible"] > risk_control["true_feasible"]
        or primary["adaptive_losses"] < risk_control["adaptive_losses"]
        or primary["adaptive_improvements"]
        > risk_control["adaptive_improvements"]
    )
    nonworse_than_v27 = bool(
        primary["true_feasible"] >= promoted["true_feasible"]
        and primary["adaptive_losses"] <= promoted["adaptive_losses"]
    )
    mechanism_gate = bool(all((
        primary["complete"], paired,
        all(source_contracts.values()),
        all(preservation_contracts.values()),
        v27_equivalent, v47_equivalent,
        primary["true_feasible"] == len(seeds),
        primary["adaptive_losses"] == 0,
        false_certificates == 0,
        strict_over_risk,
        nonworse_than_v27,
    )))
    strict_over_v27 = bool(
        primary["true_feasible"] > promoted["true_feasible"]
        or primary["adaptive_losses"] < promoted["adaptive_losses"]
        or primary["adaptive_improvements"]
        > promoted["adaptive_improvements"]
        or (
            primary["median_feasible_regret"] is not None
            and promoted["median_feasible_regret"] is not None
            and primary["median_feasible_regret"]
            < promoted["median_feasible_regret"] - 1e-12
        )
    )
    return {
        "scope": "queue_failure_sentinel",
        "seeds": list(seeds),
        "fixed_source_call_budget_per_variant": 384,
        "paired_initial_design_actions_and_target_responses": bool(paired),
        "source_contract": source_contracts,
        "preservation_contract": preservation_contracts,
        "certified_only_matches_no_preservation_v27": bool(v27_equivalent),
        "certified_only_matches_no_preservation_v47": bool(v47_equivalent),
        "certified_only_strictly_improves_v47_risk_preservation": bool(
            strict_over_risk),
        "certified_only_nonworse_than_promoted_v27": bool(nonworse_than_v27),
        "mechanism_gate_passed": bool(mechanism_gate),
        "strictly_improves_promoted_v27": bool(strict_over_v27),
        "variant_summaries": summaries,
        "promotion_eligible": (
            ["v48_certified_only"]
            if mechanism_gate and strict_over_v27 else []
        ),
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
