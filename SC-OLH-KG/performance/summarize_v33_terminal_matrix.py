"""Summarize and gate the preregistered V33 terminal-policy matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
VARIANT_CONFIGS = {
    ("legacy", "legacy"): "v32",
    ("legacy", "off"): "posterior_only",
    ("commit_before_switch", "certified_only"): "commit_before_switch",
    ("terminal_kg_1step", "certified_only"): "terminal_kg_1step",
    ("terminal_kg_depth3", "certified_only"): "terminal_kg_depth3",
}
VARIANTS = tuple(VARIANT_CONFIGS.values())
PRIMARY_BASELINE = "v32"
PRIMARY_CHALLENGER = "terminal_kg_1step"
EXPECTED_SEEDS = tuple(range(7))
INFORMATIVE_COMPLETION_MIN_GAIN = 3


def _median(values):
    clean = [float(value) for value in values if value is not None]
    return None if not clean else float(np.median(clean))


def _mean(values):
    clean = [float(value) for value in values if value is not None]
    return None if not clean else float(np.mean(clean))


def _result_files(inputs):
    files = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("result.json")))
        elif path.is_file():
            files.append(path)
    return files


def _variant(row):
    key = (
        str(row.get("finalist_replication_policy", "legacy")),
        str(row.get("finalist_empirical_override", "legacy")),
    )
    if key not in VARIANT_CONFIGS:
        raise ValueError(f"unregistered V33 policy pair {key}")
    return VARIANT_CONFIGS[key]


def load_rows(inputs):
    rows = []
    seen = set()
    for path in _result_files(inputs):
        payload = json.loads(path.read_text())
        payload_rows = payload.get("rows", [])
        if len(payload_rows) != 1:
            raise ValueError(f"one-seed shard expected in {path}")
        row = dict(payload_rows[0])
        variant = _variant(row)
        heldout = str(row["heldout"])
        seed = int(row["seed"])
        key = (variant, heldout, seed)
        if key in seen:
            raise ValueError(f"duplicate V33 shard {key}")
        if heldout not in DOMAINS:
            raise ValueError(f"unregistered V33 domain {heldout}")
        seen.add(key)
        row["_variant"] = variant
        row["_path"] = str(path)
        rows.append(row)
    return rows


def _terminal_rows(finalist):
    return [
        item for item in (finalist.get("terminal_kg_rows") or [])
        if isinstance(item, dict)
    ]


def _group_summary(variant, heldout, rows):
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    completed = []
    terminal_evaluations = []
    terminal_arm_counts = []
    terminal_selected_gains = []
    target_oracle_used = []
    uncertified_overrides = []
    certified_overrides = []
    used_overrides = []
    for row in rows:
        finalist = row.get("finalist_replication") or {}
        completed.append(int(finalist.get("completed_target_count", 0) or 0))
        terminal_evaluations.append(int(
            finalist.get("terminal_kg_evaluations", 0) or 0))
        terminal_rows = _terminal_rows(finalist)
        terminal_arm_counts.extend(
            item.get("terminal_kg_arm_count") for item in terminal_rows)
        terminal_selected_gains.extend(
            item.get("terminal_kg_selected_gain") for item in terminal_rows)
        target_oracle_used.append(bool(
            finalist.get("target_oracle_used", False)))
        used = bool(row.get("replicated_finalist_used", False))
        certified = bool(row.get(
            "replicated_finalist_empirical_certificate", False))
        used_overrides.append(used)
        certified_overrides.append(used and certified)
        uncertified_overrides.append(used and not certified)
    return {
        "variant": variant,
        "heldout": heldout,
        "n_seeds": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "true_feasible_count": int(sum(
            bool(row["true_feasible"]) for row in rows)),
        "posterior_certified_count": int(sum(
            bool(row["posterior_feasible"]) for row in rows)),
        "false_feasible_count": int(sum(
            bool(row["false_feasible"]) for row in rows)),
        "median_feasible_regret": _median(
            row.get("feasible_simple_regret") for row in rows),
        "median_simple_regret": _median(
            row.get("simple_regret") for row in rows),
        "mean_constraint_violation": _mean(
            row.get("constraint_violation") for row in rows),
        "median_true_chance_margin": _median(
            row.get("true_chance_margin") for row in rows),
        "median_algorithm_time_sec": _median(
            row.get("algorithm_time_sec") for row in rows),
        "median_wall_time_sec": _median(
            row.get("wall_time_sec") for row in rows),
        "informative_completion_count": int(sum(
            count > 0 for count in completed)),
        "completed_target_count_median": _median(completed),
        "terminal_kg_evaluations_median": _median(terminal_evaluations),
        "terminal_kg_full_budget_count": int(sum(
            count == 3 for count in terminal_evaluations)),
        "terminal_kg_arm_count_median": _median(terminal_arm_counts),
        "terminal_kg_selected_gain_median": _median(
            terminal_selected_gains),
        "target_oracle_used_count": int(sum(target_oracle_used)),
        "replicated_finalist_used_count": int(sum(used_overrides)),
        "replicated_finalist_certified_count": int(sum(
            certified_overrides)),
        "replicated_finalist_uncertified_count": int(sum(
            uncertified_overrides)),
    }


def _metric_nonworse(challenger, baseline, metric, tolerance=1e-12):
    challenger_value = challenger.get(metric)
    baseline_value = baseline.get(metric)
    return bool(
        challenger_value is not None
        and baseline_value is not None
        and float(challenger_value) <= float(baseline_value) + tolerance
    )


def summarize(rows):
    grouped = {}
    by_key = {}
    for row in rows:
        key = (row["_variant"], str(row["heldout"]))
        grouped.setdefault(key, []).append(row)
        by_key[(key[0], key[1], int(row["seed"]))] = row
    summaries = [
        _group_summary(variant, heldout, group)
        for (variant, heldout), group in sorted(grouped.items())
    ]
    summary_by_key = {
        (item["variant"], item["heldout"]): item for item in summaries
    }

    expected_cells = {
        (variant, domain) for variant in VARIANTS for domain in DOMAINS
    }
    complete_cells = {
        key for key, item in summary_by_key.items()
        if tuple(item["seeds"]) == EXPECTED_SEEDS
    }
    matrix_complete = complete_cells == expected_cells

    comparisons = []
    domain_checks = []
    for domain in DOMAINS:
        baseline = summary_by_key.get((PRIMARY_BASELINE, domain))
        challenger = summary_by_key.get((PRIMARY_CHALLENGER, domain))
        common_seeds = sorted(
            set(
                seed for variant, heldout, seed in by_key
                if variant == PRIMARY_BASELINE and heldout == domain
            )
            & set(
                seed for variant, heldout, seed in by_key
                if variant == PRIMARY_CHALLENGER and heldout == domain
            )
        )
        paired_regret_differences = []
        for seed in common_seeds:
            base_row = by_key[(PRIMARY_BASELINE, domain, seed)]
            challenger_row = by_key[(PRIMARY_CHALLENGER, domain, seed)]
            if (
                base_row.get("feasible_simple_regret") is not None
                and challenger_row.get("feasible_simple_regret") is not None
            ):
                paired_regret_differences.append(float(
                    challenger_row["feasible_simple_regret"]
                    - base_row["feasible_simple_regret"]
                ))
        comparisons.append({
            "heldout": domain,
            "common_seeds": common_seeds,
            "paired_feasible_regret_delta_median": _median(
                paired_regret_differences),
            "true_feasible_count_delta": (
                None if baseline is None or challenger is None
                else challenger["true_feasible_count"]
                - baseline["true_feasible_count"]
            ),
            "false_feasible_count_delta": (
                None if baseline is None or challenger is None
                else challenger["false_feasible_count"]
                - baseline["false_feasible_count"]
            ),
            "median_feasible_regret_delta": (
                None
                if baseline is None or challenger is None
                or baseline["median_feasible_regret"] is None
                or challenger["median_feasible_regret"] is None
                else challenger["median_feasible_regret"]
                - baseline["median_feasible_regret"]
            ),
        })
        domain_checks.append({
            "heldout": domain,
            "complete": bool(
                baseline is not None and challenger is not None
                and tuple(baseline["seeds"]) == EXPECTED_SEEDS
                and tuple(challenger["seeds"]) == EXPECTED_SEEDS
            ),
            "true_feasible_nonworse": bool(
                baseline is not None and challenger is not None
                and challenger["true_feasible_count"]
                >= baseline["true_feasible_count"]
            ),
            "false_feasible_nonworse": bool(
                baseline is not None and challenger is not None
                and challenger["false_feasible_count"]
                <= baseline["false_feasible_count"]
            ),
            "median_feasible_regret_nonworse": bool(
                baseline is not None and challenger is not None
                and _metric_nonworse(
                    challenger, baseline, "median_feasible_regret")
            ),
        })

    baseline_summaries = [
        summary_by_key.get((PRIMARY_BASELINE, domain)) for domain in DOMAINS
    ]
    challenger_summaries = [
        summary_by_key.get((PRIMARY_CHALLENGER, domain)) for domain in DOMAINS
    ]
    baseline_completion = sum(
        item["informative_completion_count"]
        for item in baseline_summaries if item is not None
    )
    challenger_completion = sum(
        item["informative_completion_count"]
        for item in challenger_summaries if item is not None
    )
    target_feasibility = bool(
        all(item is not None for item in challenger_summaries)
        and challenger_summaries[0]["true_feasible_count"] == 7
        and challenger_summaries[1]["true_feasible_count"] >= 5
        and challenger_summaries[2]["true_feasible_count"] >= 5
    )
    no_target_oracle = bool(
        all(item["target_oracle_used_count"] == 0 for item in summaries)
    )
    no_uncertified_override = bool(
        all(
            item is not None
            and item["replicated_finalist_uncertified_count"] == 0
            for item in challenger_summaries
        )
    )
    suffix_budget_complete = bool(
        all(
            item is not None and item["terminal_kg_full_budget_count"] == 7
            for item in challenger_summaries
        )
    )
    informative_completion_improved = bool(
        challenger_completion
        >= baseline_completion + INFORMATIVE_COMPLETION_MIN_GAIN
    )
    per_domain_nonworse = bool(all(
        item["complete"]
        and item["true_feasible_nonworse"]
        and item["false_feasible_nonworse"]
        and item["median_feasible_regret_nonworse"]
        for item in domain_checks
    ))
    gate = {
        "baseline": PRIMARY_BASELINE,
        "challenger": PRIMARY_CHALLENGER,
        "matrix_complete": matrix_complete,
        "target_feasibility_passed": target_feasibility,
        "per_domain_nonworse_passed": per_domain_nonworse,
        "no_target_oracle_passed": no_target_oracle,
        "no_uncertified_override_passed": no_uncertified_override,
        "terminal_suffix_budget_passed": suffix_budget_complete,
        "informative_completion_baseline": baseline_completion,
        "informative_completion_challenger": challenger_completion,
        "informative_completion_min_gain": (
            INFORMATIVE_COMPLETION_MIN_GAIN),
        "informative_completion_improved": (
            informative_completion_improved),
        "domain_checks": domain_checks,
    }
    gate["passed"] = bool(
        matrix_complete
        and target_feasibility
        and per_domain_nonworse
        and no_target_oracle
        and no_uncertified_override
        and suffix_budget_complete
        and informative_completion_improved
    )
    return {
        "schema_version": 1,
        "preregistered_primary": {
            "baseline": PRIMARY_BASELINE,
            "challenger": PRIMARY_CHALLENGER,
            "domains": list(DOMAINS),
            "seeds": list(EXPECTED_SEEDS),
        },
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "primary_gate": gate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    payload = summarize(load_rows(args.inputs))
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
