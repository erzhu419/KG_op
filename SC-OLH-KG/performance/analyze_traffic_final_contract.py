#!/usr/bin/env python3
"""Audit fresh-seed SUMO shortlists under a familywise exact certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.execution_provenance import (  # noqa: E402
    attach_execution_provenance,
)


def exact_binomial_lower(successes, trials, delta):
    """One-sided Clopper-Pearson lower bound for a Bernoulli probability."""

    successes = int(successes)
    trials = int(trials)
    delta = float(delta)
    if trials <= 0 or not 0.0 < delta < 1.0:
        raise ValueError("exact binomial certificate has invalid inputs")
    if successes < 0 or successes > trials:
        raise ValueError("success count is outside [0, trials]")
    if successes == 0:
        return 0.0
    return float(beta.ppf(delta, successes, trials - successes + 1))


def audit_oos_payload(
    payload,
    *,
    target_probability=0.95,
    familywise_delta=0.05,
    fallback_source_index=1,
):
    rows = sorted(
        payload.get("candidates", ()),
        key=lambda row: int(row["source_index"]),
    )
    if not rows:
        raise ValueError("traffic OOS payload contains no candidates")
    source_indices = [int(row["source_index"]) for row in rows]
    if source_indices != list(range(len(rows))):
        raise ValueError("traffic shortlist source indices are not contiguous")
    if len({int(row["run_seed"]) for row in rows}) != 1:
        raise ValueError("one traffic OOS payload must contain one search seed")
    seed_lists = {
        tuple(map(int, row["validation"]["seeds"])) for row in rows
    }
    if len(seed_lists) != 1:
        raise ValueError("traffic shortlist did not use common fresh seeds")

    per_candidate_delta = float(familywise_delta) / float(len(rows))
    certified_rows = []
    audit_rows = []
    for row in rows:
        validation = row["validation"]
        trials = int(validation["R"])
        successes = int(validation["feasible_count"])
        lower = exact_binomial_lower(
            successes,
            trials,
            per_candidate_delta,
        )
        audited = {
            "source_index": int(row["source_index"]),
            "shortlist_rank": int(row["source_index"]) + 1,
            "x": list(map(int, row["x"])),
            "R": trials,
            "feasible_count": successes,
            "feasible_probability": float(
                validation["feasible_probability"]),
            "familywise_exact_lower": lower,
            "certified": bool(lower >= float(target_probability)),
            "mean_vector": list(map(float, validation["mean"])),
        }
        audit_rows.append(audited)
        if audited["certified"]:
            certified_rows.append(audited)

    if certified_rows:
        deployed = certified_rows[0]
        deployment_status = "certified_fixed_shortlist_rank"
    else:
        matching = [
            row for row in audit_rows
            if row["source_index"] == int(fallback_source_index)
        ]
        deployed = matching[0] if matching else audit_rows[0]
        deployment_status = "uncertified_frozen_incumbent_fallback"
    trials_per_candidate = {row["R"] for row in audit_rows}
    if len(trials_per_candidate) != 1:
        raise ValueError("traffic shortlist candidates used unequal budgets")
    return {
        "schema_version": 1,
        "status": "ok",
        "run_seed": int(rows[0]["run_seed"]),
        "method": str(rows[0]["method"]),
        "partition": str(rows[0]["partition"]),
        "target_probability": float(target_probability),
        "familywise_delta": float(familywise_delta),
        "per_candidate_delta": per_candidate_delta,
        "iid_fresh_seed_assumption": True,
        "certificate": "one_sided_clopper_pearson_bonferroni",
        "fixed_shortlist_order": True,
        "verification_samples_update_optimizer": False,
        "verification_samples_used_to_reorder_shortlist": False,
        "candidate_rows": audit_rows,
        "certified_candidate_count": int(len(certified_rows)),
        "deployed_source_index": int(deployed["source_index"]),
        "deployed_shortlist_rank": int(deployed["shortlist_rank"]),
        "deployed_x": deployed["x"],
        "deployed_certified": bool(deployed["certified"]),
        "deployed_feasible_probability": float(
            deployed["feasible_probability"]),
        "deployed_familywise_exact_lower": float(
            deployed["familywise_exact_lower"]),
        "deployed_mean_vector": deployed["mean_vector"],
        "deployment_status": deployment_status,
        "verification_calls": int(
            len(audit_rows) * next(iter(trials_per_candidate))),
    }


def analyze(
    paths,
    *,
    target_probability=0.95,
    familywise_delta=0.05,
    source_domains=(
        "FactorShockStatePolicyRZDT1",
        "InventorySupplyChain",
    ),
    excluded_nearest_source_analogue="QueueResourceControl",
    target_domain="Ingolstadt21Traffic",
    information_track="domain_blind_external_holdout",
    source_selection_mode="domain_blind_exclude_nearest",
    source_split_heldout=None,
    heldout_task_family_identifier_used=False,
    redact_policy_vectors=True,
    target_search_calls=13,
    target_initial_design_calls=10,
    evidence_phase="development_gate",
    method_selected_using_target_domain_development_results=False,
    confirmatory_holdout_seed_disjoint_from_development=False,
):
    target_search_calls = int(target_search_calls)
    target_initial_design_calls = int(target_initial_design_calls)
    if target_search_calls <= 0:
        raise ValueError("target search calls must be positive")
    if not 0 <= target_initial_design_calls <= target_search_calls:
        raise ValueError(
            "target initial-design calls must lie in [0, target search calls]")
    evidence_phase = str(evidence_phase)
    if evidence_phase not in {
        "development_gate",
        "confirmatory_holdout",
        "diagnostic_control",
    }:
        raise ValueError("unknown traffic evidence phase")
    if (
        evidence_phase == "confirmatory_holdout"
        and not confirmatory_holdout_seed_disjoint_from_development
    ):
        raise ValueError(
            "confirmatory traffic evidence requires a disjoint seed split")
    source_payloads = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in paths
    ]
    provenances = [
        payload.get("execution_provenance")
        for payload in source_payloads
    ]
    if any(provenance is not None for provenance in provenances):
        if any(provenance != provenances[0] for provenance in provenances):
            raise ValueError(
                "traffic OOS payloads use different execution snapshots")
    rows = [
        audit_oos_payload(
            payload,
            target_probability=target_probability,
            familywise_delta=familywise_delta,
        )
        for payload in source_payloads
    ]
    if not rows:
        raise ValueError("no traffic OOS payloads were provided")
    seeds = [int(row["run_seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("traffic OOS audit contains duplicate search seeds")
    if redact_policy_vectors:
        for row in rows:
            row.pop("deployed_x", None)
            for candidate in row.get("candidate_rows", ()):
                candidate.pop("x", None)
    verification_calls = int(statistics.median(
        row["verification_calls"] for row in rows))
    target_total_calls = target_search_calls + verification_calls
    source_calls = 384
    payload = {
        "schema_version": 1,
        "status": "complete",
        "n_seeds": len(rows),
        "seeds": sorted(seeds),
        "certified_seed_count": int(sum(
            row["deployed_certified"] for row in rows)),
        "certified_seed_rate": float(sum(
            row["deployed_certified"] for row in rows) / len(rows)),
        "median_deployed_feasible_probability": float(statistics.median(
            row["deployed_feasible_probability"] for row in rows)),
        "median_deployed_familywise_exact_lower": float(statistics.median(
            row["deployed_familywise_exact_lower"] for row in rows)),
        "source_calls_per_run": source_calls,
        "target_initial_design_calls_per_run": target_initial_design_calls,
        "target_adaptive_search_calls_per_run": (
            target_search_calls - target_initial_design_calls),
        "target_search_calls_per_run": target_search_calls,
        "target_verification_calls_per_run": verification_calls,
        "target_total_calls_per_run": target_total_calls,
        "source_plus_target_total_calls_per_run": (
            source_calls + target_total_calls),
        # Backward-compatible alias. Unlike target_total_calls_per_run, this
        # legacy field includes the frozen source archive cost.
        "total_calls_per_run": source_calls + target_total_calls,
        "policy_vectors_exported": not bool(redact_policy_vectors),
        "information_contract": {
            "track": str(information_track),
            "source_selection_mode": str(source_selection_mode),
            "source_domains": list(map(str, source_domains)),
            "excluded_nearest_source_analogue": str(
                excluded_nearest_source_analogue),
            "source_split_heldout": (
                None
                if source_split_heldout is None
                else str(source_split_heldout)
            ),
            "target_domain": str(target_domain),
            "heldout_task_family_identifier_used_by_proposal": bool(
                heldout_task_family_identifier_used),
            "target_labels_used_to_fit_proposal": False,
            "target_oracle_used": False,
            "historical_target_anchor_used": False,
            "evidence_phase": evidence_phase,
            "method_selected_using_target_domain_development_results": bool(
                method_selected_using_target_domain_development_results),
            "evaluation_outcomes_used_for_method_selection": False,
            "confirmatory_holdout_seed_disjoint_from_development": bool(
                confirmatory_holdout_seed_disjoint_from_development),
        },
        "rows": sorted(rows, key=lambda row: row["run_seed"]),
    }
    attach_execution_provenance(payload)
    return payload


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-probability", type=float, default=0.95)
    parser.add_argument("--familywise-delta", type=float, default=0.05)
    parser.add_argument(
        "--source-domains",
        default="FactorShockStatePolicyRZDT1,InventorySupplyChain",
    )
    parser.add_argument(
        "--excluded-nearest-source-analogue",
        default="QueueResourceControl",
    )
    parser.add_argument("--target-domain", default="Ingolstadt21Traffic")
    parser.add_argument(
        "--information-track",
        default="domain_blind_external_holdout",
    )
    parser.add_argument(
        "--source-selection-mode",
        default="domain_blind_exclude_nearest",
    )
    parser.add_argument("--source-split-heldout", default="")
    parser.add_argument(
        "--heldout-task-family-identifier-used",
        action="store_true",
    )
    parser.add_argument(
        "--redact-policy-vectors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--target-search-calls", type=int, default=13)
    parser.add_argument("--target-initial-design-calls", type=int, default=10)
    parser.add_argument(
        "--evidence-phase",
        choices=(
            "development_gate",
            "confirmatory_holdout",
            "diagnostic_control",
        ),
        default="development_gate",
    )
    parser.add_argument(
        "--method-selected-using-target-domain-development-results",
        action="store_true",
    )
    parser.add_argument(
        "--confirmatory-holdout-seed-disjoint-from-development",
        action="store_true",
    )
    args = parser.parse_args()
    payload = analyze(
        args.paths,
        target_probability=args.target_probability,
        familywise_delta=args.familywise_delta,
        source_domains=tuple(
            value.strip()
            for value in args.source_domains.split(",")
            if value.strip()
        ),
        excluded_nearest_source_analogue=(
            args.excluded_nearest_source_analogue),
        target_domain=args.target_domain,
        information_track=args.information_track,
        source_selection_mode=args.source_selection_mode,
        source_split_heldout=(
            args.source_split_heldout.strip() or None),
        heldout_task_family_identifier_used=(
            args.heldout_task_family_identifier_used),
        redact_policy_vectors=args.redact_policy_vectors,
        target_search_calls=args.target_search_calls,
        target_initial_design_calls=args.target_initial_design_calls,
        evidence_phase=args.evidence_phase,
        method_selected_using_target_domain_development_results=(
            args.method_selected_using_target_domain_development_results),
        confirmatory_holdout_seed_disjoint_from_development=(
            args.confirmatory_holdout_seed_disjoint_from_development),
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "n_seeds": payload["n_seeds"],
        "certified_seed_count": payload["certified_seed_count"],
        "certified_seed_rate": payload["certified_seed_rate"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
