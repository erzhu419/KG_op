#!/usr/bin/env python3
"""Merge fresh-seed shards for the posthoc universal-library diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from scipy.stats import beta


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def exact_binomial_lower(successes, trials, delta):
    successes = int(successes)
    trials = int(trials)
    if trials <= 0 or not 0.0 < float(delta) < 1.0:
        raise ValueError("invalid exact-binomial inputs")
    if successes == 0:
        return 0.0
    return float(beta.ppf(
        float(delta), successes, trials - successes + 1))


def analyze(
    payloads,
    *,
    expected_library_size,
    target_probability=0.95,
    familywise_delta=0.05,
    redact_policy_vectors=True,
):
    if not payloads:
        raise ValueError("posthoc diagnostic has no shard payloads")
    provenances = [payload.get("execution_provenance") for payload in payloads]
    if any(value != provenances[0] for value in provenances):
        raise ValueError("posthoc diagnostic shards use different snapshots")
    candidates = []
    for payload in payloads:
        if payload.get("partial_observations"):
            raise ValueError("posthoc diagnostic contains incomplete candidates")
        candidates.extend(payload.get("candidates", ()))
    by_index = {}
    for row in candidates:
        index = int(row["source_index"])
        if index in by_index:
            raise ValueError("posthoc diagnostic has duplicate candidate indices")
        by_index[index] = row
    expected_indices = list(range(int(expected_library_size)))
    if sorted(by_index) != expected_indices:
        raise ValueError("posthoc diagnostic does not cover the fixed library")
    seed_lists = {
        tuple(map(int, row["validation"]["seeds"]))
        for row in by_index.values()
    }
    if len(seed_lists) != 1:
        raise ValueError("posthoc library candidates lack common fresh seeds")

    per_candidate_delta = float(familywise_delta) / len(by_index)
    rows = []
    for index in expected_indices:
        source = by_index[index]
        validation = source["validation"]
        trials = int(validation["R"])
        successes = int(validation["feasible_count"])
        probability = float(validation["feasible_probability"])
        lower = exact_binomial_lower(
            successes, trials, per_candidate_delta)
        row = {
            "source_index": index,
            "R": trials,
            "feasible_count": successes,
            "feasible_probability": probability,
            "familywise_exact_lower": lower,
            "point_feasible": bool(probability >= float(target_probability)),
            "familywise_certified": bool(lower >= float(target_probability)),
            "mean_vector": list(map(float, validation["mean"])),
        }
        if not redact_policy_vectors:
            row["x"] = list(map(int, source["x"]))
        rows.append(row)
    replications = {row["R"] for row in rows}
    if len(replications) != 1:
        raise ValueError("posthoc library candidates used unequal budgets")
    probabilities = [row["feasible_probability"] for row in rows]
    best_probability = max(probabilities)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "evidence_phase": "posthoc_certifiability_diagnostic",
        "diagnostic_only": True,
        "admissible_for_method_selection": False,
        "admissible_for_confirmatory_claim": False,
        "target_oracle_used": False,
        "historical_target_anchor_used": False,
        "policy_vectors_exported": not bool(redact_policy_vectors),
        "library_size": len(rows),
        "fresh_seed_replications_per_candidate": next(iter(replications)),
        "fresh_seed_count": len(next(iter(seed_lists))),
        "target_verification_calls": (
            len(rows) * next(iter(replications))
        ),
        "target_probability": float(target_probability),
        "familywise_delta": float(familywise_delta),
        "per_candidate_delta": per_candidate_delta,
        "point_feasible_candidate_count": sum(
            row["point_feasible"] for row in rows),
        "familywise_certified_candidate_count": sum(
            row["familywise_certified"] for row in rows),
        "maximum_empirical_feasible_probability": best_probability,
        "median_empirical_feasible_probability": float(
            statistics.median(probabilities)),
        "best_source_indices": [
            row["source_index"] for row in rows
            if row["feasible_probability"] == best_probability
        ],
        "rows": rows,
        "execution_provenance": provenances[0],
    }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-library-size", type=int, required=True)
    parser.add_argument("--target-probability", type=float, default=0.95)
    parser.add_argument("--familywise-delta", type=float, default=0.05)
    parser.add_argument(
        "--redact-policy-vectors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    payload = analyze(
        [_read_json(path) for path in args.paths],
        expected_library_size=args.expected_library_size,
        target_probability=args.target_probability,
        familywise_delta=args.familywise_delta,
        redact_policy_vectors=args.redact_policy_vectors,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "library_size": payload["library_size"],
        "point_feasible_candidate_count": payload[
            "point_feasible_candidate_count"],
        "familywise_certified_candidate_count": payload[
            "familywise_certified_candidate_count"],
        "out": str(args.out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
