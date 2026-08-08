#!/usr/bin/env python3
"""Post-decision temporal stability audit for the OPSD V2 experiment.

This audit never changes the frozen shortlist, certification decision, or
method. It reports descriptive safety across chronological start-index blocks
and a small set of physically nonoverlapping windows. Neither summary is
treated as an additional iid certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_quality import json_safe  # noqa: E402
from problems.energy_reliability import OPSDStorageReliabilityProblem  # noqa: E402


SOURCE_CONTRACT = "opsd_region_heldout_profile_design_v2"
AUDIT_CONTRACT = "opsd_postdecision_temporal_block_audit_v1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nonoverlapping_window_starts(starts, horizon):
    """Greedily retain chronological starts whose physical windows do not overlap."""

    starts = np.unique(np.asarray(starts, dtype=np.int64).reshape(-1))
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    selected = []
    next_allowed = None
    for start in starts:
        if next_allowed is None or int(start) >= next_allowed:
            selected.append(int(start))
            next_allowed = int(start) + horizon
    return np.asarray(selected, dtype=np.int64)


def _summarize(values, tau):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) == 0:
        raise ValueError("temporal audit requires nonempty objective/constraint rows")
    return {
        "window_count": int(len(values)),
        "feasibility_probability": float(np.mean(values[:, 1] <= float(tau))),
        "objective_mean": float(np.mean(values[:, 0])),
        "constraint_mean": float(np.mean(values[:, 1])),
        "constraint_q95": float(np.quantile(values[:, 1], 0.95)),
    }


def audit_frozen_policy(
    problem,
    point,
    *,
    split="verification",
    chronological_blocks=4,
    maximum_sampled_starts=512,
):
    """Audit one already-frozen policy without changing its certificate."""

    block_count = int(chronological_blocks)
    maximum = int(maximum_sampled_starts)
    if block_count < 2 or maximum < block_count:
        raise ValueError("temporal audit needs at least two populated blocks")
    starts = problem.split_window_starts(split)
    if len(starts) > maximum:
        indices = np.linspace(0, len(starts) - 1, maximum).round().astype(int)
        sampled_starts = starts[indices]
    else:
        sampled_starts = starts
    sampled_values = problem.evaluate_window_starts(point, sampled_starts)
    blocks = []
    for block_index, indices in enumerate(
        np.array_split(np.arange(len(sampled_starts)), block_count)
    ):
        if len(indices) == 0:
            continue
        summary = _summarize(sampled_values[indices], problem.tau)
        summary.update({
            "block_index": int(block_index),
            "first_start_index": int(sampled_starts[indices[0]]),
            "last_start_index": int(sampled_starts[indices[-1]]),
        })
        blocks.append(summary)
    disjoint_starts = nonoverlapping_window_starts(starts, problem.d)
    disjoint_values = problem.evaluate_window_starts(point, disjoint_starts)
    block_probabilities = [row["feasibility_probability"] for row in blocks]
    return {
        "split": str(split),
        "admissible_start_count": int(len(starts)),
        "sampled_start_count": int(len(sampled_starts)),
        "chronological_block_count": int(len(blocks)),
        "chronological_blocks": blocks,
        "sampled_distribution_summary": _summarize(
            sampled_values, problem.tau),
        "minimum_chronological_block_feasibility_probability": float(
            min(block_probabilities)),
        "maximum_chronological_block_feasibility_probability": float(
            max(block_probabilities)),
        "nonoverlapping_start_count": int(len(disjoint_starts)),
        "nonoverlapping_summary": _summarize(
            disjoint_values, problem.tau),
        "inferential_certificate_claimed": False,
        "postdecision_only": True,
        "physical_windows_within_chronological_blocks_may_overlap": True,
        "nonoverlapping_summary_is_descriptive_due_to_small_sample": True,
    }


def audit_result(
    result_path,
    *,
    data_path,
    chronological_blocks=4,
    maximum_sampled_starts=512,
):
    result_path = Path(result_path)
    row = json.loads(result_path.read_text(encoding="utf-8"))
    if row.get("contract_id") != SOURCE_CONTRACT or row.get("status") != "ok":
        raise ValueError("temporal audit received an incompatible energy result")
    payload = {
        "schema_version": 1,
        "contract_id": AUDIT_CONTRACT,
        "status": "not_certified",
        "source_result_sha256": _sha256(result_path),
        "source_result_path": str(result_path),
        "target_market": row["target_market"],
        "target_region": row["target_region"],
        "target_seed": int(row["target_seed"]),
        "arm": row["arm"],
        "independently_certified": bool(row["independently_certified"]),
        "postdecision_only": True,
        "used_to_modify_method_or_certificate": False,
    }
    selected_rank = row["verification"].get("selected_shortlist_rank")
    if not row["independently_certified"] or selected_rank is None:
        return payload
    shortlist = row.get("shortlist", [])
    if not 1 <= int(selected_rank) <= len(shortlist):
        raise ValueError("certified result has no matching frozen shortlist member")
    point = tuple(int(value) for value in shortlist[int(selected_rank) - 1]["point"])
    problem = OPSDStorageReliabilityProblem(
        data_path,
        market=row["target_market"],
        year=int(row["year"]),
        d=int(row["nominal_dimension"]),
        alpha=float(row["alpha"]),
        required_splits=("verification",),
    )
    payload.update({
        "status": "complete",
        "selected_shortlist_rank": int(selected_rank),
        "temporal_audit": audit_frozen_policy(
            problem,
            point,
            chronological_blocks=chronological_blocks,
            maximum_sampled_starts=maximum_sampled_starts,
        ),
    })
    return payload


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chronological-blocks", type=int, default=4)
    parser.add_argument("--maximum-sampled-starts", type=int, default=512)
    args = parser.parse_args()
    payload = audit_result(
        args.result,
        data_path=args.data,
        chronological_blocks=args.chronological_blocks,
        maximum_sampled_starts=args.maximum_sampled_starts,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "target_market": payload["target_market"],
        "target_seed": payload["target_seed"],
        "arm": payload["arm"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
