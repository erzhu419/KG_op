"""Summarize offline exact-KG Monte Carlo stability diagnostics.

The input files are produced by ``diagnose_exact_kg_checkpoint.py``.  This
script only compares already-computed score vectors and truth-only audit
labels; it never reconstructs a problem or calls a simulator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(a, b):
    a = _average_ranks(a)
    b = _average_ranks(b)
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def top_k_overlap(a, b, k):
    k = max(1, min(int(k), len(a), len(b)))
    top_a = set(np.argsort(-np.asarray(a, dtype=float), kind="stable")[:k])
    top_b = set(np.argsort(-np.asarray(b, dtype=float), kind="stable")[:k])
    return float(len(top_a & top_b) / k)


def median(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.median(values))


def mean(values):
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.mean(values))


def load_audits(paths):
    by_seed = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if not payload.get("offline_only") or int(payload.get("simulator_calls", -1)) != 0:
            raise ValueError(f"not an offline zero-call audit: {path}")
        seed = int(payload["seed"])
        if seed not in by_seed:
            payload = dict(payload)
            payload["rows"] = list(payload["rows"])
            payload["input_paths"] = [str(path)]
            by_seed[seed] = payload
            continue
        merged = by_seed[seed]
        old_candidates = [item["x"] for item in merged["candidate_table"]]
        new_candidates = [item["x"] for item in payload["candidate_table"]]
        if old_candidates != new_candidates:
            raise ValueError(
                f"seed {seed} changed candidate set across diagnostic files")
        existing = {
            (
                str(row["sampling_mode"]),
                int(row["mc_samples"]),
                int(row.get("repeat", 0)),
            )
            for row in merged["rows"]
        }
        for row in payload["rows"]:
            key = (
                str(row["sampling_mode"]),
                int(row["mc_samples"]),
                int(row.get("repeat", 0)),
            )
            if key in existing:
                raise ValueError(f"duplicate seed/mode/MC/repeat row: {seed}/{key}")
            merged["rows"].append(row)
            existing.add(key)
        merged["input_paths"].append(str(path))
    return [by_seed[seed] for seed in sorted(by_seed)]


def summarize(audits, reference_mode="iid", reference_mc=2, top_k=3):
    comparisons = []
    grouped = {}
    reference_ranks = {}
    for audit in audits:
        seed = int(audit["seed"])
        candidate_table = sorted(
            audit["candidate_table"], key=lambda item: int(item["index"]))
        true_margins = np.asarray([
            float(item["true_chance_margin"]) for item in candidate_table
        ], dtype=float)
        true_objectives = np.asarray([
            float(item["true_objective"]) for item in candidate_table
        ], dtype=float)
        rows = {
            (
                str(row["sampling_mode"]),
                int(row["mc_samples"]),
                int(row.get("repeat", 0)),
            ): row
            for row in audit["rows"]
        }
        ref_key = (str(reference_mode), int(reference_mc), 0)
        if ref_key not in rows:
            raise ValueError(f"seed {seed} has no reference row {ref_key}")
        reference = rows[ref_key]
        reference_ranks[seed] = reference.get(
            "highest_score_true_feasible_rank")
        for key, row in sorted(rows.items()):
            scores = row["raw_scores"]
            ref_scores = reference["raw_scores"]
            if len(scores) != len(ref_scores):
                raise ValueError(f"seed {seed} changed candidate set across modes")
            if len(scores) != len(candidate_table):
                raise ValueError(f"seed {seed} score/candidate length mismatch")
            entropy_gain = row.get("task_entropy_gain", [0.0] * len(scores))
            weight_movement = row.get(
                "task_weight_movement", [0.0] * len(scores))
            comparison = {
                "seed": seed,
                "sampling_mode": key[0],
                "mc_samples": key[1],
                "repeat": key[2],
                "selected_true_feasible": bool(row["selected_true_feasible"]),
                "selected_true_margin": float(row["selected_true_margin"]),
                "selected_source": str(row["selected_source"]),
                "highest_score_true_feasible_rank": row.get(
                    "highest_score_true_feasible_rank"),
                "best_true_feasible_rank": row.get("best_true_feasible_rank"),
                "selected_minus_highest_feasible_score": row.get(
                    "selected_minus_highest_feasible_score"),
                "raw_negative_fraction": float(row["raw_negative_fraction"]),
                "elapsed_sec": float(row["elapsed_sec"]),
                "spearman_vs_reference": spearman(scores, ref_scores),
                "top_k_overlap_vs_reference": top_k_overlap(
                    scores, ref_scores, top_k),
                "spearman_score_vs_true_safety": spearman(
                    scores, -true_margins),
                "spearman_score_vs_true_objective_quality": spearman(
                    scores, -true_objectives),
                "spearman_score_vs_task_entropy_gain": spearman(
                    scores, entropy_gain),
                "spearman_score_vs_task_weight_movement": spearman(
                    scores, weight_movement),
                "same_winner_as_reference": (
                    int(row["selected_index"])
                    == int(reference["selected_index"])
                ),
            }
            comparisons.append(comparison)
            grouped.setdefault(key[:2], []).append(comparison)

    variants = []
    for key, rows in sorted(grouped.items()):
        seeds = sorted({row["seed"] for row in rows})
        ranks = [row["highest_score_true_feasible_rank"] for row in rows]
        rank_improvements = []
        near_top = 0
        for row in rows:
            rank = row["highest_score_true_feasible_rank"]
            ref_rank = reference_ranks[row["seed"]]
            if rank is not None and ref_rank is not None:
                rank_improvements.append(float(ref_rank) - float(rank))
                near_top += int(float(rank) <= max(3, math.ceil(0.1 * len(
                    next(audit["candidate_table"] for audit in audits
                         if int(audit["seed"]) == row["seed"])
                ))))
        variants.append({
            "sampling_mode": key[0],
            "mc_samples": key[1],
            "n_seeds": len(seeds),
            "n_rows": len(rows),
            "selected_true_feasible_row_count": int(sum(
                row["selected_true_feasible"] for row in rows)),
            "selected_true_feasible_rate": mean(
                row["selected_true_feasible"] for row in rows),
            "selected_true_feasible_seed_count_all_repeats": int(sum(
                all(
                    row["selected_true_feasible"]
                    for row in rows if row["seed"] == seed
                )
                for seed in seeds
            )),
            "selected_true_feasible_seed_count_any_repeat": int(sum(
                any(
                    row["selected_true_feasible"]
                    for row in rows if row["seed"] == seed
                )
                for seed in seeds
            )),
            "selected_true_margin_median": median(
                row["selected_true_margin"] for row in rows),
            "highest_score_true_feasible_rank_median": median(ranks),
            "best_true_feasible_rank_median": median(
                row["best_true_feasible_rank"] for row in rows),
            "safe_score_gap_median": median(
                row["selected_minus_highest_feasible_score"] for row in rows),
            "safe_rank_improvement_median": median(rank_improvements),
            "near_top_safe_count": int(near_top),
            "near_top_safe_rate": float(near_top / max(len(rows), 1)),
            "negative_fraction_mean": mean(
                row["raw_negative_fraction"] for row in rows),
            "spearman_vs_reference_median": median(
                row["spearman_vs_reference"] for row in rows),
            "top_k_overlap_vs_reference_mean": mean(
                row["top_k_overlap_vs_reference"] for row in rows),
            "score_vs_true_safety_spearman_median": median(
                row["spearman_score_vs_true_safety"] for row in rows),
            "score_vs_true_objective_spearman_median": median(
                row["spearman_score_vs_true_objective_quality"] for row in rows),
            "score_vs_task_entropy_spearman_median": median(
                row["spearman_score_vs_task_entropy_gain"] for row in rows),
            "score_vs_task_weight_movement_spearman_median": median(
                row["spearman_score_vs_task_weight_movement"] for row in rows),
            "same_winner_count": int(sum(
                row["same_winner_as_reference"] for row in rows)),
            "elapsed_sec_median": median(row["elapsed_sec"] for row in rows),
        })

    reference = next(
        row for row in variants
        if row["sampling_mode"] == reference_mode
        and row["mc_samples"] == int(reference_mc)
    )
    challengers = [row for row in variants if row is not reference]
    best = min(challengers, key=lambda row: (
        -row["selected_true_feasible_rate"],
        float("inf") if row["highest_score_true_feasible_rank_median"] is None
        else row["highest_score_true_feasible_rank_median"],
        float("inf") if row["safe_score_gap_median"] is None
        else row["safe_score_gap_median"],
    )) if challengers else reference
    if (
        best["selected_true_feasible_rate"] >= 2.0 / 3.0
        and best["near_top_safe_rate"] >= 2.0 / 3.0
        and (best["safe_rank_improvement_median"] or 0.0) >= 2.0
    ):
        verdict = "estimator_noise_primary"
    elif (
        (best["safe_rank_improvement_median"] or 0.0) >= 2.0
        or (best["spearman_vs_reference_median"] or 1.0) < 0.8
        or best["selected_true_feasible_rate"]
        > reference["selected_true_feasible_rate"]
    ):
        verdict = "mixed_estimator_and_representation"
    else:
        verdict = "representation_or_terminal_model_primary"
    return {
        "schema_version": 1,
        "offline_only": True,
        "simulator_calls": 0,
        "reference": {
            "sampling_mode": str(reference_mode),
            "mc_samples": int(reference_mc),
            "top_k": int(top_k),
        },
        "predeclared_verdict": verdict,
        "best_challenger": {
            "sampling_mode": best["sampling_mode"],
            "mc_samples": best["mc_samples"],
        },
        "variants": variants,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--reference-mode", default="iid")
    parser.add_argument("--reference-mc", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    payload = summarize(
        load_audits(args.inputs),
        reference_mode=args.reference_mode,
        reference_mc=args.reference_mc,
        top_k=args.top_k,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
