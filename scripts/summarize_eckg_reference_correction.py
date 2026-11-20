#!/usr/bin/env python3
"""Pack or summarize the 24-state correction from saved candidate values, without simulation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


FAMILIES = ("Queueing_CrossHorizon", "Traffic_Ingolstadt")
STAGES = ("step_0", "step_59")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pack(corrected_dir: Path, original_dir: Path) -> dict:
    corrected_jobs = read(corrected_dir / "jobs.json")
    original_jobs = read(original_dir / "jobs.json")
    if corrected_jobs["jobs"] != original_jobs["jobs"] or len(original_jobs["jobs"]) != 12:
        raise ValueError("The correction requires the originally specified twelve family/task pairs")
    tasks = []
    for job in original_jobs["jobs"]:
        relative = Path("items") / (job["job_id"] + ".json")
        result = read(corrected_dir / relative)
        original = read(original_dir / relative)
        for key in ("job_id", "family", "target_task_idx", "task_id", "cell_seed"):
            if result[key] != original[key]:
                raise ValueError(f"{relative}: changed {key}")
        compact = {k: v for k, v in result.items() if k not in STAGES}
        for stage in STAGES:
            state, prior = result[stage], original[stage]
            for key in ("n_unique_evaluated", "q_eckg16", "q_single", "q_feas", "q_menu_ref_opt", "q_term_ref_opt"):
                if state[key] != prior[key]:
                    raise ValueError(f"{relative}/{stage}: changed {key}")
            # The high-resolution quadrature and the selected queries were already paid for.
            # Check their finite-draw results before using the recovered candidate vectors.
            for key in ("menu_ref_max", "menu_ref_at_eckg16", "menu_ref_at_single", "menu_ref_at_feas",
                        "actual_term_ref_max", "actual_term_ref_at_eckg16", "actual_term_ref_at_single", "actual_term_ref_at_feas"):
                np.testing.assert_allclose(state[key], prior[key], rtol=1e-8, atol=1e-10, err_msg=f"{relative}/{stage}: {key}")
            compact[stage] = {k: state[k] for k in (
                "fast_outer_nodes", "fast_inner_nodes", "reference_seed", "n_unique_evaluated", "screened_candidate_ids",
                "q_eckg16", "q_single", "q_feas", "fast16_wall_ms", "ref_all160_wall_sec",
                "fast_menu_scores", "reference_menu_scores", "reference_nomination_scores",
            )}
        tasks.append(compact)
    return {
        "diagnostic_name": "eckg_reference_states_v4_correction",
        "frozen_execution_sha256": original_jobs["execution_code_sha256"],
        "instrumented_execution_sha256": corrected_jobs["execution_code_sha256"],
        "selection_rule": "first argmax; Stage-A membership is read before Stage-B scores",
        "fast_nodes": [16, 128], "reference_nodes": [256, 1024],
        "tasks": tasks,
    }


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(spearmanr(left, right).statistic)


def state_statistics(state: dict) -> dict:
    fast, menu, term = (np.asarray(state[key], dtype=float) for key in
                        ("fast_menu_scores", "reference_menu_scores", "reference_nomination_scores"))
    if any(a.shape != (160,) or not np.all(np.isfinite(a)) for a in (fast, menu, term)):
        raise ValueError("Each state must contain three complete finite 160-candidate score vectors")
    if (state["fast_outer_nodes"], state["fast_inner_nodes"]) != (16, 128):
        raise ValueError("Fast scores must use the sixteen outer and 128 inner production nodes")
    screen = state["screened_candidate_ids"]
    if not 1 <= len(screen) <= 16 or len(set(screen)) != len(screen) or not all(0 <= q < 160 for q in screen):
        raise ValueError("Invalid Stage-A contender list")
    ec, single, feas = (state[k] for k in ("q_eckg16", "q_single", "q_feas"))
    m, t = int(np.argmax(menu)), int(np.argmax(term))
    out = {
        "stage_a_top16_contains_menu_opt_rate": float(m in screen),
        "stage_a_top16_contains_term_opt_rate": float(t in screen),
        "exact_match_eckg16_vs_menu_opt_rate": float(ec == m),
        "exact_match_eckg16_vs_term_opt_rate": float(ec == t),
        "quadrature_spearman_mean": rank_correlation(fast, menu),
        "menu_vs_actual_terminal_spearman_mean": rank_correlation(menu, term),
        "mean_n_unique_evaluated": state["n_unique_evaluated"],
        "mean_screened_candidates": len(screen),
        "fast16_mean_wall_ms": state["fast16_wall_ms"],
        "ref_all160_mean_wall_sec": state["ref_all160_wall_sec"],
        "menu_ref_rel_regret_eckg16_mean": (menu[m] - menu[ec]) / max(menu[m], 1e-12),
        "mean_max_abs_fast_minus_reference_menu_x1e3": 1000 * np.max(np.abs(fast-menu)),
        "mean_abs_fast_minus_reference_menu_x1e3": 1000 * np.mean(np.abs(fast-menu)),
        "reference_menu_winner_margin_mean_x1e3": 1000 * (menu[m] - np.partition(menu, -2)[-2]),
        "reference_nomination_winner_margin_mean_x1e3": 1000 * (term[t] - np.partition(term, -2)[-2]),
        "reference_nomination_screen_loss_mean_x1e3": 1000 * (term[t] - np.max(term[screen])),
        "reference_nomination_query_loss_mean_x1e3": 1000 * (term[t] - term[ec]),
        "actual_term_eckg16_beats_or_ties_single_count": int(term[ec] >= term[single] - 1e-12),
        "actual_term_eckg16_beats_or_ties_feas_count": int(term[ec] >= term[feas] - 1e-12),
    }
    for key, q in (("max", t), ("eckg16", ec), ("single", single), ("feas", feas)):
        out[f"actual_term_ref_{key}_mean_x1e3"] = 1000 * term[q]
    for key, q in (("max", m), ("eckg16", ec), ("single", single), ("feas", feas)):
        out[f"menu_ref_{key}_mean_x1e3"] = 1000 * menu[q]
    return out


def summarize(bundle: dict) -> dict:
    tasks = bundle["tasks"]
    expected = Counter((fam, i) for fam in FAMILIES for i in range(6))
    if Counter((t["family"], t["target_task_idx"]) for t in tasks) != expected:
        raise ValueError("Incomplete or repeated preselected tasks")
    families = {}
    for family in FAMILIES:
        families[family] = {}
        for stage in STAGES:
            rows = [state_statistics(t[stage]) for t in tasks if t["family"] == family]
            group = {"n_states": len(rows)}
            for key in rows[0]:
                values = [r[key] for r in rows if r[key] is not None]
                if key.endswith("_count"):
                    group[key] = sum(values)
                elif values:
                    group[key] = float(np.mean(values))
                else:
                    group[key] = None
                if "spearman" in key:
                    group[key.removesuffix("_mean") + "_defined_states"] = len(values)
            families[family][stage] = group
    fees = {key: sum(t[key] for t in tasks) for key in (
        "diagnostic_search_calls", "diagnostic_verification_calls", "diagnostic_audit_calls",
        "worker_cpu_seconds", "worker_wall_seconds",
    )}
    return {
        "diagnostic_name": "eckg_reference_validation_corrected",
        "frozen_execution_sha256": bundle["frozen_execution_sha256"],
        "instrumented_execution_sha256": bundle["instrumented_execution_sha256"],
        "fast_outer_sobol_points": 16, "fast_inner_sobol_points": 128,
        "outer_sobol_points": 256, "inner_sobol_points": 1024,
        "candidates_scored_per_state": 160, "total_states": 2 * len(tasks),
        "reference_law": "single nomination under the frozen working posterior; a single high-resolution scramble distinct from production, shared across families at matched task index and stage",
        "diagnostic_fees_completed_tasks": fees,
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("pack", help="Bind corrected worker outputs to the original state receipts")
    p.add_argument("--corrected-dir", required=True, type=Path)
    p.add_argument("--original-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("summarize", help="Rebuild all reported statistics from the portable state vectors")
    p.add_argument("--states", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "pack":
        result = pack(args.corrected_dir, args.original_dir)
    else:
        result = summarize(read(args.states))
    write(args.output, result)


if __name__ == "__main__":
    main()
