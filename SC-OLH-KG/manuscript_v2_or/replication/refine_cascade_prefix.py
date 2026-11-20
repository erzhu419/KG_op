"""Refine the fixed-menu values after identifying unstable Hermite estimates."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.special import ndtr, ndtri, roots_jacobi
from scipy.stats import qmc

from restore_cascade_prefix import INDEXES, MANIFEST
from score_cascade_prefix import (
    FANTASY_W,
    SIZES,
    _read_json,
    _write_json,
    best_average,
    binomial_power_one,
    contribution_coefficients,
    query_parts,
    shortlist_menu,
    term_bounds,
)


@lru_cache(maxsize=2)
def beta_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_jacobi(order, SIZES[0] - 2, 1)
    return ndtri((nodes + 1.0) / 2.0), weights / weights.sum()


def marginal_primary(phase, cov, noise, order) -> np.ndarray:
    thresholds, weights = beta_rule(order)
    return ndtr((thresholds * noise[0] - phase[:, 0, None]) / np.sqrt(cov[0, 0])) @ weights


@lru_cache(maxsize=2)
def normal_sobol(seed: int) -> np.ndarray:
    return ndtri(qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(15))


def joint_terms(phase, cov, noise, seed, primary) -> tuple[np.ndarray, np.ndarray]:
    z = normal_sobol(seed)
    shifted = z @ np.linalg.cholesky(cov).T
    draws = phase[:, None, :] + shifted
    a0, a1, a2 = np.moveaxis(binomial_power_one(ndtr(draws / noise), SIZES), -1, 0)
    moments = np.stack([a0 * a1, (1 - a0) * a1, (1 - a0) * (1 - a1) * a2], axis=-1)
    return tuple(np.column_stack([primary, x.mean(axis=1)]) for x in (moments[:, :4096], moments))


def refine_state(state: dict) -> dict:
    started = time.process_time()
    rosters, ref = shortlist_menu(state)
    query = query_parts(state, rosters, ref)
    lower, upper, small_rule, large_rule = [], [], [], []
    for row in query:
        lis, ups, small, large = [], [], [], []
        for coeff, gmean, gcov, noise in row:
            lo, hi = term_bounds(gmean, gcov, noise)
            pos, neg = np.maximum(coeff, 0), np.minimum(coeff, 0)
            lis.append(pos @ lo.T + neg @ hi.T)
            ups.append(pos @ hi.T + neg @ lo.T)
            small.append(np.outer(coeff[:, 0], marginal_primary(gmean, gcov, noise, 24)))
            large.append(np.outer(coeff[:, 0], marginal_primary(gmean, gcov, noise, 48)))
        lower.append(best_average(lis, FANTASY_W))
        upper.append(best_average(ups, FANTASY_W))
        small_rule.append(best_average(small, FANTASY_W))
        large_rule.append(best_average(large, FANTASY_W))
    lower, upper = np.array(lower), np.array(upper)
    retained = np.flatnonzero(upper + 1e-12 >= lower.max())
    ref_single = int(np.argmax(large_rule))
    old = state["saved_query"]
    evaluation = np.unique([*retained, ref_single, old])
    screening_seconds = time.process_time() - started
    values = {}
    for q in evaluation:
        low1, high1, low2, high2 = [], [], [], []
        for coeff, gmean, gcov, noise in query[q]:
            primary = marginal_primary(gmean, gcov, noise, 48)
            for seed, low, high in ((456121190, low1, high1), (456121193, low2, high2)):
                small, fine = joint_terms(gmean, gcov, noise, seed, primary)
                low.append(coeff @ small.T)
                high.append(coeff @ fine.T)
        values[int(q)] = [best_average(bundle, FANTASY_W) for bundle in (low1, high1, low2, high2)]

    potential = np.array([values[int(q)] for q in retained])
    winners = [int(retained[i]) for i in np.argmax(potential, axis=0)]
    fine = potential[:, [1, 3]].mean(axis=1)
    winner = int(retained[np.argmax(fine)])
    first_saved = np.array(values[ref_single])
    actual_saved = np.array(values[old])
    regrets = potential.max(axis=0) - first_saved
    baseline_pred = state["prediction"]
    baseline_g = baseline_pred.cov_con + np.diag(0.25 * np.maximum(baseline_pred.r_perp_con, 0))
    current_single = []
    current_backup = []
    for roster in rosters:
        r = np.ix_(roster, roster)
        coeff = contribution_coefficients(
            baseline_pred.mean_obj[roster][None, :], baseline_pred.cov_obj[r],
            baseline_pred.aleatoric_var_obj[roster], ref,
        )[0]
        gmean = baseline_pred.mean_con[roster][None, :]
        gcov = baseline_g[r]
        noise = np.sqrt(np.maximum(baseline_pred.ucb_aleatoric_var_con[roster], 1e-8))
        primary = marginal_primary(gmean, gcov, noise, 48)
        high = [joint_terms(gmean, gcov, noise, seed, primary)[1]
                for seed in (456121190, 456121193)]
        gate = np.mean(high, axis=0)[0]
        current_single.append(float(gate[0] * coeff[0]))
        current_backup.append(float(gate[2:] @ coeff[2:]))
    return {
        "first_slot_top": ref_single, "actual_first_slot_top": old,
        "single_24_and_48_agree": int(np.argmax(small_rule)) == ref_single,
        "full_top_at_4096_32768_two_scrambles": winners, "full_top_mean32768": winner,
        "full_numerical_top_agreement": len(set(winners)) == 1,
        "candidate_count": len(query), "retained_queries": len(retained),
        "evaluated_queries": len(evaluation),
        "single_retained": bool(ref_single in retained), "actual_single_retained": bool(old in retained),
        "proxy_regret_estimates": regrets.tolist(),
        "fine_mean_full_value_gap_from_restricted_single": float(np.max(fine) - first_saved[[1, 3]].mean()),
        "fine_mean_full_value_gap_from_actual_single": float(np.max(fine) - actual_saved[[1, 3]].mean()),
        "baseline_backup_value_refined": current_backup[int(np.argmax(current_single))],
        "largest_abs_fine_scramble_difference": float(np.max(np.abs(potential[:, 1] - potential[:, 3]))),
        "worst_fine_interval_overshoot": float(max(
            0, np.max(lower[retained] - fine), np.max(fine - upper[retained])
        )),
        "screen_and_first_value_cpu_seconds": screening_seconds,
    }


def run(job: tuple[Path, Path, dict]) -> str:
    source, directory, meta = job
    dest = directory / f"{source.stem}.json"
    with dest.open("x") as stream:
        stream.write('{"status":"started"}\n')
    result = {key: meta[key] for key in ("family", "target_task_idx", "audit_status_quo_class", "anchor_p")}
    result.update(job_id=source.stem, search_calls=0, verification_calls=0, audit_calls=0)
    start = time.process_time()
    wall = time.perf_counter()
    try:
        with source.open("rb") as stream:
            result.update(status="ok", **refine_state(pickle.load(stream)))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.perf_counter() - wall)
        _write_json(dest, result)
    return source.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(MANIFEST)
    ids = [f"{family['name']}_t{index:02d}" for family in manifest["families"] for index in INDEXES]
    jobs = [
        (args.prefix / f"{job_id}.pkl", args.output, _read_json(args.previous / f"{job_id}.json"))
        for job_id in ids[args.start:args.end]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for job_id in executor.map(run, jobs):
            print(job_id, flush=True)


if __name__ == "__main__":
    main()
