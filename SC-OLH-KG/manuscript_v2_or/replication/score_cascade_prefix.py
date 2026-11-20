"""Score a fixed shortlist menu with single-slot and guarded one-step values."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import spearmanr

from v2_or.acquisition.verification_kg import _gauss_hermite_1d
from v2_or.dispatch.campaign_runner import _read_json, _write_json
from v2_or.verification.gated_verifier_v2 import (
    exact_binomial_critical_k,
    posterior_binomial_gate_power,
    _select_empirical_feasible_incumbent_with_idx,
)

from restore_cascade_prefix import COHORT, INDEXES, MANIFEST


SIZES = np.array([112, 128, 140])
ERROR_LEVELS = (0.025, 0.015, 0.010)
normal_rule = lru_cache(maxsize=3)(_gauss_hermite_1d)
FANTASY_Z, FANTASY_W = _gauss_hermite_1d(5)
QUANTILES = ndtri(np.arange(1, 512) / 512)


def binomial_power_one(violation: np.ndarray, n: np.ndarray) -> np.ndarray:
    return (1.0 - violation) ** (n - 1) * (1.0 + (n - 1) * violation)


def shortlist_menu(state: dict) -> tuple[np.ndarray, float]:
    pred = state["prediction"]
    first = state["initial_pred"]
    alpha = state["alpha"]
    if [exact_binomial_critical_k(n, alpha, d) for n, d in zip(SIZES, ERROR_LEVELS)] != [1, 1, 1]:
        raise ValueError("The frozen exact-gate budget requires critical count 1 in each slot")
    order = np.argsort(pred.mean_obj, kind="stable")[:64]
    ref = max(state["reference_obj"], pred.mean_obj.min() + 0.20 * np.std(pred.mean_obj))
    incumbent_powers = posterior_binomial_gate_power(
        first.mean_con, first.std_con, first.ucb_aleatoric_var_con, r_perp_con=first.r_perp_con,
        n_samples=SIZES[1], alpha=alpha, delta_slot=ERROR_LEVELS[1],
    )
    _, incumbent = _select_empirical_feasible_incumbent_with_idx(
        state["records"], alpha=alpha, slot_budget=SIZES[1], delta_slot=ERROR_LEVELS[1],
        pooled_aleat_var_con=first.ucb_aleatoric_var_con, posterior_gate_power=incumbent_powers,
        posterior_mean_obj=first.mean_obj,
    )
    p0, p2 = [
        posterior_binomial_gate_power(
            pred.mean_con, pred.std_con, pred.ucb_aleatoric_var_con, r_perp_con=pred.r_perp_con,
            alpha=alpha, n_samples=SIZES[slot], delta_slot=ERROR_LEVELS[slot],
        )
        for slot in (0, 2)
    ]
    keys = [np.round(x, 5).tobytes() for x in state["candidates"]]
    primary_gain = np.maximum(ref - pred.mean_obj, 0) * p0
    primary_order = order[np.argsort(-primary_gain[order], kind="stable")]
    fallback_order = order[np.argsort(-p2[order], kind="stable")]
    seen = {keys[incumbent]}
    rosters = []
    for i in primary_order:
        if keys[i] in seen:
            continue
        fallback = next(j for j in fallback_order if keys[j] not in {keys[i], keys[incumbent]})
        rosters.append((i, incumbent, fallback))
        seen.add(keys[i])
        if len(rosters) == 4:
            break
    if len(rosters) != 4:
        raise ValueError("Expected four distinct primary policies")
    return np.array(rosters), float(ref)


@lru_cache(maxsize=2)
def tensor_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    z, weights = normal_rule(order)
    grid = np.stack(np.meshgrid(z, z, z, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid, np.einsum("i,j,k->ijk", weights, weights, weights).ravel()


def cubature_terms(gmean: np.ndarray, gcov: np.ndarray, noise: np.ndarray, order: int) -> np.ndarray:
    """Moments of A0, A0&A1, !A0&A1, and !A0&!A1&A2."""
    grid, weights = tensor_nodes(order)
    latent = gmean[:, None, :] + grid @ np.linalg.cholesky(gcov).T
    a0, a1, a2 = np.moveaxis(binomial_power_one(ndtr(latent / noise), SIZES), -1, 0)
    return np.stack([
        a0 @ weights,
        (a0 * a1) @ weights,
        ((1 - a0) * a1) @ weights,
        ((1 - a0) * (1 - a1) * a2) @ weights,
    ], axis=-1)


def term_bounds(gmean: np.ndarray, gcov: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sd = np.sqrt(np.maximum(np.diag(gcov), 0))
    latent = gmean[:, None, :] + QUANTILES[:, None] * sd
    # The integrand decreases with the Gaussian quantile, and endpoints lie in [0, 1].
    lower = binomial_power_one(ndtr(latent / noise), SIZES).sum(axis=1) / 512
    upper = lower + 1 / 512
    l0, l1, l2 = lower.T
    u0, u1, u2 = upper.T
    return (
        np.stack([l0, np.maximum(l0 + l1 - 1, 0), np.maximum(l1 - u0, 0),
                  np.maximum(l2 - u0 - u1, 0)], axis=-1),
        np.stack([u0, np.minimum(u0, u1), np.minimum(1 - l0, u1),
                  np.minimum(np.minimum(1 - l0, 1 - l1), u2)], axis=-1),
    )


def contribution_coefficients(fmean: np.ndarray, fcov: np.ndarray, noise: np.ndarray, ref: float) -> np.ndarray:
    """Integrate the objective comparison over two independent verification sample means."""
    diff = fmean[:, 0] - fmean[:, 1]
    var_diff = max(fcov[0, 0] + fcov[1, 1] - 2 * fcov[0, 1], 0)
    observed_sd = np.sqrt(max(var_diff + noise[0] / SIZES[0] + noise[1] / SIZES[1], 1e-16))
    z = diff / observed_sd
    guard_correction = diff * ndtr(z) + var_diff / observed_sd * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return np.column_stack([ref - fmean[:, 0], guard_correction, ref - fmean[:, 1], ref - fmean[:, 2]])


def single_terms(gmean: np.ndarray, gcov: np.ndarray, noise: np.ndarray, order: int) -> np.ndarray:
    z, weights = normal_rule(order)
    sd = np.sqrt(max(float(gcov[0, 0]), 0))
    latent = gmean[:, 0, None] + z * sd
    return binomial_power_one(ndtr(latent / noise[0]), SIZES[0]) @ weights


def best_average(values: list[np.ndarray], w: np.ndarray) -> float:
    return float(np.einsum("ij,i,j->", np.maximum(np.max(values, axis=0), 0), w, w))


def query_parts(state: dict, rosters: np.ndarray, ref: float):
    pred = state["prediction"]
    reps = state["family"]["batch_reps"]
    innovation_f = pred.cov_obj / np.sqrt(np.maximum(np.diag(pred.cov_obj) + pred.aleatoric_var_obj / reps, 1e-10))[:, None]
    innovation_g = pred.cov_con / np.sqrt(np.maximum(np.diag(pred.cov_con) + pred.aleatoric_var_con / reps, 1e-10))[:, None]
    gbase_cov = pred.cov_con + np.diag(0.25 * np.maximum(pred.r_perp_con, 0))
    pieces = []
    for q in range(len(state["candidates"])):
        row = []
        for roster in rosters:
            cov_idx = np.ix_(roster, roster)
            a = innovation_f[q, roster]
            b = innovation_g[q, roster]
            fmean = pred.mean_obj[roster] + FANTASY_Z[:, None] * a
            gmean = pred.mean_con[roster] + FANTASY_Z[:, None] * b
            fcov = pred.cov_obj[cov_idx] - np.outer(a, a)
            gcov = gbase_cov[cov_idx] - np.outer(b, b)
            noise_g = np.sqrt(np.maximum(pred.ucb_aleatoric_var_con[roster], 1e-8))
            coeff = contribution_coefficients(fmean, fcov, pred.aleatoric_var_obj[roster], ref)
            row.append((coeff, gmean, gcov, noise_g))
        pieces.append(row)
    return pieces


def scores_for_state(state: dict) -> dict:
    rosters, ref = shortlist_menu(state)
    pred = state["prediction"]
    n = len(state["candidates"])
    gbase_cov = pred.cov_con + np.diag(0.25 * np.maximum(pred.r_perp_con, 0))
    all_search = {}
    spent = {"interval_screen": 0.0, "single_17": 0.0, "full_9": 0.0, "full_17": 0.0}
    for q, row in enumerate(query_parts(state, rosters, ref)):
        values = {key: [] for key in ("single_9", "single_17", "full_9", "full_17", "lower", "upper")}
        for coeff, gmean, gcov, noise_g in row:
            for order in (9, 17):
                t0 = time.process_time()
                p0 = single_terms(gmean, gcov, noise_g, order)
                values[f"single_{order}"].append(np.outer(coeff[:, 0], p0))
                if order == 17:
                    spent["single_17"] += time.process_time() - t0
                t0 = time.process_time()
                terms = cubature_terms(gmean, gcov, noise_g, order)
                values[f"full_{order}"].append(coeff @ terms.T)
                spent[f"full_{order}"] += time.process_time() - t0
            t0 = time.process_time()
            lower, upper = term_bounds(gmean, gcov, noise_g)
            pos, neg = np.maximum(coeff, 0), np.minimum(coeff, 0)
            values["lower"].append(pos @ lower.T + neg @ upper.T)
            values["upper"].append(pos @ upper.T + neg @ lower.T)
            spent["interval_screen"] += time.process_time() - t0
        all_search[q] = {key: best_average(value, FANTASY_W) for key, value in values.items()}

    scores = {key: np.array([entry[key] for entry in all_search.values()])
              for key in next(iter(all_search.values()))}
    top = {name: int(np.argmax(values)) for name, values in scores.items()}
    # Common current-history offset cancels from all orderings and their bounds.
    retained = scores["upper"] + 1e-12 >= np.max(scores["lower"])
    full_best = top["full_17"]
    primary_best = top["single_17"]
    anchor_gates = []
    backup_benefits = []
    for roster in rosters:
        idx = np.ix_(roster, roster)
        coeff = contribution_coefficients(pred.mean_obj[roster][None, :], pred.cov_obj[idx],
                                          pred.aleatoric_var_obj[roster], ref)[0]
        terms = cubature_terms(pred.mean_con[roster][None, :], gbase_cov[idx],
                               np.sqrt(np.maximum(pred.ucb_aleatoric_var_con[roster], 1e-8)), 17)[0]
        anchor_gates.append(float(coeff[0] * terms[0]))
        backup_benefits.append(float(coeff[2:] @ terms[2:]))
    full = scores["full_17"]
    low, high = scores["lower"], scores["upper"]
    return {
        "candidate_count": n,
        "rosters": rosters.tolist(),
        "top": top,
        "saved_query": state["saved_query"],
        "full_value_gap_from_restricted_single": float(full[full_best] - full[primary_best]),
        "full_value_gap_from_actual_single": float(full[full_best] - full[state["saved_query"]]),
        "rank_correlation_restricted": float(spearmanr(scores["single_17"], full).statistic),
        "rank_correlation_actual": float(spearmanr(state["implemented_scores"], full).statistic),
        "value_outside_frechet_interval": float(max(np.max(low - full), np.max(full - high), 0)),
        "retained_queries": int(retained.sum()),
        "retains_full_best": bool(retained[full_best]),
        "unique_certifies_restricted_single": bool(retained.sum() == 1 and retained[primary_best]),
        "unique_certifies_actual_single": bool(retained.sum() == 1 and retained[state["saved_query"]]),
        "baseline_backup_value": backup_benefits[int(np.argmax(anchor_gates))],
        "quadrature_full_values_max_difference": float(np.max(np.abs(full - scores["full_9"]))),
        "timing": spent,
    }


def score(job: tuple[Path, Path]) -> str:
    source, directory = job
    output = directory / f"{source.stem}.json"
    with output.open("x") as stream:
        stream.write('{"status":"started"}\n')
    start = time.process_time()
    wall = time.perf_counter()
    result = {"job_id": source.stem, "search_calls": 0, "verification_calls": 0, "audit_calls": 0}
    try:
        with source.open("rb") as stream:
            state = pickle.load(stream)
        family, index = state["family"]["name"], state["task_index"]
        audits = _read_json(COHORT / "operating_policy_diagnostics.json")
        anchor = audits[family]["anchor_audits"][str(index)]
        result.update(
            status="ok", family=family, target_task_idx=index,
            audit_status_quo_class=anchor["safety_audit_class"], anchor_p=anchor["mc_violation_prob"],
            **scores_for_state(state),
        )
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.perf_counter() - wall)
        _write_json(output, result)
    return source.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(MANIFEST)
    jobs = [(args.prefix / f"{f['name']}_t{i:02d}.pkl", args.output)
            for f in manifest["families"] for i in INDEXES]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for job_id in executor.map(score, jobs[args.start:args.end]):
            print(job_id, flush=True)


if __name__ == "__main__":
    main()
