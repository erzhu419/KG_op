"""Compare next-query values with production-eligible terminal nominations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from refine_cascade_prefix import beta_rule
from restore_cascade_prefix import ARM, INDEXES, MANIFEST, _read_json, _write_json
from score_cascade_prefix import QUANTILES, SIZES, binomial_power_one
from v2_or.contracts import SufficientStatsRecord
from v2_or.models.tempered_hvd_posterior import TemperedHierarchicalHVDModel
from v2_or.verification.gated_verifier_v2 import build_feasible_aware_shortlist, exact_binomial_critical_k


OUTER_SEEDS = (78463511, 78463529)
OUTER_SIZES = (32, 64)
INNER_SEEDS = (4157201, 4157213)
INNER_SIZES = (512, 2048)


@dataclass
class RosterDraw:
    policy_indexes: tuple[int, ...]
    guard: bool
    gains: np.ndarray
    constraint_mean: np.ndarray
    constraint_covariance: np.ndarray
    constraint_sd: np.ndarray


def model_from_state(state: dict, prior_dir: Path) -> TemperedHierarchicalHVDModel:
    if "model" in state:
        return state["model"]
    family = state["family"]
    arm = next(arm for arm in family["arms"] if arm["name"] == ARM)
    with (prior_dir / family["name"] / "prior_default.pkl").open("rb") as stream:
        prior = pickle.load(stream)
    model = TemperedHierarchicalHVDModel(
        prior,
        mode=arm["model_mode"],
        use_source_basis=arm["use_source_basis"],
        use_source_coef_prior=arm.get("use_source_coef_prior"),
        use_source_hvd_prior=arm["use_source_hvd_prior"],
        variance_estimator=arm["variance_estimator"],
        use_empirical_r_perp=arm["use_empirical_r_perp"],
        n0_freeze=family["n0_init"],
        coefficient_prior=arm["coefficient_prior"],
    )
    for record in state["records"]:
        model.observe(record)
    model.freeze_matched_scales(state["records"])
    return model


@lru_cache(maxsize=4)
def normal_nodes(dimension: int, size: int, seed: int) -> np.ndarray:
    return ndtri(qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(size.bit_length() - 1))


def paired_fantasies(prediction, query: int, nodes: np.ndarray) -> np.ndarray:
    """Generate both repeats of each head; rows preserve SSQ and violations."""
    observation_vars = np.array([prediction.aleatoric_var_obj[query], prediction.aleatoric_var_con[query]])
    latent_vars = np.array([prediction.cov_obj[query, query], prediction.cov_con[query, query]])
    means = np.array([prediction.mean_obj[query], prediction.mean_con[query]])
    center = means + nodes[:, :2] * np.sqrt(latent_vars + observation_vars / 2.0)
    contrast = nodes[:, 2:] * np.sqrt(observation_vars / 2.0)
    return np.stack((center + contrast, center - contrast), axis=1)


def guarded_saving(mean: np.ndarray, covariance: np.ndarray, noise: np.ndarray) -> float:
    diff = mean[0] - mean[1]
    variance = max(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1], 0)
    observed_sd = np.sqrt(max(variance + noise[0] / SIZES[0] + noise[1] / SIZES[1], 1e-16))
    z = diff / observed_sd
    return float(diff * ndtr(z) + variance / observed_sd * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi))


def native_draw(
    model: TemperedHierarchicalHVDModel,
    record: SufficientStatsRecord,
    alpha: float,
    anchor_observation: float,
    loss_reference: float,
    nomination_mode: str,
) -> RosterDraw:
    """Use the exact next-update and nomination rules without simulator calls."""
    updated = copy.copy(model)
    updated._records_by_key = model._records_by_key.copy()
    updated._ordered_keys = model._ordered_keys.copy()
    updated.observe(record)
    records = updated.unique_records
    policies = [r.x_norm for r in records]
    prediction = updated.predict_joint(policies, [r.exposure for r in records])
    shortlist = build_feasible_aware_shortlist(
        candidate_pool=policies,
        mean_obj=prediction.mean_obj,
        mean_con=prediction.mean_con,
        std_con=prediction.std_con,
        ucb_aleat_var_con=prediction.ucb_aleatoric_var_con,
        r_perp_con=prediction.r_perp_con,
        initial_records=records,
        alpha=alpha,
        nomination_mode=nomination_mode,
        reference_obj=0.5 * (anchor_observation + float(prediction.mean_obj[0])),
    )
    position = {policy.tobytes(): i for i, policy in enumerate(policies)}
    roster = tuple(position[x.tobytes()] for _, x in shortlist)
    incumbent = len(shortlist) >= 2 and shortlist[1][0] in {"empirical_feasible_incumbent", "safe_initial_incumbent"}
    idx = np.asarray(roster)
    means = prediction.mean_obj[idx]
    values = np.zeros(4)
    values[0] = loss_reference - means[0]
    if len(roster) >= 2:
        values[2] = loss_reference - means[1]
    if len(roster) == 3:
        values[3] = loss_reference - means[2]
    if incumbent:
        values[1] = guarded_saving(
            means, prediction.cov_obj[np.ix_(idx, idx)], prediction.aleatoric_var_obj[idx],
        )
    return RosterDraw(
        roster, incumbent, values, prediction.mean_con[idx],
        prediction.cov_con[np.ix_(idx, idx)] + np.diag(0.25 * np.maximum(prediction.r_perp_con[idx], 0.0)),
        np.sqrt(np.maximum(prediction.ucb_aleatoric_var_con[idx], 1e-8)),
    )


def prove_gate_counts(alpha: float) -> None:
    """For supported roster lengths, prove the fast gate integrand uses the native cutoff."""
    for fractions in ((1.0,), (0.625, 0.375), (0.50, 0.30, 0.20)):
        if any(exact_binomial_critical_k(int(n), alpha, 0.05 * frac) != 1
               for n, frac in zip(SIZES, fractions)):
            raise ValueError(f"Unexpected exact-binomial success threshold for alpha={alpha}")


def moment_bounds(mean, covariance, noise) -> tuple[np.ndarray, np.ndarray]:
    """Fréchet intersections using marginal quantile brackets, one row per fantasy."""
    n = mean.shape[1]
    sd = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    latent = mean[:, None, :] + QUANTILES[:, None] * sd[:, None, :]
    lower = np.zeros((mean.shape[0], 3))
    upper = lower.copy()
    lower[:, :n] = binomial_power_one(ndtr(latent / noise[:, None, :]), SIZES[:n]).sum(axis=1) / 512
    upper[:, :n] = lower[:, :n] + 1 / 512
    l0, l1, l2 = lower.T
    u0, u1, u2 = upper.T
    return (
        np.column_stack([l0, np.maximum(l0 + l1 - 1, 0), np.maximum(l1 - u0, 0),
                         np.maximum(l2 - u0 - u1, 0)]),
        np.column_stack([u0, np.minimum(u0, u1), np.minimum(1 - l0, u1),
                         np.minimum(np.minimum(1 - l0, 1 - l1), u2)]),
    )


def first_marginal(mean, covariance, noise, order: int) -> np.ndarray:
    nodes, weights = beta_rule(order)
    normal_cdf = ndtr((nodes * noise[:, 0, None] - mean[:, 0, None]) / np.sqrt(covariance[:, 0, 0, None]))
    return normal_cdf @ weights


def numeric_joint_moments(mean, covariance, noise, primary, seed: int) -> list[np.ndarray]:
    """Shared Sobol points for the companion moments, with an analytic primary marginal."""
    n = mean.shape[1]
    nodes = normal_nodes(3, INNER_SIZES[-1], seed)[:, :n]
    sampled = mean[:, None, :] + nodes @ np.linalg.cholesky(covariance).transpose(0, 2, 1)
    p = binomial_power_one(ndtr(sampled / noise[:, None, :]), SIZES[:n])
    p0 = p[:, :, 0]
    p1 = p[:, :, 1] if n >= 2 else np.zeros_like(p0)
    p2 = p[:, :, 2] if n == 3 else np.zeros_like(p0)
    moments = [p0 * p1, (1 - p0) * p1, (1 - p0) * (1 - p1) * p2]
    return [np.column_stack([primary] + [m[:, :count].mean(axis=1) for m in moments]) for count in INNER_SIZES]


def _groups(draws: list[RosterDraw]):
    for slots in (1, 2, 3):
        indexes = [i for i, r in enumerate(draws) if len(r.policy_indexes) == slots]
        if indexes:
            group = [draws[i] for i in indexes]
            yield (
                indexes,
                np.stack([r.gains for r in group]),
                np.stack([r.constraint_mean for r in group]),
                np.stack([r.constraint_covariance for r in group]),
                np.stack([r.constraint_sd for r in group]),
            )


def first_and_envelope(draws: list[RosterDraw]) -> np.ndarray:
    scores = np.empty((4, len(draws)))
    for indexes, gains, mean, cov, noise in _groups(draws):
        low, high = moment_bounds(mean, cov, noise)
        pos, neg = np.maximum(gains, 0), np.minimum(gains, 0)
        scores[:, indexes] = np.stack([
            gains[:, 0] * first_marginal(mean, cov, noise, 24),
            gains[:, 0] * first_marginal(mean, cov, noise, 48),
            np.sum(pos * low + neg * high, axis=1),
            np.sum(pos * high + neg * low, axis=1),
        ])
    return scores


def integrate_full(draws: list[RosterDraw]) -> np.ndarray:
    scores = np.empty((4, len(draws)))
    for indexes, gains, mean, cov, noise in _groups(draws):
        primary = first_marginal(mean, cov, noise, 48)
        moments = [score for seed in INNER_SEEDS for score in numeric_joint_moments(mean, cov, noise, primary, seed)]
        scores[:, indexes] = np.einsum("fti,ti->ft", moments, gains)
    return scores


def score_state(state: dict, prior_dir: Path) -> dict:
    prove_gate_counts(state["alpha"])
    family = state["family"]
    if family["batch_reps"] != 2:
        raise ValueError("This working law is specified for two repeats")
    arm = next(arm for arm in family["arms"] if arm["name"] == ARM)
    model = model_from_state(state, prior_dir)
    records = model.unique_records
    existing = {model._policy_key(record.x_norm) for record in records}
    obs_anchor = float(state["records"][0].mean_obj)
    cands, exposures = state["candidates"], state["exposures"]
    predicted = state["prediction"]
    restored = model.predict_joint(cands, exposures)
    for field in ("mean_obj", "mean_con", "cov_obj", "cov_con", "aleatoric_var_obj", "aleatoric_var_con"):
        np.testing.assert_allclose(getattr(restored, field), getattr(predicted, field), rtol=1e-8, atol=1e-9)
    total = OUTER_SIZES[-1] * len(OUTER_SEEDS)
    nodes = np.concatenate([normal_nodes(4, OUTER_SIZES[-1], seed) for seed in OUTER_SEEDS])
    laws = [(seed, count, slice(offset, offset + count))
            for s, seed in enumerate(OUTER_SEEDS)
            for offset in (s * OUTER_SIZES[-1],)
            for count in OUTER_SIZES]
    all_draws = []
    summary = np.empty((len(cands), len(laws), 4))
    first_cpu, screen_cpu = 0.0, 0.0
    list_sizes, guarded_nodes = {}, 0
    new_policy_nodes = np.zeros(3, dtype=int)
    roster_histogram = []
    for q, (x, exposure) in enumerate(zip(cands, exposures, strict=True)):
        start = time.process_time()
        fantasies = paired_fantasies(predicted, q, nodes)
        key = model._policy_key(x)
        n_eligible = len(records) + int(key not in existing)
        draws = [
            native_draw(model, SufficientStatsRecord.from_samples(x, samples, exposure), state["alpha"],
                        obs_anchor, state["reference_obj"], arm.get("nomination_mode", "binomial_gate_power"))
            for samples in fantasies
        ]
        for draw in draws:
            if max(draw.policy_indexes) >= n_eligible:
                raise ValueError("A native roster exceeded the queried eligible policy set")
            for slot, policy in enumerate(draw.policy_indexes):
                new_policy_nodes[slot] += int(policy == len(records))
            guarded_nodes += draw.guard
            size = len(draw.policy_indexes)
            list_sizes[size] = list_sizes.get(size, 0) + 1
        roster_histogram.append(len({(d.policy_indexes, d.guard) for d in draws}))
        first_cpu += time.process_time() - start
        start = time.process_time()
        bound = first_and_envelope(draws)
        for l, (_, _, s) in enumerate(laws):
            summary[q, l] = bound[:, s].mean(axis=1)
        screen_cpu += time.process_time() - start
        all_draws.append(draws)

    # A survivor under any registered law is refined, along with the two comparators.
    upper, lower = summary[:, :, 3], summary[:, :, 2]
    retained = upper + 1e-12 >= lower.max(axis=0)[None, :]
    first = summary[:, :, 1].argmax(axis=0)
    first24 = summary[:, :, 0].argmax(axis=0)
    baseline = state["saved_query"]
    evaluate = np.flatnonzero(retained.any(axis=1))
    evaluate = np.unique([*evaluate, *first, baseline])
    start = time.process_time()
    full = {}
    for q in evaluate:
        values = integrate_full(all_draws[q])
        full[q] = np.stack([values[:, s].mean(axis=1) for _, _, s in laws])
    full_cpu = time.process_time() - start
    reports = []
    for k, (seed, count, _) in enumerate(laws):
        survivors = np.flatnonzero(retained[:, k])
        estimates = np.stack([full[q][k] for q in survivors])
        fine = estimates[:, [1, 3]].mean(axis=1)
        winner = int(survivors[np.argmax(fine)])
        gap = float(np.max(fine) - full[first[k]][k, [1, 3]].mean())
        gap_lower = max(0.0, float(lower[:, k].max() - upper[first[k], k]))
        gap_upper = max(0.0, float(upper[:, k].max() - lower[first[k], k]))
        reports.append(dict(
            fantasy_seed=seed, fantasy_count=count, first_query=int(first[k]),
            first_beta24_query=int(first24[k]), full_query=winner,
            numerical_winners=survivors[estimates.argmax(axis=0)].tolist(),
            first_excluded=not bool(retained[first[k], k]), retained_queries=len(survivors),
            full_matches_first=bool(winner == first[k]), model_value_gain=gap,
            model_gain_bounds=[gap_lower, gap_upper],
            largest_fine_scramble_difference=float(np.max(np.abs(estimates[:, 1] - estimates[:, 3]))),
            worst_fine_interval_overshoot=float(max(0, np.max(lower[survivors, k] - fine),
                                                    np.max(fine - upper[survivors, k]))),
        ))
    return dict(
        family=family["name"], target_task_idx=state["task_index"], cached_native_query=int(baseline),
        completed_online_steps=int(state.get("completed_online_steps", 0)),
        unique_observed_policies=len(records), query_count=len(cands),
        new_policy_queries=sum(model._policy_key(x) not in existing for x in cands),
        evaluated_by_inner=len(evaluate), fantasy_updates=len(cands) * total,
        native_roster_sizes=list_sizes, native_guarded_fantasies=guarded_nodes,
        newly_eligible_in_slot=new_policy_nodes.tolist(),
        distinct_rosters_across_fantasies=int(sum(roster_histogram)),
        cpu_seconds_by_component=dict(native_state_updates=first_cpu, first_and_envelope=screen_cpu, full_integration=full_cpu),
        laws=reports,
    )


def run(job: tuple[Path, Path, Path]) -> str:
    source, directory, prior_dir = job
    receipt = directory / f"{source.stem}.json"
    with receipt.open("x") as stream:
        stream.write('{"status":"started"}\n')
    start = time.process_time()
    wall = time.perf_counter()
    result = dict(job_id=source.stem, search_calls=0, verification_calls=0, audit_calls=0)
    try:
        with source.open("rb") as stream:
            result.update(status="ok", **score_state(pickle.load(stream), prior_dir))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.perf_counter() - wall)
        _write_json(receipt, result)
    return source.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-indexes", type=int, nargs="+", default=INDEXES)
    parser.add_argument("--priors", type=Path, default=Path("paper_artifacts/v2_or_u3_confirm_v1/source_bundles"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(MANIFEST)
    ids = [f"{family['name']}_t{index:02d}" for family in manifest["families"] for index in args.task_indexes]
    jobs = [(args.prefix / f"{job_id}.pkl", args.output, args.priors) for job_id in ids[args.start:args.end]]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for job_id in executor.map(run, jobs):
            print(job_id, flush=True)


if __name__ == "__main__":
    main()
