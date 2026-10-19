"""Read the saved V69 posterior on a shared bank without replaying or fitting.

New-posterior engines still retain the complete old proposal posterior. Calling
the base decision decomposition directly therefore compares both posteriors on
exactly the same paid history. Constraint probabilities are the original expert
mixture probabilities, not a Gaussian approximation to aggregated moments.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from acquisition.decision_backends import feasible_first_terminal_order
from algorithms.single_olhkg import SingleOLHKGAlgorithm


ARMS = ("legacy_native_v69", "same_source_gp", "current_structural_mixture",
        "initialization_only")
ROOT = Path(__file__).resolve().parents[1]


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def native_view(engine, bank):
    """Extract the original terminal posterior and ordering on ``bank``.

    This deliberately bypasses LegacyPosteriorAlgorithm's decision-context
    replacement. No history updates, simulator calls or source fits occur.
    """
    bank = [tuple(map(int, point)) for point in bank]
    components = SingleOLHKGAlgorithm._terminal_bayes_risk_components(
        engine, engine.gpr, engine.variance_model, bank,
        task_ensemble=engine.task_ensemble,
        risk_penalty=engine.config.decision_risk_penalty)
    threshold = float(engine.config.decision_terminal_maximum_violation_probability)
    order, mode = feasible_first_terminal_order(
        components["objective"], components["probability_violation"],
        maximum_violation_probability=threshold)
    observed = list(dict.fromkeys(tuple(point) for point, _ in engine.history))
    index = {point: i for i, point in enumerate(bank)}
    observed_ids = [index[point] for point in observed]
    observed_order, observed_mode = feasible_first_terminal_order(
        np.asarray(components["objective"])[observed_ids],
        np.asarray(components["probability_violation"])[observed_ids],
        maximum_violation_probability=threshold)
    observed_ranked = [observed_ids[int(i)] for i in observed_order]
    result = {
        "history_episodes": len(engine.history),
        "observed_bank_ids": observed_ids,
        "bank_rank_order": list(map(int, order)),
        "bank_selection_mode": mode,
        "observed_rank_order": observed_ranked,
        "observed_selection_mode": observed_mode,
        "observed_selected_bank_id": observed_ranked[0],
        "recorded_primary_bank_id": index[tuple(engine.frozen_terminal["primary"])],
        "terminal_maximum_violation_probability": threshold,
        "components": components,
    }
    ensemble = engine.task_ensemble
    if ensemble is None:
        raise ValueError("This diagnostic expects the restored V69 expert ensemble")
    objective = ensemble.expert_moments_many(0, bank, certification=False)
    constraint = ensemble.expert_moments_many(1, bank, certification=False)
    posterior = ensemble.posterior
    decision_weights = posterior.decision_weights()
    objective_weights = (posterior.posterior_weights() if posterior.safe_generalized
                         else decision_weights)
    z = float(norm.ppf(1. - engine.problem.alpha))
    g_mean, g_epistemic, g_noise = constraint
    g_margin = g_mean + z * np.sqrt(np.maximum(g_noise, 0.)) - engine.problem.tau
    g_probability = norm.cdf(g_margin / np.sqrt(np.maximum(g_epistemic, 1e-12)))
    g_mixture_mean = decision_weights @ g_mean
    g_within_epistemic = decision_weights @ np.maximum(g_epistemic, 0.)
    g_between_epistemic = decision_weights @ (g_mean - g_mixture_mean[None, :]) ** 2
    result["experts"] = {
        "names": list(posterior.expert_names),
        "safe_generalized": bool(posterior.safe_generalized),
        "posterior_weights": posterior.posterior_weights(),
        "safe_posterior_weights": posterior.safe_posterior_weights(),
        "constraint_decision_weights": decision_weights,
        "objective_decision_weights": objective_weights,
        "objective_mean": objective[0],
        "objective_epistemic_variance": objective[1],
        "objective_episode_variance": objective[2],
        "constraint_mean_raw": g_mean,
        "constraint_epistemic_variance": g_epistemic,
        "constraint_episode_variance": g_noise,
        "constraint_chance_margin_mean": g_margin,
        "constraint_probability_violation": g_probability,
    }
    result["aggregate_moments"] = {
        "objective_mean": objective_weights @ objective[0],
        "objective_episode_variance": objective_weights @ objective[2],
        "constraint_mean_raw": g_mixture_mean,
        "constraint_within_epistemic_variance": g_within_epistemic,
        "constraint_between_expert_variance": g_between_epistemic,
        "constraint_epistemic_variance": g_within_epistemic + g_between_epistemic,
        "constraint_episode_variance": decision_weights @ g_noise,
        "constraint_probability_violation": decision_weights @ g_probability,
        "note": "Constraint decision is the exact weighted expert CDF, not the CDF of these aggregate moments.",
    }
    if engine._certification_head_authority() != "task_joint":
        raise ValueError("Saved engine changed its constraint authority")
    np.testing.assert_allclose(
        components["probability_violation"], decision_weights @ g_probability,
        rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        components["objective"], objective_weights @ objective[0],
        rtol=1e-12, atol=1e-12)
    result["exact_component_reconstruction_passed"] = True
    return _plain(result)


def diagnose(bridge, bank):
    start = time.process_time()
    engines = {}
    rows = {}
    for arm in ARMS:
        with (bridge / arm / "search_engine.pkl").open("rb") as handle:
            engines[arm] = pickle.load(handle)
        rows[arm] = native_view(engines[arm], bank)
    for arm in ("legacy_native_v69", "initialization_only"):
        row = rows[arm]
        if row["observed_selected_bank_id"] != row["recorded_primary_bank_id"]:
            raise ValueError(f"Native primary replay differs for {arm}")
        row["recorded_primary_parity"] = True
    initial = engines["initialization_only"]
    initial_posterior = initial.pre_sampling_log["task_posterior"]["posterior"]
    initial_parity = {}
    for arm in ARMS[:-1]:
        engine = engines[arm]
        changed_config = [key for key, value in vars(initial.config).items()
                          if str(value) != str(getattr(engine.config, key))]
        initial_history_equal = all(
            tuple(point) == tuple(other_point) and np.array_equal(raw, other_raw)
            for (point, raw), (other_point, other_raw)
            in zip(initial.history, engine.history[:10]))
        posterior_equal = (initial_posterior ==
                           engine.pre_sampling_log["task_posterior"]["posterior"])
        if changed_config != ["N"] or not initial_history_equal or not posterior_equal:
            raise ValueError(f"Saved initialization contracts differ for {arm}")
        initial_parity[arm] = {
            "config_differences": changed_config,
            "same_first10_raw_history": initial_history_equal,
            "same_saved_initial_task_posterior": posterior_equal,
        }
    return {
        "scope": "Saved original V69 proposal/terminal posterior, evaluated on one physical policy bank. New-arm rows use that arm's own paid history.",
        "bank_count": len(bank), "arms": rows,
        "initialization_parity": initial_parity,
        "initialization_N_note": "Only N differs (10 versus 13). Initial belief fitting does not use N; saved initial posterior diagnostics and first10 raw observations agree exactly.",
        "accounting": {"cpu_seconds": time.process_time() - start,
                       "new_simulator_calls": 0, "source_fits": 0,
                       "target_model_refits": 0, "downloads": 0},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, default=ROOT / "paper_artifacts/mainline_legacy_backend_bridge_v1/FactorShockStatePolicyRZDT1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    bank = inputs["bank"]
    result = diagnose(args.bridge, bank)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), **result["accounting"]}))


if __name__ == "__main__":
    main()
