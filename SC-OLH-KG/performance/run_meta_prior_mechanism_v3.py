"""Paired source-prior, support-correction and verification-target experiments."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.run_meta_prior_mechanism import (  # noqa: E402
    diagnostics, loss_and_derivatives, moments, run_arm, task_truth, verify,
)
from performance.run_meta_prior_mechanism_v2 import prepare_sources  # noqa: E402
from representation.source_function_prior import SourceFunctionPrior, head_batches  # noqa: E402
from representation.conditional_function_prior import fit_conditional_priors  # noqa: E402
from representation.source_risk_ard import fit_source_risk_ard  # noqa: E402
from representation.transfer_mixture import TransferMixtureGPR  # noqa: E402
from core.verification_kg import verification_kg_scores, verification_utility  # noqa: E402

PROTOCOL = ROOT / "performance/meta_prior_mechanism_v3_protocol.json"


def recommend(mean, variance, protocol):
    return int(np.argmax(verification_utility(mean, variance, protocol)[0]))


def score_recommendation(row, index, truth, target_seed, protocol):
    margin = truth[:, 1] + norm.ppf(1 - protocol["chance_alpha"]) * np.exp(.5 * truth[:, 2])
    feasible = np.flatnonzero(margin <= 0)
    loss, _ = loss_and_derivatives(truth, np.zeros_like(truth), protocol)
    row.update({"recommendation": index, "true_feasible": bool(margin[index] <= 0),
                "true_margin": float(margin[index]), "true_objective": float(truth[index, 0]),
                "feasible_regret": float(truth[index, 0] - truth[feasible, 0].min())
                    if margin[index] <= 0 and len(feasible) else None,
                "penalized_regret": float(loss[index] - loss.min()),
                "verification": verify(truth[index], target_seed, index,
                    protocol["verification_replicates"], protocol["chance_alpha"])})


def run_shard(seed, regime, difficulty, protocol):
    start = time.monotonic()
    tasks, supports = prepare_sources(seed, regime, protocol)
    rng = np.random.default_rng(51000 + seed)
    psi = rng.uniform(-2, 2, (protocol["candidates"], protocol["psi_dim"]))
    candidates = np.arange(len(psi))[:, None]
    if "initial_bank_size" in protocol:
        initial_rng = np.random.default_rng(51000 + seed)
        initial_rng.uniform(-2, 2, (protocol["initial_bank_size"], protocol["psi_dim"]))
        initial = initial_rng.choice(protocol["initial_bank_size"], protocol["initial_points"], replace=False).tolist()
    else:
        initial = rng.choice(len(psi), protocol["initial_points"], replace=False).tolist()
    reference = psi[:protocol.get("reference_bank_size", len(psi))]
    batches = [head_batches(tasks, h) for h in range(3)]
    noises = [[task["noise"][:, h] for task in tasks] for h in range(3)]
    conditional = [fit_conditional_priors(batches[h], noises[h], reference) for h in range(3)]
    priors = {"conditional_hard": [c["conditional_hard"] for c in conditional],
              "adaptive_v2": [c["conditional_adaptive"] for c in conditional]}
    priors["adaptive_corrected"] = priors["adaptive_v2"]
    priors["fixed_source"] = [SourceFunctionPrior.fit(batches[h], noises[h], reference,
                              "fixed_source") for h in range(3)]
    priors["risk_ard"] = priors["conditional_hard"][:2] + [
        fit_source_risk_ard(batches[2], noises[2], reference)]
    priors["transfer_mixture"] = priors["conditional_hard"]
    priors["risk_ard_mixture"] = priors["risk_ard"]
    priors["oracle_structure"] = [SourceFunctionPrior.fit(batches[h], noises[h], reference,
        "oracle_structure", oracle_indices=supports[h]) for h in range(3)]
    target_seed = 61000 + seed * 100 + protocol["regimes"].index(regime)
    truth = task_truth(psi, supports, target_seed)
    truth[:, 1] += protocol["difficulty_offsets"][difficulty]
    true_utility, true_power = verification_utility(truth, np.zeros_like(truth), protocol)
    margin = truth[:, 1] + norm.ppf(1 - protocol["chance_alpha"]) * np.exp(.5 * truth[:, 2])
    rows = []
    for arm in protocol["arms"]:
        for scheme in ("penalty", "verification_kg"):
            if arm in ("transfer_mixture", "risk_ard_mixture"):
                models = [TransferMixtureGPR([priors[arm][h], priors["fixed_source"][h]], psi, reference)
                          for h in range(3)]
            else:
                models = [p.target_model(psi) for p in priors[arm]]
            records = [[] for _ in range(3)]
            initial_choice, initial_model_state = [], []

            def after_observation(live, step, index, response, noise):
                for h in range(3):
                    records[h].append(([index], response[h], noise[h]))
                if step + 1 != protocol["initial_points"]:
                    return
                if arm in ("adaptive_v2", "adaptive_corrected"):
                    for h, model in enumerate(live):
                        info = priors[arm][h].diagnostics
                        spec = {"source_pip": info["source_pip"], "source_slab_scale": info["source_slab_scale"],
                                "spike_ratio": .08, "max_iter": 40, "residual_floor_scale": .05,
                                "max_effective_fraction": .35 if arm == "adaptive_v2" else 1.,
                                "multiplicity_correction": 1. if arm == "adaptive_v2" else 0.}
                        model.enable_adaptive_sparsity(spec, [r[0] for r in records[h]],
                            [r[1] for r in records[h]], [r[2] for r in records[h]],
                            deviation_variance=priors[arm][h].discrepancy_variance)
                        d = model.adaptive_sparsity_diagnostics()
                        initial_model_state.append({k: d[k] for k in (
                            "effective_dimension", "max_effective_dimension", "active_count_0_5")})
                elif arm in ("transfer_mixture", "risk_ard_mixture"):
                    initial_model_state.extend(m.diagnostics() for m in live)
                m, v = moments(live, candidates)
                initial_choice.append(recommend(m, v, protocol))

            result = run_arm(priors[arm], initial, candidates, psi, truth, target_seed, protocol,
                on_observation=after_observation, models=models,
                acquisition=verification_kg_scores if scheme == "verification_kg" else None,
                recommendation_rule=recommend if scheme == "verification_kg" else None)
            mean, variance = moments(models, candidates)
            predicted_utility, predicted_power = verification_utility(mean, variance, protocol)
            result.pop("priors")
            result.pop("first_online_kg_scores")
            result.pop("target_support", None)
            final_state = []
            if arm.startswith("adaptive"):
                for model in models:
                    d = model.adaptive_sparsity_diagnostics()
                    final_state.append({k: d[k] for k in ("n_observations", "active_count_0_5",
                        "escaped_source_spike_count", "effective_dimension", "converged")})
            elif arm in ("transfer_mixture", "risk_ard_mixture"):
                final_state = [m.diagnostics() for m in models]
            result.update({"seed": seed, "regime": regime, "difficulty": difficulty,
                "arm": arm, "scheme": scheme, "initial_design": initial,
                "oracle_support": supports, "source_calls": 6144,
                "initial_model_state": initial_model_state, "final_model_state": final_state,
                "true_feasible_fraction": float(np.mean(margin <= 0)),
                "initial_best_expected_verified_reward": float(true_utility[initial_choice[0]]),
                "best_available_expected_verified_reward": float(true_utility.max())})
            versions = [result]
            if scheme == "penalty":
                reranked = copy.deepcopy(result)
                reranked["scheme"] = "verification_recommendation"
                choice = recommend(mean, variance, protocol)
                score_recommendation(reranked, choice, truth, target_seed, protocol)
                # Reuse the original posterior's nominal safety label at this candidate.
                from performance.run_meta_prior_mechanism import safety_margin
                reranked["posterior_safe"] = bool(safety_margin(mean, variance,
                                                   protocol["chance_alpha"])[choice] <= 0)
                versions.append(reranked)
            for row in versions:
                choice = row["recommendation"]
                row.update({"predicted_verifier_power": float(predicted_power[choice]),
                    "true_verifier_power": float(true_power[choice]),
                    "expected_verified_reward": float(true_utility[choice]),
                    "realized_verified_reward": float(row["verification"]["verified"]
                                                    * np.clip((5 - truth[choice, 0]) / 10, 0, 1)),
                    "false_verified": bool(row["verification"]["verified"] and margin[choice] > 0),
                    "initial_recommendation": initial_choice[0],
                    "source_nonintercept_means_zero": all(not np.any(p.mean[1:]) for p in priors[arm])})
                rows.append(row)
            print(json.dumps({"seed": seed, "regime": regime, "difficulty": difficulty,
                              "arm": arm, "scheme": scheme,
                              "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
    # Keep source risk diagnostics once per shard, not repeated in every row.
    source_info = {arm: [{k: p.diagnostics[k] for k in p.diagnostics if k in (
        "selected_names", "selected_indices", "source_selected_names", "source_selected_indices",
        "source_function_variance", "optimizer_converged", "optimizer_iterations", "selection_cv")}
        for p in ps] for arm, ps in priors.items()}
    return {"protocol": protocol, "rows": rows, "source_info": source_info,
            "elapsed_seconds": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--difficulty", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if (args.seed not in protocol["seeds"] or args.regime not in protocol.get("evaluation_regimes", protocol["regimes"])
            or args.difficulty not in protocol["difficulty_offsets"]):
        raise ValueError("outside frozen protocol")
    result = run_shard(args.seed, args.regime, args.difficulty, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
