"""One frozen full-method search with a predeclared search/validation split."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.verification_kg import verification_kg_scores, verification_utility
from performance.run_meta_prior_mechanism import (
    moments, run_arm, shared_support, target_supports, task_truth,
)
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from performance.run_meta_prior_mechanism_v3 import recommend
from representation.conditional_function_prior import conditional_selection
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches
from representation.source_risk_ard import fit_source_risk_ard
from representation.transfer_mixture import TransferMixtureGPR

PROTOCOL = ROOT / "performance/meta_prior_total_budget_protocol.json"


def simulator_supports(seed, regime, dim):
    support = shared_support(seed, dim)
    if regime == "separate_head_structure":
        return [support, shared_support(seed + 1000, dim),
                target_supports(support, seed, "support_mismatch", dim)[0]]
    return target_supports(support, seed, regime, dim)


def fit_priors(seed, regime, arm, reference, protocol):
    if arm == "target_only":
        return [SourceFunctionPrior.fit([], [], reference, "target_only") for _ in range(3)], None
    tasks, _ = prepare_sources(seed, regime, protocol)
    batches = [head_batches(tasks, h) for h in range(3)]
    noises = [[task["noise"][:, h] for task in tasks] for h in range(3)]
    full = [SourceFunctionPrior.fit(batches[h], noises[h], reference, "fixed_source")
            for h in range(3)]
    if arm == "fixed_source":
        return full, None
    structural = []
    for h in range(2):
        selected, history = conditional_selection(batches[h], noises[h],
            FixedFunctionBasis(reference), protocol["learned_active_dim_per_head"])
        prior = SourceFunctionPrior.from_basis(batches[h], noises[h],
            FixedFunctionBasis(reference, selected), "conditional_hard")
        prior.diagnostics.update(selected_indices=selected, selection_cv=history)
        structural.append(prior)
    structural.append(fit_source_risk_ard(batches[2], noises[2], reference))
    return structural, full


def run_shard(seed, regime, difficulty, arm, allocation, frozen):
    start = time.monotonic()
    protocol = {**frozen, **frozen["allocations"][allocation]}
    if protocol["target_points"] * protocol["replicates"] + protocol["verification_replicates"] != protocol["total_target_calls"]:
        raise ValueError("allocation violates the fixed total budget")
    rng = np.random.default_rng(51000 + seed)
    psi = rng.uniform(-2, 2, (protocol["candidates"], protocol["psi_dim"]))
    reference = psi[:protocol["reference_bank_size"]]
    initial_rng = np.random.default_rng(51000 + seed)
    initial_rng.uniform(-2, 2, (protocol["initial_bank_size"], protocol["psi_dim"]))
    initial = initial_rng.choice(protocol["initial_bank_size"], protocol["initial_points"], replace=False).tolist()
    candidates = np.arange(len(psi))[:, None]
    priors, full = fit_priors(seed, regime, arm, reference, protocol)
    models = ([TransferMixtureGPR([priors[h], full[h]], psi, reference) for h in range(3)]
              if full is not None else [p.target_model(psi) for p in priors])
    # Supports are private simulator inputs, never fitting/acquisition inputs.
    supports = simulator_supports(seed, regime, protocol["psi_dim"])
    target_seed = 61000 + 100 * seed + protocol["regimes"].index(regime)
    truth = task_truth(psi, supports, target_seed)
    truth[:, 1] += protocol["difficulty_offsets"][difficulty]
    initial_choice = []

    def after_observation(live, step, index, response, noise):
        if step + 1 == protocol["initial_points"]:
            m, v = moments(live, candidates)
            initial_choice.append(recommend(m, v, protocol))

    result = run_arm(priors, initial, candidates, psi, truth, target_seed, protocol,
        models=models, acquisition=verification_kg_scores, recommendation_rule=recommend,
        on_observation=after_observation)
    result.pop("priors")
    result.pop("first_online_kg_scores")
    result.pop("target_support", None)
    mean, variance = moments(models, candidates)
    predicted_reward, predicted_power = verification_utility(mean, variance, protocol)
    true_reward, true_power = verification_utility(truth, np.zeros_like(truth), protocol)
    margin = truth[:, 1] + norm.ppf(1 - protocol["chance_alpha"]) * np.exp(.5 * truth[:, 2])
    choice = result["recommendation"]
    result.update(seed=seed, regime=regime, difficulty=difficulty, arm=arm, allocation=allocation,
        initial_design=initial, oracle_support=supports, source_calls=0 if arm == "target_only" else
            protocol["source_tasks"] * protocol["source_points"] * protocol["replicates"],
        total_target_calls=result["target_search_calls"] + result["target_verification_calls"],
        initial_recommendation=initial_choice[0],
        initial_expected_verified_reward=float(true_reward[initial_choice[0]]),
        predicted_verified_reward=float(predicted_reward[choice]),
        predicted_verifier_power=float(predicted_power[choice]),
        true_verifier_power=float(true_power[choice]),
        expected_verified_reward=float(true_reward[choice]),
        realized_verified_reward=float(result["verification"]["verified"] * np.clip((5 - truth[choice, 0]) / 10, 0, 1)),
        best_available_expected_verified_reward=float(true_reward.max()),
        expected_search_shortfall=float(true_reward.max() - true_reward[choice]),
        true_feasible_fraction=float(np.mean(margin <= 0)),
        false_verified=bool(result["verification"]["verified"] and margin[choice] > 0),
        source_nonintercept_means_zero=all(not np.any(p.mean[1:]) for p in priors),
        final_model_state=[m.diagnostics() for m in models] if full is not None else [])
    source_info = [{k: v for k, v in p.diagnostics.items() if k in (
        "mode", "selected_names", "selected_indices", "source_selected_indices",
        "source_function_variance", "source_label_count", "selection_cv", "optimizer_converged")}
        for p in priors]
    return {"protocol": protocol, "rows": [result], "source_info": source_info,
            "elapsed_seconds": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--difficulty", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if any(value not in protocol[key] for value, key in (
        (args.seed, "seeds"), (args.regime, "regimes"), (args.difficulty, "difficulty_offsets"),
        (args.arm, "arms"), (args.allocation, "allocations"))):
        raise ValueError("outside frozen protocol")
    result = run_shard(args.seed, args.regime, args.difficulty, args.arm, args.allocation, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({"elapsed_seconds": result["elapsed_seconds"],
                      "target_calls": result["rows"][0]["total_target_calls"]}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
