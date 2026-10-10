"""Frozen risk attribution; one independent seed/regime/layer/prior shard."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.verification_kg import risk_only_kg_scores, verification_utility
from performance.run_meta_prior_mechanism import observe, task_truth, verify
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from representation.conditional_function_prior import conditional_selection
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches
from representation.source_risk_ard import fit_source_risk_ard
from representation.transfer_mixture import TransferMixtureGPR

PROTOCOL = ROOT / "performance/meta_prior_risk_isolation_protocol.json"


def source_hard_prior(batches, noise, psi):
    selected, history = conditional_selection(batches, noise, FixedFunctionBasis(psi))
    prior = SourceFunctionPrior.from_basis(batches, noise, FixedFunctionBasis(psi, selected), "conditional_hard")
    prior.diagnostics.update(selected_indices=selected, selection_cv=history)
    return prior


def checkpoint(model, fixed_mean, fixed_variance, candidates, truth, target_seed, p, count):
    mean, variance = fixed_mean.copy(), fixed_variance.copy()
    mean[:, 2] = model.posterior_mean_many(candidates)
    variance[:, 2] = model.posterior_var_many(candidates)
    utility, power = verification_utility(mean, variance, p)
    choice = int(np.argmax(utility))
    true_utility, true_power = verification_utility(truth, np.zeros_like(truth), p)
    margin = truth[:, 1] + norm.ppf(1 - p["chance_alpha"]) * np.exp(.5 * truth[:, 2])
    feasible = np.flatnonzero(margin <= 0)
    verified = verify(truth[choice], target_seed, choice, p["verification_replicates"], p["chance_alpha"])
    error = mean[:, 2] - truth[:, 2]
    return {"batches": count, "search_calls": count * p["replicates"],
            "verification_calls": p["verification_replicates"], "recommendation": choice,
            "expected_verified_reward": float(true_utility[choice]),
            "finite_bank_shortfall": float(true_utility.max() - true_utility[choice]),
            "predicted_verifier_power": float(power[choice]), "true_verifier_power": float(true_power[choice]),
            "verifier_power_absolute_error": float(abs(power[choice] - true_power[choice])),
            "risk_rmse": float(np.sqrt(np.mean(error ** 2))),
            "risk_nominal90_coverage": float(np.mean(abs(error) <= norm.ppf(.95) * np.sqrt(variance[:, 2]))),
            "local_risk_error": float(error[choice]), "true_feasible": bool(margin[choice] <= 0),
            "feasible_regret": float(truth[choice, 0] - truth[feasible, 0].min()) if margin[choice] <= 0 else None,
            "true_objective": float(truth[choice, 0]), "verification": verified,
            "realized_verified_reward": float(verified["verified"] * np.clip((5 - truth[choice, 0]) / 10, 0, 1)),
            "false_verified": bool(verified["verified"] and margin[choice] > 0),
            "risk_model_state": model.diagnostics() if isinstance(model, TransferMixtureGPR) else None}


def run_shard(seed, regime, difficulty, arm, p):
    start = time.monotonic()
    tasks, supports = prepare_sources(seed, regime, p)
    rng = np.random.default_rng(51000 + seed)
    psi = rng.uniform(-2, 2, (p["candidates"], p["psi_dim"]))
    candidates = np.arange(len(psi))[:, None]
    initial = rng.choice(len(psi), p["initial_points"], replace=False).tolist()
    target_seed = 61000 + seed * 100 + p["regimes"].index(regime)
    truth = task_truth(psi, supports, target_seed)
    truth[:, 1] += p["difficulty_offsets"][difficulty]
    batches = [head_batches(tasks, h) for h in range(3)]
    noises = [[t["noise"][:, h] for t in tasks] for h in range(3)]
    means = [source_hard_prior(batches[h], noises[h], psi).target_model(psi) for h in range(2)]
    for index in initial:
        response, noise = observe(truth[index], target_seed, index, 0, p["replicates"])
        for h, model in enumerate(means):
            model.update(candidates[index], response[h], noise[h])
    frozen_mean, frozen_var = np.zeros_like(truth), np.zeros_like(truth)
    for h, model in enumerate(means):
        frozen_mean[:, h] = model.posterior_mean_many(candidates)
        frozen_var[:, h] = model.posterior_var_many(candidates)
    if arm == "target_only":
        prior = SourceFunctionPrior.fit(batches[2], noises[2], psi, "target_only")
    elif arm in ("risk_ard", "risk_ard_mixture"):
        prior = fit_source_risk_ard(batches[2], noises[2], psi)
    else:
        prior = SourceFunctionPrior.fit(batches[2], noises[2], psi, arm,
                  oracle_indices=supports[2] if arm == "oracle_structure" else None)
    full = SourceFunctionPrior.fit(batches[2], noises[2], psi, "fixed_source") if arm == "risk_ard_mixture" else None
    true_utility, true_power = verification_utility(truth, np.zeros_like(truth), p)
    margin = truth[:, 1] + norm.ppf(1 - p["chance_alpha"]) * np.exp(.5 * truth[:, 2])
    rows, contexts, traces = [], {}, {}
    for context in p["mean_contexts"]:
        fixed_mean, fixed_var = frozen_mean.copy(), frozen_var.copy()
        if context == "known_means_diagnostic":
            fixed_mean[:, :2], fixed_var[:, :2] = truth[:, :2], 0
        contexts[context] = {"mean": fixed_mean[:, :2].tolist(), "variance": fixed_var[:, :2].tolist(),
                             "rmse": np.sqrt(np.mean((fixed_mean[:, :2] - truth[:, :2]) ** 2, axis=0)).tolist()}
        model = TransferMixtureGPR([prior, full], psi) if full is not None else prior.target_model(psi)
        visits, queries = np.zeros(len(psi), dtype=int), []
        for step in range(p["target_points"]):
            index = initial[step] if step < len(initial) else int(np.argmax(
                risk_only_kg_scores(model, candidates, fixed_mean, fixed_var, p)))
            response, noise = observe(truth[index], target_seed, index, int(visits[index]), p["replicates"])
            model.update(candidates[index], response[2], noise[2])
            visits[index] += 1
            queries.append(int(index))
            if step + 1 in p["checkpoints"]:
                row = checkpoint(model, fixed_mean, fixed_var, candidates, truth, target_seed, p, step + 1)
                row.update(seed=seed, regime=regime, difficulty=difficulty, arm=arm, mean_context=context,
                           risk_source_calls=0 if arm == "target_only" else 6144,
                           common_mean_source_calls=6144, source_archive_calls=6144)
                rows.append(row)
        traces[context] = queries
        print(json.dumps({"seed": seed, "regime": regime, "difficulty": difficulty, "arm": arm,
                          "mean_context": context, "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
    return {"protocol": p, "rows": rows, "initial_design": initial, "queries": traces, "fixed_contexts": contexts,
            "truth_summary": {"best_available_expected_verified_reward": float(true_utility.max()),
                              "best_available_verifier_power": float(true_power[np.argmax(true_utility)]),
                              "true_feasible_count": int(np.sum(margin <= 0))},
            "source_info": prior.diagnostics, "elapsed_seconds": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--difficulty", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(PROTOCOL.read_text())
    if args.seed not in p["seeds"] or args.regime not in p["regimes"] or args.difficulty not in p["difficulty_offsets"] or args.arm not in p["arms"]:
        raise ValueError("outside frozen protocol")
    result = run_shard(args.seed, args.regime, args.difficulty, args.arm, p)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
