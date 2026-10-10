"""Stage A: two noise updates on archived queries, without KG or verification."""
import gzip
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.verification_kg import verification_utility
from performance.run_meta_prior_mechanism import moments, observe, task_truth
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches
from representation.transfer_mixture import TransferMixtureGPR

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_calibration_replay_a_20261003"


def restore_context(record, protocol):
    row, info = record["row"], record["source_info"]
    seed, regime = row["seed"], row["regime"]
    psi = np.random.default_rng(51000 + seed).uniform(-2, 2,
        (protocol["candidates"], protocol["psi_dim"]))
    reference = psi[:protocol["reference_bank_size"]]
    tasks, supports = prepare_sources(seed, regime, protocol)
    batches = [head_batches(tasks, h) for h in range(3)]
    noises = [[t["noise"][:, h] for t in tasks] for h in range(3)]
    full = [SourceFunctionPrior.fit(batches[h], noises[h], reference, "fixed_source")
            for h in range(3)]
    structural = [SourceFunctionPrior.from_basis(batches[h], noises[h],
        FixedFunctionBasis(reference, info[h]["selected_indices"]), "conditional_hard")
        for h in range(2)]
    basis = FixedFunctionBasis(reference)
    tau = np.asarray(info[2]["source_function_variance"])
    signal_variance = np.mean([max(float(np.var(b.signals) - np.mean(v)), .02)
                              for b, v in zip(batches[2], noises[2])])
    mean = np.r_[np.mean([np.mean(b.signals) for b in batches[2]]), np.zeros(len(tau))]
    structural.append(SourceFunctionPrior(basis, mean,
        np.diag(np.r_[.25, tau + .01 * signal_variance / len(tau)]), .02, {}))
    if any(np.any(p.mean[1:]) for p in full + structural):
        raise ValueError("nonintercept source coefficient means must be zero")
    target_seed = 61000 + 100 * seed + protocol["regimes"].index(regime)
    truth = task_truth(psi, supports, target_seed)
    truth[:, 1] += protocol["difficulty_offsets"][row["difficulty"]]
    return psi, reference, full, structural, truth, target_seed


def replay(record, context, protocol, update):
    row = record["row"]
    psi, reference, full, structural, truth, target_seed = context
    candidates = np.arange(len(psi))[:, None]
    models = ([TransferMixtureGPR([structural[h], full[h]], psi, reference) for h in range(3)]
              if row["arm"] == "risk_ard_mixture" else [p.target_model(psi) for p in full])
    if len(row["queries"]) != protocol["target_points"] or row["queries"][:8] != row["initial_design"]:
        raise ValueError("archived query path does not match its budget/initial design")
    visits = np.zeros(len(psi), dtype=int)
    for step, index in enumerate(row["queries"]):
        x = candidates[index:index + 1]
        # Compute before generating any current response, including eta.
        if update == "pre_batch_risk_predictive":
            mu = models[2].posterior_mean_many(x)[0]
            variance = models[2].posterior_var_many(x)[0]
            predictable_noise = float(np.exp(mu + .5 * variance) / protocol["replicates"])
        response, noise = observe(truth[index], target_seed, index, int(visits[index]),
                                  protocol["replicates"])
        if update == "pre_batch_risk_predictive":
            noise[1] = predictable_noise
        for h, model in enumerate(models):
            model.update(candidates[index], response[h], noise[h])
        visits[index] += 1
        if step + 1 == protocol["initial_points"]:
            m, v = moments(models, candidates)
            initial_choice = int(np.argmax(verification_utility(m, v, protocol)[0]))
    mean, variance = moments(models, candidates)
    predicted_reward, predicted_power = verification_utility(mean, variance, protocol)
    choice = int(np.argmax(predicted_reward))
    weights = [m.diagnostics()["structural_weight"] for m in models] if row["arm"] == "risk_ard_mixture" else []
    return mean, variance, choice, initial_choice, predicted_reward, predicted_power, weights


def score(row, replayed, truth, protocol, update):
    mean, variance, choice, initial, reward, power, weights = replayed
    # Simulator truth is used only after both recommendations have been fixed.
    true_reward, true_power = verification_utility(truth, np.zeros_like(truth), protocol)
    margin = truth[:, 1] + norm.ppf(1 - protocol["chance_alpha"]) * np.exp(.5 * truth[:, 2])
    gap = float(power[choice] - true_power[choice])
    z = (mean[choice] - truth[choice]) / np.sqrt(variance[choice])
    return {**{k: row[k] for k in ("seed", "regime", "arm")}, "update": update,
        "recommendation": choice, "initial_recommendation": initial,
        "initial_expected_verified_reward": float(true_reward[initial]),
        "original_initial_expected_verified_reward": float(true_reward[row["initial_recommendation"]]),
        "expected_verified_reward": float(true_reward[choice]),
        "predicted_verified_reward": float(reward[choice]),
        "predicted_verifier_power": float(power[choice]), "true_verifier_power": float(true_power[choice]),
        "power_gap": gap, "absolute_power_gap": abs(gap), "squared_power_gap": gap ** 2,
        "true_feasible": bool(margin[choice] <= 0), "true_margin": float(margin[choice]),
        "posterior_mean": mean[choice].tolist(), "posterior_variance": variance[choice].tolist(),
        "truth": truth[choice].tolist(), "standardized_error": z.tolist(),
        "structural_weights": weights,
        "reward_minus_own_initial": float(true_reward[choice] - true_reward[initial])}


def summarize(rows, spec):
    metrics = ("expected_verified_reward", "absolute_power_gap", "power_gap", "squared_power_gap",
               "true_feasible", "reward_minus_own_initial")
    rng = np.random.default_rng(20261003)
    indices = rng.integers(len(spec["seeds"]), size=(20000, len(spec["seeds"])))
    groups, changes, source_contrasts = [], [], []
    cells = {}
    for regime in spec["regimes"]:
        for arm in spec["priors"]:
            for update in spec["updates_per_path"]:
                rr = sorted((r for r in rows if (r["regime"], r["arm"], r["update"]) ==
                    (regime, arm, update)), key=lambda r: r["seed"])
                cells[regime, arm, update] = rr
                groups.append({"regime": regime, "arm": arm, "update": update,
                    "means": {k: float(np.mean([r[k] for r in rr])) for k in metrics}})
            old, new = (cells[regime, arm, u] for u in spec["updates_per_path"])
            delta = np.array([b["expected_verified_reward"] - a["expected_verified_reward"]
                              for a, b in zip(old, new)])
            changes.append({"regime": regime, "arm": arm, "reward_delta": float(delta.mean()),
                "paired_seed_bootstrap95": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
                "power_absolute_error_delta": float(np.mean([b["absolute_power_gap"] - a["absolute_power_gap"]
                                                             for a, b in zip(old, new)])),
                "recommendations_changed": sum(a["recommendation"] != b["recommendation"] for a, b in zip(old, new)),
                "better": int(np.sum(delta > 1e-12)), "worse": int(np.sum(delta < -1e-12)),
                "seed_reward_deltas": delta.tolist()})
        for update in spec["updates_per_path"]:
            generic = cells[regime, "fixed_source", update]
            learned = cells[regime, "risk_ard_mixture", update]
            delta = np.array([a["expected_verified_reward"] - b["expected_verified_reward"]
                              for a, b in zip(learned, generic)])
            source_contrasts.append({"regime": regime, "update": update,
                "learned_minus_generic": float(delta.mean()),
                "paired_seed_bootstrap95": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()})
    return {"groups": groups, "update_changes": changes, "source_contrasts": source_contrasts,
            "intervals": "Nominal unadjusted paired seed intervals on retrospective development data."}


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    execution = json.loads((ARTIFACTS / "execution_plan.json").read_text())
    spec, frozen = execution["stage_a"], execution["base_protocol"]
    protocol = {**frozen, **frozen["allocations"][spec["allocation"]]}
    records = json.loads(gzip.decompress((ARTIFACTS / "inputs.json.gz").read_bytes()))["records"]
    expected = {(s, r, a) for s in spec["seeds"] for r in spec["regimes"] for a in spec["priors"]}
    actual = [(r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) for r in records]
    if len(actual) != len(expected) or set(actual) != expected or any(
        r["row"]["difficulty"] != spec["difficulty"] or r["row"]["allocation"] != spec["allocation"] for r in records):
        raise ValueError("replay inputs do not cover the complete planned cohort")
    cap = int(spec["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("stage A reached its local CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    result = {"stage": "meta_prior_calibration_replay_a_20261003", "complete": False,
        "old_reproduction_passed": False, "paths": len(records), "kg_calls": 0,
        "new_searches": 0, "new_random_verifications": 0, "server_experiment_jobs": 0,
        "interpretation": "Retrospective fixed-query-path development evidence only.", "rows": []}
    contexts, old_results = {}, {}
    max_error = 0.0
    try:
        # Reproduce the entire old cohort before computing any counterfactual.
        for count, record in enumerate(records, 1):
            row = record["row"]
            key = row["seed"], row["regime"]
            if key not in contexts:
                learned = next(r for r in records if (r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) ==
                               (*key, "risk_ard_mixture"))
                contexts[key] = restore_context(learned, protocol)
            old = replay(record, contexts[key], protocol, "sample_variance_plugin")
            if old[2] != row["recommendation"] or old[3] != row["initial_recommendation"]:
                raise ValueError(f"old recommendation reconstruction mismatch: {actual[count - 1]}")
            observed = [old[4][old[2]], old[5][old[2]], *old[6]]
            wanted = [row["predicted_verified_reward"], row["predicted_verifier_power"],
                      *[m["structural_weight"] for m in row["final_model_state"]]]
            np.testing.assert_allclose(observed, wanted, atol=1e-8, rtol=1e-8)
            max_error = max(max_error, float(np.max(np.abs(np.array(observed) - wanted))))
            old_results[actual[count - 1]] = old
            result["rows"].append(score(row, old, contexts[key][4], protocol, "sample_variance_plugin"))
            if count % 20 == 0:
                print(f"REPRODUCED {count}/{len(records)}", flush=True)
        result["old_reproduction_passed"] = True
        result["max_old_reproduction_error"] = max_error
        for count, record in enumerate(records, 1):
            row = record["row"]
            context = contexts[row["seed"], row["regime"]]
            new = replay(record, context, protocol, "pre_batch_risk_predictive")
            old = old_results[actual[count - 1]]
            # A one-variable replay must leave the f and eta head posteriors unchanged.
            for i in (0, 1):
                np.testing.assert_allclose(new[i][:, [0, 2]], old[i][:, [0, 2]], atol=1e-12, rtol=1e-12)
            result["rows"].append(score(row, new, context[4], protocol, "pre_batch_risk_predictive"))
            if count % 20 == 0:
                print(f"COUNTERFACTUAL {count}/{len(records)}", flush=True)
        result.update(summarize(result["rows"], spec))
        result["complete"] = True
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (ARTIFACTS / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "paths", "old_reproduction_passed",
        "max_old_reproduction_error", "cpu_seconds", "wall_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()
