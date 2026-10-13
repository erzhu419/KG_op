"""Replay stored observations to explain the frozen seed19 decision change."""
import json
import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path("/home/zhengliang01/scheduleurm_work/KG_op_meta_prior_mechanism_deploy/SC-OLH-KG")
sys.path.insert(0, str(ROOT))
import numpy as np
from performance.run_meta_prior_mechanism import observe, moments, task_truth
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.conditional_function_prior import fit_conditional_priors
from representation.source_risk_ard import fit_source_risk_ard
from representation.transfer_mixture import TransferMixtureGPR
from core.verification_kg import verification_utility

results = Path("/home/zhengliang01/scheduleurm_work/KG_op_meta_prior_mechanism_results")
output = []
for stage in ("meta_prior_mechanism_v3_20261002", "meta_prior_budget80_20261002"):
    shard = json.loads((results / stage / "seed19_support_mismatch_binding/result.json").read_text())
    protocol = shard["protocol"]
    tasks, supports = prepare_sources(19, "support_mismatch", protocol)
    rng = np.random.default_rng(51019)
    psi = rng.uniform(-2, 2, (96, 8))
    candidates = np.arange(96)[:, None]
    batches = [head_batches(tasks, h) for h in range(3)]
    noise = [[t["noise"][:, h] for t in tasks] for h in range(3)]
    full = [SourceFunctionPrior.fit(batches[h], noise[h], psi, "fixed_source") for h in range(3)]
    structural = [fit_conditional_priors(batches[h], noise[h], psi)["conditional_hard"] for h in range(2)]
    structural += [fit_source_risk_ard(batches[2], noise[2], psi)]
    target_seed = 62900 + protocol["regimes"].index("support_mismatch")
    truth = task_truth(psi, supports, target_seed)
    truth[:, 1] += protocol["difficulty_offsets"]["binding"]
    true_utility, true_power = verification_utility(truth, np.zeros_like(truth), protocol)
    for arm in ("fixed_source", "risk_ard_mixture"):
        row = next(r for r in shard["rows"] if r["arm"] == arm and r["scheme"] == "verification_kg")
        models = ([p.target_model(psi) for p in full] if arm == "fixed_source" else
                  [TransferMixtureGPR([structural[h], full[h]], psi) for h in range(3)])
        visits = np.zeros(96, dtype=int)
        for index in row["queries"]:
            response, variance = observe(truth[index], target_seed, index, int(visits[index]), 12)
            for h, model in enumerate(models):
                model.update(candidates[index], response[h], variance[h])
            visits[index] += 1
        mean, variance = moments(models, candidates)
        utility, power = verification_utility(mean, variance, protocol)
        assert int(np.argmax(utility)) == row["recommendation"]
        assert np.allclose(np.sqrt(np.mean((mean - truth) ** 2, axis=0)), row["final"]["rmse"])
        inspected = []
        for index in (67, 82):
            point = {"candidate": index, "visits": int(visits[index]), "truth": truth[index].tolist(),
                     "posterior_mean": mean[index].tolist(), "posterior_sd": np.sqrt(variance[index]).tolist(),
                     "predicted_utility": float(utility[index]), "predicted_power": float(power[index]),
                     "true_utility": float(true_utility[index]), "true_power": float(true_power[index])}
            if arm == "risk_ard_mixture":
                point["risk_component_mean"] = [float(m.posterior_mean(candidates[index])) for m in models[2].models]
                point["risk_component_sd"] = [float(np.sqrt(m.posterior_var(candidates[index]))) for m in models[2].models]
            inspected.append(point)
        output.append({"stage": stage, "arm": arm, "recommendation": row["recommendation"],
                       "risk_weights": models[2].diagnostics() if arm == "risk_ard_mixture" else None,
                       "candidates": inspected})
print(json.dumps({"replay_matches_frozen_recommendations_and_rmse": True, "rows": output}))
