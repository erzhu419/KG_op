"""A33: reuse jointly observed histories as a separated meta-training archive."""
import gzip
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.run_meta_prior_mechanism import observe, shared_support, target_supports, task_truth
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import mixture_forecasts, prequential_components
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_heterogeneous_archive_a33_20261003"


def visits_for(candidates):
    visits, result = {}, []
    for candidate in candidates:
        result.append(visits.get(candidate, 0))
        visits[candidate] = result[-1] + 1
    return result


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_heterogeneous_archive_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    cohort = spec["cohort"]
    result = {"stage": spec["stage"], "complete": False, "new_source_calls": 0,
              "new_target_calls": 0, "server_jobs": 0, "server_pulls": 0,
              "kg_calls": 0, "recommendations": 0, "evaluation_responses_saved": 0}
    try:
        records = read(spec["inputs"][0])["records"]
        base = read(spec["inputs"][1])["base_protocol"]
        protocol = {**base, **base["allocations"][cohort["allocation"]]}
        contexts = {(c["seed"], c["regime"]): c for c in read(spec["inputs"][2])["rows"]}
        previous = {(p["seed"], p["regime"], p["path_policy"]): p
                    for p in read(spec["inputs"][3])["rows"]}
        indexed = {(r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]): r["row"] for r in records}
        all_seeds = cohort["meta_training_seeds"] + cohort["held_development_evaluation_seeds"]
        expected = {(s, r, a) for s in all_seeds for r in cohort["regimes"] for a in cohort["path_policies"]}
        if len(records) != 160 or set(indexed) != expected or set(previous) != expected:
            raise ValueError("A33 original paid cohort changed")
        evaluation = []
        for s, r, a in sorted(expected):
            row = indexed[s, r, a]
            if (row["difficulty"] != cohort["difficulty"] or row["allocation"] != cohort["allocation"]
                    or len(row["queries"]) != 24 or row["queries"][:8] != row["initial_design"]):
                raise ValueError("A33 original chronology or budget changed")
            if s in cohort["held_development_evaluation_seeds"]:
                evaluation.append({"seed": s, "regime": r, "path_policy": a,
                                   "candidates": row["queries"], "initial_design": row["initial_design"],
                                   "prior_visits": visits_for(row["queries"])})
        tasks, duplicates, max_error, repeated_candidates = [], 0, 0., 0
        for s in cohort["meta_training_seeds"]:
            psi = np.random.default_rng(51000 + s).uniform(-2, 2, (protocol["candidates"], protocol["psi_dim"]))
            basis = FixedFunctionBasis(psi[:protocol["reference_bank_size"]])
            support = shared_support(s, protocol["psi_dim"])
            for regime in cohort["regimes"]:
                supports = target_supports(support, s, regime, protocol["psi_dim"])
                if regime == "separate_head_structure":
                    supports = [support, shared_support(s + 1000, protocol["psi_dim"]),
                                target_supports(support, s, "support_mismatch", protocol["psi_dim"])[0]]
                target_seed = 61000 + 100 * s + protocol["regimes"].index(regime)
                truth = task_truth(psi, supports, target_seed)
                truth[:, 1] += protocol["difficulty_offsets"][cohort["difficulty"]]
                c = contexts[s, regime]
                priors = [SourceFunctionPrior(basis, np.r_[p["intercept_mean"], np.zeros(basis.feature_dim)],
                          np.diag(p["coefficient_variance"]), p["discrepancy_variance"], {})
                          for p in c["eta_prior_components"]]
                keys, observed, noise, paths, key_index = [], [], [], [], {}
                for policy in cohort["path_policies"]:
                    ids = indexed[s, regime, policy]["queries"]
                    visits = visits_for(ids)
                    labels, variances, mapping = [], [], []
                    for candidate, visit in zip(ids, visits):
                        y, n = observe(truth[candidate], target_seed, candidate, visit, protocol["replicates"])
                        key = (candidate, visit)
                        if key in key_index:
                            k = key_index[key]
                            if not (np.array_equal(y, observed[k]) and np.array_equal(n, noise[k])):
                                raise ValueError("A33 shared random-number record differs across policies")
                            duplicates += 1
                        else:
                            k = len(keys); key_index[key] = k
                            keys.append(key); observed.append(y); noise.append(n)
                        mapping.append(k); labels.append(y); variances.append(n)
                    labels, variances = np.array(labels), np.array(variances)
                    forecast = prequential_components(priors, psi, np.array(ids), labels[:, 2], variances[:, 2])
                    score, _ = mixture_forecasts(forecast["log_evidence"], forecast["log_density"], [.5, .5])
                    old = previous[s, regime, policy]
                    max_error = max(max_error, float(np.max(np.abs(score[8:24] - old["scores"]["I"]))),
                                    float(np.max(np.abs(forecast["log_density"][8:24, 1] - old["scores"]["F"]))))
                    if max_error > 1e-8:
                        raise ValueError("A33 regenerated paid observations do not reproduce A28")
                    paths.append({"path_policy": policy, "observation_indices": mapping})
                ids = [k[0] for k in keys]
                repeated_candidates += len(ids) - len(set(ids))
                tasks.append({"seed": s, "regime": regime, "candidates": ids,
                              "prior_visits": [k[1] for k in keys], "psi": psi[ids].tolist(),
                              "response": np.array(observed).tolist(), "noise": np.array(noise).tolist(),
                              "paths": paths})
        count = sum(len(t["candidates"]) for t in tasks)
        archive = {"role": "historical_meta_training", "seed_pairs": cohort["meta_training_seed_pairs"],
                   "head_order": ["f", "g", "eta"], "tasks": tasks}
        (ARTIFACTS / "meta_training_archive.json.gz").write_bytes(gzip.compress(json.dumps(archive, allow_nan=False).encode()))
        (ARTIFACTS / "evaluation_manifest.json").write_text(json.dumps({"role": "reused_development_evaluation",
            "untouched_confirmation": False, "paths": evaluation}, indent=2) + "\n")
        result.update(complete=True, meta_training_latent_tasks=len(tasks), meta_training_paths=80,
                      evaluation_latent_tasks=40, evaluation_paths=len(evaluation),
                      meta_training_unique_batches=count, shared_policy_duplicate_batches_removed=duplicates,
                      distinct_visit_batches_at_repeated_candidates=repeated_candidates,
                      represented_historical_training_path_calls=80 * 24 * 12,
                      represented_unique_historical_training_calls=count * 12,
                      original_training_source_archives=16,
                      represented_original_training_source_calls=16 * 8 * 64 * 12,
                      max_original_i_and_f_forecast_error=max_error)
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result["cpu_seconds"] = time.process_time() - start
        (ARTIFACTS / "preparation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
