"""A39: initial-state risk-class information on the public candidate bank."""
import gzip
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.risk_class_information import candidate_risk_predictions, risk_class_information
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_risk_information_design_a39_20261003"


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_risk_information_design_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A39 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [], "new_covariance_fits": 0,
              "new_source_calls": 0, "new_target_calls": 0, "server_jobs": 0, "server_pulls": 0,
              "kg_calls": 0, "recommendations": 0, "evaluation_paths_scored": 0,
              "gaussian_conditioning_solves": 0}
    details = []
    try:
        fits = {f["pair_id"]: f for f in read(spec["inputs"][0])["fits"]}
        cache = {(p["seed"], p["regime"], p["path_policy"]): p for p in read(spec["inputs"][1])["rows"]}
        tasks = read(spec["inputs"][2])["tasks"]
        contexts = {(c["seed"], c["regime"]): c for c in read(spec["inputs"][3])["rows"] if 48 <= c["seed"] <= 55}
        reference = np.random.default_rng(51048).uniform(-2, 2, (384, 8))[:96]
        common = FixedFunctionBasis(reference)
        future_noise = float(polygamma(1, 11 / 2))
        density_error, quadrature_error, bounds_error = 0., 0., 0.
        for task in tasks:
            seed, regime = task["seed"], task["regime"]
            paths = task["paths"]
            first = np.array(paths[0]["observation_indices"][:8])
            second = np.array(paths[1]["observation_indices"][:8])
            if not np.array_equal(first, second):
                raise ValueError("A39 original two policies do not share their initial observations")
            ids = np.array(task["candidates"])[first]
            labels, noise = np.array(task["response"])[first, 2], np.array(task["noise"])[first, 2]
            bank = np.random.default_rng(51000 + seed).uniform(-2, 2, (384, 8))
            local_basis = FixedFunctionBasis(bank[:96])
            priors = [SourceFunctionPrior(local_basis, np.r_[p["intercept_mean"], np.zeros(68)],
                      np.diag(p["coefficient_variance"]), p["discrepancy_variance"], {})
                      for p in contexts[seed, regime]["eta_prior_components"]]
            saved = cache[seed, regime, paths[0]["path_policy"]]
            fitted = fits[saved["pair_id"]]
            priors += [SourceFunctionPrior(common, np.array(p["mean"]), np.diag(p["coefficient_variance"]),
                       p["discrepancy_variance"], {}) for p in fitted["priors"]]
            weights = np.array(saved["component_posterior"])[0]
            means, variance = candidate_risk_predictions(priors, bank, ids, labels, noise, future_noise)
            result["gaussian_conditioning_solves"] += 4
            information = risk_class_information(means, variance, weights, 49)
            checked = risk_class_information(means, variance, weights, 81)
            quadrature_error = max(quadrature_error, float(np.max(np.abs(information - checked))))
            state_weights = np.r_[weights[:2].sum(), weights[2:]]
            entropy = float(-state_weights @ np.log(state_weights))
            bounds_error = max(bounds_error, float(max(-information.min(), information.max() - entropy, 0.)))
            maximum = float(information.max())
            for path in paths:
                index = int(path["observation_indices"][8])
                candidate = task["candidates"][index]
                y = task["response"][index][2]
                density = -.5 * ((y - means[candidate]) ** 2 / variance[candidate] + np.log(2 * np.pi * variance[candidate]))
                old = cache[seed, regime, path["path_policy"]]
                if not np.allclose(weights, np.array(old["component_posterior"])[0], rtol=0, atol=1e-12):
                    raise ValueError("A39 initial-state posterior differs between policies")
                density_error = max(density_error, float(np.max(np.abs(density - np.array(old["risk_log_density"])[0, :4]))))
                value = float(information[candidate])
                result["rows"].append({"seed": seed, "regime": regime, "path_policy": path["path_policy"],
                    "pair_id": saved["pair_id"], "old_ninth_candidate": candidate, "information": value,
                    "bank_maximum_information": maximum, "information_ratio_to_maximum": value / maximum,
                    "information_rank": 1 + int(np.sum(information > value + 1e-12)), "state_entropy": entropy})
            details.append({"seed": seed, "regime": regime, "pair_id": saved["pair_id"],
                            "state_weights": state_weights.tolist(), "candidate_information": information.tolist()})
        controls = {"max_old_ninth_query_density_error": density_error, "max_quadrature49_81_difference": quadrature_error,
                    "max_information_entropy_bound_error": bounds_error}
        result["controls"] = controls
        if density_error > 1e-8 or quadrature_error > 1e-6 or bounds_error > 1e-6:
            raise ValueError("A39 predictive reconstruction or information integration not verified")
        bootstrap = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(4, size=(20000, 4))
        groups = []
        for regime in ["all"] + sorted({r["regime"] for r in result["rows"]}):
            for policy in ("risk_ard_mixture", "fixed_source"):
                selected = [r for r in result["rows"] if r["path_policy"] == policy and (regime == "all" or r["regime"] == regime)]
                groups.append({"regime": regime, "path_policy": policy,
                    **{name: block_contrast([np.mean([r[name] for r in selected if r["pair_id"] == j]) for j in range(4)], bootstrap)
                       for name in ("information", "bank_maximum_information", "information_ratio_to_maximum")},
                    "mean_information_rank": float(np.mean([r["information_rank"] for r in selected])),
                    "mean_state_entropy": float(np.mean([r["state_entropy"] for r in selected]))})
        result.update(complete=True, groups=groups, latent_tasks=len(tasks), old_next_queries_evaluated=len(result["rows"]))
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result["cpu_seconds"] = time.process_time() - start
        (ARTIFACTS / "candidate_information.json.gz").write_bytes(gzip.compress(json.dumps({"rows": details}, allow_nan=False).encode()))
        (ARTIFACTS / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result.get(k) for k in ("complete", "cpu_seconds", "controls", "failure")}))


if __name__ == "__main__":
    main()
