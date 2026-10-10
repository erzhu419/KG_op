"""A41: fixed latent-function and uncertainty diagnosis on the public bank."""
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.run_meta_prior_mechanism import shared_support, target_supports, task_truth
from performance.validate_meta_prior_heterogeneous_covariance import scale_matched
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.latent_risk_metrics import latent_risk_metrics
from representation.risk_class_information import candidate_risk_predictions
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_latent_risk_error_plan.json")
    artifacts = ROOT / "paper_artifacts" / spec["stage"]
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A41 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [],
        "new_covariance_fits": 0, "new_source_calls": 0, "new_target_calls": 0,
        "server_jobs": 0, "server_pulls": 0, "kg_calls": 0, "recommendations": 0,
        "evaluation_paths_scored": 0, "gaussian_conditioning_solves": 0,
        "oracle_is_deployable": False}
    try:
        archive = read(spec["inputs"][0])
        tasks = archive["tasks"]
        fitted = read(spec["inputs"][1])
        fits = {f["pair_id"]: f for f in fitted["fits"]}
        cache = {(p["seed"], p["regime"], p["path_policy"]): p for p in read(spec["inputs"][2])["rows"]}
        contexts = {(c["seed"], c["regime"]): c for c in read(spec["inputs"][3])["rows"]}
        base = read(spec["inputs"][4])["base_protocol"]
        previous = read(spec["inputs"][5])
        expected = {(s, r) for pair in archive["seed_pairs"] for s in pair for r in base["regimes"]}
        if (len(tasks) != 40 or {(t["seed"], t["regime"]) for t in tasks} != expected
                or sorted({t["seed"] for t in tasks}) != list(range(48, 56))
                or not fitted["complete"] or not previous["complete"]):
            raise ValueError("A41 requires the unchanged completed training cohort")
        common = FixedFunctionBasis(np.random.default_rng(51048).uniform(-2, 2, (384, 8))[:96])
        controls = {"max_next_paid_atom_density_error": 0., "max_next_paid_mixture_score_error": 0.,
                    "max_past_component_weight_error": 0., "max_shared_initial_evidence_error": 0.,
                    "max_meta_basis_normalization_error": 0., "max_nonintercept_source_mean": 0.,
                    "queried_candidates_scored": 0}
        for task in tasks:
            seed, regime = task["seed"], task["regime"]
            bank = np.random.default_rng(51000 + seed).uniform(-2, 2, (384, 8))
            basis = FixedFunctionBasis(bank[:96])
            local = [SourceFunctionPrior(basis, np.r_[p["intercept_mean"], np.zeros(68)],
                    np.diag(p["coefficient_variance"]), p["discrepancy_variance"], {})
                    for p in contexts[seed, regime]["eta_prior_components"]]
            paths = {p["path_policy"]: p["observation_indices"] for p in task["paths"]}
            one, two = (cache[seed, regime, p] for p in ("risk_ard_mixture", "fixed_source"))
            if paths["risk_ard_mixture"][:8] != paths["fixed_source"][:8] or one["pair_id"] != two["pair_id"]:
                raise ValueError("A41 shared initial history or source fold changed")
            controls["max_shared_initial_evidence_error"] = max(controls["max_shared_initial_evidence_error"],
                float(np.max(np.abs(np.array(one["risk_log_evidence"])[0] - np.array(two["risk_log_evidence"])[0]))))
            fit = fits[one["pair_id"]]
            if seed not in fit["held_seeds"] or seed in fit["training_seeds"]:
                raise ValueError("A41 scored seed entered its saved source covariance fit")
            meta = [SourceFunctionPrior(common, np.array(p["mean"]), np.diag(p["coefficient_variance"]),
                    p["discrepancy_variance"], {}) for p in fit["priors"]]
            for p, saved in zip(meta, fit["priors"]):
                controls["max_meta_basis_normalization_error"] = max(controls["max_meta_basis_normalization_error"],
                    float(np.max(np.abs(common.mean - saved["basis_mean"]))),
                    float(np.max(np.abs(common.scale - saved["basis_scale"]))))
            priors = local + meta + [scale_matched(p) for p in meta]
            controls["max_nonintercept_source_mean"] = max(controls["max_nonintercept_source_mean"],
                max(float(np.max(np.abs(p.mean[1:]))) for p in priors))
            high = int(np.argmax([g["high_frequency_variance_fraction"] for g in fit["diagnostics"]["geometry"]]))
            known = 2 + (high if regime == "high_frequency_risk" else 1 - high)
            pi = np.array(fit["diagnostics"]["class_weights"])
            model_specs = {"I": ([0, 1], [.5, .5]), "L": ([0, 1, 2, 3], np.r_[.25, .25, .5 * pi]),
                           "LG": ([0, 1, 4, 5], np.r_[.25, .25, .5 * pi]),
                           "O": ([0, 1, known], [.25, .25, .5])}
            ids = np.array(task["candidates"])
            labels, noise = np.array(task["response"])[:, 2], np.array(task["noise"])[:, 2]
            states = [("early", "shared_initial", paths["risk_ard_mixture"][:8], one, 0)]
            states += [("late", policy, path[:23], cache[seed, regime, policy], 15) for policy, path in paths.items()]
            predictions = []
            for phase, policy, indices, cached, step in states:
                mean, variance = candidate_risk_predictions(priors, bank, ids[indices], labels[indices], noise[indices], 0.)
                result["gaussian_conditioning_solves"] += len(priors)
                if result["gaussian_conditioning_solves"] > spec["limits"]["gaussian_conditioning_solves_max"]:
                    raise ValueError("A41 exceeds its fixed Gaussian conditioning budget")
                e = np.array(cached["risk_log_evidence"])[step:step + 1]
                models = {}
                for name, (columns, masses) in model_specs.items():
                    posterior = e[:, columns] + np.log(masses)
                    posterior = np.exp(posterior - logsumexp(posterior, axis=1, keepdims=True))[0]
                    models[name] = (columns, posterior)
                    if name in ("L", "LG"):
                        key = "component_posterior" if name == "L" else "generic_component_posterior"
                        controls["max_past_component_weight_error"] = max(controls["max_past_component_weight_error"],
                            float(np.max(np.abs(posterior - np.array(cached[key])[step]))))
                check_policies = paths.keys() if phase == "early" else [policy]
                for check_policy in check_policies:
                    future = paths[check_policy][8 if phase == "early" else 23]
                    observed_var = variance[ids[future]] + noise[future]
                    density = -.5 * ((labels[future] - mean[ids[future]]) ** 2 / observed_var + np.log(2 * np.pi * observed_var))
                    old = cache[seed, regime, check_policy]
                    controls["max_next_paid_atom_density_error"] = max(controls["max_next_paid_atom_density_error"],
                        float(np.max(np.abs(density - np.array(old["risk_log_density"])[step]))))
                    for name in ("I", "L", "LG"):
                        columns, posterior = models[name]
                        score = logsumexp(density[columns] + np.log(posterior))
                        controls["max_next_paid_mixture_score_error"] = max(controls["max_next_paid_mixture_score_error"],
                            abs(float(score) - old["scores"][name][step]))
                queried = set(ids[indices])
                unseen = np.array([j for j in range(len(bank)) if j not in queried])
                controls["queried_candidates_scored"] += len(set(unseen) & set(ids[indices]))
                predictions.append((phase, policy, unseen, mean, variance, models))
            # Original analytic truth is generated only after every state/model prediction.
            support = shared_support(seed, base["psi_dim"])
            supports = target_supports(support, seed, regime, base["psi_dim"])
            if regime == "separate_head_structure":
                supports = [support, shared_support(seed + 1000, base["psi_dim"]),
                            target_supports(support, seed, "support_mismatch", base["psi_dim"])[0]]
            truth = task_truth(bank, supports, 61000 + 100 * seed + base["regimes"].index(regime))[:, 2]
            for phase, policy, unseen, mean, variance, models in predictions:
                scores = {name: latent_risk_metrics(mean[unseen][:, columns], variance[unseen][:, columns], posterior, truth[unseen])
                          for name, (columns, posterior) in models.items()}
                result["rows"].append({"seed": seed, "regime": regime, "pair_id": one["pair_id"],
                    "phase": phase, "path_policy": policy, "unqueried_candidates": len(unseen), "models": scores,
                    "component_weights": {name: posterior.tolist() for name, (_, posterior) in models.items()}})
        result["controls"] = controls
        if max(controls.values()) > 1e-8:
            raise ValueError("A41 original prediction/chronology control failed")
        bootstrap = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(4, size=(20000, 4))
        groups = []
        for phase in ("early", "late"):
            for regime in ["all"] + base["regimes"]:
                rows = [r for r in result["rows"] if r["phase"] == phase and (regime == "all" or r["regime"] == regime)]
                metrics = list(rows[0]["models"]["I"])
                contrasts = {}
                for first, second in (("L", "I"), ("L", "LG"), ("O", "I"), ("O", "L")):
                    contrasts[f"{first}_minus_{second}"] = {metric: block_contrast(
                        [np.mean([r["models"][first][metric] - r["models"][second][metric] for r in rows if r["pair_id"] == j])
                         for j in range(4)], bootstrap) for metric in ("eta_rmse", "latent_log_score", "coverage90", "environmental_variance_bias")}
                groups.append({"phase": phase, "regime": regime, "latent_tasks": len(rows) if phase == "early" else len(rows) // 2,
                    "models": {name: {metric: float(np.mean([r["models"][name][metric] for r in rows])) for metric in metrics}
                               for name in model_specs}, "contrasts": contrasts})
        result.update(complete=True, groups=groups, latent_tasks=len(tasks), posterior_states=len(result["rows"]),
                      unqueried_state_candidate_scores=sum(r["unqueried_candidates"] for r in result["rows"]),
                      component_order=["local_structural", "local_full", "meta0", "meta1", "isotropic0", "isotropic1"])
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result["cpu_seconds"] = time.process_time() - start
        (artifacts / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result.get(k) for k in ("complete", "cpu_seconds", "posterior_states", "controls", "failure")}))


if __name__ == "__main__":
    main()
