"""A40: one retrospective query intervention using only paid records."""
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
from scipy.special import logsumexp, polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.risk_class_information import candidate_risk_predictions
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_paid_record_information_intervention_a40_20261003"


def select_information_batch(candidates, visits, initial_indices, information):
    """Selection uses input/visit identities and past-state information only."""
    earliest = {}
    initial = set(initial_indices)
    for index, (candidate, visit) in enumerate(zip(candidates, visits)):
        if index not in initial and (candidate not in earliest or visit < visits[earliest[candidate]]):
            earliest[candidate] = index
    candidate = min(earliest, key=lambda c: (-information[c], c))
    return earliest[candidate]


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_paid_record_information_intervention_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A40 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [], "new_covariance_fits": 0,
        "new_source_calls": 0, "new_target_calls": 0, "server_jobs": 0, "server_pulls": 0,
        "kg_calls": 0, "recommendations": 0, "evaluation_paths_scored": 0, "gaussian_conditioning_solves": 0}
    try:
        tasks = read(spec["inputs"][0])["tasks"]
        fits = {f["pair_id"]: f for f in read(spec["inputs"][1])["fits"]}
        cache = {(p["seed"], p["regime"], p["path_policy"]): p for p in read(spec["inputs"][2])["rows"]}
        information = {(p["seed"], p["regime"]): p for p in read(spec["inputs"][3])["rows"]}
        contexts = {(c["seed"], c["regime"]): c for c in read(spec["inputs"][4])["rows"] if 48 <= c["seed"] <= 55}
        common = FixedFunctionBasis(np.random.default_rng(51048).uniform(-2, 2, (384, 8))[:96])
        future_noise = float(polygamma(1, 11 / 2))
        density_error, posterior_error, coincidence_error = 0., 0., 0.
        for task in tasks:
            seed, regime = task["seed"], task["regime"]
            paths = {p["path_policy"]: p for p in task["paths"]}
            initial = paths["risk_ard_mixture"]["observation_indices"][:8]
            if initial != paths["fixed_source"]["observation_indices"][:8]:
                raise ValueError("A40 initial paid records differ")
            ids = np.array(task["candidates"])
            info = np.array(information[seed, regime]["candidate_information"])
            info_index = select_information_batch(task["candidates"], task["prior_visits"], initial, info)
            selected = {"risk_ard_mixture": paths["risk_ard_mixture"]["observation_indices"][8],
                        "fixed_source": paths["fixed_source"]["observation_indices"][8], "information": info_index}
            removed_candidates = set(ids[initial]) | {ids[k] for k in selected.values()}
            evaluation = np.array([k for k, candidate in enumerate(ids) if candidate not in removed_candidates])
            if not len(evaluation):
                raise ValueError("A40 has no common held records after candidate exclusion")
            labels, noise = np.array(task["response"])[:, 2], np.array(task["noise"])[:, 2]
            bank = np.random.default_rng(51000 + seed).uniform(-2, 2, (384, 8))
            local_basis = FixedFunctionBasis(bank[:96])
            priors = [SourceFunctionPrior(local_basis, np.r_[p["intercept_mean"], np.zeros(68)],
                      np.diag(p["coefficient_variance"]), p["discrepancy_variance"], {})
                      for p in contexts[seed, regime]["eta_prior_components"]]
            cached = cache[seed, regime, "risk_ard_mixture"]
            fitted = fits[cached["pair_id"]]
            priors += [SourceFunctionPrior(common, np.array(p["mean"]), np.diag(p["coefficient_variance"]),
                       p["discrepancy_variance"], {}) for p in fitted["priors"]]
            weights = np.array(cached["component_posterior"])[0]
            means, variance = candidate_risk_predictions(priors, bank, ids[initial], labels[initial], noise[initial], future_noise)
            result["gaussian_conditioning_solves"] += 4
            high_class = int(np.argmax([g["high_frequency_variance_fraction"] for g in fitted["diagnostics"]["geometry"]]))
            correct = 2 + (high_class if regime == "high_frequency_risk" else 1 - high_class)
            before = float(weights[correct] / weights[2:4].sum())
            states = {}
            for name, index in selected.items():
                candidate = ids[index]
                density = -.5 * ((labels[index] - means[candidate]) ** 2 / variance[candidate] + np.log(2 * np.pi * variance[candidate]))
                updated = weights * np.exp(density - np.max(density))
                updated /= updated.sum()
                if name != "information":
                    old = cache[seed, regime, name]
                    density_error = max(density_error, float(np.max(np.abs(density - np.array(old["risk_log_density"])[0, :4]))))
                    posterior_error = max(posterior_error, float(np.max(np.abs(updated - np.array(old["component_posterior"])[1]))))
                training = initial + [index]
                post_mean, post_var = candidate_risk_predictions(priors, bank, ids[training], labels[training], noise[training], future_noise)
                result["gaussian_conditioning_solves"] += 4
                target_mean, target_var = post_mean[ids[evaluation]], post_var[ids[evaluation]]
                predictive = -.5 * ((labels[evaluation, None] - target_mean) ** 2 / target_var + np.log(2 * np.pi * target_var))
                score = logsumexp(predictive + np.log(updated), axis=1)
                states[name] = {"selected_candidate": int(candidate), "selected_visit": task["prior_visits"][index],
                    "held_score": float(score.mean()), "meta_mass": float(updated[2:4].sum()),
                    "correct_class_probability": float(updated[correct] / updated[2:4].sum()),
                    "class_probability_change": float(updated[correct] / updated[2:4].sum() - before),
                    "query_information": float(info[candidate])}
            for one, first in selected.items():
                for two, second in selected.items():
                    if first == second:
                        coincidence_error = max(coincidence_error, abs(states[one]["held_score"] - states[two]["held_score"]))
            result["rows"].append({"seed": seed, "regime": regime, "pair_id": cached["pair_id"],
                "held_record_count": len(evaluation), "held_candidate_count": len(set(ids[evaluation])),
                "coincident_selected_batches": len(selected) - len(set(selected.values())), "histories": states,
                "information_minus_risk": states["information"]["held_score"] - states["risk_ard_mixture"]["held_score"],
                "information_minus_generic": states["information"]["held_score"] - states["fixed_source"]["held_score"],
                "information_class_probability_change": states["information"]["class_probability_change"]})
        controls = {"max_original_ninth_density_error": density_error, "max_original_after_ninth_posterior_error": posterior_error,
                    "max_coincident_history_prediction_error": coincidence_error}
        result["controls"] = controls
        if max(controls.values()) > 1e-8:
            raise ValueError("A40 unchanged history reproduction failed")
        bootstrap = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(4, size=(20000, 4))
        groups = []
        for regime in ["all"] + sorted({r["regime"] for r in result["rows"]}):
            rows = result["rows"] if regime == "all" else [r for r in result["rows"] if r["regime"] == regime]
            groups.append({"regime": regime, **{name: block_contrast(
                [np.mean([r[name] for r in rows if r["pair_id"] == j]) for j in range(4)], bootstrap)
                for name in ("information_minus_risk", "information_minus_generic", "information_class_probability_change")},
                "mean_held_records": float(np.mean([r["held_record_count"] for r in rows])),
                "history_means": {name: {metric: float(np.mean([r["histories"][name][metric] for r in rows]))
                     for metric in ("class_probability_change", "query_information", "correct_class_probability", "meta_mass")}
                     for name in ("information", "risk_ard_mixture", "fixed_source")}})
        result.update(complete=True, groups=groups, latent_tasks=len(tasks), intervention_histories=3 * len(tasks),
                      total_common_held_records=sum(r["held_record_count"] for r in result["rows"]))
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result["cpu_seconds"] = time.process_time() - start
        (ARTIFACTS / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result.get(k) for k in ("complete", "cpu_seconds", "controls", "failure")}))


if __name__ == "__main__":
    main()
