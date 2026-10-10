"""A38: family-information diagnostic from cached covariance predictions."""
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import mixture_forecasts

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_covariance_mixture_family_diagnostic_a38_20261003"


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_covariance_mixture_family_diagnostic_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    current, prior = read(spec["inputs"][0]), read(spec["inputs"][2])
    if not current["complete"] or not all(f["diagnostics"]["optimizer_converged"] for f in current["fits"]):
        raise ValueError("A38 requires the completed converged source fits")
    fits = {f["pair_id"]: f for f in current["fits"]}
    previous = {(p["seed"], p["regime"], p["path_policy"]): p for p in read(spec["inputs"][3])["rows"]}
    profiles = read(spec["inputs"][1])["rows"]
    rows, max_score_error, max_identity_error, converged_fold_error = [], 0., 0., 0.
    for profile in profiles:
        pair = profile["pair_id"]
        fitted = fits[pair]
        pi = np.array(fitted["diagnostics"]["class_weights"])
        high_class = int(np.argmax([g["high_frequency_variance_fraction"] for g in fitted["diagnostics"]["geometry"]]))
        known_class = high_class if profile["regime"] == "high_frequency_risk" else 1 - high_class
        e, p = np.array(profile["risk_log_evidence"]), np.array(profile["risk_log_density"])
        weights = np.r_[.25, .25, .5 * pi]
        l, posterior = mixture_forecasts(e[:, :4], p[:, :4], weights)
        lg, _ = mixture_forecasts(e[:, [0, 1, 4, 5]], p[:, [0, 1, 4, 5]], weights)
        i, _ = mixture_forecasts(e[:, :2], p[:, :2], [.5, .5])
        o, _ = mixture_forecasts(e[:, [0, 1, 2 + known_class]], p[:, [0, 1, 2 + known_class]], [.25, .25, .5])
        meta, _ = mixture_forecasts(e[:, 2:4], p[:, 2:4], pi)
        meta_mass = posterior[:, 2:4].sum(axis=1)
        correct_class_probability = posterior[:, 2 + known_class] / meta_mass
        accounting = np.logaddexp(np.log1p(-meta_mass), np.log(meta_mass) + meta - i)
        max_identity_error = max(max_identity_error, float(np.max(np.abs(accounting - (l - i)))))
        for name, score in {"L": l, "LG": lg, "I": i}.items():
            max_score_error = max(max_score_error, float(np.max(np.abs(score - profile["scores"][name]))))
        old = previous[profile["seed"], profile["regime"], profile["path_policy"]]
        old_converged = prior["fits"][pair]["diagnostics"]["optimizer_converged"]
        if old_converged:
            converged_fold_error = max(converged_fold_error, float(np.max(np.abs(l - old["scores"]["L"]))))
        bad = meta < i
        rows.append({"seed": profile["seed"], "regime": profile["regime"], "path_policy": profile["path_policy"],
            "pair_id": pair, "o_minus_i": float(np.mean(o - i)), "o_minus_l": float(np.mean(o - l)),
            "completion_l_change": float(np.mean(l - old["scores"]["L"])),
            "meta_mass_step9": float(meta_mass[0]), "meta_mass_step24": float(meta_mass[-1]),
            "correct_class_probability_step9": float(correct_class_probability[0]),
            "correct_class_probability_step24": float(correct_class_probability[-1]),
            "worse_meta_steps": int(bad.sum()), "mass_on_worse_meta_steps_sum": float(meta_mass[bad].sum())})
    if max_score_error > 1e-8 or max_identity_error > 1e-8 or converged_fold_error > 1e-8:
        raise ValueError("A38 cached forecasts, mixture accounting or unchanged converged fold differ")
    bootstrap = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(4, size=(20000, 4))
    groups = []
    for regime in ["all"] + sorted({r["regime"] for r in rows}):
        selected = rows if regime == "all" else [r for r in rows if r["regime"] == regime]
        count = sum(r["worse_meta_steps"] for r in selected)
        groups.append({"regime": regime, **{name: block_contrast(
            [np.mean([r[name] for r in selected if r["pair_id"] == j]) for j in range(4)], bootstrap)
            for name in ("o_minus_i", "o_minus_l", "completion_l_change")},
            **{name: float(np.mean([r[name] for r in selected])) for name in
                ("meta_mass_step9", "meta_mass_step24", "correct_class_probability_step9", "correct_class_probability_step24")},
            "fraction_steps_with_worse_meta_density": count / (16 * len(selected)),
            "mass_on_worse_meta_steps": sum(r["mass_on_worse_meta_steps_sum"] for r in selected) / count if count else None})
    optimization = [{"pair_id": f["pair_id"], "optimizer_iterations": f["diagnostics"]["optimizer_iterations"],
        "extra_training_likelihood_improvement": prior["fits"][f["pair_id"]]["diagnostics"]["source_negative_log_likelihood_per_projected_label"]
            - f["diagnostics"]["source_negative_log_likelihood_per_projected_label"]} for f in current["fits"]]
    result = {"stage": spec["stage"], "complete": True, "rows": rows, "groups": groups,
        "optimization_completion": optimization, "max_a37_forecast_error": max_score_error,
        "max_mixture_accounting_error": max_identity_error, "max_originally_converged_fold_error": converged_fold_error,
        "oracle_is_deployable": False, "new_gaussian_forecasts": 0, "new_covariance_fits": 0,
        "new_source_calls": 0, "new_target_calls": 0, "server_jobs": 0, "server_pulls": 0,
        "kg_calls": 0, "recommendations": 0, "evaluation_paths_scored": 0,
        "cpu_seconds": time.process_time() - start}
    (ARTIFACTS / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "max_a37_forecast_error", "max_mixture_accounting_error", "cpu_seconds")}))


if __name__ == "__main__":
    main()
