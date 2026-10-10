"""A44: exact cached mixture accounting and source-context identifiability."""
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.validate_meta_prior_scale_preserving_source_covariance import read


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_source_context_accounting_plan.json")
    artifacts = ROOT / "paper_artifacts" / spec["stage"]
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [],
        "new_covariance_fits": 0, "new_gaussian_forecasts": 0, "new_source_calls": 0,
        "new_target_calls": 0, "server_jobs": 0, "server_pulls": 0, "kg_calls": 0,
        "recommendations": 0, "evaluation_paths_scored": 0}
    try:
        validation = read(spec["inputs"][0])
        profiles = read(spec["inputs"][1])["rows"]
        if not validation["complete"] or len(profiles) != 80:
            raise ValueError("A44 requires completed A43 source-validation paths")
        contexts = {(c["seed"], c["regime"]): c for c in read(spec["inputs"][2])["rows"]}
        tasks = {(t["seed"], t["regime"]): t for t in read(spec["inputs"][3])["tasks"]}
        regimes = ["shared_structure", "support_mismatch", "high_frequency_risk", "separate_risk_support"]
        context_rows = []
        for seed in range(48, 56):
            first = contexts[seed, regimes[0]]
            designs = {}
            for regime in regimes:
                task = tasks[seed, regime]
                designs[regime] = {p["path_policy"]: [task["candidates"][i] for i in p["observation_indices"][:8]]
                                   for p in task["paths"]}
            context_rows.append({"seed": seed,
                "four_source_eta_contexts_equal": all(contexts[seed, r]["eta_prior_components"] == first["eta_prior_components"] for r in regimes),
                "four_source_fg_selections_equal": all(contexts[seed, r]["fg_selected_indices"] == first["fg_selected_indices"] for r in regimes),
                "four_initial_designs_equal": all(designs[r] == designs[regimes[0]] for r in regimes),
                "separate_head_eta_context_differs": contexts[seed, "separate_head_structure"]["eta_prior_components"] != first["eta_prior_components"]})
        controls = {"max_next_past_mass_error": 0., "max_cumulative_score_identity_error": 0.,
                    "max_density_identity_error": 0., "lower_bound_violations": 0}
        for profile in profiles:
            delta = np.array(profile["scores"]["U"]) - np.array(profile["scores"]["I"])
            mass = np.array(profile["meta_mass"])
            ratio = 1. + np.expm1(delta) / mass
            if np.any(ratio <= 0):
                raise ValueError("A44 cached scores cannot resolve a positive empirical predictive density")
            empirical_gain = np.log(ratio)
            updated = 1. - (1. - mass) * np.exp(-delta)
            controls["max_next_past_mass_error"] = max(controls["max_next_past_mass_error"],
                float(np.max(np.abs(updated[:-1] - mass[1:]))))
            cumulative = float(np.log1p(-mass[0]) - np.log1p(-updated[-1]))
            controls["max_cumulative_score_identity_error"] = max(controls["max_cumulative_score_identity_error"],
                abs(float(delta.sum()) - cumulative))
            reconstructed = np.logaddexp(np.log1p(-mass), np.log(mass) + empirical_gain)
            controls["max_density_identity_error"] = max(controls["max_density_identity_error"],
                float(np.max(np.abs(delta - reconstructed))))
            bound = float(np.log1p(-mass[0]))
            controls["lower_bound_violations"] += int(cumulative < bound - 1e-10)
            negative = empirical_gain < 0
            result["rows"].append({"seed": profile["seed"], "regime": profile["regime"],
                "path_policy": profile["path_policy"], "pair_id": profile["pair_id"],
                "u_minus_i": float(delta.mean()), "empirical_minus_i": float(empirical_gain.mean()),
                "early_empirical_minus_i": float(empirical_gain[:4].mean()), "late_empirical_minus_i": float(empirical_gain[-4:].mean()),
                "meta_mass_step9": float(mass[0]), "meta_mass_step24": float(mass[-1]), "meta_mass_after24": float(updated[-1]),
                "mean_score_lower_bound": bound / len(delta), "worse_empirical_steps": int(negative.sum()),
                "mass_on_worse_empirical_steps_sum": float(mass[negative].sum())})
        if max(controls.values()) > 1e-8:
            raise ValueError("A44 cached Bayesian weight or density accounting failed")
        groups = []
        for regime in ["all"] + regimes + ["separate_head_structure"]:
            rows = result["rows"] if regime == "all" else [r for r in result["rows"] if r["regime"] == regime]
            count = sum(r["worse_empirical_steps"] for r in rows)
            groups.append({"regime": regime, **{m: float(np.mean([r[m] for r in rows])) for m in
                ("u_minus_i", "empirical_minus_i", "early_empirical_minus_i", "late_empirical_minus_i",
                 "meta_mass_step9", "meta_mass_step24", "meta_mass_after24", "mean_score_lower_bound")},
                "worse_empirical_step_fraction": count / (16 * len(rows)),
                "mean_mass_on_worse_empirical_steps": sum(r["mass_on_worse_empirical_steps_sum"] for r in rows) / count if count else None})
        result.update(complete=True, controls=controls, groups=groups, source_contexts=context_rows, scored_paths=len(profiles))
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result["cpu_seconds"] = time.process_time() - start
        (artifacts / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result.get(k) for k in ("complete", "cpu_seconds", "controls", "source_contexts", "failure")}))


if __name__ == "__main__":
    main()
