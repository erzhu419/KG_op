"""A42: source-only task direction distribution versus its moment collapse."""
import gzip
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.validate_meta_prior_heterogeneous_covariance import fit_tasks, saved_prior, scale_matched
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import mixture_forecasts, prequential_components
from representation.source_task_direction_prior import task_direction_distribution


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_task_direction_prior_plan.json")
    artifacts = ROOT / "paper_artifacts" / spec["stage"]
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A42 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "fits": [], "rows": [],
              "source_task_ard_fits": 0, "prequential_component_factorizations": 0,
              "new_source_calls": 0, "new_target_calls": 0, "server_jobs": 0, "server_pulls": 0,
              "kg_calls": 0, "recommendations": 0, "evaluation_paths_scored": 0}
    profiles = []
    try:
        tasks = read(spec["inputs"][0])["tasks"]
        cache = {(p["seed"], p["regime"], p["path_policy"]): p for p in read(spec["inputs"][1])["rows"]}
        if not read(spec["inputs"][2])["complete"]:
            raise ValueError("A42 requires completed A41 diagnosis")
        expected = {(s, r) for s in spec["cohort"]["meta_training_seeds"] for r in spec["cohort"]["regimes"]}
        if len(tasks) != 40 or {(t["seed"], t["regime"]) for t in tasks} != expected:
            raise ValueError("A42 frozen source cohort changed")
        reference = np.random.default_rng(51048).uniform(-2, 2, (384, 8))[:96]
        learned = {}
        for task in tasks:
            prior = fit_tasks([task], reference)
            learned[task["seed"], task["regime"]] = prior
            result["source_task_ard_fits"] += 1
            result["fits"].append({"seed": task["seed"], "regime": task["regime"], "prior": saved_prior(prior)})
        controls = {"max_original_i_score_error": 0., "max_same_scale_covariance_trace_error": 0.,
                    "max_collapsed_covariance_error": 0., "max_common_mean_error": 0.,
                    "max_nonintercept_source_mean": 0., "scored_seeds_in_source_prior": 0}
        for fold, held in enumerate(spec["cohort"]["outer_seed_pairs"]):
            training = [t for t in tasks if t["seed"] not in held]
            scored = [t for t in tasks if t["seed"] in held]
            controls["scored_seeds_in_source_prior"] += len(set(held) & {t["seed"] for t in training})
            source_priors = [learned[t["seed"], t["regime"]] for t in training]
            components, collapsed = task_direction_distribution(source_priors)
            generic = [scale_matched(p) for p in components]
            controls["max_collapsed_covariance_error"] = max(controls["max_collapsed_covariance_error"],
                float(np.max(np.abs(collapsed.covariance - np.mean([p.covariance for p in components], axis=0)))))
            for prior, control in zip(components, generic):
                controls["max_same_scale_covariance_trace_error"] = max(controls["max_same_scale_covariance_trace_error"],
                    abs(float(np.trace(prior.covariance) - np.trace(control.covariance))))
                controls["max_common_mean_error"] = max(controls["max_common_mean_error"],
                    float(np.max(np.abs(prior.mean - collapsed.mean))), float(np.max(np.abs(prior.mean - control.mean))))
                controls["max_nonintercept_source_mean"] = max(controls["max_nonintercept_source_mean"],
                    float(np.max(np.abs(prior.mean[1:]))))
            size = len(components)
            masses = np.r_[.25, .25, np.full(size, .5 / size)]
            for task in scored:
                seed, regime = task["seed"], task["regime"]
                bank = np.random.default_rng(51000 + seed).uniform(-2, 2, (384, 8))
                labels, noise = np.array(task["response"])[:, 2], np.array(task["noise"])[:, 2]
                for path in task["paths"]:
                    policy = path["path_policy"]
                    old = cache[seed, regime, policy]
                    if old["pair_id"] != fold:
                        raise ValueError("A42 source fold changed")
                    index = np.array(path["observation_indices"])
                    ids = np.array(task["candidates"])[index]
                    # Local atoms are unchanged cached evidence; factor only new atoms.
                    prediction = prequential_components(components + generic + [collapsed], bank, ids, labels[index], noise[index])
                    result["prequential_component_factorizations"] += 2 * size + 1
                    e = np.column_stack([np.array(old["risk_log_evidence"])[:, :2], prediction["log_evidence"][8:24]])
                    p = np.column_stack([np.array(old["risk_log_density"])[:, :2], prediction["log_density"][8:24]])
                    d, posterior = mixture_forecasts(e[:, :2 + size], p[:, :2 + size], masses)
                    columns = [0, 1] + list(range(2 + size, 2 + 2 * size))
                    dg, _ = mixture_forecasts(e[:, columns], p[:, columns], masses)
                    pooled, _ = mixture_forecasts(e[:, [0, 1, -1]], p[:, [0, 1, -1]], [.25, .25, .5])
                    baseline, _ = mixture_forecasts(e[:, :2], p[:, :2], [.5, .5])
                    controls["max_original_i_score_error"] = max(controls["max_original_i_score_error"],
                        float(np.max(np.abs(baseline - np.array(old["scores"]["I"])))))
                    delta = d - baseline
                    result["rows"].append({"seed": seed, "regime": regime, "path_policy": policy, "pair_id": fold,
                        "d_minus_i": float(delta.mean()), "d_minus_dg": float(np.mean(d - dg)),
                        "d_minus_p": float(np.mean(d - pooled)), "d_minus_l": float(np.mean(d - old["scores"]["L"])),
                        "early_d_minus_i": float(delta[:4].mean()), "late_d_minus_i": float(delta[-4:].mean()),
                        "meta_mass_step9": float(posterior[0, 2:].sum()), "meta_mass_step24": float(posterior[-1, 2:].sum())})
                    profiles.append({"seed": seed, "regime": regime, "path_policy": policy, "pair_id": fold,
                        "scores": {"D": d.tolist(), "DG": dg.tolist(), "P": pooled.tolist(), "I": baseline.tolist(), "L": old["scores"]["L"]},
                        "meta_mass": posterior[:, 2:].sum(axis=1).tolist()})
            print(json.dumps({"fold": fold, "eligible_source_tasks": size, "scored_paths": len(profiles),
                              "cpu_seconds": time.process_time() - start}), flush=True)
        result["controls"] = controls
        if max(controls.values()) > 1e-8:
            raise ValueError("A42 original paths, source exclusion or controlled moments changed")
        if (result["source_task_ard_fits"] > spec["limits"]["source_task_ard_fits_max"] or
                result["prequential_component_factorizations"] > spec["limits"]["prequential_component_factorizations_max"]):
            raise ValueError("A42 exceeds its frozen local operation budget")
        bootstrap = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(4, size=(20000, 4))
        metrics = ("d_minus_i", "d_minus_dg", "d_minus_p", "d_minus_l", "early_d_minus_i", "late_d_minus_i")
        groups = []
        for regime in ["all"] + spec["cohort"]["regimes"]:
            rows = result["rows"] if regime == "all" else [r for r in result["rows"] if r["regime"] == regime]
            groups.append({"regime": regime, **{metric: block_contrast(
                [np.mean([r[metric] for r in rows if r["pair_id"] == j]) for j in range(4)], bootstrap) for metric in metrics},
                **{metric: float(np.mean([r[metric] for r in rows])) for metric in ("meta_mass_step9", "meta_mass_step24")}})
        converged = all(p.diagnostics["optimizer_converged"] for p in learned.values())
        passed = (converged and all(groups[0][m]["paired_seed_bootstrap95"][0] > 0 for m in ("d_minus_i", "d_minus_dg", "d_minus_p"))
                  and all(g[m]["paired_seed_bootstrap95"][1] >= 0 for g in groups[1:] for m in ("d_minus_i", "d_minus_dg")))
        result.update(complete=True, groups=groups, source_gate_passed=passed, all_task_optimizers_converged=converged,
                      scored_training_paths=len(profiles), scored_training_latent_tasks=40,
                      nonconverged_tasks=[{"seed": s, "regime": r} for (s, r), p in learned.items() if not p.diagnostics["optimizer_converged"]])
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result["cpu_seconds"] = time.process_time() - start
        (artifacts / "forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": profiles}, allow_nan=False).encode()))
        (artifacts / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result.get(k) for k in ("complete", "source_gate_passed", "source_task_ard_fits", "cpu_seconds", "failure")}))


if __name__ == "__main__":
    main()
