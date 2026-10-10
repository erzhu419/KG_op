"""A43: source coefficient moments retaining task directions and uncertainty."""
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
from performance.validate_meta_prior_heterogeneous_covariance import scale_matched
from performance.validate_meta_prior_heterogeneous_covariance_mixture import training_batches
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import mixture_forecasts, prequential_components
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from representation.source_risk_covariance_mixture import projected_source_blocks
from representation.source_task_direction_prior import source_coefficient_moments, task_direction_distribution


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_task_direction_moment_plan.json")
    artifacts = ROOT / "paper_artifacts" / spec["stage"]
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A43 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "fits": [], "rows": [],
              "source_coefficient_conditioning_solves": 0, "prequential_component_factorizations": 0,
              "new_covariance_optimizers": 0, "new_source_calls": 0, "new_target_calls": 0,
              "server_jobs": 0, "server_pulls": 0, "kg_calls": 0, "recommendations": 0, "evaluation_paths_scored": 0}
    profiles = []
    try:
        tasks = read(spec["inputs"][0])["tasks"]
        a37 = read(spec["inputs"][1])
        fits = {f["pair_id"]: f for f in a37["fits"]}
        key = lambda p: (p["seed"], p["regime"], p["path_policy"])
        cache = {key(p): p for p in read(spec["inputs"][2])["rows"]}
        previous = {key(p): p for p in read(spec["inputs"][3])["rows"]}
        a42 = read(str(Path(spec["inputs"][3]).with_name("validation.json")))
        old_rows = {key(p): p for p in a42["rows"]}
        if not (a37["complete"] and a42["complete"] and read(spec["inputs"][4])["complete"]):
            raise ValueError("A43 requires complete retained A37/A42 results")
        expected = {(s, r) for s in spec["cohort"]["meta_training_seeds"] for r in spec["cohort"]["regimes"]}
        if len(tasks) != 40 or {(t["seed"], t["regime"]) for t in tasks} != expected:
            raise ValueError("A43 source cohort changed")
        basis = FixedFunctionBasis(np.random.default_rng(51048).uniform(-2, 2, (384, 8))[:96])
        controls = {"max_original_i_score_error": 0., "max_cached_l_score_error": 0., "max_cached_d_contrast_error": 0.,
                    "max_basis_normalization_error": 0., "max_same_scale_trace_error": 0., "max_collapsed_covariance_error": 0.,
                    "max_common_mean_error": 0., "max_nonintercept_source_mean": 0., "scored_seeds_in_source_prior": 0}
        for fold, held in enumerate(spec["cohort"]["outer_seed_pairs"]):
            training = [t for t in tasks if t["seed"] not in held]
            scored = [t for t in tasks if t["seed"] in held]
            saved = fits[fold]
            if (saved["held_seeds"] != held or saved["training_seeds"] != sorted({t["seed"] for t in training})
                    or len(training) != 30 or not saved["diagnostics"]["optimizer_converged"]):
                raise ValueError("A43 source hyperfit includes scored seeds or did not converge")
            controls["scored_seeds_in_source_prior"] += len(set(held) & {t["seed"] for t in training})
            source_priors = [SourceFunctionPrior(basis, np.array(p["mean"]), np.diag(p["coefficient_variance"]),
                             p["discrepancy_variance"], {}) for p in saved["priors"]]
            for p in saved["priors"]:
                controls["max_basis_normalization_error"] = max(controls["max_basis_normalization_error"],
                    float(np.max(np.abs(basis.mean - p["basis_mean"]))), float(np.max(np.abs(basis.scale - p["basis_scale"]))))
            pi = np.array(saved["diagnostics"]["class_weights"])
            batches, variances = training_batches(training)
            blocks, _ = projected_source_blocks(batches, variances, basis)
            moments, records = [], []
            for task, (design, response, noise) in zip(training, blocks):
                posterior = source_coefficient_moments(source_priors, pi, design, response, noise)
                result["source_coefficient_conditioning_solves"] += len(source_priors)
                mean = np.r_[source_priors[0].mean[0], np.zeros(basis.feature_dim)]
                moment = SourceFunctionPrior(basis, mean, np.diag(np.r_[.25, posterior["second_moment"]]), .02, {})
                moments.append(moment)
                weights = posterior["class_posterior"]
                records.append({"seed": task["seed"], "regime": task["regime"], "unique_candidates": len(response) + 1,
                    "coefficient_variance": np.diag(moment.covariance).tolist(), "class_posterior": weights.tolist(),
                    "source_posterior_variance_mass": float(weights @ posterior["posterior_diagonal_variance"].sum(axis=1)),
                    "source_posterior_mean_square_mass": float(weights @ (posterior["posterior_mean"] ** 2).sum(axis=1))})
            components, collapsed = task_direction_distribution(moments)
            generic = [scale_matched(p) for p in components]
            result["fits"].append({"pair_id": fold, "held_seeds": held, "training_seeds": saved["training_seeds"],
                "mean": collapsed.mean.tolist(), "basis_mean": basis.mean.tolist(), "basis_scale": basis.scale.tolist(),
                "discrepancy_variance": .02, "source_task_moments": records})
            controls["max_collapsed_covariance_error"] = max(controls["max_collapsed_covariance_error"],
                float(np.max(np.abs(collapsed.covariance - np.mean([p.covariance for p in components], axis=0)))))
            for prior, control in zip(components, generic):
                controls["max_same_scale_trace_error"] = max(controls["max_same_scale_trace_error"],
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
                    old, old_d, old_row = cache[seed, regime, policy], previous[seed, regime, policy], old_rows[seed, regime, policy]
                    if old["pair_id"] != fold or old_d["pair_id"] != fold:
                        raise ValueError("A43 cached validation fold changed")
                    index = np.array(path["observation_indices"])
                    ids = np.array(task["candidates"])[index]
                    prediction = prequential_components(components + generic + [collapsed], bank, ids, labels[index], noise[index])
                    result["prequential_component_factorizations"] += 2 * size + 1
                    old_e, old_p = np.array(old["risk_log_evidence"]), np.array(old["risk_log_density"])
                    e = np.column_stack([old_e[:, :2], prediction["log_evidence"][8:24]])
                    p = np.column_stack([old_p[:, :2], prediction["log_density"][8:24]])
                    u, posterior = mixture_forecasts(e[:, :2 + size], p[:, :2 + size], masses)
                    columns = [0, 1] + list(range(2 + size, 2 + 2 * size))
                    ug, _ = mixture_forecasts(e[:, columns], p[:, columns], masses)
                    pooled, _ = mixture_forecasts(e[:, [0, 1, -1]], p[:, [0, 1, -1]], [.25, .25, .5])
                    baseline, _ = mixture_forecasts(e[:, :2], p[:, :2], [.5, .5])
                    old_l, _ = mixture_forecasts(old_e[:, :4], old_p[:, :4], np.r_[.25, .25, .5 * pi])
                    d = np.array(old_d["scores"]["D"])
                    controls["max_original_i_score_error"] = max(controls["max_original_i_score_error"],
                        float(np.max(np.abs(baseline - old["scores"]["I"]))), float(np.max(np.abs(baseline - old_d["scores"]["I"]))))
                    controls["max_cached_l_score_error"] = max(controls["max_cached_l_score_error"],
                        float(np.max(np.abs(old_l - old["scores"]["L"]))), float(np.max(np.abs(old_l - old_d["scores"]["L"]))))
                    cached_deltas = {"d_minus_i": d - baseline, "d_minus_l": d - old_l,
                                     "early_d_minus_i": (d - baseline)[:4], "late_d_minus_i": (d - baseline)[-4:]}
                    controls["max_cached_d_contrast_error"] = max(controls["max_cached_d_contrast_error"],
                        max(abs(float(delta.mean()) - old_row[name]) for name, delta in cached_deltas.items()))
                    delta = u - baseline
                    result["rows"].append({"seed": seed, "regime": regime, "path_policy": policy, "pair_id": fold,
                        "u_minus_i": float(delta.mean()), "u_minus_ug": float(np.mean(u - ug)), "u_minus_p": float(np.mean(u - pooled)),
                        "u_minus_d": float(np.mean(u - d)), "u_minus_l": float(np.mean(u - old_l)),
                        "early_u_minus_i": float(delta[:4].mean()), "late_u_minus_i": float(delta[-4:].mean()),
                        "meta_mass_step9": float(posterior[0, 2:].sum()), "meta_mass_step24": float(posterior[-1, 2:].sum())})
                    profiles.append({"seed": seed, "regime": regime, "path_policy": policy, "pair_id": fold,
                        "scores": {"U": u.tolist(), "UG": ug.tolist(), "P": pooled.tolist(), "I": baseline.tolist(), "L": old_l.tolist(), "D": d.tolist()},
                        "meta_mass": posterior[:, 2:].sum(axis=1).tolist()})
            print(json.dumps({"fold": fold, "scored_paths": len(profiles), "cpu_seconds": time.process_time() - start}), flush=True)
        result["controls"] = controls
        if max(controls.values()) > 1e-8:
            raise ValueError("A43 original predictions, source exclusion or moment controls changed")
        for name in ("source_coefficient_conditioning_solves", "prequential_component_factorizations"):
            if result[name] > spec["limits"][name + "_max"]:
                raise ValueError("A43 exceeds its frozen operation budget")
        bootstrap = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(4, size=(20000, 4))
        metrics = ("u_minus_i", "u_minus_ug", "u_minus_p", "u_minus_d", "u_minus_l", "early_u_minus_i", "late_u_minus_i")
        groups = []
        for regime in ["all"] + spec["cohort"]["regimes"]:
            rows = result["rows"] if regime == "all" else [r for r in result["rows"] if r["regime"] == regime]
            groups.append({"regime": regime, **{metric: block_contrast(
                [np.mean([r[metric] for r in rows if r["pair_id"] == j]) for j in range(4)], bootstrap) for metric in metrics},
                **{metric: float(np.mean([r[metric] for r in rows])) for metric in ("meta_mass_step9", "meta_mass_step24")}})
        passed = (all(groups[0][m]["paired_seed_bootstrap95"][0] > 0 for m in ("u_minus_i", "u_minus_ug", "u_minus_p"))
                  and all(g[m]["paired_seed_bootstrap95"][1] >= 0 for g in groups[1:] for m in ("u_minus_i", "u_minus_ug")))
        result.update(complete=True, groups=groups, source_gate_passed=passed,
                      scored_training_paths=len(profiles), scored_training_latent_tasks=40)
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result["cpu_seconds"] = time.process_time() - start
        (artifacts / "forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": profiles}, allow_nan=False).encode()))
        (artifacts / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result.get(k) for k in ("complete", "source_gate_passed", "cpu_seconds", "controls", "failure")}))


if __name__ == "__main__":
    main()
