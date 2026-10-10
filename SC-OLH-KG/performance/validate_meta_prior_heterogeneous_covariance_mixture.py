"""A36: fixed source-only latent covariance mixture validation."""
import gzip
import argparse
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
from performance.target_risk_residual import collapse
from performance.validate_meta_prior_heterogeneous_covariance import saved_prior, scale_matched
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import mixture_forecasts, prequential_components
from representation.source_risk_covariance_mixture import fit_source_risk_covariance_mixture
from representation.transferable_spectral import SourceDomainBatch

def training_batches(tasks):
    batches, variances = [], []
    for task in tasks:
        ids = np.array(task["candidates"])
        unique, labels, noise = collapse(ids, np.array(task["response"])[:, 2], np.array(task["noise"])[:, 2])
        first = [int(np.flatnonzero(ids == candidate)[0]) for candidate in unique]
        batches.append(SourceDomainBatch(str(len(batches)), np.array(task["psi"])[first], labels))
        variances.append(noise)
    return batches, variances


def fit_mixture(tasks, reference, initial_tau, max_iterations):
    batches, variances = training_batches(tasks)
    return fit_source_risk_covariance_mixture(batches, variances, reference, initial_tau, max_iterations=max_iterations)


def main(plan_path):
    start = time.process_time()
    spec = read(plan_path)
    artifacts = ROOT / "paper_artifacts" / spec["stage"]
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError(f"{spec['stage']} local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [], "fits": [],
              "new_source_calls": 0, "new_target_calls": 0, "server_jobs": 0,
              "server_pulls": 0, "kg_calls": 0, "recommendations": 0,
              "new_pooled_ard_fits": 0, "new_local_prior_fits": 0,
              "evaluation_paths_scored": 0, "source_mixture_fits": 0}
    profiles, max_i_error, max_q_error = [], 0., 0.
    try:
        tasks = read(spec["inputs"][0])["tasks"]
        pooled = read(spec["inputs"][1])
        attribution = read(spec["inputs"][2])
        if not pooled["complete"] or not attribution["complete"]:
            raise ValueError("A36 needs complete saved A34 fits and A35 forecasts")
        baseline = {(p["seed"], p["regime"], p["path_policy"]): p for p in read(spec["cached_baseline_components"])["rows"]}
        original = {(p["seed"], p["regime"], p["path_policy"]): p for p in read(spec["inputs"][4])["rows"]
                    if p["seed"] in spec["cohort"]["meta_training_seeds"]}
        expected = {(s, r) for s in spec["cohort"]["meta_training_seeds"] for r in spec["cohort"]["regimes"]}
        if len(tasks) != 40 or {(t["seed"], t["regime"]) for t in tasks} != expected:
            raise ValueError("A36 frozen meta-training cohort changed")
        reference = np.random.default_rng(51048).uniform(-2, 2, (384, 8))[:96]
        pooled_by_pair = {f["pair_id"]: f for f in pooled["fits"]}
        pooled_metrics = {(r["seed"], r["regime"], r["path_policy"]): r for r in pooled["rows"]}
        for fold, held in enumerate(spec["cohort"]["outer_seed_pairs"]):
            fold_start = time.process_time()
            training = [t for t in tasks if t["seed"] not in held]
            scored = [t for t in tasks if t["seed"] in held]
            prior_fit = pooled_by_pair[fold]
            if (len(training) != 30 or len(scored) != 10 or prior_fit["held_seeds"] != held
                    or prior_fit["training_seeds"] != sorted({t["seed"] for t in training})):
                raise ValueError("A36 held seeds entered mixture fitting or its initialization")
            initial_tau = prior_fit["prior"]["diagnostics"]["source_function_variance"]
            learned, class_weights, diagnostics = fit_mixture(training, reference, initial_tau, spec["optimizer_max_iterations"])
            result["source_mixture_fits"] += 1
            generic = [scale_matched(p) for p in learned]
            training_responsibilities = np.array(diagnostics["task_class_responsibilities"])
            diagnostics["training_class_audit"] = [{"regime": r, "mean_responsibilities":
                training_responsibilities[[t["regime"] == r for t in training]].mean(axis=0).tolist()}
                for r in spec["cohort"]["regimes"]]
            result["fits"].append({"pair_id": fold, "held_seeds": held,
                "training_seeds": sorted({t["seed"] for t in training}), "training_tasks": len(training),
                "priors": [saved_prior(p) for p in learned], "diagnostics": diagnostics})
            weights = np.r_[.25, .25, .5 * class_weights]
            for task in scored:
                seed, regime = task["seed"], task["regime"]
                bank = np.random.default_rng(51000 + seed).uniform(-2, 2, (384, 8))
                labels, noise = np.array(task["response"])[:, 2], np.array(task["noise"])[:, 2]
                for path in task["paths"]:
                    policy = path["path_policy"]
                    index = np.array(path["observation_indices"])
                    candidates = np.array(task["candidates"])[index]
                    cached = baseline[seed, regime, policy]
                    if cached["pair_id"] != fold:
                        raise ValueError("A36 cached source fold changed")
                    old_e, old_p = np.array(cached["risk_log_evidence"]), np.array(cached["risk_log_density"])
                    risk = prequential_components(learned + generic, bank, candidates, labels[index], noise[index])
                    e = np.column_stack([old_e[:, :2], risk["log_evidence"][8:24]])
                    p = np.column_stack([old_p[:, :2], risk["log_density"][8:24]])
                    l, post = mixture_forecasts(e[:, :4], p[:, :4], weights)
                    lg, generic_post = mixture_forecasts(e[:, [0, 1, 4, 5]], p[:, [0, 1, 4, 5]], weights)
                    i, _ = mixture_forecasts(old_e[:, :2], old_p[:, :2], [.5, .5])
                    q, _ = mixture_forecasts(old_e[:, :3], old_p[:, :3], [.25, .25, .5])
                    # A35 was verified against A34; reconstruct it again from cached atoms.
                    cached_q = np.logaddexp(np.log1p(-np.array(cached["q_component_posterior"])[:, 2]) + i,
                                           np.log(np.array(cached["q_component_posterior"])[:, 2]) + old_p[:, 2])
                    max_q_error = max(max_q_error, float(np.max(np.abs(q - cached_q))))
                    max_q_error = max(max_q_error, abs(float(np.mean(q - i)) - pooled_metrics[seed, regime, policy]["q_minus_i"]))
                    max_i_error = max(max_i_error, float(np.max(np.abs(i - original[seed, regime, policy]["scores"]["I"]))))
                    if max_i_error > 1e-8 or max_q_error > 1e-8:
                        raise ValueError("A36 original I or pooled Q forecast changed")
                    result["rows"].append({"seed": seed, "regime": regime, "path_policy": policy, "pair_id": fold,
                        "l_minus_i": float(np.mean(l - i)), "l_minus_lg": float(np.mean(l - lg)),
                        "l_minus_q": float(np.mean(l - q)), "early_l_minus_i": float(np.mean((l - i)[:4])),
                        "late_l_minus_i": float(np.mean((l - i)[-4:])),
                        "step9_component_weights": post[0].tolist(), "step24_component_weights": post[-1].tolist()})
                    profiles.append({"seed": seed, "regime": regime, "path_policy": policy, "pair_id": fold,
                        "scores": {"L": l.tolist(), "LG": lg.tolist(), "I": i.tolist(), "Q": q.tolist()},
                        "risk_log_evidence": e.tolist(), "risk_log_density": p.tolist(),
                        "component_posterior": post.tolist(), "generic_component_posterior": generic_post.tolist()})
            elapsed = time.process_time() - fold_start
            print(json.dumps({"fold": fold, "cpu_seconds": elapsed,
                "converged": diagnostics["optimizer_converged"], "iterations": diagnostics["optimizer_iterations"],
                "class_weights": diagnostics["class_weights"]}), flush=True)
            if fold == 0:
                projection = time.process_time() - start + elapsed * 4 * 1.25 + 5
                result["admission"] = {"projected_cpu_seconds": projection, "cap_cpu_seconds": cap, "admitted": projection < cap}
                if projection >= cap:
                    raise TimeoutError("A36 projected mixture fits exceed frozen CPU cap")
        bootstrap = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(4, size=(20000, 4))
        metrics = ("l_minus_i", "l_minus_lg", "l_minus_q", "early_l_minus_i", "late_l_minus_i")
        groups = []
        for regime in ["all"] + spec["cohort"]["regimes"]:
            selected = result["rows"] if regime == "all" else [r for r in result["rows"] if r["regime"] == regime]
            groups.append({"regime": regime, **{name: block_contrast(
                [np.mean([r[name] for r in selected if r["pair_id"] == j]) for j in range(4)], bootstrap) for name in metrics},
                "step9_component_weights": np.mean([r["step9_component_weights"] for r in selected], axis=0).tolist(),
                "step24_component_weights": np.mean([r["step24_component_weights"] for r in selected], axis=0).tolist()})
        converged = all(f["diagnostics"]["optimizer_converged"] for f in result["fits"])
        passed = (converged and all(groups[0][n]["paired_seed_bootstrap95"][0] > 0 for n in ("l_minus_i", "l_minus_lg"))
            and groups[0]["l_minus_q"]["mean_gain"] > 0
            and all(g[n]["paired_seed_bootstrap95"][1] >= 0 for g in groups[1:] for n in ("l_minus_i", "l_minus_lg")))
        result.update(groups=groups, source_gate_passed=passed, max_original_i_forecast_error=max_i_error,
                      max_a34_q_forecast_error=max_q_error, scored_training_paths=len(profiles),
                      scored_training_latent_tasks=40, component_order=["local_structural", "local_full", "class1", "class2"])
        if passed:
            initial_tau = np.mean([f["prior"]["diagnostics"]["source_function_variance"] for f in pooled["fits"]], axis=0)
            final, class_weights, diagnostics = fit_mixture(tasks, reference, initial_tau, spec["optimizer_max_iterations"])
            result["source_mixture_fits"] += 1
            result.update(final_priors=[saved_prior(p) for p in final], final_class_weights=class_weights.tolist(),
                          final_diagnostics=diagnostics, source_gate_passed=bool(diagnostics["optimizer_converged"]))
        result["complete"] = True
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result["cpu_seconds"] = time.process_time() - start
        (artifacts / "forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"component_order":
            ["local_structural", "local_full", "class1", "class2", "class1_isotropic", "class2_isotropic"],
            "rows": profiles}, allow_nan=False).encode()))
        (artifacts / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result.get(k) for k in ("complete", "source_gate_passed", "source_mixture_fits", "cpu_seconds", "failure")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="performance/meta_prior_heterogeneous_covariance_mixture_plan.json")
    main(parser.parse_args().plan)
