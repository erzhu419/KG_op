"""A34: covariance learning on the separated heterogeneous meta archive."""
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
from performance.target_risk_residual import collapse
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import mixture_forecasts, prequential_components
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from representation.source_risk_ard import fit_source_risk_ard
from representation.transferable_spectral import SourceDomainBatch

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_heterogeneous_covariance_a34_20261003"


def fit_tasks(tasks, reference):
    batches, variances = [], []
    for task in tasks:
        ids = np.array(task["candidates"])
        unique, labels, noise = collapse(ids, np.array(task["response"])[:, 2], np.array(task["noise"])[:, 2])
        psi = np.array(task["psi"])
        first = [int(np.flatnonzero(ids == candidate)[0]) for candidate in unique]
        batches.append(SourceDomainBatch(str((task["seed"], task["regime"])), psi[first], labels))
        variances.append(noise)
    return fit_source_risk_ard(batches, variances, reference)


def scale_matched(prior):
    covariance = np.diag(np.r_[prior.covariance[0, 0],
                              np.full(prior.basis.feature_dim, np.trace(prior.covariance[1:, 1:]) / prior.basis.feature_dim)])
    return SourceFunctionPrior(prior.basis, prior.mean.copy(), covariance, prior.discrepancy_variance, {})


def saved_prior(prior):
    return {"mean": prior.mean.tolist(), "coefficient_variance": np.diag(prior.covariance).tolist(),
            "discrepancy_variance": prior.discrepancy_variance,
            "basis_mean": prior.basis.mean.tolist(), "basis_scale": prior.basis.scale.tolist(),
            "diagnostics": prior.diagnostics}


def summarize(rows, bootstrap):
    result = []
    for regime in ["all"] + sorted({r["regime"] for r in rows}):
        selected = rows if regime == "all" else [r for r in rows if r["regime"] == regime]
        metrics = ("q_minus_i", "q_minus_g", "m_minus_i", "b_minus_i", "early_q_minus_i", "late_q_minus_i")
        result.append({"regime": regime, **{name: block_contrast(
            [np.mean([r[name] for r in selected if r["pair_id"] == i]) for i in range(4)], bootstrap)
            for name in metrics}})
    return result


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_heterogeneous_covariance_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A34 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [], "fits": [],
              "new_source_calls": 0, "new_target_calls": 0, "server_jobs": 0,
              "server_pulls": 0, "kg_calls": 0, "recommendations": 0,
              "held_development_evaluation_paths_scored": 0}
    profiles, max_error = [], 0.
    try:
        if not read(spec["inputs"][1])["complete"]:
            raise ValueError("A34 needs the completed jointly observed A33 archive")
        tasks = read(spec["inputs"][0])["tasks"]
        contexts = {(c["seed"], c["regime"]): c for c in read(spec["inputs"][2])["rows"]
                    if c["seed"] in spec["cohort"]["meta_training_seeds"]}
        previous = {(p["seed"], p["regime"], p["path_policy"]): p
                    for p in read(spec["inputs"][3])["rows"]
                    if p["seed"] in spec["cohort"]["meta_training_seeds"]}
        protocol = read(spec["inputs"][4])["base_protocol"]
        reference = np.random.default_rng(51048).uniform(-2, 2, (384, 8))[:96]
        expected = {(s, r) for s in spec["cohort"]["meta_training_seeds"] for r in spec["cohort"]["regimes"]}
        if len(tasks) != 40 or {(t["seed"], t["regime"]) for t in tasks} != expected:
            raise ValueError("A34 meta-training task split changed")
        for fold, held in enumerate(spec["cohort"]["outer_seed_pairs"]):
            fold_start = time.process_time()
            training = [t for t in tasks if t["seed"] not in held]
            scored = [t for t in tasks if t["seed"] in held]
            if len(training) != 30 or len(scored) != 10 or {t["seed"] for t in training} & set(held):
                raise ValueError("A34 held source task entered covariance learning")
            learned = fit_tasks(training, reference)
            generic = scale_matched(learned)
            if not (np.all(learned.mean[1:] == 0) and np.isclose(np.trace(learned.covariance), np.trace(generic.covariance))):
                raise ValueError("A34 learned means or scale-matched control changed")
            result["fits"].append({"pair_id": fold, "held_seeds": held,
                "training_seeds": sorted({t["seed"] for t in training}),
                "training_tasks": len(training), "prior": saved_prior(learned),
                "high_frequency_variance_fraction": float(np.diag(learned.covariance)[61:].sum() / np.trace(learned.covariance[1:, 1:]))})
            for task in scored:
                seed, regime = task["seed"], task["regime"]
                labels, noise = np.array(task["response"])[:, 2], np.array(task["noise"])[:, 2]
                bank = np.random.default_rng(51000 + seed).uniform(-2, 2, (protocol["candidates"], protocol["psi_dim"]))
                basis = FixedFunctionBasis(bank[:protocol["reference_bank_size"]])
                local = [SourceFunctionPrior(basis, np.r_[p["intercept_mean"], np.zeros(basis.feature_dim)],
                         np.diag(p["coefficient_variance"]), p["discrepancy_variance"], {})
                         for p in contexts[seed, regime]["eta_prior_components"]]
                for path in task["paths"]:
                    policy = path["path_policy"]
                    ids = np.array(path["observation_indices"])
                    candidates = np.array(task["candidates"])[ids]
                    # Candidate identities share discrepancy across distinct visit records.
                    forecast = prequential_components(local + [learned, generic], bank, candidates, labels[ids], noise[ids])
                    e, p = forecast["log_evidence"][8:24], forecast["log_density"][8:24]
                    q, _ = mixture_forecasts(e[:, :3], p[:, :3], [.25, .25, .5])
                    g, _ = mixture_forecasts(e[:, [0, 1, 3]], p[:, [0, 1, 3]], [.25, .25, .5])
                    i, _ = mixture_forecasts(e[:, :2], p[:, :2], [.5, .5])
                    max_error = max(max_error, float(np.max(np.abs(i - previous[seed, regime, policy]["scores"]["I"]))))
                    if max_error > 1e-8:
                        raise ValueError("A34 source path chronology or original local prior changed")
                    result["rows"].append({"seed": seed, "regime": regime, "path_policy": policy, "pair_id": fold,
                        "q_minus_i": float(np.mean(q - i)), "q_minus_g": float(np.mean(q - g)),
                        "m_minus_i": float(np.mean(p[:, 2] - i)), "b_minus_i": float(np.mean(p[:, 3] - i)),
                        "early_q_minus_i": float(np.mean((q - i)[:4])), "late_q_minus_i": float(np.mean((q - i)[-4:]))})
                    profiles.append({"seed": seed, "regime": regime, "path_policy": policy, "pair_id": fold,
                        "scores": {"Q": q.tolist(), "G": g.tolist(), "I": i.tolist(), "M": p[:, 2].tolist(), "B": p[:, 3].tolist()}})
            elapsed = time.process_time() - fold_start
            print(json.dumps({"fold": fold, "cpu_seconds": elapsed,
                              "converged": learned.diagnostics["optimizer_converged"]}), flush=True)
            if fold == 0:
                projection = time.process_time() - start + elapsed * 4 * 1.25 + 5
                result["admission"] = {"projected_cpu_seconds": projection, "cap_cpu_seconds": cap, "admitted": projection < cap}
                if projection >= cap:
                    raise TimeoutError("A34 projected fits exceed frozen CPU cap")
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(4, size=(20000, 4))
        groups = summarize(result["rows"], indices)
        converged = all(f["prior"]["diagnostics"]["optimizer_converged"] for f in result["fits"])
        overall = groups[0]
        passed = (converged and all(overall[n]["paired_seed_bootstrap95"][0] > 0 for n in ("q_minus_i", "q_minus_g"))
                  and all(g[n]["paired_seed_bootstrap95"][1] >= 0 for g in groups[1:] for n in ("q_minus_i", "q_minus_g")))
        result.update(groups=groups, source_gate_passed=passed, max_original_i_forecast_error=max_error,
                      scored_training_paths=len(profiles), scored_training_latent_tasks=40,
                      source_covariance_fits=len(result["fits"]))
        if passed:
            final = fit_tasks(tasks, reference)
            result.update(final_prior=saved_prior(final), source_covariance_fits=len(result["fits"]) + 1,
                          final_optimizer_converged=final.diagnostics["optimizer_converged"])
            result["source_gate_passed"] = bool(final.diagnostics["optimizer_converged"])
        result["complete"] = True
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result["cpu_seconds"] = time.process_time() - start
        (ARTIFACTS / "forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": profiles}, allow_nan=False).encode()))
        (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result.get(k) for k in ("complete", "source_gate_passed", "source_covariance_fits", "cpu_seconds", "failure")}), flush=True)


if __name__ == "__main__":
    main()
