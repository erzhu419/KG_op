"""A35: saved-fit accounting of heterogeneous risk prediction."""
import gzip
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
from representation.archive_covariance_prior import mixture_forecasts, prequential_components
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_heterogeneous_covariance_attribution_a35_20261003"


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_heterogeneous_covariance_attribution_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    fitted = read(spec["inputs"][0])
    if not fitted["complete"] or fitted["source_covariance_fits"] != 4:
        raise ValueError("A35 requires the unchanged four completed A34 fits")
    old = {(p["seed"], p["regime"], p["path_policy"]): p for p in read(spec["inputs"][1])["rows"]}
    tasks = read(spec["inputs"][2])["tasks"]
    seeds = {t["seed"] for t in tasks}
    contexts = {(c["seed"], c["regime"]): c for c in read(spec["inputs"][3])["rows"] if c["seed"] in seeds}
    reference = np.random.default_rng(51048).uniform(-2, 2, (384, 8))[:96]
    rows, cache, geometry, max_error, identity_error = [], [], [], 0., 0.
    for fold in fitted["fits"]:
        saved = fold["prior"]
        basis = FixedFunctionBasis(reference)
        learned = SourceFunctionPrior(basis, np.array(saved["mean"]), np.diag(saved["coefficient_variance"]),
                                      saved["discrepancy_variance"], saved["diagnostics"])
        trace = np.trace(learned.covariance[1:, 1:])
        generic = SourceFunctionPrior(basis, learned.mean.copy(), np.diag(np.r_[learned.covariance[0, 0],
                                      np.full(basis.feature_dim, trace / basis.feature_dim)]), learned.discrepancy_variance, {})
        variance = np.diag(learned.covariance)[1:]
        geometry.append({"pair_id": fold["pair_id"],
                         "effective_function_dimension": float(variance.sum() ** 2 / (variance @ variance)),
                         "high_frequency_variance_fraction": float(variance[60:].sum() / trace)})
        for task in tasks:
            seed, regime = task["seed"], task["regime"]
            if seed not in fold["held_seeds"]:
                continue
            bank = np.random.default_rng(51000 + seed).uniform(-2, 2, (384, 8))
            local_basis = FixedFunctionBasis(bank[:96])
            local = [SourceFunctionPrior(local_basis, np.r_[p["intercept_mean"], np.zeros(local_basis.feature_dim)],
                       np.diag(p["coefficient_variance"]), p["discrepancy_variance"], {})
                     for p in contexts[seed, regime]["eta_prior_components"]]
            for path in task["paths"]:
                index = np.array(path["observation_indices"])
                ids = np.array(task["candidates"])[index]
                risk = prequential_components(local + [learned, generic], bank, ids,
                         np.array(task["response"])[index, 2], np.array(task["noise"])[index, 2])
                e, p = risk["log_evidence"][8:24], risk["log_density"][8:24]
                q, q_weights = mixture_forecasts(e[:, :3], p[:, :3], [.25, .25, .5])
                g, g_weights = mixture_forecasts(e[:, [0, 1, 3]], p[:, [0, 1, 3]], [.25, .25, .5])
                i, _ = mixture_forecasts(e[:, :2], p[:, :2], [.5, .5])
                scores = {"Q": q, "G": g, "I": i, "M": p[:, 2], "B": p[:, 3]}
                previous = old[seed, regime, path["path_policy"]]
                for name, score in scores.items():
                    max_error = max(max_error, float(np.max(np.abs(score - previous["scores"][name]))))
                reconstructed = np.logaddexp(np.log1p(-q_weights[:, 2]), np.log(q_weights[:, 2]) + p[:, 2] - i)
                identity_error = max(identity_error, float(np.max(np.abs(reconstructed - (q - i)))))
                if max_error > 1e-8 or identity_error > 1e-8:
                    raise ValueError("A35 saved-fit forecasts or exact mixture accounting changed")
                harmful = p[:, 2] < i
                rows.append({"seed": seed, "regime": regime, "path_policy": path["path_policy"],
                    "pair_id": fold["pair_id"], "q_mass_step9": float(q_weights[0, 2]),
                    "q_mass_step24": float(q_weights[-1, 2]), "g_mass_step9": float(g_weights[0, 2]),
                    "g_mass_step24": float(g_weights[-1, 2]), "worse_m_steps": int(harmful.sum()),
                    "worse_m_mass_sum": float(q_weights[harmful, 2].sum()), "m_minus_b": float(np.mean(p[:, 2] - p[:, 3]))})
                cache.append({"seed": seed, "regime": regime, "path_policy": path["path_policy"],
                    "pair_id": fold["pair_id"], "risk_log_evidence": e.tolist(), "risk_log_density": p.tolist(),
                    "q_component_posterior": q_weights.tolist(), "g_component_posterior": g_weights.tolist()})
    bootstrap = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(4, size=(20000, 4))
    groups = []
    for regime in sorted({r["regime"] for r in rows}):
        selected = [r for r in rows if r["regime"] == regime]
        count = sum(r["worse_m_steps"] for r in selected)
        groups.append({"regime": regime,
            **{name: float(np.mean([r[name] for r in selected])) for name in ("q_mass_step9", "q_mass_step24", "g_mass_step9", "g_mass_step24")},
            "fraction_steps_with_worse_m_density": count / (16 * len(selected)),
            "q_mass_on_worse_m_steps": sum(r["worse_m_mass_sum"] for r in selected) / count if count else None,
            "m_minus_b": block_contrast([np.mean([r["m_minus_b"] for r in selected if r["pair_id"] == i]) for i in range(4)], bootstrap)})
    result = {"stage": spec["stage"], "complete": True, "rows": rows, "groups": groups,
        "geometry": geometry, "max_a34_forecast_error": max_error, "max_mixture_accounting_error": identity_error,
        "new_covariance_fits": 0, "new_source_calls": 0, "new_target_calls": 0,
        "server_jobs": 0, "server_pulls": 0, "kg_calls": 0, "recommendations": 0,
        "evaluation_paths_scored": 0, "cpu_seconds": time.process_time() - start}
    (ARTIFACTS / "component_forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"component_order": ["local_structural", "local_full", "heterogeneous_ard", "heterogeneous_scale_matched_full"], "rows": cache}, allow_nan=False).encode()))
    (ARTIFACTS / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "max_a34_forecast_error", "max_mixture_accounting_error", "cpu_seconds")}))


if __name__ == "__main__":
    main()
