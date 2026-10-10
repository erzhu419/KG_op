"""A2: isolate label likelihood and g/eta coupling at already-paid sites."""
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
from core.verification_kg import predictive_pass_probability
from performance.local_risk_likelihood import local_update
from performance.replay_meta_prior_calibration import restore_context
from performance.run_meta_prior_mechanism import observe
from performance.run_meta_prior_total_budget import simulator_supports
from representation.transfer_mixture import TransferMixtureGPR

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_local_likelihood_a2_20261003"


def components(model, x):
    members = model.models if isinstance(model, TransferMixtureGPR) else [model]
    weights = np.exp(model.log_weights) if isinstance(model, TransferMixtureGPR) else [1.]
    return [(float(w), float(m.posterior_mean(x)), float(m.posterior_var(x)))
            for w, m in zip(weights, members)]


def probe(row, index, step, kind, response, noise, models, context, protocol, spec):
    priors = [components(m, [index]) for m in models]
    inferred = {name: local_update(priors[0], priors[1], response, noise, protocol, *definition)
                for name, definition in spec["variants"].items()}
    # Algorithm inputs end here. Truth/support below are post-inference diagnostics.
    truth = context[4][index]
    true_power = float(predictive_pass_probability(truth, np.zeros(3), protocol))
    for result in inferred.values():
        result.update(g_error=result["g_mean"] - float(truth[1]),
                      eta_error=result["eta_mean"] - float(truth[2]),
                      power_error=result["predicted_verifier_power"] - true_power)
    support = simulator_supports(row["seed"], row["regime"], protocol["psi_dim"])[2]
    structural, full = context[3][2], context[2][2]
    risk_variances = [{"function": structural.basis.names[i],
        "source_structural_variance": float(structural.covariance[i + 1, i + 1]),
        "generic_variance": float(full.covariance[i + 1, i + 1])} for i in support]
    return {**{k: row[k] for k in ("seed", "regime", "arm")}, "kind": kind,
        "candidate": index, "step": step + 1, "response": response.tolist(),
        "noise": noise.tolist(), "truth": truth.tolist(), "true_verifier_power": true_power,
        "g_components_before": priors[0], "eta_components_before": priors[1],
        "eta_gaussian_gain_by_component": [c[2] / (c[2] + noise[2]) for c in priors[1]],
        "target_function_prior_variances": risk_variances, "variants": inferred}


def summarize(sites, spec):
    groups = []
    original = [r for r in sites if r["kind"] == "original_winner"]
    for regime in sorted({r["regime"] for r in original}):
        for arm in ("fixed_source", "risk_ard_mixture"):
            rr = [r for r in original if r["regime"] == regime and r["arm"] == arm]
            group = {"regime": regime, "arm": arm, "sites": len(rr), "variants": {}}
            for name in spec["variants"]:
                values = [r["variants"][name] for r in rr]
                group["variants"][name] = {
                    "mean_abs_eta_error": float(np.mean([abs(v["eta_error"]) for v in values])),
                    "mean_abs_g_error": float(np.mean([abs(v["g_error"]) for v in values])),
                    "mean_abs_power_error": float(np.mean([abs(v["power_error"]) for v in values])),
                    "mean_g_eta_correlation": float(np.mean([v["g_eta_correlation"] for v in values])),
                    "mean_eta_shift_from_legacy": float(np.mean([v["eta_mean"] - r["variants"]["legacy"]["eta_mean"]
                                                                for v, r in zip(values, rr)]))}
            groups.append(group)
    return groups


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    spec = json.loads((ROOT / "performance/meta_prior_local_likelihood_protocol.json").read_text())
    inputs = ROOT / spec["input_artifacts"]
    execution = json.loads((inputs / "execution_plan.json").read_text())
    frozen = execution["base_protocol"]
    protocol = {**frozen, **frozen["allocations"]["n24_v928"]}
    records = json.loads(gzip.decompress((inputs / "inputs.json.gz").read_bytes()))["records"]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    cap = int(spec["resource_limits"]["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A2 reached its local CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    result = {"stage": spec["stage"], "complete": False, "paths": len(records), "sites": [],
        "unobserved_recommendations": [], "scope": spec["scope"],
        "server_experiment_jobs": 0, "new_server_pulls": 0, "kg_calls": 0,
        "new_searches": 0, "new_random_verifications": 0}
    contexts, max_legacy_error = {}, 0.0
    try:
        for count, record in enumerate(records, 1):
            row = record["row"]
            key = row["seed"], row["regime"]
            if key not in contexts:
                learned = next(r for r in records if (r["row"]["seed"], r["row"]["regime"], r["row"]["arm"]) ==
                               (*key, "risk_ard_mixture"))
                contexts[key] = restore_context(learned, protocol)
            context = contexts[key]
            psi, reference, full, structural, truth, target_seed = context
            models = ([TransferMixtureGPR([structural[h], full[h]], psi, reference) for h in (1, 2)]
                      if row["arm"] == "risk_ard_mixture" else [full[h].target_model(psi) for h in (1, 2)])
            wanted = {}
            if row["recommendation"] in row["queries"]:
                last = max(i for i, x in enumerate(row["queries"]) if x == row["recommendation"])
                wanted[last] = "original_winner"
            else:
                result["unobserved_recommendations"].append({k: row[k] for k in ("seed", "regime", "arm", "recommendation")})
            anchor = spec["additional_anchor"]
            if (row["seed"], row["regime"], row["arm"]) == (anchor["seed"], anchor["regime"], anchor["arm"]):
                last = max(i for i, x in enumerate(row["queries"]) if x == anchor["candidate"])
                wanted[last] = "posthoc_regression_anchor"
            visits = np.zeros(len(psi), dtype=int)
            for step, index in enumerate(row["queries"]):
                response, noise = observe(truth[index], target_seed, index, int(visits[index]), protocol["replicates"])
                diagnostic = (probe(row, index, step, wanted[step], response, noise, models,
                                    context, protocol, spec) if step in wanted else None)
                for head, model in zip((1, 2), models):
                    model.update([index], response[head], noise[head])
                visits[index] += 1
                if diagnostic is not None:
                    legacy = diagnostic["variants"]["legacy"]
                    old_moments = [models[0].posterior_mean_many([[index]])[0],
                                   models[0].posterior_var_many([[index]])[0],
                                   models[1].posterior_mean_many([[index]])[0],
                                   models[1].posterior_var_many([[index]])[0]]
                    new_moments = [legacy[k] for k in ("g_mean", "g_variance", "eta_mean", "eta_variance")]
                    np.testing.assert_allclose(new_moments, old_moments, atol=1e-8, rtol=1e-8)
                    max_legacy_error = max(max_legacy_error, float(np.max(np.abs(np.array(new_moments) - old_moments))))
                    m = np.array([0., old_moments[0], old_moments[2]])
                    v = np.array([0., old_moments[1], old_moments[3]])
                    diagnostic["legacy_moment_matched_power"] = float(predictive_pass_probability(m, v, protocol))
                    result["sites"].append(diagnostic)
            if count % 20 == 0:
                print(f"PROBED {count}/{len(records)} paths, {len(result['sites'])} paid sites", flush=True)
        result.update(groups=summarize(result["sites"], spec), complete=True,
                      max_legacy_moment_error=max_legacy_error,
                      max_integration_relative_error=max(v["integration_relative_error"] for r in result["sites"]
                                                         for v in r["variants"].values()))
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (ARTIFACTS / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "max_legacy_moment_error", "max_integration_relative_error",
                                            "cpu_seconds", "wall_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()
