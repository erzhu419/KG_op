"""Run the frozen A45 acquisition diagnostic on saved eight-observation states."""
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.verification_kg import posterior_covariances
from performance.risk_fantasy_consistency import gaussian_scores, mixture_moments, probe_scores
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from representation.transfer_mixture import TransferMixtureGPR


def restore_state(task, context, protocol):
    psi = np.random.default_rng(51000 + task["seed"]).uniform(-2, 2, (384, 8))
    reference = psi[:96]
    basis = FixedFunctionBasis(reference)
    priors = [SourceFunctionPrior(basis,
        np.r_[p["intercept_mean"], np.zeros(68)], np.diag(p["coefficient_variance"]),
        p["discrepancy_variance"], {}) for p in context["eta_prior_components"]]
    risk = TransferMixtureGPR(priors, psi, reference)
    generic = SourceFunctionPrior.fit([], [], reference, "target_only")
    heads = [generic.target_model(psi) for _ in range(2)]
    past = task["paths"][0]["observation_indices"][:8]
    ids = [task["candidates"][i] for i in past]
    if any([task["candidates"][i] for i in p["observation_indices"][:8]] != ids for p in task["paths"]):
        raise ValueError("A45 requires identical initial designs on both historical paths")
    if not np.array_equal(psi[ids], [task["psi"][i] for i in past]):
        raise ValueError("A45 public candidate coordinates changed")
    for row, candidate in zip(past, ids):
        response, noise = task["response"][row], task["noise"][row]
        for h, model in enumerate(heads):
            model.update([candidate], response[h], noise[h])
        risk.update([candidate], response[2], noise[2])
    candidates = np.arange(384)[:, None]
    means = np.column_stack([m.posterior_mean_many(candidates) for m in risk.models])
    covariance = np.moveaxis(posterior_covariances(risk.models, candidates), -1, 0)
    weights = np.exp(risk.log_weights)
    fixed_mean = np.column_stack([m.posterior_mean_many(candidates) for m in heads] + [np.zeros(384)])
    fixed_variance = np.column_stack([m.posterior_var_many(candidates) for m in heads] + [np.zeros(384)])
    noise = float(polygamma(1, (protocol["replicates"] - 1) / 2))
    _, moment_covariance = mixture_moments(means, covariance, weights)
    diagonal = np.diagonal(covariance, axis1=1, axis2=2).T
    disagreement = np.abs(means[:, 0] - means[:, 1]) / np.sqrt(diagonal.sum(axis=1) + 2 * noise)
    requested = [task["candidates"][p["observation_indices"][8]] for p in task["paths"]]
    requested += [int(np.argmax(np.diag(moment_covariance))), int(np.argmax(disagreement)), 0, 127, 255, 383]
    probes = list(dict.fromkeys(requested))
    return means, covariance, weights, probes, noise, fixed_mean, fixed_variance


def summarize(rows, regimes):
    groups = []
    for regime in regimes:
        selected = [row for row in rows if row["regime"] == regime]
        if not selected:
            continue
        groups.append({"regime": regime, "states": len(selected),
            "probe_count": sum(len(row["probes"]) for row in selected),
            "B9_F9_ranking_changes": sum(row["best_probe"]["B9"] != row["best_probe"]["F9"] for row in selected),
            "B9_G9_ranking_changes": sum(row["best_probe"]["B9"] != row["best_probe"]["G9"] for row in selected),
            "G9_G3_ranking_changes": sum(row["best_probe"]["G9"] != row["best_probe"]["G3"] for row in selected),
            **{name: {"mean_state_max_absolute": float(np.mean([row["differences"][name]["max_absolute"] for row in selected])),
                      "max_absolute": max(row["differences"][name]["max_absolute"] for row in selected)}
               for name in ("B9-F9", "F9-G9", "G9-G3")},
            "B9_reference_probe_gaps": reference_probe_gaps(selected)})
    return groups


def reference_probe_gaps(rows):
    """Value of each approximation's best probe under the same B9 reference.

These are within-probe numerical gaps, not simulator or deployment regret.
"""
    result = {}
    for arm in ("F9", "G9", "G3"):
        gaps = [max(row["scores"]["B9"]) - row["scores"]["B9"][
            row["probes"].index(row["best_probe"][arm])] for row in rows]
        result[arm] = {"mean": float(np.mean(gaps)), "max": float(np.max(gaps)),
            "best_probe_changes": sum(row["best_probe"]["B9"] != row["best_probe"][arm] for row in rows)}
    return result


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_risk_fantasy_consistency_plan.json")
    artifacts = ROOT / "paper_artifacts" / spec["stage"]
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A45 frozen CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [], "higher_order_controls": [],
        "new_source_calls": 0, "new_target_calls": 0, "server_jobs": 0, "server_pulls": 0,
        "live_policy_kg_calls": 0, "live_recommendations": 0, "evaluation_paths_scored": 0,
        "posterior_states": 0, "distinct_sampling_probes": 0, "primary_probe_scores": 0,
        "higher_order_probe_scores": 0, "frozen_risk_model": "original_local_I"}
    regimes = []
    try:
        tasks = read(spec["inputs"][0])["tasks"]
        contexts = {(row["seed"], row["regime"]): row for row in read(spec["inputs"][1])["rows"]}
        frozen = read(spec["inputs"][2])["base_protocol"]
        protocol = {**frozen, **frozen["allocations"]["n24_v928"]}
        regimes = frozen["regimes"]
        if len(tasks) != 40 or {(t["seed"], t["regime"]) for t in tasks} != {(s, r) for s in range(48, 56) for r in regimes}:
            raise ValueError("A45 historical cohort changed")
        if not read(spec["inputs"][3])["complete"]:
            raise ValueError("A45 requires completed A44")
        for task in tasks:
            task_start = time.process_time()
            state = restore_state(task, contexts[task["seed"], task["regime"]], protocol)
            means, covariance, weights, probes, noise, fmean, fvariance = state
            scores9 = probe_scores(*state, protocol, 9)
            collapsed_mean, collapsed_covariance = mixture_moments(means, covariance, weights)
            scores3 = gaussian_scores(collapsed_mean, collapsed_covariance, probes, noise,
                                       fmean, fvariance, protocol, 3)
            scores = {"B9": scores9["B"], "F9": scores9["F"], "G9": scores9["G"], "G3": scores3["G"]}
            differences = {}
            for left, right in (("B9", "F9"), ("F9", "G9"), ("G9", "G3")):
                delta = scores[left] - scores[right]
                differences[f"{left}-{right}"] = {"values": delta.tolist(), "max_absolute": float(np.max(np.abs(delta)))}
            row = {"seed": task["seed"], "regime": task["regime"], "probes": probes,
                "component_weights": weights.tolist(), "scores": {k: v.tolist() for k, v in scores.items()},
                "mixture_current_max": scores9["current_max"], "gaussian_current_max": scores9["G_current_max"],
                "expected_future_max": {"B9": scores9["B_future_max"].tolist(), "F9": scores9["F_future_max"].tolist(),
                    "G9": scores9["G_future_max"].tolist(), "G3": scores3["future_max"].tolist()},
                "best_probe": {k: probes[int(np.argmax(v))] for k, v in scores.items()}, "differences": differences}
            result["rows"].append(row)
            result["posterior_states"] += 1
            result["distinct_sampling_probes"] += len(probes)
            result["primary_probe_scores"] += 4 * len(probes)
            primary_cpu = time.process_time() - task_start
            higher_cpu = 0.
            if task["seed"] == 48:
                high_start = time.process_time()
                scores13 = probe_scores(*state, protocol, 13)
                errors = {k: float(np.max(np.abs(scores13[k] - scores9[k]))) for k in ("B", "F")}
                result["higher_order_controls"].append({"seed": task["seed"], "regime": task["regime"],
                    "probes": probes, "errors_13_minus_9": errors,
                    "differences_13_minus_9": {k: (scores13[k] - scores9[k]).tolist() for k in ("B", "F")},
                    "passed_0_001": max(errors.values()) <= .001})
                result["higher_order_probe_scores"] += 2 * len(probes)
                higher_cpu = time.process_time() - high_start
            if len(result["rows"]) == 1:
                projection = time.process_time() - start + 1.25 * (39 * primary_cpu + 4 * higher_cpu) + 5.
                result["admission"] = {"first_primary_cpu_seconds": primary_cpu,
                    "first_higher_control_cpu_seconds": higher_cpu, "projected_with_margin_cpu_seconds": projection,
                    "passed": projection <= cap}
                print(json.dumps({"admission": result["admission"]}), flush=True)
                if projection > cap:
                    raise TimeoutError("A45 projected cost exceeds frozen cap; partial diagnostic retained")
            print(json.dumps({"state": result["posterior_states"], "seed": task["seed"], "regime": task["regime"]}), flush=True)
        result["complete"] = True
    except TimeoutError as error:
        result["stop_reason"] = str(error)
    result["groups"] = summarize(result["rows"], regimes)
    result["overall_B9_reference_probe_gaps"] = reference_probe_gaps(result["rows"]) if result["rows"] else {}
    result["cpu_seconds"] = time.process_time() - start
    accuracy = bool(result["complete"] and len(result["higher_order_controls"]) == 5
                    and all(c["passed_0_001"] for c in result["higher_order_controls"]))
    decision = {"stage": spec["stage"], "complete": result["complete"],
        "integration_accuracy_resolved": accuracy, "promote_live_acquisition": False,
        "new_decision_performance_established": False,
        "finding": "Risk-only numerical diagnostic; compare probability refresh, moment approximation and integration separately.",
        "next_step": "Close A-series diagnosis and freeze one structural-meta-prior method and decisive closed-loop protocol."}
    (artifacts / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (artifacts / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({"complete": result["complete"], "cpu_seconds": result["cpu_seconds"],
                      "integration_accuracy_resolved": accuracy, "artifacts": str(artifacts)}), flush=True)


if __name__ == "__main__":
    main()
