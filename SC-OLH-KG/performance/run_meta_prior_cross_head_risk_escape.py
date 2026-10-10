"""A15 source calibration and fixed-path validation, using saved evidence only."""
import argparse
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
from performance.cross_head_risk_escape import (
    assemble_profile, concatenate_profiles, fit_source_prior, macro_prior,
    mixture_scores, posterior_macro_masses,
)
from performance.source_calibrated_adaptation import fit_source_calibration

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_cross_head_risk_escape_a15_20261003"
PLAN = ROOT / "performance/meta_prior_cross_head_risk_escape_plan.json"


def read(path):
    path = ROOT / path
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def save(name, data):
    (ARTIFACTS / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def indexed(rows, fields, expected):
    result = {tuple(row[f] for f in fields): row for row in rows}
    if len(result) != len(rows) or set(result) != expected:
        raise ValueError(f"saved cohort changed for {fields}")
    if any([f["step"] for f in row["forecasts"]] != list(range(9, 25)) for row in rows):
        raise ValueError("chronological forecast steps changed")
    return result


def contrast(gains, indices):
    gains = np.asarray(gains)
    return {"mean_gain": float(gains.mean()),
            "paired_seed_bootstrap95": np.quantile(gains[indices].mean(axis=1), [.025, .975]).tolist(),
            "better": int(np.sum(gains > 1e-12)), "worse": int(np.sum(gains < -1e-12)),
            "seed_gains": gains.tolist()}


def source_phase(spec, result):
    cohort = spec["cohort"]
    seed, groups, arms = cohort["seeds"], cohort["source_archive_groups"], cohort["priors"]
    heads = read(spec["source_inputs"][0])
    risk = read(spec["source_inputs"][1])
    source14 = read("paper_artifacts/meta_prior_cross_head_structure_mixture_a14_20261003/source_summary.json")
    if not risk["complete"] or not source14["source_complete"]:
        raise ValueError("source artifacts are incomplete")
    hh = indexed(heads["rows"], ("seed", "archive_group", "held_task"),
                 {(s, g, h) for s in seed for g in groups for h in range(8)})
    rr = indexed(risk["rows"], ("seed", "archive_group", "held_task", "arm"),
                 {(s, g, h, a) for s in seed for g in groups for h in range(8) for a in arms})
    # First retained benchmark is the learned prior, whose two heads differ.
    keys = [(s, g, a) for s in seed for g in groups for a in reversed(arms)]
    fits, alternatives, folds = [], {"32": [], "96": []}, []
    result.update(calibrations=fits, alternative_calibrations=alternatives, folds=folds)
    control_error = generic_q_error = generic_alpha_error = 0.
    numerical = {"32": 0., "96": 0.}
    for i, (s, g, a) in enumerate(keys):
        started = time.process_time()
        profiles = [assemble_profile(hh[s, g, h]["forecasts"], rr[s, g, h, a]["forecasts"],
                                     a, "null_posterior") for h in range(8)]
        evidence, predictive = concatenate_profiles(profiles)
        fit = {"seed": s, "archive_group": g, "arm": a,
               **fit_source_prior(evidence, predictive, 64)}
        fits.append(fit)
        if i == 0:
            first = time.process_time() - started
            # 32^2 + 64^2 + 96^2 = 3.5 times the primary tensor work.
            projected = result["startup_cpu_seconds"] + 1.25 * first * len(keys) * 3.5 + 30.
            result["admission"] = {"retained_first_archive_cpu_seconds": first,
                "projected_stage_cpu_seconds": projected, "cap_cpu_seconds": 900.,
                "safety_multiplier": 1.25, "target_cpu_reserve": 30., "admitted": projected < 900.}
            print(json.dumps({"admission": result["admission"]}), flush=True)
            if projected >= 900.:
                raise TimeoutError("first archive projects above the complete-stage CPU cap")
        scores = mixture_scores(evidence, predictive, fit["macro_prior_mass"])
        for n in (32, 96):
            other = {"seed": s, "archive_group": g, "arm": a,
                     **fit_source_prior(evidence, predictive, n)}
            alternatives[str(n)].append(other)
            alternative_scores = mixture_scores(evidence, predictive, other["macro_prior_mass"])
            numerical[str(n)] = max(numerical[str(n)], float(np.max(np.abs(scores - alternative_scores))))
        control_error = max(control_error, *(p["max_independent_control_error"] for p in profiles))
        if a == "fixed_source":
            old = fit_source_calibration([f for h in range(8) for f in rr[s, g, h, a]["forecasts"]], 64)
            generic_q_error = max(generic_q_error, abs(fit["q_mean"] - old["q_mean"]))
            generic_alpha_error = max(generic_alpha_error, abs(fit["alpha_mean"] - .5),
                                      abs(fit["alpha_sd"] - np.sqrt(1. / 12.)),
                                      abs(fit["alpha_q_covariance"]))
        for h in range(8):
            part = scores[16 * h:16 * (h + 1)]
            folds.append({"seed": s, "archive_group": g, "arm": a, "held_task": h,
                          "mean_gain": float(np.mean(part - profiles[h]["log_predictive"][:, 0]))})
        if (i + 1) % 8 == 0:
            print(json.dumps({"source_calibrations_completed": i + 1}), flush=True)
    indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(len(seed), size=(20000, len(seed)))
    summaries = []
    for g in groups:
        for a in arms:
            gains = [np.mean([r["mean_gain"] for r in folds
                              if (r["seed"], r["archive_group"], r["arm"]) == (s, g, a)]) for s in seed]
            ff = [f for f in fits if (f["archive_group"], f["arm"]) == (g, a)]
            summaries.append({"archive_group": g, "arm": a, "archives": len(seed), **contrast(gains, indices),
                "mean_alpha": float(np.mean([f["alpha_mean"] for f in ff])),
                "mean_q": float(np.mean([f["q_mean"] for f in ff])),
                "mean_macro_prior_mass": np.mean([f["macro_prior_mass"] for f in ff], axis=0).tolist(),
                "mean_alpha_q_correlation": float(np.mean([f["alpha_q_correlation"] for f in ff]))})
    passed = max(numerical.values()) <= 1e-4 and max(generic_q_error, generic_alpha_error) <= 1e-8
    result.update(groups=summaries, source_prior_profiles=len(folds), source_forecasts=16 * len(folds),
        max_independent_control_error=control_error, max_generic_q_control_error=generic_q_error,
        max_generic_flat_alpha_error=generic_alpha_error,
        numerical_check={"orders": [32, 64, 96], "tolerance": 1e-4,
                         "max_source_score_change": numerical, "passed": passed})
    if not passed:
        raise ValueError("source calibration quadrature/control check failed")
    result["complete"] = True


def target_phase(spec, result, source):
    gate = read(ARTIFACTS / "source_gate.json")
    if not source["complete"] or not gate["proceed_to_fixed_target_validation"]:
        raise ValueError("source gate does not allow target validation")
    cohort = spec["cohort"]
    expected = {(s, r, a) for s in cohort["seeds"] for r in cohort["target_regimes"] for a in cohort["priors"]}
    heads, risk = [read(p) for p in spec["target_inputs"]]
    old12 = read("paper_artifacts/meta_prior_source_calibrated_adaptation_a12_20261003/validation.json")
    if not all(d["complete"] for d in (heads, risk, old12)):
        raise ValueError("target artifacts are incomplete")
    fields = ("seed", "regime", "arm")
    hh, rr, aa = [indexed(d["rows"], fields, expected) for d in (heads, risk, old12)]
    fits = {(f["seed"], f["archive_group"], f["arm"]): f for f in source["calibrations"]}
    alt = {n: {(f["seed"], f["archive_group"], f["arm"]): f for f in fs}
           for n, fs in source["alternative_calibrations"].items()}
    rows = []
    result["rows"] = rows
    control_i = control14 = control12 = generic12 = 0.
    numerical = {n: 0. for n in alt}
    for s in cohort["seeds"]:
        for r in cohort["target_regimes"]:
            g = "separate_heads" if r == "separate_head_structure" else "shared"
            for a in cohort["priors"]:
                key = s, r, a
                h, risk_row, old = hh[key], rr[key], aa[key]
                if [(f["step"], f["candidate"]) for f in old["forecasts"]] != [
                        (f["step"], f["candidate"]) for f in h["forecasts"]]:
                    raise ValueError("A12 secondary control identities differ")
                p = assemble_profile(h["forecasts"], risk_row["forecasts"], a, "a9_null_posterior")
                fit = fits[s, g, a]
                scores = mixture_scores(p["log_evidence"], p["log_predictive"], fit["macro_prior_mass"])
                posterior = posterior_macro_masses(p["log_evidence"], fit["macro_prior_mass"])
                old14 = np.array([f["joint_log_score"] for f in h["forecasts"]])
                old12scores = np.array([f["calibrated_log_score"] for f in old["forecasts"]])
                reconstructed14 = mixture_scores(p["log_evidence"], p["log_predictive"], macro_prior(h["alpha_source"], 0.))
                control14 = max(control14, float(np.max(np.abs(reconstructed14 - old14))))
                reconstructed12 = mixture_scores(p["log_evidence"], p["log_predictive"], macro_prior(0., old["forecasts"][0]["q_source"]))
                control12 = max(control12, float(np.max(np.abs(reconstructed12 - old12scores))))
                control_i = max(control_i, p["max_independent_control_error"])
                if a == "fixed_source":
                    generic12 = max(generic12, float(np.max(np.abs(scores - old12scores))))
                for n in alt:
                    other = mixture_scores(p["log_evidence"], p["log_predictive"], alt[n][s, g, a]["macro_prior_mass"])
                    numerical[n] = max(numerical[n], float(np.max(np.abs(scores - other))))
                forecasts = [{"step": step, "candidate": candidate,
                    "component_log_density": p["log_predictive"][i].tolist(),
                    "relative_past_log_evidence": p["log_evidence"][i].tolist(),
                    "posterior_macro_mass": posterior[i].tolist(), "log_score": float(scores[i]),
                    "a14_log_score": float(old14[i]), "a12_log_score": float(old12scores[i])}
                    for i, (step, candidate) in enumerate(p["identity"])]
                rows.append({"seed": s, "regime": r, "arm": a, "forecasts": forecasts,
                    "source_macro_prior_mass": fit["macro_prior_mass"],
                    "mean_gain": float(np.mean(scores - p["log_predictive"][:, 0])),
                    "mean_gain_vs_a14": float(np.mean(scores - old14)),
                    "mean_gain_vs_a12": float(np.mean(scores - old12scores))})
    indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(16, size=(20000, 16))
    groups = []
    for r in cohort["target_regimes"]:
        for a in cohort["priors"]:
            selected = [row for row in rows if (row["regime"], row["arm"]) == (r, a)]
            groups.append({"regime": r, "arm": a, "paths": len(selected),
                **contrast([row["mean_gain"] for row in selected], indices),
                "versus_a14": contrast([row["mean_gain_vs_a14"] for row in selected], indices),
                "versus_a12": contrast([row["mean_gain_vs_a12"] for row in selected], indices),
                "mean_posterior_macro_mass": np.mean([f["posterior_macro_mass"] for row in selected
                                                       for f in row["forecasts"]], axis=0).tolist(),
                "mean_step9_macro_mass": np.mean([row["forecasts"][0]["posterior_macro_mass"] for row in selected], axis=0).tolist(),
                "mean_step24_macro_mass": np.mean([row["forecasts"][-1]["posterior_macro_mass"] for row in selected], axis=0).tolist()})
    passed = max(numerical.values()) <= 1e-4 and max(control_i, control14, control12, generic12) <= 1e-8
    result.update(groups=groups, target_forecasts=16 * len(rows),
        controls={"max_independent_score_error": control_i, "max_a14_no_escape_error": control14,
                  "max_a12_no_coordination_error": control12, "max_generic_a12_error": generic12},
        numerical_check={"orders": [32, 64, 96], "tolerance": 1e-4,
                         "max_target_score_change": numerical, "passed": passed})
    if not passed:
        raise ValueError("target quadrature/control check failed")
    result["complete"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("source", "target"))
    phase = parser.parse_args().phase
    cpu_start, wall_start = time.process_time(), time.monotonic()
    spec = read(PLAN)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    source = read(ARTIFACTS / "source_summary.json") if phase == "target" else None
    spent = source["cpu_seconds"] if source else 0.
    cap = int(spec["limits"]["local_cpu_hours_cap"] * 3600 - spent)
    if cap <= 0:
        raise TimeoutError("no remaining local CPU budget")
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A15 reached its complete-stage local CPU budget")
    signal.signal(signal.SIGXCPU, stop)
    result = {"stage": spec["stage"], "phase": phase, "complete": False,
        "startup_cpu_seconds": cpu_start,
        "scope": (spec["source_gate_scope"] if phase == "source" else
                  "Retrospective fixed-path development validation; source prior frozen before all target scores."),
        **{k: 0 for k in spec["limits"] if k not in ("local_threads", "local_cpu_hours_cap")}}
    save("execution_plan.json", spec)
    try:
        if phase == "source":
            source_phase(spec, result)
        else:
            target_phase(spec, result, source)
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start,
            wall_seconds=time.monotonic() - wall_start,
            current_process_vm_hwm_mb=float(next(line.split()[1] for line in
                Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))) / 1024,
            raw_ru_maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        save("source_summary.json" if phase == "source" else "target_validation.json", result)
        print(json.dumps({k: result[k] for k in ("phase", "complete", "cpu_seconds", "wall_seconds",
                                               "current_process_vm_hwm_mb")}), flush=True)


if __name__ == "__main__":
    main()
