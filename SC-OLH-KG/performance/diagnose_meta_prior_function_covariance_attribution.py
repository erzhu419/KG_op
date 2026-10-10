"""A18 source-prediction attribution using only retained A17 densities/evidence."""
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
from performance.function_covariance_attribution import decompose
from performance.run_meta_prior_cross_head_risk_escape import contrast

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_function_covariance_attribution_a18_20261003"
METRICS = ("fg_effect", "q_effect", "f_accounting_effect", "g_accounting_effect",
           "pure_f_gain", "pure_g_gain", "pure_r_gain", "same_q_ir_gain")


def main():
    start, wall = time.process_time(), time.monotonic()
    spec = json.loads((ROOT / "performance/meta_prior_function_covariance_attribution_plan.json").read_text())
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A18 source-attribution CPU budget reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "scope": spec["scope"],
              "rows": [], **spec["limits"]}
    try:
        with gzip.open(ROOT / spec["inputs"][0], "rt") as stream:
            forecasts = json.load(stream)["rows"]
        old = json.loads((ROOT / spec["inputs"][1]).read_text())
        if not old["complete"] or not old["numerical_check"]["passed"]:
            raise ValueError("A18 requires completed numerically verified A17 evidence")
        seeds = sorted({f["held_seed"] for f in old["calibrations"]})
        groups, arms = ("shared", "separate_heads"), ("fixed_source", "risk_ard_mixture")
        fields = ("seed", "archive_group", "held_task", "arm")
        expected = {(s, g, h, a) for s in seeds for g in groups for h in range(8) for a in arms}
        previous = {tuple(r[k] for k in fields): r for r in old["rows"]}
        current = {tuple(r[k] for k in fields): r for r in forecasts}
        priors = {(f["held_seed"], f["arm"]): f["macro_prior_mass"] for f in old["calibrations"]}
        if len(seeds) != 16 or len(expected) != 512 or len(previous) != len(old["rows"]) or len(current) != len(forecasts) or set(previous) != expected or set(current) != expected:
            raise ValueError("A18 source-profile cohort changed")
        if len(priors) != len(old["calibrations"]) or set(priors) != {(s, a) for s in seeds for a in arms}:
            raise ValueError("A18 macro-prior cohort changed")
        error = total_error = head_error = generic_error = 0.
        for key in sorted(expected):
            data = current[key]["scores_by_rho"]["32"]
            evidence, predictive = np.array(data["relative_log_evidence"]), np.array(data["component_log_density"])
            if evidence.shape != (16, 4) or predictive.shape != (16, 4):
                raise ValueError("A18 requires all 16 fixed chronological forecasts")
            original = previous[key]
            baseline = np.array(original["same_pool_ir_scores"])
            values = decompose(evidence, predictive, priors[key[0], key[3]], baseline)
            actual = np.array(original["log_scores"])
            error = max(error, float(np.max(np.abs(values["log_score"] - actual))))
            total_error = max(total_error, float(np.max(np.abs(actual - baseline - values["fg_effect"] - values["q_effect"]))))
            head_error = max(head_error, float(np.max(np.abs(values["fg_effect"] - values["f_accounting_effect"] - values["g_accounting_effect"]))))
            if key[3] == "fixed_source":
                generic_error = max(generic_error, float(np.max(np.abs(values["fg_effect"]))))
            weights = values["posterior_macro_mass"]
            result["rows"].append({**dict(zip(fields, key)),
                **{name: float(np.mean(values[name])) for name in METRICS},
                "prior_macro_mass": priors[key[0], key[3]],
                "mean_posterior_macro_mass": weights.mean(axis=0).tolist(),
                "step9_posterior_macro_mass": weights[0].tolist(),
                "step24_posterior_macro_mass": weights[-1].tolist(),
                "step9_f_g_r_log_evidence_ratio": evidence[0, 1:].tolist(),
                "step24_f_g_r_log_evidence_ratio": evidence[-1, 1:].tolist(),
                "mean_r_mass_effect": float(np.mean(weights[:, 3] - values["same_q_ir_posterior_macro_mass"][:, 3]))})
        result["controls"] = {"max_a17_score_error": error, "max_total_decomposition_error": total_error,
            "max_two_head_decomposition_error": head_error, "max_generic_fg_effect": generic_error}
        if max(result["controls"].values()) > 1e-10:
            raise ValueError("A18 frozen-score reproduction or accounting identity failed")
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(16, size=(20000, 16))
        summaries = []
        for g in groups:
            for a in arms:
                archives = []
                for s in seeds:
                    held = [r for r in result["rows"] if (r["seed"], r["archive_group"], r["arm"]) == (s, g, a)]
                    archives.append({"seed": s, **{name: float(np.mean([r[name] for r in held])) for name in METRICS},
                        **{name: np.mean([r[name] for r in held], axis=0).tolist() for name in
                           ("mean_posterior_macro_mass", "step9_posterior_macro_mass", "step24_posterior_macro_mass",
                            "step9_f_g_r_log_evidence_ratio", "step24_f_g_r_log_evidence_ratio")}})
                summaries.append({"archive_group": g, "arm": a, "seed_archives": 16,
                    **{name: contrast([r[name] for r in archives], indices) for name in METRICS},
                    **{name: np.mean([r[name] for r in archives], axis=0).tolist() for name in
                       ("mean_posterior_macro_mass", "step9_posterior_macro_mass", "step24_posterior_macro_mass",
                        "step9_f_g_r_log_evidence_ratio", "step24_f_g_r_log_evidence_ratio")}})
        result.update(complete=True, groups=summaries, prior_profiles=len(result["rows"]), source_forecasts=16 * len(result["rows"]))
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.monotonic() - wall,
            current_process_vm_hwm_mb=float(next(line.split()[1] for line in
                Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))) / 1024)
        (ARTIFACTS / "attribution.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "current_process_vm_hwm_mb")}), flush=True)


if __name__ == "__main__":
    main()
