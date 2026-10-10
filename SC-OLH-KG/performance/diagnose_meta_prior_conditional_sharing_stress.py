"""A32: recombined source-head records, with no recalibration or new samples."""
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from performance.conditional_sharing_calibration import cached_context
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.validate_meta_prior_scale_preserving_source_covariance import read
from representation.archive_covariance_prior import mixture_forecasts, sharing_branches, sharing_forecasts

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_conditional_sharing_stress_a32_20261003"


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_conditional_sharing_stress_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU,(cap,cap+2))
    def stop(signum,frame):
        raise TimeoutError("A32 local CPU cap reached")
    signal.signal(signal.SIGXCPU,stop)
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec,indent=2)+"\n")
    result = {"stage":spec["stage"],"complete":False,"rows":[],"new_gaussian_fits":0,
        "new_samples":0,"server_jobs":0,"server_pulls":0,"target_paths_scored":0,"kg_calls":0}
    try:
        cache = read(spec["inputs"][0])
        rows = cache["rows"]
        source = read(spec["inputs"][2])
        if not source["complete"] or len(rows) != 256 or not read(spec["inputs"][3])["proceed_to_cross_head_record_stress"]:
            raise ValueError("A32 requires the complete A30/A31 source evidence")
        indexed = {(r["seed"],r["archive_group"],r["held_task"]):r for r in rows}
        old = {(r["seed"],r["archive_group"],r["held_task"]):r for r in read(spec["inputs"][1])["rows"] if r["condition"] == "transfer"}
        fits = {f["held_pair_id"]:f for f in source["calibrations"]}
        pairs = [fits[i]["held_seeds"] for i in range(8)]
        controls = {"max_eta_only_reproduction_error":0.,"max_local_i_reproduction_error":0.}
        for row in rows:
            pair = pairs[row["pair_id"]]
            partner = next(s for s in pair if s != row["seed"])
            fg = indexed[partner,row["archive_group"],row["held_task"]]
            combined = {**row,"auxiliary_log_evidence":fg["auxiliary_log_evidence"]}
            risk,auxiliary,weights = cached_context(combined,cache["archive_keys"],pair,partner)
            branches = sharing_branches(risk,auxiliary,weights)
            c,state,posterior = sharing_forecasts(branches,fits[row["pair_id"]]["q_mean"])
            u,u_state,_ = sharing_forecasts(branches)
            i,_ = mixture_forecasts(risk["log_evidence"][:,-2:],risk["log_density"][:,-2:],[.5,.5])
            j,a = branches["scores"]["J"],branches["scores"]["A"]
            previous = old[row["seed"],row["archive_group"],row["held_task"]]["scores"]
            controls["max_eta_only_reproduction_error"] = max(controls["max_eta_only_reproduction_error"],float(np.max(np.abs(a-previous["A"]))))
            controls["max_local_i_reproduction_error"] = max(controls["max_local_i_reproduction_error"],float(np.max(np.abs(i-previous["I"]))))
            full = {name:np.repeat(v[:,1::2],2,axis=1) for name,v in risk.items()}
            cg,_,_ = sharing_forecasts(sharing_branches(full,auxiliary,weights),fits[row["pair_id"]]["q_mean"])
            delta = c-i
            metrics = {"c_minus_i":float(delta.mean()),"c_minus_j":float(np.mean(c-j)),"c_minus_a":float(np.mean(c-a)),
                "a_minus_i":float(np.mean(a-i)),"u_minus_i":float(np.mean(u-i)),"u_minus_a":float(np.mean(u-a)),
                "c_minus_cg":float(np.mean(c-cg)),"early_c_minus_i":float(delta[:4].mean()),"late_c_minus_i":float(delta[-4:].mean())}
            summary = {"seed":row["seed"],"fg_seed":partner,"pair_id":row["pair_id"],"archive_group":row["archive_group"],"held_task":row["held_task"],**metrics}
            for name,t in (("step9",0),("step24",-1)):
                summary[name] = {"c_shared_probability":float(state[t,0]),"u_shared_probability":float(u_state[t,0]),
                    "log_shared_to_separate_bayes_factor":float(branches["log_evidence"][t,0]-branches["log_evidence"][t,1]),
                    "c_population_risk_mass":float(posterior[t,:56].sum())}
            result["rows"].append(summary)
        if max(controls.values()) > 1e-8:
            raise ValueError("A32 recombination changed eta-only or local risk predictions")
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(8,size=(20000,8))
        groups = []
        for group in ("shared","separate_heads"):
            selected = [r for r in result["rows"] if r["archive_group"] == group]
            summary = {"archive_group":group,
                **{k:block_contrast([np.mean([r[k] for r in selected if r["pair_id"] == pair]) for pair in range(8)],indices) for k in metrics},
                **{name:{k:float(np.mean([r[name][k] for r in selected])) for k in selected[0][name]} for name in ("step9","step24")}}
            if summary["a_minus_i"]["mean_gain"] > 0 and summary["c_minus_i"]["mean_gain"] > 0:
                summary["fraction_independent_gain_recovered"] = summary["c_minus_i"]["mean_gain"]/summary["a_minus_i"]["mean_gain"]
            groups.append(summary)
        result.update(complete=True,groups=groups,controls=controls,record_profiles=256,
            record_forecasts=4096,physical_coobserved_task=False,calibration_changed=False)
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result["cpu_seconds"] = time.process_time()-start
        (ARTIFACTS / "diagnostic.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:result[k] for k in ("complete","cpu_seconds")}),flush=True)


if __name__ == "__main__":
    main()
