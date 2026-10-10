"""A26: cached continuous local/population allocation; no mass selection."""
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.validate_meta_prior_scale_preserving_source_covariance import read

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_archive_covariance_frontier_a26_20261003"


def main():
    start = time.process_time()
    spec = read("performance/meta_prior_archive_covariance_frontier_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap+2))
    def stop(signum, frame):
        raise TimeoutError("A26 local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2)+"\n")
    result = {"stage": spec["stage"], "complete": False, "groups": [],
              "new_predictions": 0, "new_samples": 0, "server_jobs": 0, "server_pulls": 0}
    try:
        source = read(spec["inputs"][1])
        rows = read(spec["inputs"][0])["rows"]
        expected = {(s, g, h, c) for s in range(48,64) for g in ("shared", "separate_heads")
                    for h in range(8) for c in ("matched", "transfer")}
        if not source["complete"] or len(rows) != 512 or {(r["seed"], r["archive_group"], r["held_task"], r["condition"]) for r in rows} != expected:
            raise ValueError("A26 requires unchanged complete A25 evidence")
        nodes = spec["grid"]["lambda_nodes"]
        lam = np.linspace(0,1,nodes)
        weights = np.column_stack((.5*lam, .5*(1-lam), np.full(nodes, .5)))
        with np.errstate(divide="ignore"):
            logs = np.log(weights)
        gains, error = [], 0.
        for r in rows:
            e, p = np.array(r["log_evidence"]), np.array(r["log_density"])
            branch_e = np.column_stack((logsumexp(e[:,:28],axis=1)-np.log(28), e[:,28], e[:,29]))
            branch_j = np.column_stack((logsumexp((e+p)[:,:28],axis=1)-np.log(28), (e+p)[:,28], (e+p)[:,29]))
            score = logsumexp(branch_j[:,:,None]+logs.T[None,:,:],axis=1)-logsumexp(branch_e[:,:,None]+logs.T[None,:,:],axis=1)
            for j, name in ((0,"I"),((nodes-1)//2,"H"),(nodes-1,"M")):
                error = max(error,float(np.max(np.abs(score[:,j]-r["scores"][name]))))
            delta = score-np.array(r["scores"]["I"])[:,None]
            gains.append({"pair_id":r["pair_id"],"condition":r["condition"],"archive_group":r["archive_group"],
                "mean":delta.mean(axis=0),"early_h":float(delta[:4,(nodes-1)//2].mean()),
                "late_h":float(delta[-4:,(nodes-1)//2].mean())})
        if error>1e-10:
            raise ValueError("A26 endpoint reconstruction failed")
        indices=np.random.default_rng(spec["paired_bootstrap_seed"]).integers(8,size=(20000,8))
        matched_means,matched_bounds=[],[]
        for condition in ("matched","transfer"):
            for group in ("shared","separate_heads"):
                selected=[r for r in gains if (r["condition"],r["archive_group"])==(condition,group)]
                blocks=np.array([np.mean([r["mean"] for r in selected if r["pair_id"]==i],axis=0) for i in range(8)])
                means=blocks.mean(axis=0)
                # Average resampled blocks without materializing 20000 x 8 x 201.
                counts=np.stack([(indices==i).sum(axis=1) for i in range(8)],axis=1)/8
                intervals=np.quantile(counts@blocks,[.025,.975],axis=0).T
                result["groups"].append({"condition":condition,"archive_group":group,
                    "lambda":lam.tolist(),"mean_gain_vs_i":means.tolist(),"paired_block95":intervals.tolist(),
                    "early_h":block_contrast([np.mean([r["early_h"] for r in selected if r["pair_id"]==i]) for i in range(8)],indices),
                    "late_h":block_contrast([np.mean([r["late_h"] for r in selected if r["pair_id"]==i]) for i in range(8)],indices)})
                if condition=="matched":
                    matched_means.append(means)
                    matched_bounds.append(intervals[:,1])
        acceptable=np.all(np.array(matched_means)>=-1e-12,axis=0)&(lam>0)
        result.update(complete=True,max_endpoint_error=error,
            nonzero_masses_without_matched_mean_loss=lam[acceptable].tolist(),
            all_tested_nonzero_masses_have_negative_matched_upper95=bool(np.all(np.array(matched_bounds)[:,1:]<0)),
            primary_mass_changed=False)
    except Exception as exc:
        result["failure"]=str(exc)
        raise
    finally:
        result["cpu_seconds"]=time.process_time()-start
        (ARTIFACTS/"diagnostic.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({k:result[k] for k in ("complete","cpu_seconds")}),flush=True)


if __name__=="__main__":
    main()
