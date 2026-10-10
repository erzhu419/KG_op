"""A29: exact inherited-risk and incremental joint-model accounting."""
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from performance.diagnose_meta_prior_source_cross_archive import block_contrast
from performance.validate_meta_prior_scale_preserving_source_covariance import read

ARTIFACTS=ROOT/"paper_artifacts/meta_prior_joint_archive_attribution_a29_20261003"


def main():
    start=time.process_time()
    spec=read("performance/meta_prior_joint_archive_attribution_plan.json")
    cap=spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU,(cap,cap+2))
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    (ARTIFACTS/"execution_plan.json").write_text(json.dumps(spec,indent=2)+"\n")
    source=read(spec["inputs"][1])
    rows=read(spec["inputs"][0])["rows"]
    if not source["complete"] or len(rows)!=160:
        raise ValueError("A29 requires the complete A28 cohort")
    values,error=[],0.
    for r in rows:
        s={k:np.array(v) for k,v in r["scores"].items()}
        metrics={"j_minus_g":s['J']-s['G'],"j_minus_i":s['J']-s['I'],"i_minus_g":s['I']-s['G'],
                 "a_minus_i":s['A']-s['I'],"j_minus_a":s['J']-s['A']}
        error=max(error,float(np.max(np.abs(metrics['j_minus_g']-metrics['j_minus_i']-metrics['i_minus_g']))),
                  float(np.max(np.abs(metrics['j_minus_i']-metrics['j_minus_a']-metrics['a_minus_i']))))
        values.append({**{k:r[k] for k in ('seed','pair_id','regime','path_policy')},**{k:float(v.mean()) for k,v in metrics.items()}})
    if error>1e-12:
        raise ValueError("A29 exact accounting identity failed")
    indices=np.random.default_rng(spec["paired_bootstrap_seed"]).integers(8,size=(20000,8))
    groups=[]
    for regime in sorted({r['regime'] for r in values}):
        for policy in ('risk_ard_mixture','fixed_source'):
            selected=[r for r in values if (r['regime'],r['path_policy'])==(regime,policy)]
            groups.append({'regime':regime,'path_policy':policy,
                **{k:block_contrast([np.mean([r[k] for r in selected if r['pair_id']==i]) for i in range(8)],indices) for k in metrics}})
    high=next(g for g in groups if (g['regime'],g['path_policy'])==('high_frequency_risk','fixed_source'))
    result={'stage':spec['stage'],'complete':True,'groups':groups,'rows':values,'max_identity_error':error,
        'high_frequency_generic_path_inherited_share':high['i_minus_g']['mean_gain']/high['j_minus_g']['mean_gain'],
        'new_forecasts':0,'server_jobs':0,'server_pulls':0,'new_samples':0,'cpu_seconds':time.process_time()-start}
    (ARTIFACTS/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','cpu_seconds','max_identity_error','high_frequency_generic_path_inherited_share')}),flush=True)


if __name__=='__main__':
    main()
