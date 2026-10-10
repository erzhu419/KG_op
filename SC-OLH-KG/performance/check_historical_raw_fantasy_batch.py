"""Compare merged fantasy quadrature to the actual sequential paid update."""

import copy
import time

import numpy as np

from core.positive_mixture_lookahead import predictive_values
from performance.complete_historical_source_heads import ROOT,read,save
from performance.connect_complete_historical_reuse import run as restore_connected
from representation.vectorized_joint_function_risk import raw_fantasy_moments,apply_raw_moments


def run():
    out=ROOT/'paper_artifacts/historical_control_variate_action_comparison_v1';out.mkdir(parents=True,exist_ok=True)
    path=out/'batch_equivalence.json'
    if path.exists():return read(str(path.relative_to(ROOT)))
    rows=[];models=restore_connected('performance/manifests/historical_source_variance_resolution_v4.json')
    for state,model,*_ in models:
        query={'5':7,'6':4}[state['layout']]
        values,_=predictive_values(model,query,4,202610078)
        for identity,pair in zip(model.names,model.components):
            for head,field in enumerate(pair):
                y=values[:,head];labels=y[[int(np.argmin(y)),int(np.argmax(y)),len(y)//2,0]]
                started=time.process_time();moments,gain=raw_fantasy_moments(field,query,labels)
                batched=time.process_time()-started;error=0.;started=time.process_time()
                for label,local in zip(labels,moments):
                    exact=copy.deepcopy(field);score=exact.update_batch(query,float(label),None,1)
                    merged=copy.deepcopy(field);new_score=apply_raw_moments(merged,query,local,gain)
                    error=max(error,abs(score-new_score),float(np.max(abs(exact.means-merged.means))),
                        float(np.max(abs(exact.covariances-merged.covariances))),
                        float(np.max(abs(exact.log_weights-merged.log_weights))))
                passed=error<=1e-8
                rows.append({'layout':state['layout'],'identity':identity,'head':head,'labels':labels.tolist(),
                    'maximum_paid_update_difference':error,'equivalence_passed':passed,
                    'batch_cpu_seconds':batched,'scalar_cpu_seconds':time.process_time()-started})
                if not passed:raise RuntimeError('Merged raw fantasies changed the actual paid filter')
    result={'rows':rows,'all32_heads_equivalent':len(rows)==32 and all(r['equivalence_passed'] for r in rows),
        'batch_cpu_seconds':sum(r['batch_cpu_seconds'] for r in rows),
        'scalar_cpu_seconds':sum(r['scalar_cpu_seconds'] for r in rows),
        'new_source_samples':0,'new_outcomes':0}
    save(path,result);print('fantasy batch equivalence',result['all32_heads_equivalent'],
        'batch/scalar CPU',result['batch_cpu_seconds'],result['scalar_cpu_seconds'],flush=True)
    return result


if __name__=='__main__':run()
