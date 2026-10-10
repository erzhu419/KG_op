"""Paired local-likelihood/projection reference at the opened Inventory node."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from core.raw_likelihood_lookahead import likelihood_query_lookahead
from performance.replay_legacy_mixture_budget import build_models,terminal_at_prefix


def run():
    spec=json.loads((ROOT/'performance/manifests/legacy_mixture_likelihood_v1.json').read_text())
    replay=json.loads((ROOT/spec['replay']).read_text()); protocol=replay['protocol']
    old=json.loads((ROOT/protocol['paid_paths']).read_text()); base=old['protocol']
    domain=spec['domain']; prefix=spec['prefix']; started=time.process_time()
    problem,initial,candidates,model,sources=build_models(domain,replay['source_fit_contract'],base,protocol['hypotheses'])
    paid=next(r for r in old['rows'] if r['domain']==domain and r['arm']=='source_orthogonal' and r['search_calls']==13)
    for index,value in zip(paid['paid_candidate_indices'][:prefix],paid['paid_observations'][:prefix]):
        model.update_raw(index,[value[0],value[1]-problem.tau])
    terminal,_=terminal_at_prefix(problem,initial,paid,prefix,replay['source_fit_contract'],len(candidates))
    original=next(r for r in replay['rows'] if r['domain']==domain and r['prefix']==prefix)
    weight_error=float(np.max(np.abs(model.weights-original['model_weights'])))
    rows=[]
    for seed in spec['seeds']:
        for index in spec['candidates']:
            print('likelihood reference starts',seed,index,flush=True)
            row=likelihood_query_lookahead(model,terminal,protocol['total_target_call_cap']-prefix,index,
                draw_power=spec['draw_power'],fantasy_power=spec['fantasy_power'],seed=seed)
            row['seed']=seed; row['observed']=index in paid['paid_candidate_indices'][:prefix]
            rows.append(row)
            print(json.dumps({k:row[k] for k in ['seed','candidate','adaptation_value','likelihood_tower_drift',
                'projected_baseline_drift','projection_minus_likelihood_baseline','effective_sample_size_min']}),flush=True)
    projected=json.loads((ROOT/spec['projected_reference']).read_text())
    saved=next(r['lookahead'] for r in projected['rows'] if r['domain']==domain and r['prefix']==prefix)
    result={'protocol':spec,'model_weights':model.weights.tolist(),'weight_replay_max_error':weight_error,
        'source_fit_diagnostics':sources,'rows':rows,'saved_projected_lookahead':saved,
        'new_calls':0,'cpu_seconds':time.process_time()-started}
    output=ROOT/spec['output']; output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'new_calls':0,'cpu_seconds':result['cpu_seconds']}),flush=True)


if __name__=='__main__':
    run()
