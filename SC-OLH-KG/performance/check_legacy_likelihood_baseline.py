"""Higher precision for the common, frozen Inventory baseline only."""

import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from core.raw_likelihood_lookahead import fixed_action_likelihood_reference
from performance.replay_legacy_mixture_budget import build_models,terminal_at_prefix


def check():
    spec=json.loads((ROOT/'performance/manifests/legacy_likelihood_baseline_v1.json').read_text())
    replay=json.loads((ROOT/spec['replay']).read_text()); protocol=replay['protocol']
    old=json.loads((ROOT/protocol['paid_paths']).read_text()); started=time.process_time()
    problem,initial,candidates,model,_=build_models(spec['domain'],replay['source_fit_contract'],old['protocol'],protocol['hypotheses'])
    paid=next(r for r in old['rows'] if r['domain']==spec['domain'] and r['arm']=='source_orthogonal' and r['search_calls']==13)
    for index,value in zip(paid['paid_candidate_indices'][:spec['prefix']],paid['paid_observations'][:spec['prefix']]):
        model.update_raw(index,[value[0],value[1]-problem.tau])
    terminal,_=terminal_at_prefix(problem,initial,paid,spec['prefix'],replay['source_fit_contract'],len(candidates))
    rows=[]
    for seed in spec['seeds']:
        for index in spec['candidates']:
            print('fixed baseline refinement starts',seed,index,flush=True)
            row=fixed_action_likelihood_reference(model,spec['action'],index,draw_power=spec['draw_power'],
                fantasy_power=spec['fantasy_power'],seed=seed,protocol=terminal.protocol)
            rows.append(row); print(json.dumps(row),flush=True)
    output={'protocol':spec,'rows':rows,'new_calls':0,'cpu_seconds':time.process_time()-started}
    (ROOT/spec['output']).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'new_calls':0,'cpu_seconds':output['cpu_seconds']}),flush=True)


if __name__=='__main__':
    check()
