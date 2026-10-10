"""Independently verify the already retained policy under a fixed direct rule."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive,resample_normalized_profiles
from core.designs import load_frozen_source_informed_design
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.run_legacy_one_step_closed_loop import execute_terminal


def run():
    started=time.process_time()
    spec=json.loads((ROOT/'performance/manifests/legacy_retained_direct_control_v1.json').read_text())
    closed=json.loads((ROOT/spec['closed_loop']).read_text()); p=closed['protocol']
    pilot=json.loads((ROOT/p['pilot']).read_text()); replay=json.loads((ROOT/pilot['protocol']['replay']).read_text())
    base=json.loads((ROOT/replay['protocol']['paid_paths']).read_text())['protocol']
    domain=pilot['protocol']['domain']; inputs=ROOT/'paper_artifacts/legacy_joint_bridge_v1/inputs'/domain
    archive=FrozenTransferArchive.load(inputs/'archive.json')
    initial,_=load_frozen_source_informed_design(inputs/'design.json',heldout=domain,
        seed=base['target_seed'],n0=10,dimension=base['target_dimension'])
    problem=build_scalarized_problem(domain,base['target_dimension'],base['L'],base['sigma'],base['alpha'],base['weights'])
    lifted=resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]),problem.d)
    candidates=list(dict.fromkeys(list(initial)+[tuple(map(int,problem.continuous_to_int(x))) for x in lifted]))
    retained=p['retained_policy']; source=replay['source_fit_contract']
    action={'kind':'direct','primary':retained,'shortlist':[retained],
        'incumbent_position':0,'budgets':[spec['verification_budget']]}
    protocol={'familywise_delta':source['familywise_delta'],'comparison_budget':source['comparison_budget'],
        'comparison_delta':source['comparison_delta']}
    calls=0; simulate=problem.simulate
    def counted_simulate(point,rng=None):
        nonlocal calls
        calls+=1
        return simulate(point,rng)
    problem.simulate=counted_simulate
    rows=[]
    for group,seed in enumerate(spec['verification_seeds']):
        deployed,verification=execute_terminal(problem,candidates,action,p['starting_paid_calls'],seed,protocol,p['total_target_call_cap'])
        rows.append({'group':group,'verification_seed':seed,'action':action,'deployed':deployed,'verification':verification})
        print('retained direct verification complete',group,verification['certified'],verification['total_evaluation_count'],flush=True)
    if calls!=len(rows)*spec['verification_budget']:
        raise RuntimeError('direct control simulator calls do not match the fixed budget')
    feasible=bool(problem.is_truly_feasible(candidates[retained]))
    quality=float(np.clip(1-problem.true_objective(candidates[retained])/np.dot(problem.weights,problem.ref_point),0,1))
    for row in rows:
        certified=row['verification']['certified']
        row['truth']={'deployed_feasible':feasible,'safe_reward':float(certified*feasible*quality),
            'false_certificate':bool(certified and not feasible)}
    output={'protocol':spec,'rows':rows,'new_calls':{'source':0,'search':0,'terminal':calls},
        'source_refits':0,'cpu_seconds':time.process_time()-started}
    (ROOT/spec['output']).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'new_calls':output['new_calls'],'cpu_seconds':output['cpu_seconds']}),flush=True)


if __name__=='__main__':
    run()
