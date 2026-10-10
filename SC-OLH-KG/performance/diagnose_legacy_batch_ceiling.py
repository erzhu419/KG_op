"""Post-freeze finite-bank ceiling; no source fit or new simulations."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive,resample_normalized_profiles
from core.designs import load_frozen_source_informed_design
from core.guarded_deployment_kg import deployment_values
from performance.benchmark_lodo_meta_prior import build_scalarized_problem


def run():
    started=time.process_time()
    spec=json.loads((ROOT/'performance/manifests/legacy_batch_ceiling_audit_v1.json').read_text())
    closed=json.loads((ROOT/spec['closed_loop']).read_text()); p=closed['protocol']
    pilot=json.loads((ROOT/p['pilot']).read_text()); replay=json.loads((ROOT/pilot['protocol']['replay']).read_text())
    base=json.loads((ROOT/replay['protocol']['paid_paths']).read_text())['protocol']
    domain=pilot['protocol']['domain']; inputs=ROOT/'paper_artifacts/legacy_joint_bridge_v1/inputs'/domain
    archive=FrozenTransferArchive.load(inputs/'archive.json')
    initial,_=load_frozen_source_informed_design(inputs/'design.json',heldout=domain,
        seed=base['target_seed'],n0=10,dimension=base['target_dimension'])
    problem=build_scalarized_problem(domain,base['target_dimension'],base['L'],base['sigma'],base['alpha'],base['weights'])
    def forbidden(*args,**kwargs):
        raise RuntimeError('ceiling scoring cannot execute simulations')
    problem.simulate=forbidden
    lifted=resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]),problem.d)
    candidates=list(dict.fromkeys(list(initial)+[tuple(map(int,problem.continuous_to_int(x))) for x in lifted]))
    truth=np.array([[problem.true_objective(x),np.log(problem.true_sigma(x)[0]**2),
        problem.true_constraint_mean(x)-problem.tau,np.log(problem.true_sigma(x)[1]**2)] for x in candidates])
    feasible=np.array([problem.is_truly_feasible(x) for x in candidates])
    reference=float(np.dot(problem.weights,problem.ref_point)); quality=np.clip(1-truth[:,0]/reference,0,1)
    def ceiling(indices):
        allowed=np.array(indices)[feasible[indices]]
        best=int(allowed[np.argmax(quality[allowed])])
        return {'candidate_count':len(indices),'feasible_count':len(allowed),
            'best_feasible_policy':best,'best_feasible_quality':float(quality[best])}
    source=replay['source_fit_contract']
    protocol={'objective_reference':reference,'familywise_delta':source['familywise_delta'],
        'chance_alpha':problem.alpha,'comparison_budget':source['comparison_budget'],'comparison_delta':source['comparison_delta'],
        'candidate_budgets':[spec['direct_budget']]}
    retained=spec['retained_policy']
    reward,cost=deployment_values(truth[[retained]][None],np.array([[0]]),np.array([0]),protocol)
    output={'protocol':spec,'bank':ceiling(list(range(len(candidates)))),
        'primary_roster':ceiling(spec['primary_roster']),
        'retained':{'candidate':retained,'feasible':bool(feasible[retained]),'quality':float(quality[retained])},
        'direct_retained':{'verification_calls':int(cost[0]),'total_calls':int(cost[0]+p['starting_paid_calls']),
            'expected_terminal_reward':float(reward[0])},
        'candidate_truth':{'feasible_policies':np.flatnonzero(feasible).tolist(),
            'quality':quality.tolist(),'g_mean':truth[:,2].tolist(),'eta_g':truth[:,3].tolist()},
        'new_simulator_calls':0,'source_refits':0,'cpu_seconds':time.process_time()-started}
    (ROOT/spec['output']).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('bank','primary_roster','retained','direct_retained','cpu_seconds')}))


if __name__=='__main__':
    run()
