"""One frozen objective-pair contrast on all six already-paid task histories."""
import argparse
import json
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr, softmax
from scipy.stats import norm

from performance.diagnose_mainline_initialization import merge, stream_head
from performance.diagnose_mainline_role_choice import score_choice
from performance.mainline_moment_policy import moment_state
from performance.mainline_role_prior import role_inputs, _history
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture

MANIFEST = ROOT/'performance/manifests/mainline_objective_pair_v1.json'
NAMES = ('original_role', 'head_specific_objective_pair')


def objective_donors(identities):
    names = {name:i for i,name in enumerate(identities)}
    return [names['/'.join((name.split('/')[0], 'head_specific', name.split('/')[2]))]
            for name in identities]


def objective_trajectory(record, design, history, baseline, neighbors):
    contrasts = [None]*(len(history)+1)

    def observe(prefix, start, mass, field, LF, unobserved, local):
        delta = LF[neighbors]-LF[baseline]
        mean = field.means[:,0::2]@delta.T
        variance = np.einsum('ai,kij,aj->ka', delta, field.covariances[:,0::2,0::2], delta)
        variance += record['law']['white']*(unobserved[neighbors].astype(int)+int(unobserved[baseline]))*(neighbors!=baseline)
        weights = np.exp(field.log_weights)
        mixed = weights@mean
        probability = ndtr(-mean/np.sqrt(np.maximum(variance,1e-300)))
        probability[:,neighbors==baseline] = 0.
        contrasts[prefix] = merge(contrasts[prefix], mass, mixed,
            weights@(variance+(mean-mixed)**2), weights@probability)

    trajectory, events = stream_head(record, *design, history, 0, 64, observer=observe)
    return trajectory, contrasts, events


def mix(source, heads, contrasts, donors):
    rows = []
    for prefix in range(len(heads[0][0])):
        masses, moment, difference = [], None, None
        for i,donor in enumerate(donors):
            f,g = heads[donor][0][prefix], heads[i][1][prefix]
            mass = float(source['log_weights'][i]+f[0]+g[0])
            masses.append(mass)
            moment = merge(moment,mass,np.column_stack([f[1],g[1]]),
                np.column_stack([f[2],g[2]]),np.column_stack([f[3],g[3]]))
            _,dm,dv,prob = contrasts[donor][prefix]
            difference = merge(difference,mass,dm,dv,prob)
        rows.append(dict(mean=moment[1],variance=moment[2],noise=moment[3],
            identity_weights=softmax(masses),difference_mean=difference[1],
            difference_variance=difference[2],probability_improving=difference[3]))
    return rows


def direct_field(record, D, E):
    """Independent full physical-bank projection for first-task wiring check."""
    count,n = len(record['mean']),len(D)
    means = np.stack([record['mean']@D.T,record['eta_mean']@E.T],axis=2)
    cov = np.zeros((count,n,2,n,2))
    cov[:,:,0,:,0] = np.einsum('np,kpq,mq->knm',D,
        record['covariance']+record['law']['task'],D)+record['law']['white']*np.eye(n)
    cov[:,:,1,:,1] = E@(record['eta_covariance']+record['risk']['task'])@E.T+record['risk']['white']*np.eye(n)
    field = JointFunctionRiskPosterior.from_moments(means[0],cov[0])
    field.means,field.covariances = means.reshape(count,2*n),cov.reshape(count,2*n,2*n)
    field.log_weights = record['log_weights'].copy()
    field.component_pairs = [(0,0)]*count
    return field


def validate_candidate(source, designs, donors, bank, initial_rows, candidate):
    points = list(dict.fromkeys([tuple(r['point']) for r in initial_rows]+bank[-2:]))
    ids = [bank.index(p) for p in points]
    pairs = []
    for i,donor in enumerate(donors):
        pairs.append((direct_field(source['components'][donor][0],
            *[d[ids] for d in designs[donor][0]]),
            direct_field(source['components'][i][1],*[d[ids] for d in designs[i][1]])))
    model = JointResponseMixture(source['identities'],pairs,np.exp(source['log_weights']))
    errors = []
    for prefix in range(11):
        if prefix in (0,10):
            state = moment_state(model)
            variance = state['cov'][np.arange(len(points)),:,np.arange(len(points)),:].diagonal(axis1=1,axis2=2)
            row = {}
            for key,value in dict(mean=state['mean'],variance=variance,noise=state['noise'],
                                  identity_weights=model.weights).items():
                expected = candidate[prefix][key]
                if key!='identity_weights': expected = expected[ids]
                np.testing.assert_allclose(value,expected,rtol=1e-8,atol=1e-10,err_msg=key)
                row[key] = float(np.max(abs(value-expected)))
            errors.append(dict(paid_prefix=prefix,maximum_errors=row))
        if prefix<10:
            raw = initial_rows[prefix]['raw']
            model.update_raw(points.index(tuple(initial_rows[prefix]['point'])),raw)
    events = [dict(event,identity=identity,head=head) for identity,pair in zip(model.names,model.components)
              for head,field in enumerate(pair) for event in getattr(field,'roundoff_events',[])]
    return errors,events


def run_case(plan,seed):
    if seed not in plan['target_seeds']: raise ValueError('Unregistered task')
    parent = ROOT/plan['input_dir']/f'seed_{seed}'
    output = ROOT/plan['output_dir']/f'seed_{seed}'
    case = parent/'legacy_role_full'
    with attempt(output,'objective-pair-replay','FactorShockStatePolicyRZDT1') as cost:
        cost.update(seed=seed,new_source_fits=0,new_truth_function_calls=0,download_bytes=0,
                    new_target_initial_calls=0,source_model_restorations=1,saved_raw_rows_replayed=12)
        parent_plan = read(ROOT/plan['parent_manifest'])
        engine = load_pickle(case/'search_engine.pkl')
        if float(engine.problem.tau)!=0.: raise ValueError('This saved protocol uses tau0')
        bundle = load_pickle(ROOT/parent_plan['frozen_assets']['source_models_file'])
        rows = read(case/'search.json')['history'][:12]
        audit = read(case/'query_audit.json')['stages']
        policies = [tuple(p) for p in np.load(case/'query_policies.npz')['points']]
        neighbors = [tuple(p) for p in np.load(parent/'neighbors.npz')['points']]
        bank = list(dict.fromkeys([tuple(r['point']) for r in rows]+neighbors+
            [policies[i] for stage in audit for i in stage['action_ids']]))
        position = {p:i for i,p in enumerate(bank)}
        neighbor_ids = np.array([position[p] for p in neighbors])
        baseline = position[tuple(rows[0]['point'])]
        history = _history(engine,bank,rows)
        with ExitStack() as stack:
            for name in ('simulate','true_objective','true_constraint_mean','true_sigma'):
                stack.enter_context(patch.object(engine.problem,name,side_effect=AssertionError(name)))
            source,designs = role_inputs(bundle,engine,bank)
            donors = objective_donors(source['identities'])
            heads,contrasts,events = [],[],[]
            for identity,pair,design in zip(source['identities'],source['components'],designs):
                f,delta,fe = objective_trajectory(pair[0],design[0],history,baseline,neighbor_ids)
                g,ge = stream_head(pair[1],*design[1],history,1,64)
                heads.append((f,g));contrasts.append(delta)
                events.extend(dict(event,identity=identity) for event in fe+ge)
                print(seed,identity,'complete',flush=True)
            variants = dict(zip(NAMES,[mix(source,heads,contrasts,list(range(len(donors)))),
                                       mix(source,heads,contrasts,donors)]))
            source_risk_errors = {}
            for key,selector in (('mean',slice(2,4)),('variance',slice(2,4)),('noise',slice(1,2))):
                a,b = [variants[n][0][key][:,selector] for n in NAMES]
                np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12,err_msg='source0 g '+key)
                source_risk_errors[key] = float(np.max(abs(a-b)))
            parity,rankings = [],[]
            for stage in audit:
                t = stage['paid_prefix'];ids = [position[policies[i]] for i in stage['action_ids']]
                before = variants['original_role'][t]
                expected = dict(objective_mean=before['mean'][ids,0],constraint_mean=before['mean'][ids,2],
                    objective_epistemic=before['variance'][ids,0],constraint_epistemic=before['variance'][ids,2],
                    constraint_aleatoric=before['noise'][ids,1])
                errors = {}
                for key,value in expected.items():
                    recorded = stage['score'][key]
                    np.testing.assert_allclose(value,recorded,rtol=1e-8,atol=1e-10,err_msg=key)
                    errors[key] = float(np.max(abs(value-recorded)))
                parity.append(dict(paid_prefix=t,maximum_errors=errors))
                ranked = {}
                for name,snapshots in variants.items():
                    state = snapshots[t]
                    margin = state['mean'][ids,2]+norm.ppf(.95)*np.sqrt(np.maximum(state['noise'][ids,1],1e-12))
                    chosen,_,mode,count = score_choice(state['mean'][ids,0],margin)
                    ranked[name] = dict(selected_bank_index=ids[chosen],selected_id=stage['action_ids'][chosen],
                                        mode=mode,feasible_count=count)
                rankings.append(dict(paid_prefix=t,candidate_count=len(ids),variants=ranked))
            direct,check_events = None,[]
            if seed==plan['target_seeds'][0]:
                direct,check_events = validate_candidate(source,designs,donors,bank,rows[:10],
                                                         variants[NAMES[1]])
                cost['source_model_restorations'] += 1
                cost['saved_raw_rows_replayed'] += 10
            for name,states in variants.items():
                for state in states:
                    np.testing.assert_allclose(state['difference_mean'],
                        state['mean'][neighbor_ids,0]-state['mean'][baseline,0],rtol=1e-8,atol=1e-10)
        result = dict(seed=seed,prefixes=plan['prefixes'],baseline_index=baseline,
            neighbor_bank_indices=neighbor_ids,identity_names=source['identities'],
            objective_donor_names=[source['identities'][i] for i in donors],
            variants={name:{key:np.array([states[t][key] for t in plan['prefixes']])
                            for key in states[0]} for name,states in variants.items()},
            query_rankings=rankings,original_score_parity=parity,direct_candidate_parity=direct,
            source0_risk_parity=source_risk_errors,conditional_g_unchanged='Both variants reuse the same eight g trajectories.',
            roundoff_events=events,direct_validation_roundoff_events=check_events,
            semantics=plan['limits'])
        np.savez_compressed(output/'bank.npz',points=np.asarray(bank,dtype=np.int16))
        save(output/'projection.json',result)
    print(seed,'done',cost['cpu_seconds'],flush=True)


def complete():
    plan = read(MANIFEST)
    output = ROOT/plan['output_dir']
    summary = read(output/'summary.json')
    projections = [read(output/f'seed_{seed}'/'projection.json') for seed in plan['target_seeds']]
    attempts = []
    for directory in [output]+[output/f'seed_{seed}' for seed in plan['target_seeds']]:
        attempts.extend(json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines())
    result = dict(status='complete',tasks=len(projections),cohort_count=summary['cohort_count'],
        attempts=len(attempts),failed_attempts=[r for r in attempts if r['status']!='completed'],
        cost={key:sum(r.get(key,0) for r in attempts) for key in (
            'cpu_seconds','wall_seconds','new_source_calls','new_target_initial_calls',
            'new_target_search_calls','new_verification_calls','new_source_fits',
            'new_truth_function_calls','download_bytes','source_model_restorations','saved_raw_rows_replayed')},
        original_score_prefix_count=sum(len(p['original_score_parity']) for p in projections),
        original_score_max_absolute_error=max(v for p in projections for r in p['original_score_parity']
                                              for v in r['maximum_errors'].values()),
        direct_candidate_max_absolute_error=max(v for p in projections for r in (p['direct_candidate_parity'] or [])
                                                 for v in r['maximum_errors'].values()),
        source0_risk_max_absolute_error=max(v for p in projections for v in p['source0_risk_parity'].values()),
        roundoff_events=sum(len(p['roundoff_events'])+len(p['direct_validation_roundoff_events']) for p in projections),
        recovery='First direct projection check read weights from moment_state; corrected to model.weights. Failed zero-simulation replay CPU retained.')
    save(output/'completion.json',result)
    plan['status']='complete'
    plan['entrypoints']=['performance/run_mainline_objective_pair.py','performance/summarize_mainline_objective_pair.py']
    save(MANIFEST,plan)
    print(result)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    args = parser.parse_args()
    run_case(read(MANIFEST),args.seed)
