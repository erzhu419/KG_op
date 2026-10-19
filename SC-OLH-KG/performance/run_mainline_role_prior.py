"""One registered legacy g prior contrast on saved initial and query paths."""
import argparse
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from performance.diagnose_mainline_initialization import marginal
from performance.mainline_role_prior import role_prior, role_trajectory, role_moment_state
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save
from performance.run_mainline_query_channels import compact, constrained_ts, joint_draws, same


MANIFEST = ROOT/'performance/manifests/mainline_role_prior_v1.json'
ARM = 'current_structural_mixture'


def run_case(plan, seed):
    assert seed in plan['target_seeds']
    parent = ROOT/plan['parent_output']/f'seed_{seed}'
    directory = ROOT/plan['output_dir']/f'seed_{seed}'
    with attempt(directory, 'role-initialization', plan['domain'], 'legacy_g_role_prior') as cost:
        cost.update(seed=seed, new_truth_function_calls=0, new_source_fits=0, download_bytes=0,
                    saved_raw_rows_replayed=10, source_model_restorations=1)
        engine = load_pickle(parent/'initialization_only/search_engine.pkl')
        problem = engine.problem
        bundle = load_pickle(ROOT/plan['frozen_assets']['source_models_file'])
        rows = read(parent/'initialization_only/search.json')['history']
        neighbors = [tuple(map(int, row)) for row in np.load(parent/'neighbors.npz')['points']]
        initial = [tuple(row['point']) for row in rows]
        bank = list(dict.fromkeys(initial+neighbors)); index = {x:i for i,x in enumerate(bank)}
        baseline = read(ROOT/plan['baseline_output']/f'seed_{seed}'/ARM/'initialization.json')
        assert len(rows) == 10 and len(bank) == baseline['bank_count']
        with ExitStack() as stack:
            for name in ('simulate', 'true_objective', 'true_constraint_mean', 'true_sigma'):
                stack.enter_context(patch.object(problem, name, side_effect=AssertionError(name)))
            prior = role_prior(engine)
            native = engine.gpr[1]._finite_mixture_component_priors[0]
            # Detect omitted legacy discrepancy calibration or target-state leakage.
            parity = {f'native_source_{key}': same(prior[key], native[key], name=key)
                      for key in ('mean', 'covariance', 'deviation_variance')}
            snapshots, scores, events = role_trajectory(bundle, engine, bank, rows)
            D = engine.gpr[1].basis_matrix(bank)
            g = D@prior['mean']-problem.tau
            variance = np.einsum('ni,ij,nj->n', D, prior['covariance'], D)+prior['deviation_variance']
            parity['source0_g_mean'] = same(snapshots[0]['mean'][:,2], g, name='native projected g')
            parity['source0_g_variance'] = same(snapshots[0]['variance'][:,2], variance,
                                               name='native projected g variance')
            for key in ('mean', 'variance'):
                parity['source0_unchanged_'+key] = same(snapshots[0][key][:,[0,1,3]],
                    np.asarray(baseline['snapshots'][0][key])[:,[0,1,3]], name='unchanged '+key)
            parity['source0_unchanged_noise'] = same(snapshots[0]['noise'],
                baseline['snapshots'][0]['noise'], name='unchanged noise')
            if seed == plan['target_seeds'][0]:
                cost['source_model_restorations'] += 1
                cost['saved_raw_rows_replayed'] += len(rows)
                endpoint = role_moment_state(bundle, engine, bank, rows)
                events.extend(endpoint['roundoff_events'])
                for key, value in marginal(endpoint).items():
                    parity['initial10_endpoint_'+key] = same(snapshots[-1][key], value, name=key)
                parity['initial10_endpoint_weights'] = same(snapshots[-1]['identity_weights'],
                                                            endpoint['weights'], name='weights')
        observations = []
        for prefix, row in enumerate(rows):
            i = index[tuple(row['point'])]; before, after = snapshots[prefix:prefix+2]
            mean, epistemic = before['mean'][i,[0,2]], before['variance'][i,[0,2]]
            noise = before['noise'][i]; centered = np.array([row['raw'][0],row['raw'][1]-problem.tau])
            observations.append(dict(initialization_row=prefix, bank_index=i, raw=row['raw'],
                centered_raw=centered, predictive_mean=mean, predictive_epistemic=epistemic,
                predictive_noise=noise, predictive_total_variance=epistemic+noise,
                standardized_residual=(centered-mean)/np.sqrt(epistemic+noise),
                log_predictive=scores[prefix], identity_weights_before=before['identity_weights'],
                identity_weights_after=after['identity_weights']))
        result = dict(status='complete', seed=seed, bank_count=len(bank),
            initial_bank_indices=[index[x] for x in initial], neighbor_bank_indices=[index[x] for x in neighbors],
            new_neighbor_indices=[i for i,x in enumerate(neighbors) if x not in set(initial)],
            snapshots=snapshots, pre_observation=observations, parity=parity, roundoff_events=events,
            source_prior=prior, identity_names=bundle['identities'])
        np.savez_compressed(directory/'bank.npz', points=np.asarray(bank,dtype=np.int16))
    result['cost'] = cost
    save(directory/'trajectory.json', result)
    print('initialization',seed,'complete',flush=True)

    with attempt(directory, 'role-query-scores', plan['domain'], 'legacy_g_role_prior') as cost:
        cost.update(seed=seed,new_truth_function_calls=0,new_source_fits=0,download_bytes=0,
                    saved_raw_rows_replayed=0)
        source = parent/ARM
        history = read(source/'search.json')['history']
        audit = read(source/'query_audit.json')
        recovery = read(ROOT/plan['query_channels_output']/f'seed_{seed}'/ARM/'recovery.json')
        points = [tuple(map(int,x)) for x in np.load(source/'query_policies.npz')['points']]
        stages = []; events = []
        with ExitStack() as stack:
            for name in ('simulate','true_objective','true_constraint_mean','true_sigma'):
                stack.enter_context(patch.object(problem,name,side_effect=AssertionError(name)))
            for saved,recovered in zip(audit['stages'],recovery['stages'],strict=True):
                prefix = saved['paid_prefix']; ids = saved['action_ids']
                assert prefix == recovered['paid_prefix'] and recovered['full_score_parity']
                candidates = [points[i] for i in ids]
                bank = list(dict.fromkeys([tuple(row['point']) for row in history[:prefix]]+candidates))
                indices = [bank.index(x) for x in candidates]
                anchor = role_moment_state(bundle,engine,bank,history[:10])
                full = anchor if prefix == 10 else role_moment_state(bundle,engine,bank,history[:prefix])
                cost['source_model_restorations'] += 1+(prefix!=10)
                cost['saved_raw_rows_replayed'] += 10+(prefix if prefix!=10 else 0)
                events.extend(anchor['roundoff_events'])
                if full is not anchor:
                    events.extend(full['roundoff_events'])
                controls = {}
                for name,state in [('full',full),('frozen10',anchor)]:
                    f,g = joint_draws(bank,candidates,state,problem.tau,recovered['rng_state'])
                    controls[name] = constrained_ts(f,g,state['noise'][indices,1],problem,ids)
                stages.append(dict(paid_prefix=prefix,action_ids=ids,
                    baseline_selected_id=saved['selected_id'], **controls,
                    full_predictions=compact(full,indices),frozen10_predictions=compact(anchor,indices)))
                print('query',seed,prefix,controls['full']['selected_id'],flush=True)
        result = dict(status='complete',seed=seed,stages=stages,roundoff_events=events,
            policy_bank_file=str((source/'query_policies.npz').relative_to(ROOT)))
    result['cost']=cost
    save(directory/'query_scores.json',result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default=MANIFEST)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    run_case(read(args.manifest),args.seed)
