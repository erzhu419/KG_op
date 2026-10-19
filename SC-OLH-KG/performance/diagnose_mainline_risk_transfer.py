"""Replay completed ranked-policy observations to locate risk transfer errors."""

import argparse
import json
from pathlib import Path
import resource
import signal
import time
import traceback

import numpy as np
from scipy.stats import norm

from performance.diagnose_mainline_objective_transfer import (
    ROOT, decompose_mean_shift, internal_decomposition, parity, read, save, score,
)
from performance.mainline_cross_dimension_prior import restore_cross_dimension_prior
from performance.mainline_moment_policy import moment_state
from performance.run_legacy_related_inventory_source import problem_for

SOURCE = ROOT/'paper_artifacts/mainline_d1000_ranked_v1'
OUTPUT = ROOT/'paper_artifacts/mainline_d1000_risk_transfer_diagnostic_v1'
FOCUS = [83, 102, 84, 98]


def capture_risk(model, query):
    """Keep both native transport columns; neither is an observation gain alone."""
    locations, masses, query_means, gains = [], [], [], []
    ix = [2*query, 2*query+1]
    for _, field in model.components:
        locations.append(field.means[:, ::2].copy())
        masses.append(np.exp(field.log_weights).copy())
        query_means.append(field.means[:, ix].copy())
        covariance = field.covariances[:, ix][:, :, ix]
        gains.append(np.linalg.solve(covariance, field.covariances[:, ix, ::2]).transpose(0, 2, 1))
    return {'weights': model.weights.copy(), 'locations': locations, 'masses': masses,
            'identity_mean': np.asarray([w@m for w, m in zip(masses, locations)]),
            'query_means': query_means, 'native_transport': gains}


def native_transport(before, after):
    """Exact location changes from local posterior g and eta displacements."""
    terms, coefficients = [], []
    for mass, gain, old, new in zip(before['masses'], before['native_transport'],
                                   before['query_means'], after['query_means']):
        terms.append(np.einsum('k,knh,kh->nh', mass, gain, new-old))
        coefficients.append(np.einsum('k,knh->nh', mass, gain))
    shift = np.einsum('i,inh->nh', before['weights'], np.asarray(terms))
    coefficients = np.einsum('i,inh->nh', before['weights'], np.asarray(coefficients))
    return {'from_query_g_displacement': shift[:, 0], 'from_query_eta_displacement': shift[:, 1],
            'mean_native_g_coefficient': coefficients[:, 0], 'mean_native_eta_coefficient': coefficients[:, 1]}


def run_case(index, frozen, completed):
    started = time.process_time()
    case = read(SOURCE/f'task{index:02d}/source_structural_mixture/case.json')
    paid = read(SOURCE/f'task{index:02d}/initialization.json')
    original = frozen['dynamic_candidates']['original_count']
    generated = list(range(original, len(frozen['bank'])))
    observed = paid['candidate_bank_indices']+[r['decision']['query'] for r in case['rounds']]
    indices = sorted(set(observed+generated+FOCUS)); local = {g: j for j, g in enumerate(indices)}
    spec = frozen['plan']; reuse = read(ROOT/spec['reuse_plan'])
    pilot = read(ROOT/read(ROOT/reuse['independent_execution_plan'])['pilot_plan'])
    coordinates = read(ROOT/spec['query_coordinates'])
    coordinates = {k: np.asarray(coordinates[k])[indices].tolist() for k in ('mean', 'risk')}
    model, restoration = restore_cross_dimension_prior(reuse, pilot, spec['layout'], 'source_structural_mixture', coordinates)
    assert len(model.components) == 8 and restoration['positive_components_per_head'] == [1024]*16
    stages = [{'phase': 'zero_target_prior', 'state': moment_state(model), 'weights': model.weights.tolist()}]
    events = [(f'initialization{j+1}', q, raw, None) for j, (q, raw) in
              enumerate(zip(paid['candidate_bank_indices'], paid['paired_observations']))]
    events += [(f"query{r['round']}", r['decision']['query'], r['raw_observation'], r) for r in case['rounds']]
    checks, updates = [], []
    for phase, global_query, raw, trace in events:
        query = local[global_query]; state0 = stages[-1]['state']; before = capture_risk(model, query)
        if trace:
            checks.append({'phase': phase+'_before', **parity(state0, model.weights, trace['before'], indices)})
        values = np.asarray([raw[0], raw[1]-paid['tau']])
        innovation = values[1]-state0['mean'][query, 2]
        variance = state0['cov'][query, 2, query, 2]+state0['noise'][query, 1]
        evidence = model.update_raw(query, values)
        state1 = moment_state(model); after = capture_risk(model, query)
        if trace:
            checks.append({'phase': phase+'_after', **parity(state1, model.weights, trace['after'], indices)})
            np.testing.assert_allclose(evidence['head_log_scores'], trace['evidence']['head_log_scores'], atol=1e-9, rtol=1e-9)
        elif phase == 'initialization2':
            checks.append({'phase': 'initialization', **parity(state1, model.weights, case['initial'], indices)})
        decomposition = decompose_mean_shift(before['weights'], before['identity_mean'], after['weights'], after['identity_mean'])
        inner = internal_decomposition(before, after); transport = native_transport(before, after)
        np.testing.assert_allclose(sum(inner.values()), decomposition['within_identity'], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(transport['from_query_g_displacement']+transport['from_query_eta_displacement'],
                                   inner['component_location'], atol=1e-10, rtol=1e-10)
        covariance = state0['cov'][:, 2, query, 2]; moment_gain = covariance/variance
        updates.append({'phase': phase, 'query': global_query, 'centered_raw_g': float(values[1]),
            'predictive_g': float(state0['mean'][query, 2]), 'g_innovation': float(innovation),
            'standardized_innovation': float(innovation/np.sqrt(variance)),
            'observation_variance': float(variance), 'episode_noise_variance': float(state0['noise'][query, 1]),
            'identity_weights_before': before['weights'].tolist(), 'identity_weights_after': after['weights'].tolist(),
            'decomposition': decomposition, 'internal': inner, 'native_transport': transport,
            'covariance_to_query': covariance, 'moment_gain': moment_gain,
            'moment_linear_shift': moment_gain*innovation})
        stages.append({'phase': phase, 'state': state1, 'weights': model.weights.tolist()})
    assert model.paid_episodes == 4
    names = list(model.names); del model
    # This experiment has already frozen and verified every recommendation.
    # Truth scores these saved paths only, after successful replay parity.
    problem = problem_for(frozen['problem_plan'], frozen['tasks'][index])
    truth = np.array([problem.true_constraint_mean(frozen['bank'][i])-problem.tau for i in indices])
    sigma = np.array([problem.true_sigma(frozen['bank'][i])[1] for i in indices])
    chance_margin = truth+norm.ppf(1-frozen['terminal_protocol']['chance_alpha'])*sigma
    unseen = [i for i in generated if i not in observed]
    origins = frozen['dynamic_candidates']['origins']
    groups = {'all_generated_unobserved': unseen,
              'source_ranked_inverse_pool': [i for i in unseen if origins[i]['route'] == 'source_ranked_inverse_pool'],
              'public_proposals': [i for i in unseen if origins[i]['route'] != 'source_ranked_inverse_pool']}
    scored_stages = []
    for stage in stages:
        state = stage['state']
        scored_stages.append({'phase': stage['phase'], 'g_mean': state['mean'][:, 2].tolist(),
            'g_variance': np.diag(state['cov'][:, 2, :, 2]).tolist(), 'identity_weights': stage['weights'],
            'groups': {name: score(state['mean'][[local[i] for i in ids], 2], truth[[local[i] for i in ids]])
                       for name, ids in groups.items()}})
    for j, update in enumerate(updates):
        error0 = stages[j]['state']['mean'][:, 2]-truth
        error1 = stages[j+1]['state']['mean'][:, 2]-truth
        update['query_true_g'] = float(truth[local[update['query']]])
        update['query_raw_minus_truth'] = update['centered_raw_g']-update['query_true_g']
        update['points'] = []
        for i in generated:
            k = local[i]
            update['points'].append({'global_index': i, 'final_unobserved': i in unseen,
                'before_error': float(error0[k]), 'after_error': float(error1[k]),
                'squared_error_change': float(error1[k]**2-error0[k]**2),
                'covariance_to_query': float(update['covariance_to_query'][k]),
                'moment_gain': float(update['moment_gain'][k]), 'moment_linear_shift': float(update['moment_linear_shift'][k]),
                **{key: float(value[k]) for block in ('decomposition', 'internal', 'native_transport')
                   for key, value in update[block].items() if isinstance(value, np.ndarray)}})
        update['decomposition_max_error'] = update['decomposition']['reconstruction_max_absolute_error']
        for key in ('decomposition', 'internal', 'native_transport', 'covariance_to_query', 'moment_gain', 'moment_linear_shift'):
            del update[key]
    return {'task_index': index, 'source_results_completed_at': completed['completed_at'],
        'subset_global_indices': indices, 'final_unobserved_groups': groups, 'focus_indices': FOCUS,
        'true_centered_g': truth.tolist(), 'true_sigma_g': sigma.tolist(), 'true_chance_margin': chance_margin.tolist(),
        'identity_names': names, 'restoration': restoration, 'subset_parity': checks,
        'stages': scored_stages, 'updates': updates,
        'cost': {'cpu_seconds': time.process_time()-started, 'replayed_saved_episodes': 4, 'new_simulator_calls': 0, 'source_fits': 0},
        'interpretation': 'g is centered constraint mean, not chance margin. Every stage is scored on identical final-unobserved points. Moment gain C(p,q)/(Var(g_q)+Eexp(eta_q)) is a linear diagnostic, not the actual mixture update. Exact component location changes transport both local posterior g and eta displacements using pre-observation component covariance; component and identity mass changes are separate. Truth has no role in replay, selection or prior fitting.'}


def summarize(cases):
    rows = []
    for group in cases[0]['final_unobserved_groups']:
        rows.append({'group': group, 'counts': [len(c['final_unobserved_groups'][group]) for c in cases],
            'initialization_to_final_mse_improved_cases': sum(c['stages'][-1]['groups'][group]['mse'] <
                c['stages'][2]['groups'][group]['mse'] for c in cases),
            'stages': [{'phase': stage['phase'],
                **{metric: float(np.mean([c['stages'][j]['groups'][group][metric] for c in cases]))
                   for metric in ('bias', 'mse')}} for j, stage in enumerate(cases[0]['stages'])]})
    focus = []
    for point in FOCUS:
        positions = [c['subset_global_indices'].index(point) for c in cases]
        effects = [next(p for p in u['points'] if p['global_index'] == point)
                   for c in cases for u in c['updates'] if u['phase'].startswith('query') and u['query'] == 83]
        focus.append({'global_index': point, 'true_g': cases[0]['true_centered_g'][positions[0]],
            'true_chance_margin': cases[0]['true_chance_margin'][positions[0]],
            'prior_error_in_latent_sd': (cases[0]['stages'][0]['g_mean'][positions[0]]-
                cases[0]['true_centered_g'][positions[0]])/np.sqrt(cases[0]['stages'][0]['g_variance'][positions[0]]),
            'stages': [{'phase': stage['phase'], 'mean_predicted_g': float(np.mean([
                c['stages'][j]['g_mean'][k] for c, k in zip(cases, positions)]))}
                for j, stage in enumerate(cases[0]['stages'])],
            'observing83_effect': {'count': len(effects),
                'squared_error_increased_cases': sum(e['squared_error_change'] > 0 for e in effects),
                **{key+'_range': [min(e[key] for e in effects), max(e[key] for e in effects)]
                   for key in ('covariance_to_query', 'moment_gain', 'actual_shift', 'before_error', 'after_error',
                               'component_location', 'identity_reweight')}},
            'final_errors': [c['stages'][-1]['g_mean'][k]-c['true_centered_g'][k] for c, k in zip(cases, positions)]})
    terms = ('component_location', 'component_mass', 'component_interaction', 'identity_reweight',
             'identity_interaction', 'from_query_eta_displacement')
    rms = {term: [float(np.sqrt(np.mean([p[term]**2 for p in u['points'] if p['final_unobserved']])))
                  for c in cases for u in c['updates']] for term in terms}
    return {'source': str(SOURCE), 'cases': len(cases), 'groups': rows, 'focus': focus,
        'rms_ranges_over_final_unobserved_points': {k: [min(v), max(v)] for k, v in rms.items()},
        'innovations': [{k: u[k] for k in ('phase', 'query', 'g_innovation', 'query_raw_minus_truth', 'standardized_innovation')}
                        | {'task_index': c['task_index']} for c in cases for u in c['updates']],
        'maximum_subset_parity_error': max(r[k] for c in cases for r in c['subset_parity'] for k in ('mean', 'point_covariance', 'identity_weights')),
        'maximum_additive_decomposition_error': max(u['decomposition_max_error'] for c in cases for u in c['updates']),
        'new_simulator_calls': 0, 'source_fits': 0,
        'interpretation': 'Four observation seeds on one nominal task. Overall unsampled risk MSE improves, while observing83 harms84 and102 locally. Initial optimism is a mean-error finding, not proof of interval undercoverage or statistically identified wrong covariance. Detailed task files retain every point covariance, moment gain, exact native transport, innovation and true error at every saved update.'}


def run(case_index=None):
    OUTPUT.mkdir(parents=True, exist_ok=True); ledger = OUTPUT/'attempts.jsonl'
    attempts = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    remaining = 120-sum(a['cpu_seconds'] for a in attempts)
    if remaining <= 0:
        raise RuntimeError('Risk diagnostic CPU budget exhausted')
    started = time.process_time(); attempt = {'requested_case': case_index}
    def timeout(_sig, _frame):
        raise TimeoutError('Risk diagnostic CPU budget exhausted')
    previous = signal.signal(signal.SIGPROF, timeout); signal.setitimer(signal.ITIMER_PROF, remaining)
    try:
        completed = read(SOURCE/'results.json')
        assert completed['all_recommendations_frozen_before_verification'] and completed['all_verification_finished_before_truth_scoring']
        assert len(completed['rows']) == 16
        frozen = read(SOURCE/'frozen_inputs.json')
        for index in range(4) if case_index is None else [case_index]:
            path = OUTPUT/f'task{index:02d}.json'
            if not path.exists():
                result = run_case(index, frozen, completed); save(path, result)
                print(json.dumps({'task_index': index, 'subset': len(result['subset_global_indices']), **result['cost']}), flush=True)
        attempt['status'] = 'completed'
    except Exception:
        attempt.update(status='failed', traceback=traceback.format_exc()); raise
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0); signal.signal(signal.SIGPROF, previous)
        attempt.update(cpu_seconds=time.process_time()-started, maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with ledger.open('a') as stream:
            stream.write(json.dumps(attempt)+'\n')
    paths = [OUTPUT/f'task{i:02d}.json' for i in range(4)]
    if all(p.exists() for p in paths):
        summary = summarize([read(p) for p in paths])
        summary['total_attempt_cpu_seconds'] = sum(a['cpu_seconds'] for a in [*attempts, attempt])
        summary['failed_attempts'] = sum(a['status'] == 'failed' for a in [*attempts, attempt])
        summary['successful_attempts'] = sum(a['status'] == 'completed' for a in [*attempts, attempt])
        save(OUTPUT/'summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-index', type=int, choices=range(4))
    run(parser.parse_args().case_index)
