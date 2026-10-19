"""Development-only replay of frozen source correction on old ranked paths."""

import argparse
import json
import resource
import signal
import time
import traceback

import numpy as np

from performance.diagnose_mainline_objective_transfer import ROOT, read, save, score
from performance.diagnose_mainline_risk_transfer import SOURCE, FOCUS, OUTPUT as BASELINE
from performance.mainline_cross_dimension_prior import restore_cross_dimension_prior
from performance.mainline_risk_correction import apply_public_mean_correction

CALIBRATION = ROOT/'paper_artifacts/mainline_d1000_risk_basis_diagnostic_v1/calibration.json'
OUTPUT = ROOT/'paper_artifacts/mainline_d1000_risk_corrected_replay_v1'


def risk_moments(model, query=None):
    """Exact nested-mixture g marginals and one covariance column, without f."""
    means, variances, columns, noises = [], [], [], []
    for _, field in model.components:
        masses = np.exp(field.log_weights); locations = field.means[:, ::2]
        mean = masses@locations; centered = locations-mean
        ids = 2*np.arange(len(mean))
        means.append(mean)
        variances.append(masses@field.covariances[:, ids, ids]+masses@(centered**2))
        if query is not None:
            columns.append(masses@field.covariances[:, ids, 2*query]+(masses*centered[:, query])@centered)
            eta = 2*query+1
            noises.append(masses@np.exp(field.means[:, eta]+field.covariances[:, eta, eta]/2))
    means = np.asarray(means); mean = model.weights@means; centered = means-mean
    result = {'mean': mean, 'variance': model.weights@np.asarray(variances)+model.weights@(centered**2)}
    if query is not None:
        result.update(covariance_to_query=model.weights@np.asarray(columns)+(model.weights*centered[:, query])@centered,
                      episode_noise=float(model.weights@noises))
    return result


def run_case(index, frozen, calibration):
    started = time.process_time()
    old = read(BASELINE/f'task{index:02d}.json')
    case = read(SOURCE/f'task{index:02d}/source_structural_mixture/case.json')
    paid = read(SOURCE/f'task{index:02d}/initialization.json')
    indices = old['subset_global_indices']; local = {i: k for k, i in enumerate(indices)}
    spec = frozen['plan']; reuse = read(ROOT/spec['reuse_plan'])
    pilot = read(ROOT/read(ROOT/reuse['independent_execution_plan'])['pilot_plan'])
    coordinates = read(ROOT/spec['query_coordinates'])
    coordinates = {k: np.asarray(coordinates[k])[indices].tolist() for k in ('mean', 'risk')}
    model, restoration = restore_cross_dimension_prior(reuse, pilot, spec['layout'], 'source_structural_mixture', coordinates)
    assert len(model.components) == 8 and restoration['positive_components_per_head'] == [1024]*16
    prior = risk_moments(model); saved_prior = old['stages'][0]
    errors = {}
    for key, actual, expected in [('g_mean', prior['mean'], saved_prior['g_mean']),
        ('g_variance', prior['variance'], saved_prior['g_variance']),
        ('identity_weights', model.weights, saved_prior['identity_weights'])]:
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        errors[key] = float(np.max(abs(actual-np.asarray(expected))))
    correction = apply_public_mean_correction(model, coordinates, calibration)
    assert len(correction['affected_identities']) == 2
    state = risk_moments(model); stages = {0: state}; effects = []
    stage_costs = {'restore_and_correct': time.process_time()-started}
    events = [(q, raw) for q, raw in zip(paid['candidate_bank_indices'], paid['paired_observations'])]
    events += [(r['decision']['query'], r['raw_observation']) for r in case['rounds']]
    for j, (global_query, raw) in enumerate(events):
        update_started = time.process_time()
        query = local[global_query]; before = risk_moments(model, query)
        values = np.asarray([raw[0], raw[1]-paid['tau']])
        innovation = float(values[1]-before['mean'][query])
        observation_variance = before['variance'][query]+before['episode_noise']
        model.update_raw(query, values); state = risk_moments(model)
        if j+1 in (2, 4):
            stages[j+1] = state
        if global_query == 83 and j >= 2:
            previous = old['updates'][j]
            for point in (102, 84):
                k = local[point]; base = next(p for p in previous['points'] if p['global_index'] == point)
                covariance = float(before['covariance_to_query'][k])
                effects.append({'global_index': point, 'phase': previous['phase'],
                    'baseline': {key: base[key] for key in ('covariance_to_query', 'moment_gain', 'actual_shift')}
                        | {'g_innovation': previous['g_innovation']},
                    'corrected': {'covariance_to_query': covariance, 'moment_gain': float(covariance/observation_variance),
                        'actual_shift': float(state['mean'][k]-before['mean'][k]), 'g_innovation': innovation}})
        stage_costs[f'observation{j+1}_at_{global_query}'] = time.process_time()-update_started
        print(json.dumps({'task_index': index, 'saved_observation': j+1, 'cpu_seconds': time.process_time()-started}), flush=True)
    assert model.paid_episodes == 4
    truth = np.asarray(old['true_centered_g'])
    unseen = old['final_unobserved_groups']['all_generated_unobserved']; selected = [local[i] for i in unseen]
    metrics, points = [], []
    for j, name in [(0, 'prior'), (2, 'initialization'), (4, 'final')]:
        metrics.append({'phase': name, 'baseline': old['stages'][j]['groups']['all_generated_unobserved'],
            'corrected': score(stages[j]['mean'][selected], truth[selected])})
        points.append({'phase': name, 'points': [{'global_index': i, 'true_g': float(truth[local[i]]),
            'baseline_g': old['stages'][j]['g_mean'][local[i]], 'corrected_g': float(stages[j]['mean'][local[i]])}
            for i in FOCUS]})
    return {'task_index': index, 'subset_global_indices': indices, 'fixed_unobserved_indices': unseen,
        'baseline_prior_parity': errors, 'restoration': restoration,
        'affected_identities': correction['affected_identities'], 'point_predictions': points,
        'observing83_effects': effects, 'fixed_unobserved_metrics': metrics,
        'cost': {'cpu_seconds': time.process_time()-started, 'saved_raw_episodes_replayed': 4,
                 'new_simulator_calls': 0, 'new_truth_calls': 0, 'source_fits': 0,
                 'stage_cpu_seconds': stage_costs}}


def summarize(cases):
    metrics, points = [], []
    for j, phase in enumerate(('prior', 'initialization', 'final')):
        metrics.append({'phase': phase, **{arm: {metric: float(np.mean([
            c['fixed_unobserved_metrics'][j][arm][metric] for c in cases])) for metric in ('mse', 'bias')}
            for arm in ('baseline', 'corrected')}, 'mse_improved_cases': sum(
                c['fixed_unobserved_metrics'][j]['corrected']['mse'] < c['fixed_unobserved_metrics'][j]['baseline']['mse'] for c in cases)})
        for k, point in enumerate(FOCUS):
            points.append({'phase': phase, 'global_index': point,
                'true_g': cases[0]['point_predictions'][j]['points'][k]['true_g'],
                **{arm: float(np.mean([c['point_predictions'][j]['points'][k][arm] for c in cases]))
                   for arm in ('baseline_g', 'corrected_g')}})
    return {'development_case_diagnostic_not_new_experiment': True, 'calibration': str(CALIBRATION),
        'parameters_tuned_on_target': False, 'cases': len(cases),
        'all_four_cases_completed': len(cases) == 4,
        'completed_case_indices': [c['task_index'] for c in cases], 'point_predictions': points,
        'fixed_unobserved_metrics': metrics, 'observing83_effects': [dict(task_index=c['task_index'], **effect)
            for c in cases for effect in c['observing83_effects']],
        'maximum_baseline_prior_parity_error': max(v for c in cases for v in c['baseline_prior_parity'].values()),
        'new_simulator_calls': 0, 'new_truth_calls': 0, 'source_fits': 0,
        'native_update_arithmetic': 'Cases0-2 retained their original saved replay. Case3 uses separately validated equivalent component-blocked native arithmetic. All cases retain8 identities x1024 components per head, identical raw-integration precision, frozen correction and saved observation order; this implementation change does not define a new scientific method.',
        'interpretation': 'Frozen source-only correction applied once before all target updates. Both variants follow the same old observations and compare the same final-unobserved points. The corrected model did not choose these queries; this does not estimate its fresh adaptive policy performance. Covariance/observation-variance gain is a moment diagnostic; actual shift is the native mixture posterior difference.'}


def run(case_index=None):
    OUTPUT.mkdir(parents=True, exist_ok=True); ledger = OUTPUT/'attempts.jsonl'
    attempts = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    remaining = min(120, 500-sum(a['cpu_seconds'] for a in attempts))
    if remaining <= 0:
        raise RuntimeError('Replay CPU budget exhausted')
    started = time.process_time(); attempt = {'requested_case': case_index, 'cumulative_cpu_cap': 500,
        'native_update': 'component-blocked equivalent arithmetic; source components and raw integration unchanged'}
    def timeout(_sig, _frame):
        raise TimeoutError('Replay CPU budget exhausted')
    previous = signal.signal(signal.SIGPROF, timeout); signal.setitimer(signal.ITIMER_PROF, remaining)
    try:
        completed = read(SOURCE/'results.json')
        assert completed['all_recommendations_frozen_before_verification'] and completed['all_verification_finished_before_truth_scoring']
        calibration = read(CALIBRATION); frozen = read(SOURCE/'frozen_inputs.json')
        assert not calibration['target_observations_used'] and calibration['source_rows'] == 48
        for index in range(4) if case_index is None else [case_index]:
            path = OUTPUT/f'task{index:02d}.json'
            if not path.exists():
                attempt['active_case'] = index
                result = run_case(index, frozen, calibration); save(path, result)
                print(json.dumps({'task_index': index, **result['cost']}), flush=True)
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
        save(OUTPUT/'summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-index', type=int, choices=range(4))
    run(parser.parse_args().case_index)
