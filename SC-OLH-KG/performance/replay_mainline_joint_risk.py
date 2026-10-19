"""Compare three frozen risk laws along the same completed development paths."""

import argparse
import json
import resource
import signal
import time
import traceback

import numpy as np

from performance.diagnose_mainline_objective_transfer import ROOT, read, save, score
from performance.diagnose_mainline_risk_transfer import SOURCE, FOCUS, OUTPUT as ORIGINAL
from performance.mainline_cross_dimension_prior import restore_cross_dimension_prior
from performance.replay_mainline_risk_correction import risk_moments, OUTPUT as RESIDUAL

CALIBRATION = ROOT/'paper_artifacts/mainline_joint_risk_source_v1/calibration.json'
OUTPUT = ROOT/'paper_artifacts/mainline_d1000_joint_risk_replay_v1'
MODELS = ('original', 'residual_addition', 'joint_source')


def run_case(index, frozen, calibration):
    from performance.mainline_joint_risk_source import apply_joint_risk_source

    started = time.process_time()
    old = read(ORIGINAL/f'task{index:02d}.json'); residual = read(RESIDUAL/f'task{index:02d}.json')
    case = read(SOURCE/f'task{index:02d}/source_structural_mixture/case.json')
    paid = read(SOURCE/f'task{index:02d}/initialization.json')
    indices = old['subset_global_indices']; local = {i: k for k, i in enumerate(indices)}
    unseen = old['final_unobserved_groups']['all_generated_unobserved']
    assert residual['subset_global_indices'] == indices and residual['fixed_unobserved_indices'] == unseen
    spec = frozen['plan']; reuse = read(ROOT/spec['reuse_plan'])
    pilot = read(ROOT/read(ROOT/reuse['independent_execution_plan'])['pilot_plan'])
    coordinates = read(ROOT/spec['query_coordinates'])
    coordinates = {k: np.asarray(coordinates[k])[indices].tolist() for k in ('mean', 'risk')}
    model, restoration = restore_cross_dimension_prior(reuse, pilot, spec['layout'], 'source_structural_mixture', coordinates)
    assert len(model.components) == 8 and restoration['positive_components_per_head'] == [1024]*16
    prior = risk_moments(model); parity = {}
    for key, actual, expected in [('g_mean', prior['mean'], old['stages'][0]['g_mean']),
        ('g_variance', prior['variance'], old['stages'][0]['g_variance']),
        ('identity_weights', model.weights, old['stages'][0]['identity_weights'])]:
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        parity[key] = float(np.max(abs(actual-np.asarray(expected))))
    joint_diagnostics = apply_joint_risk_source(model, coordinates, calibration)
    assert model.paid_episodes == 0 and not model.history
    assert len(model.components) == 8
    component_counts = [len(field.means) for pair in model.components for field in pair]
    stages = {0: risk_moments(model)}; weights = {0: model.weights.tolist()}; propagation = []
    events = list(zip(paid['candidate_bank_indices'], paid['paired_observations']))
    events += [(r['decision']['query'], r['raw_observation']) for r in case['rounds']]
    for j, (global_query, raw) in enumerate(events):
        query = local[global_query]; before = risk_moments(model, query)
        values = np.asarray([raw[0], raw[1]-paid['tau']])
        innovation = float(values[1]-before['mean'][query])
        observation_variance = before['variance'][query]+before['episode_noise']
        model.update_raw(query, values); state = risk_moments(model)
        if j+1 in (2, 4):
            stages[j+1] = state; weights[j+1] = model.weights.tolist()
        if global_query == 83 and j >= 2:
            previous = old['updates'][j]
            base = next(p for p in previous['points'] if p['global_index'] == 102)
            added = next(p for p in residual['observing83_effects'] if p['global_index'] == 102)
            assert added['phase'] == previous['phase']
            k = local[102]; covariance = float(before['covariance_to_query'][k])
            propagation.append({'phase': previous['phase'], 'query': 83, 'predicted_point': 102,
                'original': {name: base[name] for name in ('covariance_to_query', 'moment_gain', 'actual_shift')}
                    | {'g_innovation': previous['g_innovation']},
                'residual_addition': added['corrected'],
                'joint_source': {'covariance_to_query': covariance, 'moment_gain': float(covariance/observation_variance),
                    'actual_shift': float(state['mean'][k]-before['mean'][k]), 'g_innovation': innovation}})
    assert model.paid_episodes == 4
    # Only existing post-verification truth is read; neither source fit nor
    # the saved observation path can depend on these diagnostic scores.
    truth = np.asarray(old['true_centered_g']); selected = [local[i] for i in unseen]
    metrics, points = [], []
    for p, (j, phase) in enumerate([(0, 'prior'), (2, 'initialization'), (4, 'final')]):
        added = residual['fixed_unobserved_metrics'][p]
        assert added['phase'] == phase
        metrics.append({'phase': phase, 'original': old['stages'][j]['groups']['all_generated_unobserved'],
            'residual_addition': added['corrected'], 'joint_source': score(stages[j]['mean'][selected], truth[selected])})
        added_points = {v['global_index']: v for v in residual['point_predictions'][p]['points']}
        point_rows = [{'global_index': i, 'true_g': float(truth[local[i]]),
            'original': old['stages'][j]['g_mean'][local[i]], 'residual_addition': added_points[i]['corrected_g'],
            'joint_source': float(stages[j]['mean'][local[i]])} for i in FOCUS]
        for row in point_rows:
            row['signed_errors'] = {name: row[name]-row['true_g'] for name in MODELS}
            row['optimistic'] = {name: row['signed_errors'][name] < 0 for name in MODELS}
        points.append({'phase': phase, 'points': point_rows})
    return {'task_index': index, 'subset_global_indices': indices, 'fixed_unobserved_indices': unseen,
        'original_prior_parity': parity, 'restoration': restoration, 'joint_source_diagnostics': joint_diagnostics,
        'positive_components_per_head': component_counts, 'identity_weights_by_stage': weights,
        'point_predictions': points, 'fixed_unobserved_metrics': metrics, 'observing83_to102': propagation,
        'cost': {'cpu_seconds': time.process_time()-started, 'saved_raw_episodes_replayed': 4,
                 'new_simulator_calls': 0, 'new_truth_calls': 0, 'source_fits_in_replay': 0}}


def summarize(cases, calibration_path):
    metrics, points = [], []
    for j, phase in enumerate(('prior', 'initialization', 'final')):
        metrics.append({'phase': phase, **{name: {metric: float(np.mean([
            c['fixed_unobserved_metrics'][j][name][metric] for c in cases])) for metric in ('mse', 'bias')}
            for name in MODELS}, **{f'joint_mse_improved_vs_{name}_cases': sum(
                c['fixed_unobserved_metrics'][j]['joint_source']['mse'] < c['fixed_unobserved_metrics'][j][name]['mse']
                for c in cases) for name in MODELS[:2]}})
        for k, point in enumerate(FOCUS):
            row = {'phase': phase, 'global_index': point,
                'true_g': cases[0]['point_predictions'][j]['points'][k]['true_g'],
                **{name: float(np.mean([c['point_predictions'][j]['points'][k][name] for c in cases])) for name in MODELS}}
            row['signed_errors'] = {name: row[name]-row['true_g'] for name in MODELS}
            row['optimistic_cases'] = {name: sum(c['point_predictions'][j]['points'][k]['optimistic'][name] for c in cases) for name in MODELS}
            points.append(row)
    return {'development_case_diagnostic_not_new_experiment': True, 'calibration': str(calibration_path),
        'parameters_tuned_on_target': False, 'cases': len(cases), 'all_four_cases_completed': len(cases) == 4,
        'completed_case_indices': [c['task_index'] for c in cases], 'point_predictions': points,
        'fixed_unobserved_metrics': metrics,
        'observing83_to102': [dict(task_index=c['task_index'], **p) for c in cases for p in c['observing83_to102']],
        'maximum_original_prior_parity_error': max(v for c in cases for v in c['original_prior_parity'].values()),
        'new_simulator_calls': 0, 'new_truth_calls': 0, 'source_fits_in_replay': 0,
        'interpretation': 'Signed error and bias mean predicted g minus true g; negative is optimistic about the constraint mean, not a full chance-feasibility verdict. Three frozen source laws share the original ranked observations and fixed final-unobserved sets. Original and residual results are reused, not rerun. Source-fit costs and any source eta refresh are in the joint calibration artifacts; this replay only restores them. It does not estimate a newly adaptive policy or use target outcomes to choose the correction. Covariance/observation-variance gain is a moment diagnostic; actual shift is the native mixture posterior difference.'}


def run(calibration_path=CALIBRATION, case_index=None):
    OUTPUT.mkdir(parents=True, exist_ok=True); ledger = OUTPUT/'attempts.jsonl'
    attempts = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    remaining = min(60, 120-sum(a['cpu_seconds'] for a in attempts))
    if remaining <= 0:
        raise RuntimeError('Joint-risk replay CPU budget exhausted')
    started = time.process_time(); attempt = {'requested_case': case_index, 'cumulative_cpu_cap': 120}
    def timeout(_sig, _frame):
        raise TimeoutError('Joint-risk replay CPU budget exhausted')
    previous = signal.signal(signal.SIGPROF, timeout); signal.setitimer(signal.ITIMER_PROF, remaining)
    try:
        completed = read(SOURCE/'results.json')
        assert completed['all_recommendations_frozen_before_verification'] and completed['all_verification_finished_before_truth_scoring']
        calibration = read(calibration_path); frozen = read(SOURCE/'frozen_inputs.json')
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
        summary = summarize([read(p) for p in paths], calibration_path)
        summary['total_attempt_cpu_seconds'] = sum(a['cpu_seconds'] for a in [*attempts, attempt])
        summary['failed_attempts'] = sum(a['status'] == 'failed' for a in [*attempts, attempt])
        save(OUTPUT/'summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-index', type=int, choices=range(4))
    parser.add_argument('--calibration', default=str(CALIBRATION))
    args = parser.parse_args()
    run(ROOT/args.calibration, args.case_index)
