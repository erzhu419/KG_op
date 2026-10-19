"""Explain saved objective transfer without fitting or new simulation."""

import argparse
import json
from pathlib import Path
import resource
import signal
import time
import traceback

import numpy as np
from scipy.special import logsumexp

from performance.mainline_cross_dimension_prior import restore_cross_dimension_prior
from performance.mainline_moment_policy import moment_state
from performance.mainline_propagation import point_covariance
from performance.run_legacy_related_inventory_source import problem_for

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'paper_artifacts/mainline_d1000_dynamic_v1'
OUTPUT = ROOT/'paper_artifacts/mainline_d1000_transfer_diagnostic_v1'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def decompose_mean_shift(w0, m0, w1, m1):
    """Exact identity-weight/identity-mean decomposition, retaining interaction."""
    w0, m0, w1, m1 = map(np.asarray, (w0, m0, w1, m1))
    within = w0@(m1-m0)
    reweight = (w1-w0)@m0
    interaction = (w1-w0)@(m1-m0)
    actual = w1@m1-w0@m0
    return {'within_identity': within, 'identity_reweight': reweight,
            'identity_interaction': interaction, 'actual_shift': actual,
            'reconstruction_max_absolute_error': float(np.max(abs(actual-within-reweight-interaction)))}


def identity_predictions(weights, scores, post_means):
    """Head-only weight interventions on the same native conditional fields."""
    log_weights = np.full(len(weights), -np.inf)
    np.log(weights, out=log_weights, where=np.asarray(weights) > 0)
    weights_by_rule = {'fixed': np.asarray(weights)}
    for name, evidence in [('f_only', scores[:, 0]), ('g_only', scores[:, 1]), ('joint', scores.sum(axis=1))]:
        logs = log_weights+evidence
        weights_by_rule[name] = np.exp(logs-logsumexp(logs))
    return weights_by_rule, {name: weight@post_means for name, weight in weights_by_rule.items()}


def capture(model):
    locations = [field.means[:, ::2].copy() for field, _ in model.components]
    masses = [np.exp(field.log_weights).copy() for field, _ in model.components]
    return {'weights': model.weights.copy(), 'locations': locations, 'masses': masses,
            'identity_mean': np.asarray([w@m for w, m in zip(masses, locations)])}


def internal_decomposition(before, after):
    terms = []
    for w0, m0, w1, m1 in zip(before['masses'], before['locations'], after['masses'], after['locations']):
        terms.append(decompose_mean_shift(w0, m0, w1, m1))
    return {name: before['weights']@np.asarray([r[key] for r in terms]) for name, key in [
        ('component_location', 'within_identity'), ('component_mass', 'identity_reweight'),
        ('component_interaction', 'identity_interaction')]}


def parity(state, weights, saved, indices):
    actual = [state['mean'], point_covariance(state), weights]
    expected = [np.asarray(saved['mean'])[indices], np.asarray(saved['point_cov'])[indices],
                np.asarray(saved['identity_weights'])]
    errors = {}
    for name, lhs, rhs in zip(('mean', 'point_covariance', 'identity_weights'), actual, expected):
        np.testing.assert_allclose(lhs, rhs, atol=1e-9, rtol=1e-9)
        errors[name] = float(np.max(abs(lhs-rhs)))
    return errors


def score(prediction, truth):
    error = np.asarray(prediction)-truth
    return {key: np.asarray(value).tolist() for key, value in {
        'prediction_mean': np.mean(prediction, axis=0), 'truth_mean': np.mean(truth, axis=0),
        'bias': np.mean(error, axis=0), 'mse': np.mean(error**2, axis=0)}.items()}


def run_case(index, frozen, completed):
    started = time.process_time()
    case = read(SOURCE/f'task{index:02d}/source_structural_mixture/case.json')
    paid = read(SOURCE/f'task{index:02d}/initialization.json')
    original = frozen['dynamic_candidates']['original_count']
    generated = list(range(original, len(frozen['bank'])))
    observed = paid['candidate_bank_indices']+[r['decision']['query'] for r in case['rounds']]
    indices = sorted(set(observed+generated))
    local = {global_index: i for i, global_index in enumerate(indices)}
    spec = frozen['plan']; reuse = read(ROOT/spec['reuse_plan'])
    pilot = read(ROOT/read(ROOT/reuse['independent_execution_plan'])['pilot_plan'])
    coordinates = read(ROOT/spec['query_coordinates'])
    coordinates = {key: np.asarray(coordinates[key])[indices].tolist() for key in ('mean', 'risk')}
    model, restoration = restore_cross_dimension_prior(reuse, pilot, spec['layout'], 'source_structural_mixture', coordinates)
    assert len(model.components) == 8 and restoration['positive_components_per_head'] == [1024]*16
    for global_index, raw in zip(paid['candidate_bank_indices'], paid['paired_observations']):
        model.update_raw(local[global_index], [raw[0], raw[1]-paid['tau']])
    initial_state = moment_state(model)
    checks = [{'phase': 'initial', **parity(initial_state, model.weights, case['initial'], indices)}]
    initial_weights = model.weights.copy()
    cumulative_scores = np.zeros((8, 2))
    records = []
    for trace in case['rounds']:
        state0, before = moment_state(model), capture(model)
        checks.append({'phase': f"round{trace['round']}_before",
                       **parity(state0, model.weights, trace['before'], indices)})
        global_query = trace['decision']['query']; query = local[global_query]
        raw = trace['raw_observation']; values = np.asarray([raw[0], raw[1]-paid['tau']])
        mean0 = state0['mean'][query, [0, 2]]
        variance0 = np.diag(state0['cov'][query, :, query, :])[[0, 2]]
        evidence = model.update_raw(query, values)
        state1, after = moment_state(model), capture(model)
        checks.append({'phase': f"round{trace['round']}_after",
                       **parity(state1, model.weights, trace['after'], indices)})
        head_scores = np.asarray(evidence['head_log_scores'])
        np.testing.assert_allclose(head_scores, trace['evidence']['head_log_scores'], atol=1e-9, rtol=1e-9)
        cumulative_scores += head_scores
        decomposition = decompose_mean_shift(before['weights'], before['identity_mean'], after['weights'], after['identity_mean'])
        inner = internal_decomposition(before, after)
        np.testing.assert_allclose(sum(inner.values()), decomposition['within_identity'], atol=1e-10, rtol=1e-10)
        weights, predictions = identity_predictions(before['weights'], head_scores, after['identity_mean'])
        np.testing.assert_allclose(weights['joint'], after['weights'], atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(predictions['joint'], state1['mean'][:, 0], atol=1e-10, rtol=1e-10)
        # Symmetric allocation of the f/g identity-weight interaction, evaluated
        # after the same actual native f update. These are not alternative runs.
        shapley = {'f': .5*((weights['f_only']-weights['fixed'])+(weights['joint']-weights['g_only']))@after['identity_mean'],
                   'g': .5*((weights['g_only']-weights['fixed'])+(weights['joint']-weights['f_only']))@after['identity_mean']}
        records.append({'round': trace['round'], 'query': global_query, 'raw_observation': raw,
            'centered_observation': values.tolist(), 'predictive_mean_f_g': mean0.tolist(),
            'raw_innovation_f_g': (values-mean0).tolist(),
            'predictive_episode_noise_variance_f_g': state0['noise'][query].tolist(),
            'standardized_innovation_f_g': ((values-mean0)/np.sqrt(variance0+state0['noise'][query])).tolist(),
            'head_log_scores': head_scores.tolist(), 'identity_weights': {k: v.tolist() for k, v in weights.items()},
            'decomposition': decomposition, 'within_identity_breakdown': inner,
            'identity_head_shapley': shapley, 'predictions': predictions, 'before_state': state0, 'after_state': state1})
    assert model.paid_episodes == len(observed) == 4
    final_weights, final_predictions = identity_predictions(initial_weights, cumulative_scores, after['identity_mean'])
    np.testing.assert_allclose(final_weights['joint'], model.weights, atol=1e-12, rtol=1e-10)
    identity_names = list(model.names)
    del model
    # The already completed actual experiment is the only source of queries.
    # Truth is now used for scoring, after successful marginal replay checks.
    problem = problem_for(frozen['problem_plan'], frozen['tasks'][index])
    truth = np.asarray([[problem.true_objective(frozen['bank'][i]), np.log(problem.true_sigma(frozen['bank'][i])[0]**2),
        problem.true_constraint_mean(frozen['bank'][i])-problem.tau, np.log(problem.true_sigma(frozen['bank'][i])[1]**2)] for i in indices])
    unseen = [i for i in generated if i not in observed]
    origins = frozen['dynamic_candidates']['origins']
    groups = {'all_generated_unobserved': unseen,
              'source_profile_refinement': [i for i in unseen if origins[i]['route'] == 'source_profile_refinement'],
              'public_proposals': [i for i in unseen if origins[i]['route'] != 'source_profile_refinement']}
    scored = []
    for record in records:
        row = {k: v for k, v in record.items() if k not in ('before_state', 'after_state', 'predictions', 'decomposition', 'within_identity_breakdown', 'identity_head_shapley')}
        row['scored_global_indices'] = unseen
        selected = [local[i] for i in unseen]
        for key in ('decomposition', 'within_identity_breakdown', 'identity_head_shapley'):
            row[key] = {k: v[selected].tolist() if isinstance(v, np.ndarray) else v for k, v in record[key].items()}
        row['groups'] = {}
        for name, global_indices in groups.items():
            selected = [local[i] for i in global_indices]
            row['groups'][name] = {'global_indices': global_indices,
                'before_all_heads': score(record['before_state']['mean'][selected], truth[selected]),
                'after_all_heads': score(record['after_state']['mean'][selected], truth[selected]),
                'objective_weight_interventions': {k: score(v[selected], truth[selected, 0]) for k, v in record['predictions'].items()}}
        scored.append(row)
    final_scores = {}
    for name, global_indices in groups.items():
        selected = [local[i] for i in global_indices]
        final_scores[name] = {'global_indices': global_indices,
            'initial_all_heads': score(initial_state['mean'][selected], truth[selected]),
            'final_all_heads': score(state1['mean'][selected], truth[selected]),
            'objective_weight_interventions': {k: score(v[selected], truth[selected, 0]) for k, v in final_predictions.items()}}
    return {'task_index': index, 'source_results_completed_at': completed['completed_at'],
        'subset_global_indices': indices, 'head_order': ['f', 'eta_f', 'g', 'eta_g'],
        'identity_names': identity_names,
        'restoration': restoration, 'subset_parity': checks, 'rounds': scored,
        'initial_to_final': final_scores, 'cumulative_online_identity_weights': {k: v.tolist() for k, v in final_weights.items()},
        'cost': {'cpu_seconds': time.process_time()-started, 'replayed_saved_episodes': 4,
                 'new_simulator_calls': 0, 'source_fits': 0, 'truth_scored_points': len(indices)},
        'interpretation': 'All metrics use the same final-unobserved generated points per case. Within-identity changes include native f conditioning and internal positive-component reweighting. Head-only identity interventions follow the saved query/raw path and retain all actual f field updates; g-only weights do not mean only g was observed. Mean-shift terms are additive, intervention MSE differences are not. No new selection is made.'}


def run(case_index=None):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ledger = OUTPUT/'attempts.jsonl'
    attempts = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    remaining = 120-sum(a['cpu_seconds'] for a in attempts)
    if remaining <= 0:
        raise RuntimeError('Diagnostic total CPU budget exhausted')
    started = time.process_time(); attempt = {'requested_case': case_index}
    def timeout(_sig, _frame):
        raise TimeoutError('Diagnostic total CPU budget exhausted')
    previous = signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_PROF, remaining)
    try:
        completed = read(SOURCE/'results.json')
        assert completed['all_recommendations_frozen_before_verification'] and completed['all_verification_finished_before_truth_scoring']
        assert len(completed['rows']) == 16
        frozen = read(SOURCE/'frozen_inputs.json')
        for index in range(4) if case_index is None else [case_index]:
            path = OUTPUT/f'task{index:02d}.json'
            if not path.exists():
                result = run_case(index, frozen, completed)
                save(path, result)
                print(json.dumps({'task_index': index, 'subset': len(result['subset_global_indices']),
                    'max_parity_error': max(max(r[k] for k in ('mean', 'point_covariance', 'identity_weights')) for r in result['subset_parity']),
                    'cpu_seconds': result['cost']['cpu_seconds']}), flush=True)
        attempt['status'] = 'completed'
    except Exception:
        attempt.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0); signal.signal(signal.SIGPROF, previous)
        attempt.update(cpu_seconds=time.process_time()-started, maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with ledger.open('a') as stream:
            stream.write(json.dumps(attempt)+'\n')
    paths = [OUTPUT/f'task{i:02d}.json' for i in range(4)]
    if all(path.exists() for path in paths):
        cases = [read(path) for path in paths]
        save(OUTPUT/'results.json', {'all_four_saved_cases_replayed': True, 'case_files': [str(p) for p in paths],
            'new_simulator_calls': 0, 'source_fits': 0,
            'total_attempt_cpu_seconds': sum(a['cpu_seconds'] for a in [*attempts, attempt]),
            'max_subset_parity_error': max(r[k] for c in cases for r in c['subset_parity'] for k in ('mean', 'point_covariance', 'identity_weights')),
            'final_all_generated': [{'task_index': c['task_index'], **c['initial_to_final']['all_generated_unobserved']} for c in cases]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-index', type=int, choices=range(4))
    run(parser.parse_args().case_index)
