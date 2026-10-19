"""Post-verification fixed-history propagation and terminal-noise diagnosis."""
import json
import time
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from performance.diagnose_mainline_legacy_backend_bridge import terminal_action
from performance.mainline_legacy_backend import restore_point_only_noise
from performance.mainline_legacy_source_models import restore_model
from performance.mainline_moment_policy import moment_state
from performance.run_mainline_legacy_backend_bridge import ROOT, load_pickle, read, save
from performance.run_mainline_local_noise_control import (
    FULL, LOCAL, MANIFEST, case_dir, require_all,
)


def replay_seed(output, plan, seed, bundle, truth, extra, cost):
    searches = {arm['name']: read(case_dir(output, seed, arm['name'])/'search.json')
                for arm in plan['arms']}
    history = searches[FULL]['history']
    if history != searches[LOCAL]['history']:
        raise AssertionError('Paired arms have different search histories')
    bank, provenance = [], []
    for arm, search in searches.items():
        for raw_index, row in enumerate(search['history']):
            point = tuple(row['point'])
            if point not in bank:
                bank.append(point)
                provenance.append({'arm': arm, 'history_index': raw_index})
    lookup = {point: i for i, point in enumerate(bank)}
    observed = [lookup[tuple(row['point'])] for row in history]
    unseen = sorted(set(range(len(bank)))-set(observed))
    engine = load_pickle(case_dir(output, seed, FULL)/'search_engine.pkl')
    problem = engine.problem
    action_pool = engine._terminal_action_pool(engine._last_terminal_pool,
                                               observations=engine.observations)
    action_ids = [lookup[tuple(point)] for point in action_pool]
    guard = max(float(problem.pilot_constraint_guard()), 0.) if hasattr(
        problem, 'pilot_constraint_guard') else 0.
    threshold = float(engine.config.decision_terminal_maximum_violation_probability)
    states = {}
    with patch.object(problem, 'simulate', side_effect=AssertionError('Replay cannot simulate')):
        model = restore_model(bundle, 'source_structural_mixture', problem, bank)
        for step, row in enumerate(history, start=1):
            model.update_raw(lookup[tuple(row['point'])], [row['raw'][0], row['raw'][1]-problem.tau])
            if step in (10, 13):
                states[step] = moment_state(model)
        del model
        # Only observed terminal candidates need their own noise marginal.
        local_noise = restore_point_only_noise(bundle, problem, action_pool, history)
    full = states[13]
    local = {**full, 'noise': full['noise'].copy()}
    local['noise'][action_ids, 1] = local_noise[:, 1]
    actions = {name: terminal_action(state, action_ids, guard, problem.alpha, threshold)
               for name, state in [(FULL, full), (LOCAL, local)]}
    for name, action in actions.items():
        if list(bank[action['bank_index']]) != searches[name]['primary']:
            raise AssertionError(f'{seed} {name}: replay does not reproduce frozen primary')

    def get_truth(index, noise=False):
        point = bank[index]
        value = dict(truth.get(point, {}))
        changed = False
        if 'objective' not in value:
            value['objective'] = float(problem.true_objective(point))
            cost['truth_function_calls'] += 1
            value['constraint_mean_centered'] = float(problem.true_constraint_mean(point))-problem.tau
            cost['truth_function_calls'] += 1
            changed = True
        if noise and 'constraint_sigma' not in value:
            value['constraint_sigma'] = float(problem.true_sigma(point)[1])
            cost['truth_function_calls'] += 1
            changed = True
        truth[point] = value
        if changed:
            extra[point] = value
            save(output/'diagnostic_truth.json', [{'point': list(k), 'truth': v} for k, v in extra.items()])
        return value

    details = []
    selected = sorted({0, *(action['bank_index'] for action in actions.values())})
    for index in selected:
        epistemic = float(full['cov'][index, 2, index, 2])
        variants = {}
        for name, state in [(FULL, full), (LOCAL, local)]:
            margin = float(state['mean'][index, 2]+guard
                           +norm.ppf(1-problem.alpha)*np.sqrt(state['noise'][index, 1]))
            z = margin/np.sqrt(max(epistemic, 1e-12))
            variants[name] = {'constraint_episode_variance': float(state['noise'][index, 1]),
                'posterior_violation_probability': float(norm.cdf(z)), 'standardized_margin': float(z)}
        details.append({'bank_index': index, 'first_bank_occurrence': provenance[index],
            'structure_raw_history_indices': [i for i, value in enumerate(observed) if value == index],
            'objective_mean': float(full['mean'][index, 0]),
            'constraint_mean_centered': float(full['mean'][index, 2]),
            'constraint_epistemic_variance': epistemic,
            'objective_episode_variance': float(full['noise'][index, 0]),
            'terminal_variants': variants, 'truth': get_truth(index, noise=True)})
    propagation = {'bank_indices': unseen, 'policy_count': len(unseen),
        'scope': 'Fixed union of all arms histories; these policies have no structural-arm raw at any step.'}
    if unseen:
        target = np.array([[get_truth(i)['objective'], get_truth(i)['constraint_mean_centered']] for i in unseen])
        for count in (10, 13):
            prediction = states[count]['mean'][unseen][:, [0, 2]]
            propagation[str(count)] = {'mean_bias_f_g': np.mean(prediction-target, axis=0),
                'mse_f_g': np.mean((prediction-target)**2, axis=0)}
        before = np.asarray(propagation['10']['mse_f_g'])
        after = np.asarray(propagation['13']['mse_f_g'])
        propagation['mse_reduction_f_g'] = before-after
        propagation['individual_policies'] = [{'bank_index': i,
            'first_bank_occurrence': provenance[i], 'truth_f_g': target[j],
            'mean10_f_g': states[10]['mean'][i, [0, 2]],
            'mean13_f_g': states[13]['mean'][i, [0, 2]]} for j, i in enumerate(unseen)]
    return {'seed': seed, 'bank_policy_count': len(bank), 'bank_first_occurrences': provenance,
        'structure_history_bank_indices': observed, 'terminal_action_bank_order': action_ids,
        'saved_primary_parity': True, 'same_search_history': True, 'actions': actions,
        'point_diagnostics': details, 'never_sampled_prediction_change': propagation}


def main():
    plan = read(MANIFEST)
    output = ROOT/plan['output_dir']
    require_all(output, plan, 'verification.json')
    destination = output/'diagnostic_replay.json'
    previous = read(destination) if destination.exists() else {}
    if previous.get('status') == 'complete':
        print('Reusing completed diagnostic replay', flush=True)
        return
    result = {'status': 'running', 'rows': previous.get('rows', []),
        'interpretation': 'Descriptive fixed-bank prediction changes; terminal noise pairing does not identify a causal benefit of mean propagation.'}
    cost = {'status': 'failed', 'cpu_seconds': None, 'truth_function_calls': 0,
            'new_simulator_calls': 0, 'new_source_fits': 0, 'download_bytes': 0, 'cases': []}
    started = time.process_time()
    try:
        truth = {tuple(row['point']): row['truth'] for row in read(output/'truth.json')}
        extra_path = output/'diagnostic_truth.json'
        extra = {tuple(row['point']): row['truth'] for row in read(extra_path)} if extra_path.exists() else {}
        truth.update(extra)
        bundle = load_pickle(ROOT/plan['frozen_assets']['source_models_file'])
        complete = {row['seed'] for row in result['rows']}
        for seed in plan['target_seeds']:
            if seed in complete:
                continue
            case_started = time.process_time()
            case = {'seed': seed, 'status': 'failed', 'truth_function_calls': 0}
            try:
                result['rows'].append(replay_seed(output, plan, seed, bundle, truth, extra, case))
                case['status'] = 'complete'
            finally:
                case['cpu_seconds'] = time.process_time()-case_started
                cost['truth_function_calls'] += case['truth_function_calls']
                cost['cases'].append(case)
                save(destination, result)
            print('diagnosed', seed, flush=True)
        result['status'] = cost['status'] = 'complete'
    except BaseException as error:
        cost['error'] = repr(error)
        result['status'] = 'failed'
        raise
    finally:
        cost['cpu_seconds'] = time.process_time()-started
        result['cost'] = cost
        save(destination, result)
        with (output/'diagnostic_attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(cost, allow_nan=False)+'\n')
    print(json.dumps(cost), flush=True)


if __name__ == '__main__':
    main()
