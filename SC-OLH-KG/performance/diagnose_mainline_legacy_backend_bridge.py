"""Saved-history posterior replay, gated on all independent verifications."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from acquisition.decision_backends import feasible_first_terminal_order
from performance.mainline_legacy_source_models import restore_moment_state
from performance.run_mainline_legacy_backend_bridge import (
    MANIFEST, OUTPUT, load_pickle, read, require_all, save,
)


def terminal_action(state, indices, guard, alpha, threshold):
    mean = state['mean'][indices]
    variance = np.array([state['cov'][i, 2, i, 2] for i in indices])
    margin = mean[:, 2] + guard + norm.ppf(1-alpha)*np.sqrt(state['noise'][indices, 1])
    violation = norm.cdf(margin/np.sqrt(np.maximum(variance, 1e-12)))
    order, mode = feasible_first_terminal_order(mean[:, 0], violation,
                                               maximum_violation_probability=threshold)
    selected = int(order[0])
    return {'bank_index': int(indices[selected]), 'mode': mode,
            'objective_mean': float(mean[selected, 0]),
            'constraint_mean_centered': float(mean[selected, 2]),
            'posterior_violation_probability': float(violation[selected])}


def replay_case(output, family, arm, cost):
    directory = output/family/arm
    search = read(directory/'search.json')
    engine = load_pickle(directory/'search_engine.pkl')
    bundle = load_pickle(output/'source_models'/family/'source_models.pkl')
    history = search['history']
    bank = list(dict.fromkeys(tuple(row['point']) for row in history))
    lookup = {point: i for i, point in enumerate(bank)}
    initial = list(dict.fromkeys(lookup[tuple(row['point'])] for row in history[:10]))
    seen11 = {tuple(row['point']) for row in history[:11]}
    unseen = list(dict.fromkeys(lookup[tuple(row['point'])] for row in history[11:13]
                               if tuple(row['point']) not in seen11))
    model = {'same_source_gp': 'source_common_task_rbf',
             'current_structural_mixture': 'source_structural_mixture'}[arm]
    states = {}
    for count in (10, 11, 13):
        started = time.process_time()
        states[count] = restore_moment_state(bundle, model, engine.problem, bank, history[:count])
        cost['posterior_replay_cpu_seconds'][str(count)] = time.process_time()-started
        if states[count]['paid_episodes'] != count:
            raise AssertionError('Replay paid episode count differs')
    guard = max(float(engine.problem.pilot_constraint_guard()), 0.) if hasattr(
        engine.problem, 'pilot_constraint_guard') else 0.
    threshold = float(engine.config.decision_terminal_maximum_violation_probability)
    assert threshold == .5
    actions = {'initial': terminal_action(states[10], initial, guard, engine.problem.alpha, threshold),
               'final': terminal_action(states[13], list(range(len(bank))), guard,
                                        engine.problem.alpha, threshold)}
    if list(bank[actions['final']['bank_index']]) != search['primary']:
        raise AssertionError('Final replay action does not match frozen native primary')
    truth = {}

    def true_fg(index):
        if index not in truth:
            cost['truth_function_calls'] += 1
            f = float(engine.problem.true_objective(bank[index]))
            cost['truth_function_calls'] += 1
            g = float(engine.problem.true_constraint_mean(bank[index]))-engine.problem.tau
            truth[index] = np.array([f, g])
        return truth[index]

    for action in actions.values():
        index = action['bank_index']
        action['first_history_index'] = next(i for i, row in enumerate(history)
                                             if tuple(row['point']) == bank[index])
        action['true_f_g_centered'] = true_fg(index).tolist()
    propagation = {'eligible_count': len(unseen), 'bank_indices': unseen,
        'scope': 'Own later query policies still unobserved after the first post-init raw; retrospective fixed set.'}
    if unseen:
        target = np.array([true_fg(index) for index in unseen])
        before = states[10]['mean'][unseen][:, [0, 2]]
        after = states[11]['mean'][unseen][:, [0, 2]]
        propagation.update(truth_f_g_centered=target.tolist(), before=before.tolist(),
            after=after.tolist(), mean_change=(after-before).tolist(),
            before_bias=np.mean(before-target, axis=0).tolist(),
            after_bias=np.mean(after-target, axis=0).tolist(),
            before_mse=np.mean((before-target)**2, axis=0).tolist(),
            after_mse=np.mean((after-target)**2, axis=0).tolist())
    return {'family': family, 'arm': arm, 'actual_unique_observed_policy_count': len(bank),
            'history_bank_indices': [lookup[tuple(row['point'])] for row in history],
            'first_post_init_query_new': tuple(history[10]['point']) not in {
                tuple(row['point']) for row in history[:10]},
            'pilot_constraint_guard': guard, 'terminal_threshold': threshold,
            'initial_and_final_actions': actions, 'saved_primary_matches': True,
            'strict_unobserved_10_to_11': propagation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--family')
    args = parser.parse_args()
    plan = read(MANIFEST)
    if args.family and args.family not in plan['domains']:
        parser.error('--family must be one of the registered domains')
    stem = 'diagnostic_replay' + (f'_{args.family}' if args.family else '')
    result = {'status': 'running', 'rows': [], 'interpretation':
        'Cross-policy mean changes exclude self-observation. Their retrospective MSE is descriptive; '
        'initial-to-final action changes also include new candidates and their own raw updates.'}
    ledger = {'status': 'failed', 'cpu_seconds': None, 'new_source_calls': 0,
              'new_target_simulator_calls': 0, 'new_verification_calls': 0,
              'source_fits': 0, 'truth_function_calls': 0, 'cases': []}
    started = time.process_time()
    try:
        require_all(args.output, plan, 'verification.json')
        for family in ([args.family] if args.family else plan['domains']):
            for arm in ('same_source_gp', 'current_structural_mixture'):
                cost = {'family': family, 'arm': arm, 'status': 'failed',
                        'truth_function_calls': 0, 'posterior_replay_cpu_seconds': {}}
                case_start = time.process_time()
                try:
                    result['rows'].append(replay_case(args.output, family, arm, cost))
                    cost['status'] = 'completed'
                finally:
                    cost['cpu_seconds'] = time.process_time()-case_start
                    ledger['truth_function_calls'] += cost['truth_function_calls']
                    ledger['cases'].append(cost)
                    save(args.output/f'{stem}.json', result)
        ledger['status'] = result['status'] = 'completed'
    except BaseException as error:
        ledger['error'] = repr(error)
        result['status'] = 'failed'
        raise
    finally:
        ledger['cpu_seconds'] = time.process_time()-started
        result['cost'] = ledger
        save(args.output/f'{stem}.json', result)
        with (args.output/f'{stem}_attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(ledger, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'cases': len(result['rows']),
                      'cpu_seconds': ledger['cpu_seconds'], 'truth_calls': ledger['truth_function_calls']}))


if __name__ == '__main__':
    main()
