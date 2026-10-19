"""Own-point-only filtering on the completed full model's observation paths.

Restore at most six policies per task. Check the saved initialization and the
actual no-cross path before diagnosing the fixed full path. No simulation,
source fitting, query optimizer or terminal verifier is invoked.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.guarded_deployment_kg import conditional_safety_power
from performance.mainline_cross_dimension_prior import restore_cross_dimension_prior
from performance.mainline_joint_risk_source import apply_joint_risk_source
from performance.mainline_moment_policy import moment_state, terminal_action
from performance.mainline_point_only import PointOnlyResponseMixture
from performance.mainline_propagation import point_covariance
from performance.run_legacy_related_inventory_source import problem_for

OUTPUT = ROOT/'paper_artifacts/mainline_d1000_no_cross_v1'
FULL = 'source_structural_mixture'
CONTROL = 'source_structural_no_cross'
POLICY = 'fixed_two_round_moment_kg'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


def parity(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0.)
    return float(np.max(abs(actual-expected)))


def fill_terminal_state(initial, own, local_indices, updated):
    mean, point_cov = (np.asarray(initial[k]).copy() for k in ('mean', 'point_cov'))
    local_state = own.moment_state()
    local_cov = point_covariance(local_state)
    for index in set(updated):
        local = local_indices[index]
        mean[index], point_cov[index] = local_state['mean'][local], local_cov[local]
    count = len(mean)
    covariance = np.zeros((count, 4, count, 4))
    indices = np.arange(count)
    covariance[indices, :, indices, :] = point_cov
    return {'mean': mean, 'cov': covariance}


def conditional_score(primary, task_index, frozen, completed):
    for row in completed['rows']:
        if (row['task_index'] == task_index and row['action']['primary'] == primary
                and row['action']['budgets'] == [358]):
            return {'conditional_safe_reward': row['scores']['conditional_safe_reward'],
                'true_feasible': row['scores']['true_feasible'], 'quality': row['scores']['quality'],
                'score_source': 'existing358_episode_recommendation_conditional_score',
                'existing_arm': row['arm'], 'new_truth_policy_evaluations': 0}
    problem = problem_for(frozen['problem_plan'], frozen['tasks'][task_index])
    point, protocol = tuple(frozen['bank'][primary]), frozen['terminal_protocol']
    objective = problem.true_objective(point)
    centered_g = problem.true_constraint_mean(point)-problem.tau
    sigma = problem.true_sigma(point)[1]
    feasible = bool(problem.is_truly_feasible(point))
    quality = float(np.clip(1-objective/protocol['objective_reference'], 0, 1))
    power = float(conditional_safety_power(centered_g, sigma, 358,
        protocol['chance_alpha'], protocol['familywise_delta']))
    return {'conditional_safe_reward': float(feasible*quality*power),
        'true_feasible': feasible, 'quality': quality, 'true_certificate_power': power,
        'score_source': 'post_verification_analytic_truth_score', 'new_truth_policy_evaluations': 1}


def replay_task(task_index, frozen, completed, reuse, pilot, coordinates):
    started = time.process_time()
    folder = OUTPUT/f'task{task_index:02d}'
    full = read(folder/FULL/'case.json'); control = read(folder/CONTROL/'case.json')
    paid = read(folder/'initialization.json')
    full_path = [r['decision']['query'] for r in full['rounds']]
    control_path = [r['decision']['query'] for r in control['rounds']]
    indices = list(dict.fromkeys(paid['candidate_bank_indices']+full_path+control_path))
    assert len(indices) <= 6
    local_indices = {q: i for i, q in enumerate(indices)}
    subset = {key: np.asarray(coordinates[key])[indices] for key in ('mean', 'risk')}
    model, restoration = restore_cross_dimension_prior(reuse, pilot, frozen['plan']['layout'], FULL, subset)
    apply_joint_risk_source(model, subset, frozen['source_joint_risk_fit'])
    for query, raw in zip(paid['candidate_bank_indices'], paid['paired_observations']):
        model.update_raw(local_indices[query], [raw[0], raw[1]-paid['tau']])
    state = moment_state(model)
    checks = {'initial_mean': parity(state['mean'], np.asarray(full['initial']['mean'])[indices]),
        'initial_point_cov': parity(point_covariance(state), np.asarray(full['initial']['point_cov'])[indices]),
        'initial_identity_weights': parity(model.weights, full['initial']['identity_weights'])}
    own_control = PointOnlyResponseMixture.from_model(model)
    own_full_path = PointOnlyResponseMixture.from_model(model)
    del model
    for number, row in enumerate(control['rounds']):
        query, raw = row['decision']['query'], row['raw_observation']
        own_control.update_raw(local_indices[query], [raw[0], raw[1]-paid['tau']])
        after = own_control.moment_state()
        checks[f'control_round{number+1}_mean'] = parity(after['mean'], np.asarray(row['after']['mean'])[indices])
        checks[f'control_round{number+1}_point_cov'] = parity(
            point_covariance(after), np.asarray(row['after']['point_cov'])[indices])
        checks[f'control_round{number+1}_identity_weights'] = parity(
            own_control.weights, np.asarray(row['after']['identity_weights'])[indices])
    actual_control = next(r for r in completed['rows'] if
        r['task_index'] == task_index and r['arm'] == CONTROL and r['policy'] == POLICY)
    actual_full = next(r for r in completed['rows'] if
        r['task_index'] == task_index and r['arm'] == FULL and r['policy'] == POLICY)
    protocol = frozen['terminal_protocol']
    control_terminal = terminal_action(fill_terminal_state(full['initial'], own_control, local_indices, control_path), 358, protocol)
    checks['control_final_utilities'] = parity(control_terminal['utilities'], actual_control['utilities'])
    assert control_terminal['action'] == actual_control['action']
    del own_control
    for row in full['rounds']:
        query, raw = row['decision']['query'], row['raw_observation']
        own_full_path.update_raw(local_indices[query], [raw[0], raw[1]-paid['tau']])
    assert own_full_path.paid_episodes == 4
    decision = terminal_action(fill_terminal_state(full['initial'], own_full_path, local_indices, full_path), 358, protocol)
    primary = decision['action']['primary']
    scores = conditional_score(primary, task_index, frozen, completed)
    base_reward = actual_control['scores']['conditional_safe_reward']
    full_reward = actual_full['scores']['conditional_safe_reward']
    replay_reward = scores['conditional_safe_reward']
    return {'task_index': task_index, 'task_id': frozen['tasks'][task_index]['task_id'],
        'subset_global_indices': indices, 'restored_policy_count': len(indices),
        'parity_max_errors': checks, 'full_query_path': full_path, 'control_query_path': control_path,
        'control_actual_primary': actual_control['action']['primary'],
        'full_actual_primary': actual_full['action']['primary'],
        'full_path_own_only_primary': primary,
        'full_path_own_only_primary_is_sampled': primary in paid['candidate_bank_indices']+full_path,
        'full_path_own_only_scores': scores,
        'full_path_own_only_terminal': decision,
        'conditional_rewards': {'control_actual_path': base_reward, 'own_only_on_full_path': replay_reward,
            'full_actual_path': full_reward},
        'ordered_differences': {'change_path_with_own_only_updates': replay_reward-base_reward,
            'enable_propagation_on_fixed_full_path': full_reward-replay_reward,
            'total_full_minus_control': full_reward-base_reward},
        'cost': {'cpu_seconds': time.process_time()-started, 'saved_raw_observation_updates': 6,
            'new_simulator_calls': 0, 'new_verifier_calls': 0, 'source_fits': 0,
            'query_optimizer_calls': 0, 'new_truth_policy_evaluations': scores['new_truth_policy_evaluations']}}


def run():
    started = time.process_time()
    attempt = {'status': 'failed', 'new_simulator_calls': 0, 'new_verifier_calls': 0, 'source_fits': 0}
    try:
        frozen, completed = read(OUTPUT/'frozen_inputs.json'), read(OUTPUT/'results.json')
        assert completed['all_recommendations_frozen_before_verification']
        assert completed['all_verification_finished_before_truth_scoring']
        spec = frozen['plan']; reuse = read(ROOT/spec['reuse_plan'])
        pilot = read(ROOT/read(ROOT/reuse['independent_execution_plan'])['pilot_plan'])
        coordinates = read(ROOT/spec['query_coordinates'])
        rows = []
        for task_index in range(4):
            row = replay_task(task_index, frozen, completed, reuse, pilot, coordinates)
            rows.append(row); attempt['completed_tasks'] = len(rows)
            print(task_index, row['full_path_own_only_primary'], row['conditional_rewards'], flush=True)
        result = {'source_results': str(OUTPUT.relative_to(ROOT)/'results.json'),
            'diagnostic_kind': 'Ordered path-fixed replay after completed five-arm verification; not another verified optimizer arm.',
            'interpretation': 'First replace the no-cross query/observation path by the saved full-model path while retaining own-point updates; then enable the full propagation channels on that same path. The two ordered differences sum to the full-minus-control conditional reward. They are path-dependent contrasts, not independent causal shares; no new adaptive queries or verifier outcomes are generated.',
            'rows': rows,
            'mean_conditional_rewards': {key: float(np.mean([r['conditional_rewards'][key] for r in rows]))
                for key in rows[0]['conditional_rewards']},
            'mean_ordered_differences': {key: float(np.mean([r['ordered_differences'][key] for r in rows]))
                for key in rows[0]['ordered_differences']},
            'maximum_parity_error': max(max(r['parity_max_errors'].values()) for r in rows),
            'cost': {'cpu_seconds': time.process_time()-started, 'saved_raw_observation_updates': 24,
                'new_simulator_calls': 0, 'new_verifier_calls': 0, 'source_fits': 0, 'query_optimizer_calls': 0,
                'new_truth_policy_evaluations': sum(r['cost']['new_truth_policy_evaluations'] for r in rows)}}
        save(OUTPUT/'path_replay.json', result)
        attempt['status'] = 'completed'
        return result
    except Exception as error:
        attempt['exception'] = repr(error)
        raise
    finally:
        attempt['cpu_seconds'] = time.process_time()-started
        with (OUTPUT/'path_replay_attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(attempt)+'\n')


if __name__ == '__main__':
    run()
