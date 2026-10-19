"""Fixed two-round mechanism pilots, reusing source fits and real verifier."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.guarded_deployment_kg import conditional_safety_power
from performance.historical_common_bank import initialization_batches, replay_raw_batches, restore_zero_paid_prior
from performance.historical_full_bank_development import context, read
from performance.mainline_moment_policy import moment_state, terminal_action, query_action, available_state
from performance.mainline_propagation import snapshot, propagation
from performance.run_legacy_one_step_closed_loop import execute_terminal
from performance.run_legacy_related_inventory_source import problem_for, task_configs

PLAN = 'performance/manifests/mainline_two_round73_v1.json'
NO_CROSS = 'source_structural_no_cross'
DUAL_CHANNEL = 'source_structural_dual_channel'
FROZEN_QUERY = 'source_structural_frozen_query'


def base_model(arm):
    return 'source_structural_mixture' if arm in (NO_CROSS, DUAL_CHANNEL, FROZEN_QUERY) else arm


def policies_for(spec, arm):
    return spec.get('policies_by_model', {}).get(arm, spec['policies'])


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load(plan=PLAN):
    spec = read(plan)
    out = ROOT / spec['output']
    reuse = read(spec['reuse_plan'])
    return spec, out, reuse


def prepare(plan=PLAN):
    spec, out, reuse = load(plan)
    assert not ('source_joint_risk_fit_file' in spec and 'source_risk_correction_file' in spec), \
        'Joint source risk fitting replaces the previous residual correction'
    pilot, _, states, public, _, _, bank = context(reuse)
    if 'cross_dimension_inputs' in spec:
        bridge = read(spec['cross_dimension_inputs'])
        bank = bridge['bank']
        public = dict(public, source_dimension=spec['target_dimension'])
    assert len(bank) == spec['candidate_count']
    templates = {arm: next(s for s in states if s['layout'] == spec['layout'] and s['arm'] == base_model(arm))
                 for arm in spec['models']}
    protocol = templates[spec['models'][0]]['terminal_protocol']
    assert all(s['terminal_protocol'] == protocol for s in templates.values())
    task_plan = dict(public, task_seed=spec['task_seed'], source_task_count=spec['task_count'])
    tasks = spec.get('target_tasks')
    if tasks is None:
        tasks = task_configs(task_plan)
        for i, task in enumerate(tasks):
            task['task_id'] = f'mainline_20261015_{i:02d}'
    assert len(tasks) == spec['task_count']
    assert all(len(point) == public['source_dimension'] for point in bank)
    frozen = {'plan': spec, 'tasks': tasks, 'problem_plan': public, 'bank': bank,
              'terminal_protocol': protocol, 'source_training_tasks': pilot['training_tasks']}
    if 'source_risk_correction_file' in spec:
        frozen['source_risk_correction'] = read(spec['source_risk_correction_file'])
    if 'source_joint_risk_fit_file' in spec:
        calibration = read(spec['source_joint_risk_fit_file'])
        assert calibration['source_importance_usable'] and len(calibration['rows']) == 2
        frozen['source_joint_risk_fit'] = calibration
    if 'cross_dimension_inputs' in spec and 'dynamic_candidates' in bridge:
        frozen['dynamic_candidates'] = bridge['dynamic_candidates']
    path = out / 'frozen_inputs.json'
    if path.exists():
        assert read(str(path)) == frozen, 'Existing observations belong to a different frozen plan'
    else:
        save(path, frozen)
    cached_inputs = (read(spec['initialization_cache'] + '/frozen_inputs.json')
                     if 'initialization_cache' in spec else None)
    for i, task in enumerate(tasks):
        path = out / f'task{i:02d}' / 'initialization.json'
        if path.exists():
            continue
        if cached_inputs is not None:
            assert cached_inputs['problem_plan'] == public
            old_task = cached_inputs['tasks'][i]
            assert all(old_task[k] == task[k] for k in ('family', 'problem_kwargs'))
            source_path = spec['initialization_cache'] + f'/task{i:02d}/initialization.json'
            cached = read(source_path)
            assert cached['candidate_bank_indices'] == spec['initialization_indices']
            assert all(cached_inputs['bank'][q] == bank[q] for q in spec['initialization_indices'])
            save(path, dict(cached, task_id=task['task_id'], physical_calls=0,
                            reused_initialization_calls=len(cached['paired_observations']),
                            initialization_source=source_path))
            continue
        problem = problem_for(public, task)
        rng = np.random.default_rng(np.random.SeedSequence([spec['initialization_seed'], i]))
        values = [problem.simulate(tuple(bank[q]), rng).tolist() for q in spec['initialization_indices']]
        save(path, {'task_id': task['task_id'], 'candidate_bank_indices': spec['initialization_indices'],
                    'paired_observations': values, 'physical_calls': len(values), 'tau': problem.tau})
    print(f'PREPARED {len(tasks)} evaluation units; ' + ('cached initialization replay, zero new calls'
          if cached_inputs is not None else
          f'{len(tasks) * len(spec["initialization_indices"])} shared initialization calls'), flush=True)


def run_case(case_index, plan=PLAN):
    spec, out, reuse = load(plan)
    frozen = read(str(out / 'frozen_inputs.json'))
    assert frozen['plan'] == spec
    task_index, model_index = divmod(case_index, len(spec['models']))
    arm = spec['models'][model_index]
    source_arm = base_model(arm)
    no_cross = arm == NO_CROSS
    frozen_query = arm == FROZEN_QUERY
    dual_channel = arm in (DUAL_CHANNEL, FROZEN_QUERY)
    point_terminal = no_cross or dual_channel
    folder = out / f'task{task_index:02d}' / arm
    path = folder / 'case.json'
    if path.exists():
        return read(str(path))
    pilot, _, states, _, connection, old, _ = context(reuse)
    template = next(s for s in states if s['layout'] == spec['layout'] and s['arm'] == source_arm)
    if 'cross_dimension_inputs' in spec:
        from performance.mainline_cross_dimension_prior import restore_cross_dimension_prior
        coordinates = read(spec['query_coordinates'])
        model, restore = restore_cross_dimension_prior(reuse, pilot, spec['layout'], source_arm, coordinates)
        if source_arm == 'source_structural_mixture':
            if 'source_joint_risk_fit' in frozen:
                from performance.mainline_joint_risk_source import apply_joint_risk_source
                restore['source_joint_risk_fit'] = {
                    'calibration_file': spec['source_joint_risk_fit_file'],
                    **apply_joint_risk_source(model, coordinates, frozen['source_joint_risk_fit'])}
            elif 'source_risk_correction' in frozen:
                from performance.mainline_risk_correction import apply_public_mean_correction
                restore['source_risk_correction'] = apply_public_mean_correction(
                    model, coordinates, frozen['source_risk_correction'])
    else:
        model, restore = restore_zero_paid_prior(reuse, pilot, template, connection, old, 73)
    paid = read(str(out / f'task{task_index:02d}' / 'initialization.json'))
    replay_raw_batches(model, initialization_batches(paid), paid['tau'])
    problem = problem_for(frozen['problem_plan'], frozen['tasks'][task_index])
    bank = [tuple(p) for p in frozen['bank']]
    protocol = frozen['terminal_protocol']
    state = moment_state(model)
    selection_state = state
    if point_terminal:
        from performance.mainline_point_only import PointOnlyResponseMixture
        from performance.mainline_propagation import point_covariance
        initialized = state
        if frozen_query:
            from performance.mainline_dual_channel import FrozenQueryDualChannelResponseMixture
            model = FrozenQueryDualChannelResponseMixture.from_model(model)
        elif dual_channel:
            from performance.mainline_dual_channel import DualChannelResponseMixture
            model = DualChannelResponseMixture.from_model(model)
        else:
            model = PointOnlyResponseMixture.from_model(model)
        state = model.moment_state()
        errors = {}
        for key, before, after in [('mean', initialized['mean'], state['mean']),
                                   ('point_cov', point_covariance(initialized), point_covariance(state)),
                                   ('noise', initialized['noise'], state['noise'])]:
            np.testing.assert_allclose(after, before, atol=1e-12, rtol=1e-12)
            errors[key] = float(np.max(abs(after-before)))
        control_key = 'frozen_query_control' if frozen_query else ('dual_channel' if dual_channel else 'no_cross_control')
        restore[control_key] = {'factorized_after_shared_initialization': True,
            'initial_marginal_max_errors': errors,
            'identity_weights_scope': 'one independent terminal mixture per policy',
            'query_model': ('frozen initialized full coupled moments' if frozen_query else
                            ('full coupled model' if dual_channel else 'point-only model'))}
        del initialized
    initial = snapshot(state, model.weights)
    stopped = terminal_action(state, spec['total_target_call_cap'] - model.paid_episodes, protocol)
    rows, rounds = [], []
    observed = list(spec['initialization_indices'])
    seed = int(np.random.SeedSequence([spec['verification_seed'], task_index]).generate_state(1)[0])
    common = {'task_index': task_index, 'task_id': frozen['tasks'][task_index]['task_id'],
              'layout': spec['layout'], 'arm': arm, 'verification_seed': seed}
    if spec['policies'][0] in policies_for(spec, arm):
        rows.append({**common, 'policy': spec['policies'][0], 'search_calls': model.paid_episodes,
                     'observed_indices': observed.copy(), 'snapshot': initial, **stopped})
    for r in range(spec['rounds']):
        started = time.process_time()
        available = frozen.get('dynamic_candidates', {}).get('round_end_indices', [len(bank)] * spec['rounds'])[r]
        query_state = selection_state if dual_channel else state
        if dual_channel:
            selection_weights_before = model.selection_weights.copy()
        decision = query_action(available_state(query_state, available), spec['total_target_call_cap'] - model.paid_episodes, protocol)
        decision['available_candidate_count'] = available
        query = decision['query']
        observation_path = folder / f'round{r + 1}_observation.json'
        if observation_path.exists():
            record = read(str(observation_path))
            assert record['decision']['query'] == query, 'Saved observation is for a different query'
        else:
            rng = np.random.default_rng(np.random.SeedSequence([spec['search_seed'], task_index, r, query]))
            raw = problem.simulate(bank[query], rng).tolist()
            record = {'round': r + 1, 'decision': decision, 'raw_observation': raw, 'physical_calls': 1}
            save(observation_path, record)
        before, weights_before = state, model.weights.copy()
        raw = record['raw_observation']
        evidence = model.update_raw(query, [raw[0], raw[1] - problem.tau])
        state = model.moment_state() if point_terminal else moment_state(model)
        if dual_channel:
            selection_state = model.query_state()
        if frozen_query:
            frozen_changes = {key: float(np.max(abs(selection_state[key] - query_state[key])))
                              for key in ('mean', 'cov', 'noise')}
            frozen_changes['identity_weights'] = float(np.max(abs(model.selection_weights - selection_weights_before)))
            assert max(frozen_changes.values()) == 0., 'Frozen query state changed after an observation'
        remaining = spec['total_target_call_cap'] - model.paid_episodes
        terminal = terminal_action(available_state(state, available), remaining, protocol)
        # A terminal-only mechanism intervention, not another optimizer arm:
        # retain the real update at observed policies; freeze unsampled marginals.
        unseen = [i for i in range(len(bank)) if i not in set(observed + [query])]
        withheld = dict(state, mean=state['mean'].copy(), cov=state['cov'].copy())
        withheld['mean'][unseen] = before['mean'][unseen]
        withheld['cov'][unseen, :, unseen, :] = before['cov'][unseen, :, unseen, :]
        withheld_terminal = terminal_action(available_state(withheld, available), remaining, protocol)
        trace = {'round': r + 1, 'decision': decision, 'raw_observation': raw, 'paid_episodes': model.paid_episodes,
                 'evidence': evidence, 'before': snapshot(before, weights_before),
                 'after': snapshot(state, model.weights),
                 'propagation': propagation(before, state, query, observed),
                 'terminal_before_same_budget': terminal_action(available_state(before, available), remaining, protocol),
                 'terminal_with_unsampled_marginals_held_fixed': withheld_terminal,
                 'terminal': terminal, 'cpu_seconds': time.process_time() - started}
        if point_terminal:
            other = np.arange(len(bank)) != query
            changes = {'mean': float(np.max(abs(state['mean'][other]-before['mean'][other]))),
                'point_cov': float(np.max(abs(point_covariance(state)[other]-point_covariance(before)[other]))),
                'noise': float(np.max(abs(state['noise'][other]-before['noise'][other]))),
                'identity_weights': float(np.max(abs(model.weights[other]-weights_before[other])))}
            assert max(changes.values()) <= 1e-12, 'Point-only update leaked to another policy'
            trace['no_cross_nonqueried_max_changes'] = changes
        if dual_channel:
            trace.update(selection_before=snapshot(query_state, selection_weights_before),
                selection_after=snapshot(selection_state, model.selection_weights),
                selection_propagation=propagation(query_state, selection_state, query, observed),
                selection_rule=('Initialized coupled moments remain frozen; KG uses the released candidates and actual remaining budget; terminal uses own-point updates.'
                                if frozen_query else 'Unchanged coupled-model moment KG proxy; terminal rule uses the point-only posterior.'))
        if frozen_query:
            trace['frozen_query_max_changes'] = frozen_changes
        if 'dynamic_candidates' in frozen:
            original_count = frozen['dynamic_candidates']['original_count']
            new_unseen = [i for i in unseen if original_count <= i < available]
            withheld['mean'][:] = state['mean']
            withheld['cov'][:] = state['cov']
            withheld['mean'][new_unseen] = before['mean'][new_unseen]
            withheld['cov'][new_unseen, :, new_unseen, :] = before['cov'][new_unseen, :, new_unseen, :]
            trace['terminal_with_new_unsampled_marginals_held_fixed'] = terminal_action(
                available_state(withheld, available), remaining, protocol)
            trace['new_unsampled_indices'] = new_unseen
        rounds.append(trace)
        save(folder / f'round{r + 1}.json', trace)
        observed.append(query)
        print(json.dumps({'case': case_index, 'round': r + 1, 'query': query,
                          'recommendation': terminal['action']['primary'], 'cpu': trace['cpu_seconds']}), flush=True)
    assert model.paid_episodes == 4
    rows.append({**common, 'policy': spec['policies'][1], 'search_calls': model.paid_episodes,
                 'observed_indices': observed, 'snapshot': snapshot(state, model.weights), **terminal})
    result = {'case_index': case_index, 'restoration': restore, 'initial': initial, 'rounds': rounds,
              'rows': rows, 'recommendations_frozen_at': timestamp(), 'new_source_fits': 0}
    save(path, result)
    return result


def assemble(plan=PLAN):
    spec, out, _ = load(plan)
    rows = []
    for i in range(spec['task_count']):
        for arm in spec['models']:
            rows.extend(read(str(out / f'task{i:02d}' / arm / 'case.json'))['rows'])
    expected = {(i, arm, policy) for i in range(spec['task_count']) for arm in spec['models']
                for policy in policies_for(spec, arm)}
    assert len(rows) == len(expected)
    assert {(r['task_index'], r['arm'], r['policy']) for r in rows} == expected
    for row in rows:
        paid = 2 if row['policy'] == spec['policies'][0] else 4
        assert row['search_calls'] == paid and row['action']['budgets'] == [362 - paid]
    save(out / 'recommendations.json', {'rows': rows, 'all_recommendations_frozen_at': timestamp()})
    print(f'FROZEN all{len(rows)} final recommendations; no verification or truth scoring yet', flush=True)


def verify(plan=PLAN):
    spec, out, _ = load(plan)
    frozen = read(str(out / 'frozen_inputs.json'))
    recommendations = read(str(out / 'recommendations.json'))
    assert len(recommendations['rows']) == spec['task_count'] * sum(
        len(policies_for(spec, arm)) for arm in spec['models'])
    bank = [tuple(p) for p in frozen['bank']]
    protocol = frozen['terminal_protocol']
    results = []
    for row in recommendations['rows']:
        path = out / f"task{row['task_index']:02d}" / row['arm'] / (row['policy'] + '_verification.json')
        if path.exists():
            verified = read(str(path))
        else:
            problem = problem_for(frozen['problem_plan'], frozen['tasks'][row['task_index']])
            simulate, count = problem.simulate, [0]
            def counted(point, rng=None):
                count[0] += 1
                return simulate(point, rng)
            problem.simulate = counted
            deployed, verification = execute_terminal(problem, bank, row['action'], row['search_calls'],
                row['verification_seed'], protocol, spec['total_target_call_cap'])
            assert count[0] == verification['verification_budget']
            assert not verification['search_samples_reused'] and not verification['posterior_updated_from_verification']
            verified = {'deployed': deployed, 'verification': verification, 'physical_calls': count[0],
                        'verified_at': timestamp()}
            save(path, verified)
        results.append({**row, **verified})
    # Truth is opened only after every adaptive decision AND every independent verifier finished.
    truths = {}
    for i, task in enumerate(frozen['tasks']):
        problem = problem_for(frozen['problem_plan'], task)
        truth = np.array([[problem.true_objective(p), np.log(problem.true_sigma(p)[0] ** 2),
                           problem.true_constraint_mean(p) - problem.tau, np.log(problem.true_sigma(p)[1] ** 2)] for p in bank])
        feasible = np.array([problem.is_truly_feasible(p) for p in bank])
        truths[i] = truth, feasible
    for row in results:
        truth, feasible = truths[row['task_index']]
        j = row['deployed']
        quality = np.clip(1 - truth[j, 0] / protocol['objective_reference'], 0, 1)
        power = conditional_safety_power(truth[j, 2], np.exp(truth[j, 3] / 2),
            row['action']['budgets'][0], protocol['chance_alpha'], protocol['familywise_delta'])
        certified = row['verification']['certified']
        row['scores'] = {'actual_safe_reward': float(certified * feasible[j] * quality),
                         'conditional_safe_reward': float(power * feasible[j] * quality),
                         'true_feasible': bool(feasible[j]), 'false_certificate': bool(certified and not feasible[j]),
                         'quality': float(quality), 'feasible_bank_count': int(feasible.sum()),
                         'recommendation_unobserved': j not in row['observed_indices']}
    prediction, conversion = [], []
    for i in range(spec['task_count']):
        truth, feasible = truths[i]
        for arm in spec['models']:
            case = read(str(out / f'task{i:02d}' / arm / 'case.json'))
            observed = list(spec['initialization_indices'])
            for trace in case['rounds']:
                observed.append(trace['decision']['query'])
                unseen = [j for j in range(len(bank)) if j not in observed]
                before = np.asarray(trace['before']['mean'])[unseen] - truth[unseen]
                after = np.asarray(trace['after']['mean'])[unseen] - truth[unseen]
                prediction.append({'task_index': i, 'arm': arm, 'round': trace['round'], 'unobserved_count': len(unseen),
                    'rmse_before_f_eta_f_g_eta_g': np.sqrt(np.mean(before ** 2, axis=0)).tolist(),
                    'rmse_after_f_eta_f_g_eta_g': np.sqrt(np.mean(after ** 2, axis=0)).tolist(),
                    'mse_change_f_eta_f_g_eta_g': np.mean(after ** 2 - before ** 2, axis=0).tolist()})
                if 'terminal_with_unsampled_marginals_held_fixed' in trace:
                    full = trace['terminal']['action']
                    held = trace['terminal_with_unsampled_marginals_held_fixed']['action']
                    def conditional_reward(action):
                        j = action['primary']
                        q = np.clip(1 - truth[j, 0] / protocol['objective_reference'], 0, 1)
                        power = conditional_safety_power(truth[j, 2], np.exp(truth[j, 3] / 2),
                            action['budgets'][0], protocol['chance_alpha'], protocol['familywise_delta'])
                        return float(feasible[j] * q * power)
                    conversion.append({'task_index': i, 'arm': arm, 'round': trace['round'],
                        'full_primary': full['primary'], 'held_unsampled_primary': held['primary'],
                        'full_primary_is_unsampled': full['primary'] in unseen,
                        'recommendation_changed_by_unsampled_updates': full['primary'] != held['primary'],
                        'full_conditional_safe_reward': conditional_reward(full),
                        'held_unsampled_conditional_safe_reward': conditional_reward(held),
                        'interpretation': 'Frozen terminal marginal intervention for this observation only; not an independently verified no-coupling optimizer.'})
    summary = []
    for arm in spec['models']:
        for policy in policies_for(spec, arm):
            subset = [r for r in results if r['arm'] == arm and r['policy'] == policy]
            summary.append({'arm': arm, 'policy': policy, 'tasks': len(subset),
                'certified': sum(r['verification']['certified'] for r in subset),
                'false_certificates': sum(r['scores']['false_certificate'] for r in subset),
                'unobserved_recommendations': sum(r['scores']['recommendation_unobserved'] for r in subset),
                'mean_actual_safe_reward': float(np.mean([r['scores']['actual_safe_reward'] for r in subset])),
                'mean_conditional_safe_reward': float(np.mean([r['scores']['conditional_safe_reward'] for r in subset]))})
    comparisons = []
    source, gp = spec['models'][:2]; stop, online = spec['policies']
    rng = np.random.default_rng(spec['bootstrap_seed'])
    samples = rng.integers(0, spec['task_count'], (spec['bootstrap_draws'], spec['task_count']))
    pairs = [((source, online), (gp, online)), ((source, online), (source, stop)), ((gp, online), (gp, stop))]
    if NO_CROSS in spec['models']:
        pairs.extend([((source, online), (NO_CROSS, online)),
                      ((NO_CROSS, online), (source, stop))])
    if DUAL_CHANNEL in spec['models']:
        pairs.extend([((DUAL_CHANNEL, online), rhs) for rhs in
            [(source, online), (NO_CROSS, online), (gp, online), (source, stop)]])
    if FROZEN_QUERY in spec['models']:
        pairs.append(((DUAL_CHANNEL, online), (FROZEN_QUERY, online)))
        pairs.extend([((FROZEN_QUERY, online), rhs) for rhs in
            [(NO_CROSS, online), (source, stop), (gp, online)]])
    for lhs, rhs in pairs:
        differences = [next(r['scores']['actual_safe_reward'] for r in results if (r['task_index'], r['arm'], r['policy']) == (i, *lhs))
                     - next(r['scores']['actual_safe_reward'] for r in results if (r['task_index'], r['arm'], r['policy']) == (i, *rhs)) for i in range(spec['task_count'])]
        boot = np.asarray(differences)[samples].mean(axis=1)
        comparisons.append({'lhs': lhs, 'rhs': rhs, 'task_differences': differences,
                            'pairing_unit': spec.get('pairing_unit', 'independent task'),
                            'mean': float(np.mean(differences)), 'descriptive_95pct_paired_bootstrap': np.quantile(boot, [.025, .975]).tolist()})
    physical_init = sum(read(str(out / f'task{i:02d}' / 'initialization.json'))['physical_calls'] for i in range(spec['task_count']))
    reused_init = sum(read(str(out / f'task{i:02d}' / 'initialization.json')).get('reused_initialization_calls', 0)
                      for i in range(spec['task_count']))
    physical_search = sum(read(str(p))['physical_calls'] for p in out.glob('task*/*/round*_observation.json'))
    physical_verify = sum(r['physical_calls'] for r in results)
    physical = physical_init + physical_search + physical_verify
    assert physical <= spec['maximum_physical_calls']
    result = {'plan': plan, 'completed_at': timestamp(), 'all_recommendations_frozen_before_verification': True,
        'all_verification_finished_before_truth_scoring': True, 'rows': results, 'arm_summary': summary,
        'paired_comparisons': comparisons, 'unobserved_prediction_scores': prediction,
        'unsampled_information_decision_diagnostics': conversion,
        'cost': {'physical_initialization': physical_init, 'reused_initialization_calls': reused_init,
                 'physical_search': physical_search,
                 'physical_verification': physical_verify, 'physical_total': physical,
                 'logical_total': sum(r['verification']['total_evaluation_count'] for r in results), 'source_fits': 0},
        'limitations': [spec.get('evaluation_scope', 'Four new same-population d50 tasks, one predeclared layout, fixed73 bank.'),
            'Fixed two-round approximate acquisition and moment-projected actual paid updates; old exact raw HOLD remains.',
            f'Bootstrap intervals describe the declared {spec["task_count"]} paired units, not a confirmatory superiority result.']}
    if 'source_risk_correction' in frozen:
        result['cost']['source_correction_fits'] = 1
        result['source_risk_correction_file'] = spec['source_risk_correction_file']
    if 'source_joint_risk_fit' in frozen:
        fit_count = spec.get('source_fit_scope', {}).get('source_fits', len(frozen['source_joint_risk_fit']['rows']))
        result['cost'].update(source_fits=fit_count, source_joint_risk_identity_fits=fit_count,
                              original_source_law_refits=0, gp_refits=0)
        result['source_joint_risk_fit_file'] = spec['source_joint_risk_fit_file']
    if 'dynamic_candidates' in frozen:
        from performance.mainline_dynamic_diagnostics import dynamic_diagnostics
        cases = [dict(read(str(out / f'task{i:02d}' / arm / 'case.json')), task_index=i, arm=arm)
                 for i in range(spec['task_count']) for arm in spec['models']]
        result['dynamic_candidate_diagnostics'] = dynamic_diagnostics(frozen, cases, truths, results, protocol)
    save(out / 'results.json', result)
    print(json.dumps({'arm_summary': summary, 'cost': result['cost']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'case', 'assemble', 'verify'])
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--plan', default=PLAN)
    args = parser.parse_args()
    spec, out, _ = load(args.plan)
    started, wall = time.process_time(), time.monotonic()
    attempt = {'stage': args.stage, 'case_index': args.case_index, 'started_at': timestamp()}
    if args.stage == 'case':
        if args.case_index is None:
            parser.error('case requires --case-index')
        if not 0 <= args.case_index < spec['task_count'] * len(spec['models']):
            parser.error('case index exceeds registered task/model combinations')
        def timeout(_sig, _frame):
            raise TimeoutError('Registered per-case CPU budget exhausted')
        signal.signal(signal.SIGPROF, timeout)
        signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds'])
    try:
        if args.stage == 'case':
            run_case(args.case_index, args.plan)
        else:
            globals()[args.stage](args.plan)
        attempt['status'] = 'completed'
    except Exception:
        attempt.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        attempt.update(cpu_seconds=time.process_time() - started, wall_seconds=time.monotonic() - wall,
                       maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(attempt) + '\n')
