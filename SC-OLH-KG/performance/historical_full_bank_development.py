"""Restore the historical complete policy support before expensive KG work."""

import argparse
import copy
import json
import shlex
import shutil
import signal
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_common_bank import (common_bank_mapping, initialization_batches,
    replay_raw_batches, restore_zero_paid_prior)
from performance.connect_historical_learned_prior_kg import restore_learned_source_prior
from performance.connect_historical_matched_gp import controls_from_fit
from performance.connect_historical_prior_posterior import snapshot
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.prepare_historical_independent_kg import check_replayed_snapshot, terminal_from_initialization
from performance.prepare_historical_independent_query_parallel import LOCAL, REMOTE, PYTHON, NODES
from performance.run_historical_independent_kg_pilot import inputs, utility_identities
from performance.run_legacy_related_inventory_source import problem_for
from performance.score_legacy_related_inventory_source import safe_terminal_values
from core.guarded_deployment_kg import conditional_safety_power
from scipy.special import ndtri

PLAN = 'performance/manifests/historical_full_bank_development_v1.json'


def context(spec):
    pilot, frozen, states, public = inputs(read(spec['independent_execution_plan']))
    connection = read(pilot['source_connection_plan'])
    old = read(connection['legacy_connection_plan'])
    bank = read(old['public_inputs'])['bank']
    return pilot, frozen, states, public, connection, old, bank


def restricted_snapshot(model, indices):
    result = snapshot(model)
    for component in result['components']:
        for head in ('f', 'g'):
            component[head+'_mean'] = np.asarray(component[head+'_mean'])[indices].tolist()
            covariance = np.asarray(component[head+'_covariance'])
            component[head+'_covariance'] = covariance[indices][:, :, indices].tolist()
    remap = {full: local for local, full in enumerate(indices)}
    result['history'] = [dict(record, candidate=remap[record['candidate']]) for record in result['history']]
    return result


def restore_full_prior(spec, pilot, state, connection, old, count):
    if state['arm'] == 'source_structural_mixture':
        model, _, checks = restore_learned_source_prior(state['layout'], list(range(count)),
                                                       pilot['source_connection_plan'])
        return model, {'positive_components_per_head': [len(f.means) for pair in model.components for f in pair],
                       'maximum_source_projection_error': max(c['common_projection_max_error'] for c in checks)}
    selection = read(read(connection['source_plan'])['selection_plan'])
    mean_x = np.asarray(read(selection['source_inputs_by_layout'][state['layout']])['public_mean_reference'])
    risk_x = np.asarray(read(old['query_risk_inputs_by_layout'][state['layout']])['public_reference_psi'])
    fit = read(spec['gp_fit'])['by_layout'][state['layout']]
    assert fit['source_task_ids'] == [f'related_inventory_{i:02d}' for i in pilot['training_tasks']]
    model = controls_from_fit(fit, mean_x, risk_x, .04)[state['arm']]
    return model, {'frozen_gp_fit': spec['gp_fit'], 'positive_components_per_head': [1, 1]}


def terminal_rule(model, terminal, remaining, identities, power, seed, certificate_caches=None):
    values, costs = [], []
    for identity in identities:
        value, cost, actions = terminal.evaluate_posterior(*model.components[identity], remaining, power,
            seed, guarded_mean_position=0, direct_certificate_cache=(None if certificate_caches is None
                else certificate_caches.setdefault(identity, {})))
        values.append(value); costs.append(cost)
    values = model.weights[identities] @ values
    costs = model.weights[identities] @ costs
    best = int(np.lexsort((costs, -values))[0])
    return {'power': power, 'values': values.tolist(), 'costs': costs.tolist(), 'actions': actions,
            'best_action_index': best, 'best_action': actions[best], 'best_reward': float(values[best])}


def run_case(plan):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'
    if path.exists() and read(str(path)).get('recommendation_frozen'):
        return read(str(path))
    started = time.process_time()
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds'])
    try:
        pilot, frozen, states, public, connection, old, bank = context(spec)
        state = next(s for s in states if s['name'] == spec['state_name'])
        task = next(t for t in frozen['tasks'] if t['task_id'] == state['task_id'])
        paid = next(r for r in read(spec['initializations'])['rows'] if r['task_id'] == state['task_id'])
        provider = no_outcomes(problem_for(public, task))
        candidates = [tuple(p) for p in bank]
        frozen_bank = next(b for b in frozen['banks'] if b['layout'] == state['layout'])
        mapping = common_bank_mapping(bank, frozen_bank)
        model, restoration = restore_zero_paid_prior(spec, pilot, state, connection, old, len(bank))
        replay = replay_raw_batches(model, initialization_batches(paid), paid['tau'])
        remaining = replay['remaining_calls']
        queries = mapping['queries']
        check_replayed_snapshot(restricted_snapshot(model, state['candidate_bank_indices']),
                                state['posterior_moment_diagnostic'], roundoff_units=max(64, 2*len(bank)))
        restoration_cpu = time.process_time()-started
        mean, covariance = model.predict_joint()
        variance = np.asarray([np.diag(covariance[i, :, i, :]) for i in range(len(bank))])
        terminal = terminal_from_initialization(provider, candidates, paid['paired_observations'],
            state['terminal_protocol'], paid_indices=paid['candidate_bank_indices'])
        identities = utility_identities(model, state['arm'], spec['maximum_identity_integration_error_bound'])
        omitted = float(sum(w for i, w in enumerate(model.weights) if i not in identities))
        record = out/'terminal_rules.json'
        rules = read(str(record))['rows'] if record.exists() else []
        certificate_caches = {}
        for power in spec['terminal_powers']:
            if not any(r['power'] == power for r in rules):
                rules.append(terminal_rule(model, terminal, remaining, identities, power, spec['integration_seed'], certificate_caches))
                save(record, {'rows': rules})
                print('SCHEDULEURM_PROGRESS '+json.dumps({'step': len(rules), 'total': len(spec['terminal_powers']),
                    'unit': 'terminal_rules', 'metrics': {'state': state['name'], 'cpu_seconds': time.process_time()-started}}), flush=True)
        low, high = [next(r for r in rules if r['power'] == p) for p in spec['terminal_powers']]
        difference = float(np.max(abs(np.asarray(high['values'])-low['values'])))
        gates = {'action_resolution': difference <= spec['maximum_action_resolution_difference'],
                 'choice_stable': high['best_action_index'] == low['best_action_index'],
                 'identity_remainder': omitted <= spec['maximum_identity_integration_error_bound']}
        row = {k: state[k] for k in ('name', 'task_id', 'layout', 'arm')}
        row.update(bank_size=len(bank), paid_bank_indices=paid['candidate_bank_indices'],
            public_query_indices=queries, raw_replay=replay,
            paid_episodes=model.paid_episodes, source_weights=model.weights.tolist(), native_subset_replay_passed=True,
            native_replay_roundoff_units=max(64, 2*len(bank)),
            restoration=restoration, posterior_mean=mean.tolist(), posterior_variance=variance.tolist(),
            terminal_protocol=state['terminal_protocol'], utility_identities=identities, omitted_utility_mass=omitted,
            terminal_action_count=len(high['actions']), action=high['best_action'], action_index=high['best_action_index'],
            predicted_reward=high['best_reward'], predicted_calls=model.paid_episodes+high['costs'][high['best_action_index']],
            all_action_resolution_difference=difference, gates=gates, numerical_diagnostics_passed=all(gates.values()),
            terminal_rules=str(record.relative_to(ROOT)), recommendation_frozen=True, scoring_truth_opened=False,
            cpu_seconds=time.process_time()-started, restoration_cpu_seconds=restoration_cpu,
            new_physical_calls=0, new_source_fits=0, independent_confirmation=False)
        save(path, row)
        print('SCHEDULEURM_DONE '+json.dumps({'state': state['name'], 'cpu_seconds': row['cpu_seconds'],
            'numerical_diagnostics_passed': row['numerical_diagnostics_passed']}), flush=True)
        return row
    except TimeoutError:
        result = {'name': spec['state_name'], 'cpu_budget_stopped': True,
                  'cpu_seconds': time.process_time()-started, 'recommendation_frozen': False,
                  'independent_confirmation': False}
        save(out/'cpu_limit.json', result); save(path, result)
        print('SCHEDULEURM_DONE '+json.dumps(result), flush=True)
        return result
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)


def prepare():
    spec = read(PLAN); _, _, states, _, _, _, _ = context(spec)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    plans = ROOT/'performance/manifests/historical_full_bank_development_v1'; plans.mkdir(exist_ok=True)
    files = {PLAN, 'performance/historical_full_bank_development.py', 'performance/historical_common_bank.py', 'performance/prepare_historical_independent_kg.py',
             'core/conditional_terminal_kg.py',
             spec['gp_fit'], spec['initializations']}
    shards, specs = [], []
    for state in states:
        plan = dict(spec, state_name=state['name'], output=spec['output']+'/shards/'+state['name'])
        path = plans/(state['name']+'.json'); save(path, plan); files.add(str(path.relative_to(ROOT)))
        command = ['env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'PYTHONPATH=.',
                   PYTHON, '-u', '-m', 'performance.historical_full_bank_development', 'case', '--plan', str(path.relative_to(ROOT))]
        specs.append({'description': 'Full-bank development '+state['name'], 'cmd': shlex.join(command),
            'cwd': str(LOCAL), 'signature': 'KG_op/historical_full_bank_development_v1/'+state['name'],
            'project': 'KG-SYNTH', 'cpu': 1, 'ram_mb': 8192 if state['arm'] == 'source_structural_mixture' else 2048,
            'vram': 0, 'allowed_nodes': NODES, 'skip_launch_staging': True})
        shards.append({'state_name': state['name'], 'plan': str(path.relative_to(ROOT)), 'output': plan['output']})
    deployment = {'plan': PLAN, 'shards': shards, 'remote_code_root': REMOTE}
    save(out/'deployment.json', deployment); files.add(spec['output']+'/deployment.json')
    (out/'scheduler_specs.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in specs))
    (out/'upload_files.txt').write_text('\n'.join(sorted(files))+'\n')
    for name in files:
        if name.endswith('.py') or name.startswith('performance/manifests/'):
            target = LOCAL/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, target)
    print(json.dumps({'shards': len(shards), 'upload_bytes': sum((ROOT/f).stat().st_size for f in files)}), flush=True)


def collect():
    spec = read(PLAN); _, frozen, states, public, _, _, bank = context(spec)
    out = ROOT/spec['output']; deployment = read(str(out/'deployment.json'))
    rows = [read(s['output']+'/results.json') for s in deployment['shards']]
    if {r['name'] for r in rows} != {s['name'] for s in states} or len(rows) != len(states) or not all(r['recommendation_frozen'] for r in rows):
        raise ValueError('All24 full-bank recommendations must freeze before scoring')
    # Also freeze the direct-only control using the same posterior and budgets.
    # Its one-dimensional eta integral is independent of guarded Sobol power.
    for row in rows:
        rules = read(row['terminal_rules'])['rows']
        low, fine = [next(r for r in rules if r['power'] == p) for p in spec['terminal_powers']]
        direct = np.asarray([i for i, a in enumerate(fine['actions']) if a['kind'] == 'direct'])
        selected = int(direct[np.lexsort((np.asarray(fine['costs'])[direct],
                                          -np.asarray(fine['values'])[direct]))[0]])
        row['direct_stop_action_index'] = selected
        row['direct_stop_action'] = fine['actions'][selected]
        row['direct_stop_predicted_reward'] = fine['values'][selected]
        row['direct_rule_power_difference'] = float(np.max(abs(np.asarray(fine['values'])[direct]-
                                                              np.asarray(low['values'])[direct])))
    save(out/'recommendations.json', {'rows': rows, 'all_recommendations_frozen': True, 'scoring_truth_opened': False})
    for row in rows:
        task = next(t for t in frozen['tasks'] if t['task_id'] == row['task_id'])
        problem = problem_for(public, task)
        truth = np.asarray([[problem.true_objective(p), np.log(problem.true_sigma(p)[0]**2),
            problem.true_constraint_mean(p)-problem.tau, np.log(problem.true_sigma(p)[1]**2)] for p in bank])
        feasible = np.asarray([problem.is_truly_feasible(p) for p in bank])
        rules = read(row['terminal_rules'])['rows']; fine = next(r for r in rules if r['power'] == spec['terminal_powers'][-1])
        from core.conditional_terminal_kg import ConditionalTerminalActions
        guarded = fine['actions'][:len(bank)]
        terminal = ConditionalTerminalActions([a['shortlist'] for a in guarded],
            [a['incumbent_position'] for a in guarded], row['terminal_protocol'])
        safe, costs = safe_terminal_values(terminal, truth, feasible, 360)
        best = int(np.lexsort((costs, -safe))[0]); selected = row['action_index']
        scores = np.asarray(fine['values']); priorities = np.full(len(bank), -np.inf)
        for value, action in zip(scores, fine['actions']):
            priorities[action['primary']] = max(priorities[action['primary']], value)
        ranking = np.argsort(-priorities, kind='stable').tolist()
        error = np.asarray(row['posterior_mean'])-truth
        row['diagnostic_scores'] = {'conditional_safe_reward': float(safe[selected]),
            'conditional_total_calls': float(costs[selected]+2), 'primary_truly_feasible': bool(feasible[row['action']['primary']]),
            'best_same_class_safe_reward': float(safe[best]), 'loss_to_same_class_best': float(safe[best]-safe[selected]),
            'feasible_policy_count': int(feasible.sum()),
            'best_predicted_feasible_primary_rank': min((ranking.index(i)+1 for i in np.flatnonzero(feasible)), default=None),
            'head_mean_errors_f_eta_f_g_eta_g': error.mean(axis=0).tolist(),
            'feasible_head_mean_errors_f_eta_f_g_eta_g': error[feasible].mean(axis=0).tolist()}
        direct = row['direct_stop_action_index']
        row['diagnostic_scores'].update(direct_conditional_safe_reward=float(safe[direct]),
            direct_primary_truly_feasible=bool(feasible[row['direct_stop_action']['primary']]))
        row['scoring_truth_opened'] = True
    result = {'plan': PLAN, 'rows': rows, 'all_recommendations_frozen': True,
        'native_subset_replay_passed': all(r['native_subset_replay_passed'] for r in rows),
        'numerical_diagnostics_passed_count': sum(r['numerical_diagnostics_passed'] for r in rows),
        'cpu_seconds': sum(r['cpu_seconds'] for r in rows), 'new_physical_calls': 0, 'new_source_fits': 0,
        'scoring_truth_opened': True, 'independent_confirmation': False,
        'limitations': ['Already opened six development tasks; conditional utility is not a new independent deployment outcome.',
            'Coarse terminal integration retains all failed resolution gates; no KG fantasies or strict ranking certification.',
            'Earlier paid Gaussian moment projections remain.']}
    result['summary'] = []
    for arm in dict.fromkeys(s['arm'] for s in states):
        for layout in dict.fromkeys(s['layout'] for s in states):
            group = [r for r in rows if (r['arm'], r['layout']) == (arm, layout)]
            result['summary'].append({'arm': arm, 'layout': layout, 'task_count': len(group),
                'mean_conditional_safe_reward': float(np.mean([r['diagnostic_scores']['conditional_safe_reward'] for r in group])),
                'feasible_primaries': sum(r['diagnostic_scores']['primary_truly_feasible'] for r in group),
                'mean_loss_to_same_class_best': float(np.mean([r['diagnostic_scores']['loss_to_same_class_best'] for r in group])),
                'primary_indices': [r['action']['primary'] for r in group]})
    task_ids = [t['task_id'] for t in frozen['tasks']]
    models = list(dict.fromkeys(s['arm'] for s in states))
    differences = [float(np.mean([r['diagnostic_scores']['conditional_safe_reward'] for r in rows
        if r['task_id'] == task and r['arm'] == models[0]])-np.mean([
        r['diagnostic_scores']['conditional_safe_reward'] for r in rows
        if r['task_id'] == task and r['arm'] == models[1]])) for task in task_ids]
    rng = np.random.default_rng(2026100905)
    draws = rng.integers(len(task_ids), size=(20000, len(task_ids)))
    result['development_paired_source_minus_gp'] = {'task_differences': differences,
        'mean_difference': float(np.mean(differences)), 'whole_task_bootstrap_95_interval':
        np.quantile(np.asarray(differences)[draws].mean(axis=1), [.025, .975]).tolist(),
        'independent_confirmation': False}
    save(out/'results.json', result)
    print(json.dumps({'rows': len(rows), 'numerical_diagnostics_passed': result['numerical_diagnostics_passed_count'],
                      'cpu_seconds': result['cpu_seconds'], 'new_physical_calls': 0}), flush=True)
    return result


def diagnose():
    """Explain frozen recommendations using the already opened scoring truth."""
    spec = read(PLAN); _, frozen, _, public, _, _, bank = context(spec)
    out = ROOT/spec['output']; result = read(str(out/'results.json'))
    if not result['scoring_truth_opened']:
        raise ValueError('Calibration diagnosis requires completed development scoring')
    rows = []
    for row in result['rows']:
        task = next(t for t in frozen['tasks'] if t['task_id'] == row['task_id'])
        problem = problem_for(public, task)
        primary = row['action']['primary']; point = bank[primary]
        protocol = row['terminal_protocol']
        g = float(problem.true_constraint_mean(point)-problem.tau)
        sigma = float(problem.true_sigma(point)[1]); eta = float(np.log(sigma**2))
        mean = row['posterior_mean'][primary]; variance = row['posterior_variance'][primary]
        quality = float(np.clip(1-problem.true_objective(point)/protocol['objective_reference'], 0, 1))
        margin = float(g+ndtri(1-protocol['chance_alpha'])*sigma)
        diagnostic = {k: row[k] for k in ('task_id', 'layout', 'arm', 'predicted_reward',
                                         'numerical_diagnostics_passed')}
        diagnostic.update(primary=primary, kind=row['action']['kind'],
            conditional_safe_reward=row['diagnostic_scores']['conditional_safe_reward'],
            true_quality=quality, posterior_g_mean=mean[2], true_g_mean=g,
            posterior_g_sd=float(np.sqrt(variance[2])), posterior_eta_g_mean=mean[3],
            true_eta_g=eta, posterior_eta_g_sd=float(np.sqrt(variance[3])),
            true_sigma_g=sigma, true_chance_margin=margin,
            posterior_g_mean_error=float(mean[2]-g), posterior_eta_g_mean_error=float(mean[3]-eta))
        if row['action']['kind'] == 'direct':
            budget = row['action']['budgets'][0]
            diagnostic['true_certificate_probability'] = float(conditional_safety_power(
                g, sigma, budget, protocol['chance_alpha'], protocol['familywise_delta']))
        rows.append(diagnostic)
    diagnosis = {'post_outcome_development_diagnostic': True, 'new_physical_calls': 0, 'rows': rows}
    save(out/'selected_policy_calibration.json', diagnosis)
    print(json.dumps({'diagnosed_recommendations': len(rows), 'new_physical_calls': 0}), flush=True)
    return diagnosis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'case', 'collect', 'diagnose'])
    parser.add_argument('--plan', default=PLAN); args = parser.parse_args()
    if args.stage == 'case': run_case(args.plan)
    elif args.stage == 'prepare': prepare()
    elif args.stage == 'collect': collect()
    else: diagnose()
