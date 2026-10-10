"""Matched paid-batch development replay on the restored historical policy bank."""

import argparse
import copy
import json
import shlex
import shutil
import signal
import time

import numpy as np

from core.direct_terminal_reference import expected_quality, fast_expected_certificates
from core.guarded_deployment_kg import conditional_safety_power
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.historical_full_bank_development import context, restore_full_prior, restricted_snapshot
from performance.prepare_historical_independent_kg import check_replayed_snapshot, terminal_from_initialization
from performance.prepare_historical_independent_query_parallel import LOCAL, REMOTE, PYTHON, NODES
from performance.run_historical_independent_kg_pilot import utility_identities
from performance.run_legacy_related_inventory_source import problem_for
from performance.diagnose_legacy_risk_transfer import no_outcomes
from representation.joint_function_risk import local_joint_moments
from representation.vectorized_joint_function_risk import local_raw_moments
from scipy.special import digamma

PLAN = 'performance/manifests/historical_full_bank_replicates_v1.json'


def freeze_queries(rows):
    """Both learners receive the union of their already frozen stopping choices."""
    groups = {}
    for row in rows:
        groups.setdefault((row['task_id'], row['layout']), {})[row['arm']] = row
    return [{'task_id': task, 'layout': layout, 'queries': [
        {'role': role, 'candidate': group[arm]['direct_stop_action']['primary']}
        for role, arm in [('source_recommendation', 'source_structural_mixture'),
                          ('gp_recommendation', 'source_common_task_rbf')]]}
        for (task, layout), group in groups.items()]


def direct_rule(model, terminal, remaining, arm, bound):
    actions = [{'kind': kind, 'primary': int(indices[0]), 'shortlist': indices.tolist(),
                'incumbent_position': int(incumbent), 'budgets': budgets}
        for kind, table, incumbents, budgets in terminal.allocations(remaining) if kind == 'direct'
        for indices, incumbent in zip(table, incumbents)]
    candidates = [a['primary'] for a in actions]; budgets = [a['budgets'][0] for a in actions]
    identities = utility_identities(model, arm, bound); values = []; checks = []
    for identity in identities:
        f, g = model.components[identity]
        quality = {i: expected_quality(f, i, terminal.protocol['objective_reference']) for i in set(candidates)}
        certificate, info = fast_expected_certificates(g, candidates, budgets, terminal.protocol)
        values.append(np.asarray([quality[i] for i in candidates])*certificate); checks.append(info)
    rewards = model.weights[identities] @ values
    best = int(np.lexsort((budgets, -rewards))[0])
    omitted = float(sum(w for i, w in enumerate(model.weights) if i not in identities))
    eta_error = max(max(c['maximum_accepted_nested_difference'], c['adaptive_absolute_error_bound']) for c in checks)
    if omitted > bound or eta_error > 1e-8:
        raise ValueError('Direct terminal integration remains unresolved')
    return {'action': actions[best], 'predicted_reward': float(rewards[best]),
            'direct_action_count': len(actions), 'omitted_utility_mass': omitted,
            'maximum_eta_integral_diagnostic': eta_error, 'utility_identities': identities}


def batch_equivalence(model, candidate, centered):
    count = len(centered); k = (count-1)/2
    eta = np.log(centered.var(axis=0, ddof=1))-digamma(k)+np.log(k)
    errors = []
    for pair in model.components:
        for head, field in enumerate(pair):
            positions = sorted({0, len(field.means)//2, len(field.means)-1})
            ix = [2*candidate, 2*candidate+1]
            means = field.means[positions][:, ix]
            cov = field.covariances[positions][:, ix][:, :, ix]
            fast = local_raw_moments(means, cov, centered[:, head].mean(), eta[head], count)
            for i in range(len(positions)):
                reference = local_joint_moments(means[i], cov[i], centered[:, head].mean(), eta[head], count)
                errors.extend(float(np.max(abs(fast[key][i]-reference[key])))
                              for key in ('mean', 'covariance', 'log_evidence'))
    error = max(errors)
    if error > 1e-8:
        raise ValueError('Replicated positive-source update differs from historical scalar likelihood')
    return error


def prepare():
    spec = read(PLAN); base = read(spec['full_bank_plan']); _, _, states, _, _, _, _ = context(base)
    baseline = read(base['output']+'/results.json')
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    queries = {'rows': freeze_queries(baseline['rows']), 'query_truth_used': False}
    save(out/'frozen_queries.json', queries)
    plans = ROOT/'performance/manifests/historical_full_bank_replicates_v1'; plans.mkdir(exist_ok=True)
    files = {PLAN, 'performance/historical_full_bank_replicates.py',
             'representation/joint_function_risk.py', 'representation/vectorized_joint_function_risk.py',
             spec['output']+'/frozen_queries.json'}
    specs = []; shards = []
    for state in states:
        plan = dict(spec, state_name=state['name'], output=spec['output']+'/shards/'+state['name'])
        path = plans/(state['name']+'.json'); save(path, plan); files.add(str(path.relative_to(ROOT)))
        cmd = shlex.join(['env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
            'PYTHONPATH=.', PYTHON, '-u', '-m', 'performance.historical_full_bank_replicates', 'case',
            '--plan', str(path.relative_to(ROOT))])
        specs.append({'description': 'Full-bank paid batches '+state['name'], 'cmd': cmd, 'cwd': str(LOCAL),
            'signature': 'KG_op/historical_full_bank_replicates_v1/'+state['name'], 'project': 'KG-SYNTH',
            'cpu': 1, 'ram_mb': 12288 if state['arm'] == 'source_structural_mixture' else 2048,
            'vram': 0, 'allowed_nodes': NODES, 'skip_launch_staging': True})
        shards.append({'name': state['name'], 'plan': str(path.relative_to(ROOT)), 'output': plan['output']})
    save(out/'deployment.json', {'shards': shards, 'remote_code_root': REMOTE})
    files.add(spec['output']+'/deployment.json')
    (out/'scheduler_specs.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in specs))
    (out/'upload_files.txt').write_text('\n'.join(sorted(files))+'\n')
    for name in files:
        target = LOCAL/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, target)
    print(json.dumps({'shards': len(shards), 'expected_recommendations': len(shards)*5,
                      'upload_bytes': sum((ROOT/f).stat().st_size for f in files)}), flush=True)


def observations():
    spec = read(PLAN); base = read(spec['full_bank_plan'])
    _, frozen, _, public, _, _, bank = context(base); out = ROOT/spec['output']
    path = out/'observations.json'
    if path.exists(): return read(str(path))
    queries = read(spec['output']+'/frozen_queries.json')['rows']; rows = []
    for task_index, task in enumerate(frozen['tasks']):
        problem = problem_for(public, task)
        candidates = sorted({q['candidate'] for group in queries if group['task_id'] == task['task_id']
                              for q in group['queries']})
        for candidate in candidates:
            rng = np.random.default_rng(np.random.SeedSequence([spec['observation_seed'], task_index, candidate]))
            values = [list(map(float, problem.simulate(bank[candidate], rng))) for _ in range(4)]
            rows.append({'task_id': task['task_id'], 'candidate': candidate, 'paired_observations': values,
                         'tau': float(problem.tau), 'physical_calls': 4})
    result = {'rows': rows, 'new_physical_calls': 4*len(rows), 'truth_scoring_used': False}
    save(path, result)
    print('SCHEDULEURM_DONE '+json.dumps({'paired_batches': len(rows), 'new_physical_calls': 4*len(rows)}), flush=True)
    return result


def run_case(plan):
    spec = read(plan); base = read(spec['full_bank_plan']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'
    if path.exists() and read(str(path)).get('recommendation_frozen'): return read(str(path))
    started = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_case_cpu_seconds'])
    progress = {'rows': []}; rows = progress['rows']
    try:
        pilot, frozen, states, public, connection, old, bank = context(base)
        state = next(s for s in states if s['name'] == spec['state_name'])
        paid = next(r for r in read(base['initializations'])['rows'] if r['task_id'] == state['task_id'])
        task = next(t for t in frozen['tasks'] if t['task_id'] == state['task_id'])
        model, restoration = restore_full_prior(base, pilot, state, connection, old, len(bank))
        for candidate, value in zip(paid['candidate_bank_indices'], paid['paired_observations']):
            model.update_raw(candidate, [value[0], value[1]-paid['tau']])
        check_replayed_snapshot(restricted_snapshot(model, state['candidate_bank_indices']),
            state['posterior_moment_diagnostic'], roundoff_units=max(64, 2*len(bank)))
        terminal = terminal_from_initialization(no_outcomes(problem_for(public, task)), [tuple(p) for p in bank],
            paid['paired_observations'], state['terminal_protocol'], paid_indices=paid['candidate_bank_indices'])
        queries = next(r['queries'] for r in read(spec['registration_output']+'/frozen_queries.json')['rows']
                       if (r['task_id'], r['layout']) == (state['task_id'], state['layout']))
        batches = read(spec['registration_output']+'/observations.json')['rows']
        baseline = next(r for r in read(base['output']+'/results.json')['rows'] if r['name'] == state['name'])
        equivalence = []; interventions = [{'policy': 'stop', 'candidate': None, 'replicates': 0}]+[
            {'policy': q['role']+'_batch'+str(n), 'candidate': q['candidate'], 'replicates': n}
            for q in queries for n in spec['batch_sizes']]
        for intervention in interventions:
            future = copy.deepcopy(model) if intervention['replicates'] else model
            evidence = None
            if intervention['replicates']:
                batch = next(r for r in batches if (r['task_id'], r['candidate']) ==
                             (state['task_id'], intervention['candidate']))
                centered = np.asarray(batch['paired_observations'])[:intervention['replicates']].copy()
                centered[:, 1] -= batch['tau']
                if intervention['replicates'] == 4:
                    equivalence.append(batch_equivalence(model, intervention['candidate'], centered))
                evidence = future.update_replicates(intervention['candidate'], centered)
            if future.paid_episodes != 2+intervention['replicates']:
                raise ValueError('Fresh replicated batch was debited incorrectly')
            rule = direct_rule(future, terminal, spec['total_call_cap']-future.paid_episodes,
                               state['arm'], spec['maximum_identity_integration_error_bound'])
            if intervention['policy'] == 'stop' and (
                rule['action'] != baseline['direct_stop_action'] or
                abs(rule['predicted_reward']-baseline['direct_stop_predicted_reward']) > 1e-4):
                raise ValueError('Native direct stopping rule changed on restoration')
            mean, covariance = future.predict_joint()
            row = {**intervention, **rule, 'paid_episodes': future.paid_episodes, 'evidence': evidence,
                'source_weights': future.weights.tolist(), 'posterior_mean': mean.tolist(),
                'posterior_variance': [np.diag(covariance[i, :, i, :]).tolist() for i in range(len(bank))],
                'maximum_raw_integration_relative_error': max(f.max_integration_relative_error
                                                             for pair in future.components for f in pair)}
            rows.append(row); save(out/'recommendation_progress.json', {'rows': rows})
            if intervention['replicates']: del future
            print('SCHEDULEURM_PROGRESS '+json.dumps({'step': len(rows), 'total': len(interventions),
                'unit': 'recommendations', 'metrics': {'state': state['name'], 'cpu_seconds': time.process_time()-started}}), flush=True)
        result = {k: state[k] for k in ('name', 'task_id', 'layout', 'arm')}
        result.update(rows=rows, terminal_protocol=state['terminal_protocol'], restoration=restoration,
            native_subset_replay_passed=True, native_batch_equivalence_max_error=max(equivalence),
            recommendation_frozen=True, cpu_seconds=time.process_time()-started, independent_confirmation=False,
            new_physical_calls=0, full_guarded_precision_HOLD_unchanged=True)
        save(path, result)
        print('SCHEDULEURM_DONE '+json.dumps({'state': state['name'], 'cpu_seconds': result['cpu_seconds']}), flush=True)
        return result
    except TimeoutError:
        result = {'name': spec['state_name'], 'recommendation_frozen': False, 'cpu_budget_stopped': True,
                  'cpu_seconds': time.process_time()-started, 'completed_recommendations': len(rows)}
        save(path, result)
        print('SCHEDULEURM_DONE '+json.dumps(result), flush=True)
        return result
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)


def collect():
    spec = read(PLAN); out = ROOT/spec['output']; deployment = read(str(out/'deployment.json'))
    cases = [read(s['output']+'/results.json') for s in deployment['shards']]
    if (len(cases) != 24 or not all(c.get('recommendation_frozen') and len(c['rows']) == 5 for c in cases)
        or {c['name'] for c in cases} != {s['name'] for s in deployment['shards']}):
        raise ValueError('All120 development recommendations must freeze before scoring')
    save(out/'recommendations.json', {'cases': cases, 'all_recommendations_frozen': True})
    _, frozen, _, public, _, _, bank = context(read(spec['full_bank_plan']))
    rows = []
    for case in cases:
        task = next(t for t in frozen['tasks'] if t['task_id'] == case['task_id'])
        problem = problem_for(public, task); protocol = case['terminal_protocol']
        truth = np.asarray([[problem.true_objective(p), np.log(problem.true_sigma(p)[0]**2),
            problem.true_constraint_mean(p)-problem.tau, np.log(problem.true_sigma(p)[1]**2)] for p in bank])
        feasible = np.asarray([problem.is_truly_feasible(p) for p in bank])
        for row in case['rows']:
            index = row['action']['primary']; budget = row['action']['budgets'][0]
            power = float(conditional_safety_power(truth[index, 2], np.exp(truth[index, 3]/2),
                          budget, protocol['chance_alpha'], protocol['familywise_delta']))
            mean = np.asarray(row['posterior_mean']); error = mean-truth
            score = {'conditional_safe_reward': float(np.clip(1-truth[index, 0]/protocol['objective_reference'],
                        0, 1)*feasible[index]*power), 'primary_truly_feasible': bool(feasible[index]),
                'true_certificate_probability': power, 'selected_g_mean_error': float(error[index, 2]),
                'bank_g_mean_rmse': float(np.sqrt(np.mean(error[:, 2]**2))),
                'bank_eta_g_mean_rmse': float(np.sqrt(np.mean(error[:, 3]**2)))}
            score['selected_posterior_mean'] = mean[index].tolist()
            if row['candidate'] is not None:
                query = row['candidate']
                score.update(query_g_mean_error=float(error[query, 2]),
                    query_eta_g_mean_error=float(error[query, 3]), query_posterior_mean=mean[query].tolist())
            rows.append({**{k: case[k] for k in ('name', 'task_id', 'layout', 'arm')}, **row, 'scores': score})
    summaries = []
    for arm in dict.fromkeys(c['arm'] for c in cases):
        for policy in dict.fromkeys(r['policy'] for r in rows):
            group = [r for r in rows if (r['arm'], r['policy']) == (arm, policy)]
            summaries.append({'arm': arm, 'policy': policy, 'count': len(group),
                'mean_conditional_safe_reward': float(np.mean([r['scores']['conditional_safe_reward'] for r in group])),
                'feasible_primaries': sum(r['scores']['primary_truly_feasible'] for r in group),
                'mean_bank_g_rmse': float(np.mean([r['scores']['bank_g_mean_rmse'] for r in group])),
                'mean_bank_eta_g_rmse': float(np.mean([r['scores']['bank_eta_g_mean_rmse'] for r in group])),
                'primary_indices': [r['action']['primary'] for r in group]})
    result = {'rows': rows, 'summary': summaries, 'independent_confirmation': False,
        'all_recommendations_frozen': True, 'cpu_seconds': sum(c['cpu_seconds'] for c in cases),
        'new_physical_calls': read(str(out/'observations.json'))['new_physical_calls'],
        'new_verifier_calls': 0, 'new_source_fits': 0, 'full_guarded_precision_HOLD_unchanged': True,
        'maximum_native_batch_equivalence_error': max(c['native_batch_equivalence_max_error'] for c in cases)}
    task_ids = [t['task_id'] for t in frozen['tasks']]
    draws = np.random.default_rng(2026100907).integers(len(task_ids), size=(20000, len(task_ids)))
    def task_rewards(arm, policy):
        return np.asarray([np.mean([r['scores']['conditional_safe_reward'] for r in rows
            if (r['task_id'], r['arm'], r['policy']) == (task, arm, policy)]) for task in task_ids])
    def comparison(label, differences):
        return {'comparison': label, 'task_differences': differences.tolist(),
            'mean_difference': float(differences.mean()),
            'whole_task_bootstrap_95_interval': np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist()}
    policies = list(dict.fromkeys(r['policy'] for r in rows))
    models = list(dict.fromkeys(c['arm'] for c in cases))
    result['paired_comparisons'] = [comparison(arm+'/'+policy+' minus stop',
        task_rewards(arm, policy)-task_rewards(arm, 'stop')) for arm in models for policy in policies if policy != 'stop']+[
        comparison('source minus gp/'+policy, task_rewards(models[0], policy)-task_rewards(models[1], policy))
        for policy in policies]
    save(out/'results.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'summary')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'observations', 'case', 'collect'])
    parser.add_argument('--plan', default=PLAN); args = parser.parse_args()
    if args.stage == 'case': run_case(args.plan)
    else: {'prepare': prepare, 'observations': observations, 'collect': collect}[args.stage]()
