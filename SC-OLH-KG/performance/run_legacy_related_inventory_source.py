"""Registered related-task source pilot, before any new target closed loop."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive, resample_normalized_profiles
from core.designs import load_frozen_source_informed_design
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes, sample_variance_pit
from performance.diagnose_legacy_source_terminal import freeze_roster
from performance.replay_legacy_mixture_budget import terminal_at_prefix
from performance.replay_legacy_observable_budget import fit_arms
from performance.score_legacy_related_inventory_source import add_safe_scores
from representation.joint_function_risk import JointFunctionRiskPosterior, joint_batch_log_score
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_function_prior import legacy_public_coordinates, replicated_source_labels
from representation.legacy_observable_prior import legacy_exposure_descriptors


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def task_configs(plan):
    rng = np.random.default_rng(plan['task_seed'])
    return [{'task_id': f"related_inventory_{i:02d}", 'family': plan['family'],
             'problem_kwargs': {'task_geometry_shift': rng.uniform(-.035, .035, 3).tolist(),
                                'task_geometry_radius_scale': float(rng.uniform(.95, 1.05))}}
            for i in range(plan['source_task_count'])]


def problem_for(plan, task):
    d = plan['task_distribution']
    return build_scalarized_problem(task['family'], plan['source_dimension'], d['L'],
                                    d['sigma'], d['alpha'], d['weights'],
                                    problem_kwargs=task['problem_kwargs'])


def public_bank(plan, replay):
    base = json.loads((ROOT / replay['protocol']['paid_paths']).read_text())['protocol']
    folder = ROOT / 'paper_artifacts/legacy_joint_bridge_v1/inputs' / plan['family']
    archive = FrozenTransferArchive.load(folder / 'archive.json')
    initial, _ = load_frozen_source_informed_design(folder / 'design.json', heldout=plan['family'],
        seed=base['target_seed'], n0=10, dimension=base['target_dimension'])
    provider = no_outcomes(build_scalarized_problem(plan['family'], base['target_dimension'],
        base['L'], base['sigma'], base['alpha'], base['weights']))
    lifted = resample_normalized_profiles(np.vstack([t.X for t in archive.tasks]), provider.d)
    original = list(dict.fromkeys(list(initial) +
        [tuple(map(int, provider.continuous_to_int(x))) for x in lifted]))
    curves = resample_normalized_profiles(np.asarray(original) / provider.L, plan['source_dimension'])
    source = no_outcomes(problem_for(plan, task_configs(plan)[0]))
    bank = list(dict.fromkeys(tuple(map(int, source.continuous_to_int(x))) for x in curves))
    return bank, len(original)


def prepare_data(plan, execution, replay, out):
    bank, original_count = public_bank(plan, replay)
    configs = task_configs(plan)
    total = len(bank) * len(configs) * plan['replicates_per_site']
    if total > plan['maximum_new_source_calls']:
        raise ValueError('registered source-call budget exceeded before simulation')
    frozen = {'plan': plan, 'execution': execution, 'original_bank_size': original_count,
              'bank': [list(x) for x in bank], 'tasks': configs, 'source_calls': total}
    inputs_path, raw_path = out / 'frozen_inputs.json', out / 'observations.json'
    if inputs_path.exists():
        if json.loads(inputs_path.read_text()) != frozen:
            raise ValueError('existing source data belong to a different frozen input plan')
    else:
        save(inputs_path, frozen)
    if raw_path.exists():
        return frozen, json.loads(raw_path.read_text()), 0
    tasks = []
    for i, config in enumerate(configs):
        provider = problem_for(plan, config)
        rng = np.random.default_rng(np.random.SeedSequence([plan['observation_seed'], i]))
        values = [[provider.simulate(x, rng).tolist() for _ in range(plan['replicates_per_site'])]
                  for x in bank]
        tasks.append({**config, 'tau': provider.tau, 'Y_replicates': values})
        print('source observations generated', config['task_id'], len(bank), flush=True)
    data = {'bank': frozen['bank'], 'tasks': tasks, 'generation_calls': total}
    save(raw_path, data)
    return frozen, data, total


def training_view(data, indices, L):
    """Only selected source records enter the fit; configs are not features."""
    profiles = np.asarray(data['bank']) / L
    return [replicated_source_labels(data['tasks'][i]['task_id'], profiles,
            data['tasks'][i]['Y_replicates'], data['tasks'][i]['tau']) for i in indices]


def head_moments(model):
    views = {}
    for name, (f, g) in zip(model.names, model.components):
        fm, fv = f.predict(); gm, gv = g.predict()
        views[name] = (np.column_stack([fm, gm]), np.column_stack([fv, gv]))
    mean, covariance = model.predict_joint()
    views['paid_evidence_mixture'] = (mean, np.asarray([np.diag(covariance[i, :, i, :])
                                                     for i in range(len(mean))]))
    return views


def prediction_metrics(model, task, bank, L, truth, count):
    labels = replicated_source_labels(task['task_id'], np.asarray(bank) / L,
                                      task['Y_replicates'], task['tau'])
    test = np.arange(count, len(bank))
    raw = np.asarray(task['Y_replicates'])
    observed = labels['response'][:, [0, 3, 1, 2]]
    nrep = raw.shape[1]
    f_scores, g_scores, f_pits, g_pits = [], [], [], []
    for f, g in model.components:
        fs, gs, fp, gp = [], [], [], []
        for i in test:
            fs.append(joint_batch_log_score(f.local_components(i), observed[i, 0], observed[i, 1], nrep))
            gs.append(joint_batch_log_score(g.local_components(i), observed[i, 2], observed[i, 3], nrep))
            fp.append(sum(w * float(sample_variance_pit(m[1], c[1, 1], raw[i, :, 0].var(ddof=1), nrep))
                          for w, m, c in f.local_components(i)))
            gp.append(sum(w * float(sample_variance_pit(m[1], c[1, 1], raw[i, :, 1].var(ddof=1), nrep))
                          for w, m, c in g.local_components(i)))
        f_scores.append(fs); g_scores.append(gs); f_pits.append(fp); g_pits.append(gp)
    f_scores, g_scores = np.asarray(f_scores), np.asarray(g_scores)
    f_pits, g_pits = np.asarray(f_pits), np.asarray(g_pits)
    weights = model.weights[:, None]
    views = head_moments(model)
    result = {}
    for name, (mean, variance) in views.items():
        if name in model.names:
            i = model.names.index(name)
            fs, gs, joint, fp, gp = f_scores[i], g_scores[i], f_scores[i] + g_scores[i], f_pits[i], g_pits[i]
        else:
            logw = np.log(weights)
            fs, gs = logsumexp(logw + f_scores, axis=0), logsumexp(logw + g_scores, axis=0)
            joint = logsumexp(logw + f_scores + g_scores, axis=0)
            fp, gp = model.weights @ f_pits, model.weights @ g_pits
        error = mean[test] - truth[test]
        result[name] = {'head_order': ['f', 'eta_f', 'g', 'eta_g'], 'test_site_count': len(test),
            'truth_bias': error.mean(axis=0).tolist(), 'truth_mse': np.mean(error**2, axis=0).tolist(),
            'moment_interval_95_truth_coverage': np.mean(np.abs(error) <= 1.96 * np.sqrt(variance[test]), axis=0).tolist(),
            'noisy_observed_label_mse': np.mean((mean[test] - observed[test])**2, axis=0).tolist(),
            'joint_batch_log_score_f': float(np.mean(fs)), 'joint_batch_log_score_g': float(np.mean(gs)),
            'complete_identity_joint_batch_log_score': float(np.mean(joint)),
            'sample_variance_pit_f': fp.tolist(), 'sample_variance_pit_g': gp.tolist()}
    return result


def run(execution_path='performance/manifests/legacy_related_inventory_source_execution_v1.json'):
    started = time.process_time()
    execution = json.loads((ROOT / execution_path).read_text())
    plan = json.loads((ROOT / execution['plan']).read_text())
    replay = json.loads((ROOT / execution['legacy_replay']).read_text())
    source_spec = replay['source_fit_contract']
    names = replay['protocol']['hypotheses']
    out = ROOT / plan['output']; out.mkdir(parents=True, exist_ok=True)
    frozen, data, new_calls = prepare_data(plan, execution, replay, out)
    bank = [tuple(x) for x in data['bank']]
    train = [i for i in range(len(data['tasks'])) if i not in plan['source_holdouts']]
    tasks = training_view(data, train, plan['task_distribution']['L'])
    descriptors = [legacy_exposure_descriptors(no_outcomes(problem_for(plan, data['tasks'][i])),
                    bank, source_spec['dynamics_grid_dimension']) for i in train]
    print('source-only fit starts', [t['name'] for t in tasks], flush=True)
    coordinate, arms = fit_arms(tasks, np.vstack(descriptors), source_spec)
    print('source-only fit complete', {a: [p.diagnostics.get('optimizer_converged') for p in ps]
          for a, ps in arms.items()}, flush=True)
    source = {'training_task_ids': [t['name'] for t in tasks], 'coordinate': coordinate.diagnostics(),
              'fit_diagnostics': {a: [p.diagnostics for p in ps] for a, ps in arms.items()}}
    folds, rows, jobs = [], [], []
    count, cap = execution['adaptation_calls'], execution['total_target_call_cap']
    for i in plan['source_holdouts']:
        task = data['tasks'][i]
        provider = no_outcomes(problem_for(plan, task))
        desc = legacy_exposure_descriptors(provider, bank, source_spec['dynamics_grid_dimension'])
        observable = coordinate.transform(desc)
        legacy = legacy_public_coordinates(np.asarray(bank) / provider.L)
        components = []
        for name in names:
            ps = arms[name]; psi = legacy if name == 'legacy_source_orthogonal' else observable
            components.append((JointFunctionRiskPosterior([ps[0]], [ps[3]], psi),
                               JointFunctionRiskPosterior([ps[1]], [ps[2]], psi)))
        model = JointResponseMixture(names, components)
        observed = np.asarray([value[0] for value in task['Y_replicates'][:count]])
        for index, value in enumerate(observed):
            model.update_raw(index, [value[0], value[1] - provider.tau])
        original, _ = terminal_at_prefix(provider, bank[:count], {'paid_observations': observed.tolist()},
                                         count, source_spec, len(bank))
        terminal, retained, roster = freeze_roster(model, original, list(range(count)), cap,
            execution['retention_draw_power'], execution['integration_seed'], len(bank))
        reward, cost, actions = model.terminal_components(terminal, cap-count,
            execution['terminal_draw_power'], execution['integration_seed'])
        views = {name: (reward[k], cost[k]) for k, name in enumerate(names)}
        views['paid_evidence_mixture'] = (model.weights @ reward, model.weights @ cost)
        views['fixed_retained_direct'] = views['paid_evidence_mixture']
        direct = next(k for k, a in enumerate(actions) if a['kind'] == 'direct' and
                      a['primary'] == retained and a['budgets'] == [min(128, cap-count)])
        fold = {'task_id': task['task_id'], 'model_weights': model.weights.tolist(),
                'paid_history': model.history, 'retained_policy': retained,
                'primary_roster': roster, 'action_count': len(actions),
                'prediction': {name: {'mean': m.tolist(), 'variance': v.tolist()}
                               for name, (m, v) in head_moments(model).items()}}
        folds.append(fold)
        current = []
        for selector in execution['selectors']:
            r, c = views[selector]
            best = direct if selector == 'fixed_retained_direct' else int(np.lexsort((c, -r))[0])
            row = {'task_id': task['task_id'], 'selector': selector, 'search_calls': count,
                   'total_target_call_cap': cap, 'action': actions[best], 'action_index': best,
                   'predicted_reward': float(r[best]), 'predicted_total_calls': float(c[best]+count)}
            current.append(row); rows.append(row)
        jobs.append((task, model, terminal, fold, current))
        print('heldout recommendations frozen', task['task_id'], retained, flush=True)
    # Save every recommendation before the first heldout scoring truth query.
    save(out / 'recommendations.json', {'execution': execution, 'source': source, 'folds': folds, 'rows': rows})
    for task, model, terminal, fold, current in jobs:
        provider = problem_for(plan, task)
        truth = np.array([[provider.true_objective(x), np.log(provider.true_sigma(x)[0]**2),
                           provider.true_constraint_mean(x)-provider.tau, np.log(provider.true_sigma(x)[1]**2)]
                          for x in bank])
        feasible = [k for k, x in enumerate(bank) if provider.is_truly_feasible(x)]
        r, c, actions = terminal.evaluate(truth[None], cap-count)
        best = int(np.lexsort((c, -r))[0])
        fold['scoring_truth'] = truth.tolist()
        fold['feasible_policies'] = feasible
        fold['best_same_class_action'] = {'action': actions[best], 'reward': float(r[best]),
                                           'total_calls': float(c[best]+count)}
        fold['calibration'] = prediction_metrics(model, task, bank, provider.L, truth, count)
        for row in current:
            a = row['action_index']
            row['truth'] = {'reward': float(r[a]), 'total_calls': float(c[a]+count),
                'primary_feasible': row['action']['primary'] in feasible,
                'reward_estimation_error': float(row['predicted_reward']-r[a]),
                'call_estimation_error': float(row['predicted_total_calls']-c[a]-count),
                'loss_to_best_same_class_action': float(r[best]-r[a])}
        add_safe_scores({'execution': execution}, terminal, fold, current)
        print('heldout scoring complete', task['task_id'], 'feasible policies', len(feasible), flush=True)
    result = {'plan': plan, 'execution': execution, 'source': source, 'folds': folds, 'rows': rows,
        'cost': {'dataset_generation_calls': data['generation_calls'], 'new_source_calls_this_run': new_calls,
                 'training_source_calls': len(train)*len(bank)*plan['replicates_per_site'],
                 'reused_heldout_adaptation_calls': len(plan['source_holdouts'])*count,
                 'new_target_search_calls': 0, 'new_terminal_calls': 0, 'new_independent_certificates': 0},
        'bank_size': len(bank), 'cpu_seconds': time.process_time()-started}
    result['scoring_convention'] = ('truth.reward retains certificate-weighted quality including false certificates; '
        'truth.safe_reward gives truly infeasible policies zero utility with unchanged verifier/comparison branches.')
    save(out / 'results.json', result)
    print(json.dumps({'bank_size': len(bank), 'decisions': len(rows), 'cost': result['cost'],
                      'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
