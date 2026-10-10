"""Source-gated complete mean/risk uncertainty on the two covered layouts."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.check_legacy_source_layout_robustness import models_at_query
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save, training_view
from representation.joint_response_mixture import JointResponseMixture
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, fit_orthogonal_function_prior
from representation.public_control_mean import source_prior_moments
from representation.source_function_prior import head_batches
from representation.source_joint_population import (fit_joint_source_population, fit_joint_common_source_population,
    continue_joint_common_source_population, fit_flat_common_source_population)

EXECUTION = 'performance/manifests/legacy_joint_population_covered_complete_mean_uncertainty_v1.json'


def fit_means(spec, inputs, out):
    out.mkdir(parents=True, exist_ok=True); path = out/'source_fit.json'
    fitted = read(path) if path.exists() else {'heads': [], 'cpu_seconds': 0.}
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    for h in range(len(fitted['heads']), 2):
        started = time.process_time(); batches = head_batches(tasks, h); noises = [t['noise'][:, h] for t in tasks]
        regularizer = fit_orthogonal_function_prior(batches, noises, basis, max_iterations=spec['regularizer_max_iterations'])
        fixed = fit_joint_source_population(regularizer, batches, noises, max_iterations=spec['fixed_population_max_iterations'])
        joint = fit_joint_common_source_population(regularizer, batches, noises, fixed.diagnostics, optimizer=spec['optimizer'])
        if not joint.diagnostics['optimizer_converged']:
            save(out/f'head{h}_joint_before_completion.json', joint.diagnostics)
            joint = continue_joint_common_source_population(regularizer, batches, noises, joint.diagnostics, optimizer=spec['optimizer'])
        flat = fit_flat_common_source_population(regularizer, batches, noises, joint.diagnostics, optimizer=spec['optimizer'])
        fitted['heads'].append({'regularizer': regularizer.diagnostics, 'fixed_structural': fixed.diagnostics,
            'joint_structural': joint.diagnostics, 'flat': flat.diagnostics})
        fitted['cpu_seconds'] += time.process_time()-started; save(path, fitted)
    if not all(d['optimizer_converged'] for h in fitted['heads'] for d in h.values()):
        raise RuntimeError('complete failed native source mean objectives before forecasts')
    return fitted


def complete_models(group, means, mean_queries, risk_queries, arms):
    fields = {a: [] for a in arms}
    for (inp, _, risks), fit, mean_psi, risk_psi in zip(group, means, mean_queries, risk_queries):
        mb, rb = OrthogonalLowOrderBasis(inp['public_mean_reference']), OrthogonalLowOrderBasis(inp['public_risk_reference'])
        eta = [source_prior_moments(restore_joint(rb, h['joint_structural']), risk_psi) for h in risks['heads']]
        for arm in arms:
            moments = [source_prior_moments(restore_joint(mb, h[arm]), mean_psi) for h in fit['heads']]
            fields[arm].append(paired_fields(moments+eta))
    names = ['full144']+[f'drop{i}' for i in range(6)]
    return [(a, JointResponseMixture(names, fields[a])) for a in arms]


def run():
    started = time.process_time(); spec = read(EXECUTION); source = read(spec['source_execution'])
    data = read(source['source_data']); public = read(Path(source['source_data']).with_name('frozen_inputs.json'))
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); groups, fits, rows, jobs = {}, {}, [], {}
    for name in spec['layouts']:
        folder = Path(source['output'])/f'layout{name}'
        group = [(read(folder/'inputs.json'), read(folder/'source_fit.json'), read(folder/'risk_source_fit.json'))]
        group += [(read(folder/f'task{i}'/'inputs.json'), read(folder/f'task{i}'/'gp_source_fit.json'),
                   read(folder/f'task{i}'/'risk_source_fit.json')) for i in source['training_tasks']]
        means = [group[0][1]]+[fit_means(group[0][0]['execution'], g[0], out/f'layout{name}'/f'task{i}')
            for i, g in zip(source['training_tasks'], group[1:])]
        groups[name], fits[name] = group, means
        print('covered complete mean fits frozen', name, flush=True)
    for name, group in groups.items():
        for i in source['heldout_tasks']:
            task = data['tasks'][i]; query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source',
                'tau': task['tau'], 'candidate_indices': [0, 1], 'paired_observations': [task['Y_replicates'][j][0] for j in (0, 1)]}
            mq, rq = [g[0]['heldout_mean_psi'][str(i)] for g in group], [g[0]['heldout_risk_psi'][str(i)] for g in group]
            controls = dict(models_at_query(group, mq, rq))
            for arm, model in complete_models(group, fits[name], mq, rq, spec['mean_arms']):
                prior = forecast(query, arm, 'full_plus_six_complete_source_deletions', model)
                for j, value in zip(query['candidate_indices'], query['paired_observations']):
                    centered = [value[0], value[1]-query['tau']]
                    model.update_raw(j, centered); controls[arm].update_raw(j, centered)
                # Detect a changed full identity or swapped response/risk head before comparing ensembles.
                for field, old in zip(model.components[0], controls[arm].components[0]):
                    for actual, expected in zip(field.predict(full_cov=True), old.predict(full_cov=True)):
                        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
                row = forecast(query, arm, 'full_plus_six_complete_source_deletions', model)
                row.update(layout=name, prior=prior); rows.append(row); jobs[(name, i, arm)] = model
    save(out/'source_predictions.json', {'execution': spec, 'rows': rows})
    # All12 deletion fits and8 priors/posteriors precede any new source holdout score.
    references = read(spec['reference_profiles'])['rows']; contrasts = []
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in source['heldout_tasks']}
    for row in rows:
        i, arm, name = row['task_index'], row['mean_arm'], row['layout']; model = jobs[(name, i, arm)]
        score_source(row, labels[i], model, 3)
        old = next(r for r in references if r['layout'] == name and r['task_index'] == i and r['mean_arm'] == arm)
        contrast = {'layout': name, 'task_id': row['task_id'], 'mean_arm': arm,
            'g_mse_difference': row['scores']['observed_head_mse'][1]-old['scores']['observed_head_mse'][1],
            'joint_batch_score_difference': row['scores']['joint_batch_log_score']-old['scores']['joint_batch_log_score'],
            'g_batch_score_difference': row['scores']['response_batch_log_scores'][1]-old['scores']['response_batch_log_scores'][1]}
        contrast['passed'] = (contrast['g_mse_difference'] <= 0 and contrast['joint_batch_score_difference'] >= 0
                              and contrast['g_batch_score_difference'] > 0); contrasts.append(contrast)
    gates = {a: all(c['passed'] for c in contrasts if c['mean_arm'] == a) for a in spec['mean_arms']}
    summary = {name: {a: {k: np.mean([r['scores'][k] for r in rows if r['layout'] == name and r['mean_arm'] == a], axis=0).tolist()
        for k in rows[0]['scores']} for a in spec['mean_arms']} for name in spec['layouts']}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows})
    continuations = sum('continued_from_optimizer_iterations' in h['joint_structural']
        for means in fits.values() for fit in means[1:] for h in fit['heads'])
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'source_gates': gates,
        'full_identity_reproduction_passed': True, 'native_mean_fits': 96+continuations,
        'source_mean_fit_cpu_seconds': sum(f['cpu_seconds'] for means in fits.values() for f in means[1:]),
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result); print({'summary': summary, 'contrasts': contrasts, 'source_gates': gates,
        'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
