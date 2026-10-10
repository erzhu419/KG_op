"""Additional crossed source labels,with fixed sparse and GP population laws."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_source_spectral_uncertainty import source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_coherent_common_compression import condition
from performance.diagnose_legacy_f_level_shape_mean_error import decompose
from performance.diagnose_legacy_f_source_latent_noise_feasibility import conditional_pair, ARMS
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.source_joint_spectral_geometry import JointSpectralGeometryLikelihood
from representation.source_kernel_population import rbf_cross, shared_kernel

PLAN = 'performance/manifests/legacy_joint_population_f_crossed_source_reconditioning_plan_v1.json'


def fixed_gp(tasks, query, fit):
    anchor = fit['anchor']; scale2 = anchor['normalization']['scale'] ** 2
    center = anchor['normalization']['mean'] + anchor['normalization']['scale'] * anchor['constant']
    amplitude, intercept, discrepancy = fit['residual_parameters']; xs = np.vstack([t['features'] for t in tasks])
    residual = block_diag(*[scale2 * (amplitude * rbf_cross(np.asarray(t['features']), np.asarray(t['features']), anchor['lengthscale']) + intercept)
        + np.diag(np.asarray(t['noise'])[:, 0] + scale2 * discrepancy) for t in tasks])
    error = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks]) - center
    nll, alpha, factor = condition(scale2 * shared_kernel(xs, xs, anchor), residual, error)
    cross = scale2 * shared_kernel(query, xs, anchor)
    covariance = (scale2 * (shared_kernel(query, query, anchor) + amplitude * rbf_cross(query, query, anchor['lengthscale']) + intercept)
        + scale2 * discrepancy * np.eye(len(query)) - cross @ cho_solve(factor, cross.T))
    return center + cross @ alpha, .5 * (covariance + covariance.T), nll / len(error)


def run():
    started = time.process_time(); spec = read(PLAN); geometry = read(spec['information_results']); baseline = read(spec['latent_noise_results'])
    assert geometry['diagnostic_interpreted'] and baseline['diagnostic_interpreted']
    base_priors = read(Path(baseline['execution']['output']) / 'frozen_inputs.json')['rows']
    registry = read(spec['source_gp_registry'])['rows']; source_spec = read(spec['source_probe_plan'])
    coverage, _ = source_paths(read(spec['source_context_execution'])); query = spec['source_validation_task']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True); work, rosters = [], []
    for layout in spec['layouts']:
        model, law, public = source_model(source_spec, layout, query)
        g = next(r for r in geometry['rows'] if (r['layout'], r['source_task_count']) == (layout, 4))
        assert g['source_task_names'] == law['source_task_names'] and len(g['union_policies']) == 24
        inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        old = [t for t in inputs['source_tasks'] if t['name'] in law['source_task_names']]
        record = next(r for r in registry if (r['layout'], r['inner_query_task']) == (layout, f'related_inventory_{query:02d}'))
        assert record['inner_training_task_names'] == law['source_task_names']; gp = read(record['source_fit'])['fits']['gp']
        query_psi = public[coverage['layouts'][layout][query]]
        old_gp_mean, old_gp_covariance, _ = fixed_gp(old, query_psi, gp)
        np.testing.assert_allclose(old_gp_mean, record['prediction']['source_gp']['mean'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(old_gp_covariance, record['prediction']['source_gp']['covariance'], atol=1e-8, rtol=1e-8)
        source_ids = [int(name.rsplit('_', 1)[1]) for name in law['source_task_names']]
        expanded = [coverage['layouts'][layout][q] + sorted(set(g['union_policies']) - set(coverage['layouts'][layout][q])) for q in source_ids]
        assert all(len(sites) == 24 for sites in expanded) and query not in source_ids and 1 not in source_ids
        rosters.append({'layout': layout, 'source_ids': source_ids, 'source_task_names': law['source_task_names'], 'policy_indices_by_source': expanded,
            'historical_mean_calls_per_arm': 288, 'baseline_historical_mean_calls_per_arm': 144, 'unchanged_risk_calls_per_arm': 96})
        work.append({'layout': layout, 'model': model, 'law': law, 'public': public, 'old': old, 'gp': gp,
            'old_gp_covariance': old_gp_covariance, 'query_psi': query_psi, 'source_ids': source_ids, 'expanded': expanded})
    save(out / 'source_rosters.json', {'execution': spec, 'rows': rosters, 'rosters_frozen_before_added_source_labels': True,
        'both_old_gp_conditioner_reproduction_checks_passed': True})
    raw = read(coverage['source_data']); fields, laws = [], []
    for item in work:
        layout, model, law = item['layout'], item['model'], item['law']; tasks = []
        for q, sites, old in zip(item['source_ids'], item['expanded'], item['old']):
            values = np.asarray([[value[0] for value in raw['tasks'][q]['Y_replicates'][j]] for j in sites]); assert values.shape == (24, 3)
            task = {'name': old['name'], 'psi': item['public'][sites], 'features': item['public'][sites],
                'response': values.mean(axis=1)[:, None], 'noise': (values.var(axis=1, ddof=1) / 3)[:, None]}
            for key in ('psi', 'features'): np.testing.assert_allclose(task[key][:12], old[key], atol=1e-12, rtol=1e-12)
            for key in ('response', 'noise'): np.testing.assert_allclose(task[key][:12, 0], np.asarray(old[key])[:, 0], atol=1e-12, rtol=1e-12)
            tasks.append(task)
        expanded_model = JointSpectralGeometryLikelihood(model.basis, tasks, *model.templates, model.columns, model.center, model.white, .01)
        _, _, state = expanded_model.evaluate(np.asarray(law['parameters'])); assert expanded_model.label_count == 96
        for key in ('common_prior_coefficient_covariance', 'task_coefficient_covariance'):
            np.testing.assert_allclose(state[key], law[key], atol=1e-10, rtol=1e-10)
        h = state['source_posterior_common_covariance']; assert np.linalg.eigvalsh(np.asarray(law['source_posterior_common_covariance']) - h).min() >= -1e-10
        design = np.column_stack([np.ones(12), model.basis.transform(item['query_psi'])])
        f_mean = design @ state['source_posterior_coefficient_mean']
        f_covariance = design @ (h + state['task_coefficient_covariance']) @ design.T + model.white * np.eye(12)
        gp_mean, gp_covariance, gp_nll = fixed_gp(tasks, item['query_psi'], item['gp'])
        assert np.linalg.eigvalsh(item['old_gp_covariance'] - gp_covariance).min() >= -1e-10
        base_prior = next(r for r in base_priors if r['layout'] == layout)
        f_means = np.stack([f_mean, gp_mean], axis=1); f_variances = np.stack([np.diag(f_covariance), np.diag(gp_covariance)], axis=1)
        assert np.all(f_variances > 0)
        fields.append({'layout': layout, 'sites': base_prior['sites'], 'f_means': f_means.tolist(), 'f_variances': f_variances.tolist(),
            'eta_mean': base_prior['eta_mean'], 'eta_variance': base_prior['eta_variance'], 'source_task_names': law['source_task_names'],
            'common_H_and_GP_predictive_covariance_information_order_verified': True})
        laws.append({'layout': layout, 'source_task_names': law['source_task_names'], 'frozen_parameters': law['parameters'],
            'source96_beta': state['source_posterior_coefficient_mean'].tolist(), 'source96_H': h.tolist(), 'gp_source96_nll_per_label': gp_nll,
            'no_covariance_template_or_parameter_refit': True})
    del raw
    save(out / 'predictions.json', {'execution': spec, 'rows': fields, 'all_four_zero_shot_forecasts_frozen_before_source_validation_labels': True})
    save(out / 'source_laws.json', {'execution': spec, 'rows': laws})
    rows = []; base = {'execution': spec, 'no_model_adopted': True, 'outer_task1_outcomes_indexed': False, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False, 'extra_g_or_risk_labels_used': False}
    for field in fields:
        layout = field['layout']; inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == f'related_inventory_{query:02d}'); labels = np.asarray(task['response'])
        f_means, f_variances, eta_mean, eta_variance = (np.asarray(field[k]) for k in ('f_means', 'f_variances', 'eta_mean', 'eta_variance'))
        local = {arm: [] for arm in ARMS}
        for j, label in enumerate(labels):
            if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False, 'stopped_at_cpu_budget': True,
                    'cpu_seconds': time.process_time() - started}); return
            result = conditional_pair(f_means[j], f_variances[j], eta_mean[j], eta_variance[j], label[0], label[3])
            for r in result: local[r['arm']].append({'policy': field['sites'][j], **r})
        old_field = next(r for r in base_priors if r['layout'] == layout)
        for index, arm in enumerate(ARMS):
            mse = decompose(f_means[:, index], labels[:, 0]); old_mse = decompose(np.asarray(old_field['f_means'])[:, index], labels[:, 0])
            density = float(np.mean([r['conditional_mean_density'] for r in local[arm]])); old_density = baseline['summary'][layout][arm]['conditional_mean_density']
            rows.append({'layout': layout, 'arm': arm, 'source_validation_task': query, 'historical_calls_per_arm': 288, 'mean_error': mse,
                'conditional_mean_density': density, 'sites': local[arm], 'baseline_mean_error': old_mse, 'baseline_conditional_mean_density': old_density,
                'mse_relative_change_vs_baseline': mse['mse'] / old_mse['mse'] - 1,
                'conditional_mean_density_change_vs_baseline': density - old_density})
    comparisons, passed = {}, True
    for layout in spec['layouts']:
        s, g = (next(r for r in rows if (r['layout'], r['arm']) == (layout, arm)) for arm in ARMS)
        ratio = s['mean_error']['mse'] / g['mean_error']['mse']; old_ratio = s['baseline_mean_error']['mse'] / g['baseline_mean_error']['mse']
        gap = s['conditional_mean_density'] - g['conditional_mean_density']; old_gap = s['baseline_conditional_mean_density'] - g['baseline_conditional_mean_density']
        own_improved = s['mean_error']['mse'] <= s['baseline_mean_error']['mse'] and s['conditional_mean_density'] >= s['baseline_conditional_mean_density']
        matched_improved = ratio < old_ratio or gap > old_gap
        comparisons[layout] = {'structure_GP_mse_ratio': ratio, 'baseline_structure_GP_mse_ratio': old_ratio,
            'structure_minus_GP_conditional_mean_density': gap, 'baseline_structure_minus_GP_conditional_mean_density': old_gap,
            'own_baseline_gate_passed': own_improved, 'matched_GP_gap_improved': matched_improved, 'screening_passed': own_improved and matched_improved}
        passed = passed and comparisons[layout]['screening_passed']
    save(out / 'results.json', {**base, 'rows': rows, 'comparisons': comparisons, 'source_coverage_screening_passed': passed,
        'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started, 'cost': spec['cost'],
        'interpretation': 'Fixed-law source conditioning contrast at144 versus288 historical calls,not equal-budget algorithm superiority. Twelve-site opened source validation,zero paid updates. Extra source label uncertainty still treated with the frozen Gaussian plug-in approximation.'})
    print('crossed source reconditioning', [(r['layout'], r['arm'], r['mse_relative_change_vs_baseline'], r['conditional_mean_density_change_vs_baseline']) for r in rows],
        'screening', comparisons, 'passed', passed, 'CPU', time.process_time() - started, flush=True)


if __name__ == '__main__':
    run()
