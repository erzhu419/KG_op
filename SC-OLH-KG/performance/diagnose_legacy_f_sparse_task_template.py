"""Source-only sparse task covariance direction in the original dictionary."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights

EXECUTION = 'performance/manifests/legacy_joint_population_f_sparse_task_template_execution_v1.json'


def source_paths(spec):
    reference = read(read(spec['mixture_execution'])['reference_execution'])
    centering = read(read(reference['source_execution'])['centering_execution'])
    return read(centering['coverage_execution']), read(centering['profile_execution'])


def sparse_template(basis, native, gp, compressed):
    anchor = gp['anchor']; scale2 = anchor['normalization']['scale']**2
    amplitude, intercept, _ = gp['residual_parameters']
    projected = np.asarray(compressed['prior_coefficient_covariance']).copy()
    projected[0, 0] -= scale2*anchor['constant_variance']
    projected *= amplitude/anchor['signal_variance']; projected[0, 0] += scale2*intercept
    projected = .5*(projected+projected.T)
    active = (np.asarray(native['source_group_variances']) > 0) | (np.asarray(native['common_group_variances']) > 0)
    columns = np.r_[0, 1+np.flatnonzero(active[basis.group_index])]
    principal = projected[np.ix_(columns, columns)]; diagonal = np.diag(principal)
    assert np.all(diagonal > 0)
    correlation = principal/np.sqrt(np.outer(diagonal, diagonal))
    budgets = np.bincount(basis.group_index, weights=np.diag(projected)[1:]) * active
    weights = spectral_weights(basis, native['spectral_exponent'])[0]
    marginal = np.r_[projected[0, 0], budgets[basis.group_index]*weights]
    template = np.zeros_like(projected)
    template[np.ix_(columns, columns)] = correlation*np.sqrt(np.outer(marginal[columns], marginal[columns]))
    tail = native['tail_variance_budget']/basis.feature_dim
    excess = np.r_[0., np.asarray(native['source_group_variances'])[basis.group_index]*weights]
    np.testing.assert_allclose(excess[1:]+tail, np.asarray(native['residual_coefficient_diagonal'])[1:], atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(np.diag(template), marginal, atol=1e-14, rtol=1e-12)
    assert np.linalg.eigvalsh(template).min() >= -1e-12
    inactive = np.flatnonzero(~active[basis.group_index])+1
    np.testing.assert_array_equal(template[inactive], 0.)
    return template, excess, tail, {'strong_columns': columns.tolist(), 'strong_group_names':
        [name for name, enabled in zip(basis.group_names, active) if enabled],
        'template_group_budgets': budgets.tolist(), 'task_spectral_exponent': native['spectral_exponent'],
        'template_level_variance': float(template[0, 0]), 'independent_tail_per_coefficient': float(tail),
        'template_minimum_eigenvalue': float(np.linalg.eigvalsh(template).min()),
        'psd_support_frequency_marginals_and_original_tail_verified': True}


def source_system(inputs, head, template, excess, tail, deleted):
    tasks = inputs['source_tasks']; native = head['joint_structural']
    assert {t['name'] for t in tasks} == {f'related_inventory_{i:02d}' for i in range(6) if i != deleted}
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    designs = [np.column_stack([np.ones(len(t['psi'])), basis.transform(np.asarray(t['psi']))]) for t in tasks]
    d = np.vstack(designs); m0 = np.asarray(head['regularizer']['coefficient_mean'])
    error = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks])-d@m0
    assert len(tasks) == 5 and len(error) == 60
    a = np.diag(native['common_prior_coefficient_diagonal'])
    weak = np.diag(np.r_[0., np.full(basis.feature_dim, tail)])
    base = weak+np.diag(excess); direction = template-np.diag(excess)
    c0 = d@a@d.T+block_diag(*[local@base@local.T+np.diag(np.asarray(t['noise'])[:, 0]+native['discrepancy_variance'])
        for local, t in zip(designs, tasks)])
    u = block_diag(*[np.ones((len(local), 1)) for local in designs])
    cq = block_diag(*[local@direction@local.T for local in designs])
    return {'d': d, 'm0': m0, 'error': error, 'a': a, 'c0': c0, 'u': u, 'cq': cq,
        'base': base, 'direction': direction, 'designs': designs, 'tasks': tasks, 'native': native}


def condition(system, q, variance, *, derivative=False):
    d, a, error = (system[key] for key in ('d', 'a', 'error'))
    c = system['c0']+q*system['cq']+variance*system['u']@system['u'].T
    factor = cho_factor(c, lower=True); alpha = cho_solve(factor, error)
    nll = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha)/len(error)
    beta = system['m0']+a@d.T@alpha
    h = a-a@d.T@cho_solve(factor, d)@a; h = .5*(h+h.T)
    value = {'source_negative_log_likelihood_per_label': float(nll), 'coefficient_mean': beta.tolist(),
        'population_mean_covariance': h.tolist()}
    if derivative:
        inverse_direction = cho_solve(factor, system['cq'])
        value['q_gradient_per_label'] = float(.5*(np.trace(inverse_direction)-alpha@system['cq']@alpha)/len(error))
        value['q_total_fisher_information'] = float(.5*np.trace(inverse_direction@inverse_direction))
    return value


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage, profile = source_paths(spec)
    geometry = read(spec['geometry_execution']); compressed = read(geometry['common_compression_laws'])['rows']
    rows, templates = [], []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
            native = head['joint_structural']; variance = native['residual_coefficient_diagonal'][0]
            saved = next(r for r in compressed if r['layout'] == layout and r['deleted_task'] == deleted and r['scope'] == 'full_dictionary')
            template, excess, tail, geometry_record = sparse_template(OrthogonalLowOrderBasis(inputs['public_mean_reference']), native, head['joint_gp'], saved)
            system = source_system(inputs, head, template, excess, tail, deleted)
            baseline = condition(system, 0., variance, derivative=True)
            for key in ('source_negative_log_likelihood_per_label', 'coefficient_mean', 'population_mean_covariance'):
                np.testing.assert_allclose(baseline[key], native[key], **spec['native_reproduction_tolerance'])
            q = spec['fixed_probe_q']; probe = condition(system, q, variance)
            r = system['base']+q*system['direction']; r[0, 0] += variance
            dense = system['d']@system['a']@system['d'].T+block_diag(*[local@r@local.T+np.diag(np.asarray(t['noise'])[:, 0]+native['discrepancy_variance'])
                for local, t in zip(system['designs'], system['tasks'])])
            np.testing.assert_allclose(dense, system['c0']+q*system['cq']+variance*system['u']@system['u'].T, atol=1e-12, rtol=1e-12)
            verification = {'native_beta_H_nll_reproduced': True, 'dense_and_component_source_covariance_match': True}
            if [layout, deleted] in spec['derivative_verification_folds']:
                step = spec['derivative_step']; f0 = baseline['source_negative_log_likelihood_per_label']
                f1, f2 = [condition(system, x, variance)['source_negative_log_likelihood_per_label'] for x in (step, 2*step)]
                finite_difference = (4*f1-f2-3*f0)/(2*step)
                np.testing.assert_allclose(finite_difference, baseline['q_gradient_per_label'], **spec['derivative_tolerance'])
                verification.update(q_gradient_one_sided_finite_difference=float(finite_difference), q_gradient_verified=True)
            rows.append({'layout': layout, 'deleted_task': deleted, 'native_task_intercept_variance': variance,
                **geometry_record, 'baseline': baseline, 'fixed_probe': probe,
                'probe_source_log_likelihood_gain_per_label': baseline['source_negative_log_likelihood_per_label']-probe['source_negative_log_likelihood_per_label'],
                'verification': verification})
            templates.append({'layout': layout, 'deleted_task': deleted, 'source_task_names': [t['name'] for t in system['tasks']],
                **geometry_record, 'template_coefficient_covariance': template.tolist(), 'original_task_excess_diagonal': excess.tolist()})
    summary = {layout: {'mean_probe_source_log_likelihood_gain_per_label': float(np.mean([r['probe_source_log_likelihood_gain_per_label'] for r in rows if r['layout'] == layout])),
        'improving_folds': sum(r['probe_source_log_likelihood_gain_per_label'] > 0 for r in rows if r['layout'] == layout),
        'mean_q_gradient_at_zero_per_label': float(np.mean([r['baseline']['q_gradient_per_label'] for r in rows if r['layout'] == layout]))}
        for layout in spec['layouts']}
    passed = all(r['mean_probe_source_log_likelihood_gain_per_label'] > 0 for r in summary.values())
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'source_templates.json', {'execution': spec, 'rows': templates, 'training_source_only': True})
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'source_direction_gate_passed': passed,
        'all12_source_constructions_verified': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'no_model_adopted': True, 'query_paid_outcomes_read': False, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'summary': summary, 'gate': passed, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
