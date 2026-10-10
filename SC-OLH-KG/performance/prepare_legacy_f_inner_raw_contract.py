"""Honest source4 risk laws and finite-moment intercept reference rules."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import encode_diverse_source
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.fit_legacy_f_source_predictive_shape import outer_law
from performance.prepare_legacy_f_task_level_reference_quadrature import posterior_density_u, rule_on_intervals
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, fit_orthogonal_function_prior
from representation.public_control_mean import source_prior_moments
from representation.source_function_prior import head_batches
from representation.source_joint_population import fit_joint_source_population, fit_joint_common_source_population
from representation.task_level_reference import TaskLevelReferencePosterior

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_raw_source_execution_v1.json'


def finite_reference(inputs, native, spec):
    """For four source tasks integrate E[v] and bounded common moments,not E[v²]."""
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); tasks = inputs['source_tasks']
    designs = [np.column_stack([np.ones(len(t['psi'])), basis.transform(np.asarray(t['psi']))]) for t in tasks]
    d = np.vstack(designs); m0 = np.asarray(native['empirical_center_coefficient_mean'])
    a = np.diag(native['common_prior_coefficient_diagonal']); r = np.asarray(native['residual_coefficient_diagonal'])
    signal, variance = native['source_signal_scale'], float(r[0]); r0 = r.copy(); r0[0] = 0.
    error = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks])-d@m0
    c0 = d@a@d.T+block_diag(*[(z*r0)@z.T+np.diag(np.asarray(t['noise'])[:, 0]+native['discrepancy_variance']) for z, t in zip(designs, tasks)])
    u = block_diag(*[np.ones((len(z), 1)) for z in designs]); factor = cho_factor(c0, lower=True)
    inverse_u, alpha = cho_solve(factor, u), cho_solve(factor, error)
    b = u.T@inverse_u; eigen, vectors = np.linalg.eigh(.5*(b+b.T))
    assert len(eigen) == len(tasks) == 4 and np.all(eigen > 0)
    rhs = vectors.T@u.T@alpha; update = a@d.T@inverse_u@vectors
    beta0 = m0+a@d.T@alpha; h0 = a-a@d.T@cho_solve(factor, d@a); h0 = .5*(h0+h0.T)
    nll0 = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha)
    posterior = TaskLevelReferencePosterior(eigen, rhs, signal, variance, spec['quadrature'], signal)
    def conditional(v):
        w = v/(1+v*eigen)
        return beta0-update@(w*rhs), h0+(update*w)@update.T
    beta, h = conditional(variance)
    np.testing.assert_allclose((nll0+posterior.nll_difference(variance))/len(error), native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(beta, native['coefficient_mean'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(h, native['population_mean_covariance'], atol=1e-8, rtol=1e-8)
    dense = cho_factor(c0+signal*u@u.T, lower=True); dense_alpha = cho_solve(dense, error)
    db, dh = conditional(signal)
    np.testing.assert_allclose(db, m0+a@d.T@dense_alpha, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(dh, a-a@d.T@cho_solve(dense, d@a), atol=1e-8, rtol=1e-8)
    dnll = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(dense[0])).sum()+error@dense_alpha)
    np.testing.assert_allclose(nll0+posterior.nll_difference(signal), dnll, atol=1e-8, rtol=1e-8)
    dense_b = u.T@cho_solve(dense, u)
    np.testing.assert_allclose(posterior.information(signal), .5*np.trace(dense_b@dense_b), atol=1e-8, rtol=1e-8)
    law = {'source_task_names': [t['name'] for t in tasks], 'native_law': native,
        'task_variance_eigenvalues': eigen.tolist(), 'task_variance_rhs': rhs.tolist(),
        'coefficient_mean_at_zero_task_intercept_variance': beta0.tolist(),
        'common_covariance_at_zero_task_intercept_variance': h0.tolist(),
        'common_coefficient_update_columns': update.tolist(), 'source_signal': signal,
        'native_task_intercept_variance': variance,
        'reference_log_normalizer_relative_to_native_nll': float(posterior.log_scale+np.log(posterior.normalization))}
    def integrand(x):
        if x == 0.: return np.zeros(2+len(eigen)+len(eigen)**2)
        v = signal*(1-x*x)/(x*x); shrink = v*eigen/(1+v*eigen)
        return posterior_density_u(x, law)*np.r_[1., v/signal, shrink, np.outer(shrink, shrink).ravel()]
    points = sorted(set(np.sqrt(signal/(signal+v)) for v in (variance, signal) if v > 0))
    t = spec['quadrature']
    values, error_bound, info = quad_vec(integrand, 0., 1., points=points, norm='max', full_output=True, workers=1,
        epsabs=t['absolute_tolerance_for_scaled_integrals'], epsrel=t['relative_tolerance'], limit=t['maximum_subintervals'])
    if not info.success: raise RuntimeError(info.message)
    assert error_bound <= max(t['absolute_tolerance_for_scaled_integrals'], t['relative_tolerance']*np.max(np.abs(values)))
    np.testing.assert_allclose(values[0], 1., atol=2*t['relative_tolerance'], rtol=0.)
    shrink = values[2:6]; shrink_cov = values[6:].reshape(4, 4)-np.outer(shrink, shrink)
    mean_update = update*(rhs/eigen)
    integrated_beta = beta0-mean_update@shrink
    integrated_h = h0+(update*(shrink/eigen))@update.T+mean_update@shrink_cov@mean_update.T
    integrated_h = .5*(integrated_h+integrated_h.T); assert np.linalg.eigvalsh(integrated_h).min() >= -1e-10
    law.update(integrated_common_coefficient_mean=integrated_beta.tolist(),
        integrated_common_coefficient_covariance=integrated_h.tolist(), integrated_task_intercept_variance=float(signal*values[1]))
    record = {'source_task_count': 4, 'source_label_count': len(error),
        'reference_posterior': {'mean': float(signal*values[1]), 'variance_second_moment': 'infinite',
            'variance_standard_deviation': 'infinite', 'density_tail_power': -3,
            'normalization_relative_error': posterior.normalization_error/posterior.normalization,
            'normalized_vector_integral_error': float(error_bound), 'normalized_mass_error': float(abs(values[0]-1))},
        'native_and_interior_dense_likelihood_fisher_and_common_moments_verified': True,
        'finite_full_tail_moments_integrated': True, 'adaptive_u_intervals': sorted(info.intervals.tolist())}
    return record, law


def finite_rule_errors(nodes, weights, law):
    mass = weights.sum(); probability = weights/mass
    e, rhs = np.asarray(law['task_variance_eigenvalues']), np.asarray(law['task_variance_rhs'])
    update = np.asarray(law['common_coefficient_update_columns']); w = nodes[:, None]/(1+nodes[:, None]*e)
    beta = np.asarray(law['coefficient_mean_at_zero_task_intercept_variance'])-(w*rhs)@update.T
    mean = probability@beta; displacement = beta-mean
    h = np.asarray(law['common_covariance_at_zero_task_intercept_variance'])+(update*(probability@w))@update.T+(displacement.T*probability)@displacement
    expected_h = np.asarray(law['integrated_common_coefficient_covariance'])
    return {'raw_mass_absolute': float(abs(mass-1)),
        'mean_variance_relative': float(abs(probability@nodes/law['integrated_task_intercept_variance']-1)),
        'coefficient_mean_max_absolute': float(np.max(np.abs(mean-law['integrated_common_coefficient_mean']))),
        'common_covariance_relative_frobenius': float(np.linalg.norm(h-expected_h)/np.linalg.norm(expected_h))}


def run(execution=EXECUTION):
    started = time.process_time(); spec = read(execution); plan = read(spec['plan'])
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    coverage, _ = source_paths(read(plan['source_context_execution'])); parent = read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    roster = [i for i in range(6) if i != plan['outer_deleted_task']]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank, 50) for i in roster}
    raw = read(coverage['source_data']); data = {'bank': raw['bank'], 'tasks': {i: raw['tasks'][i] for i in roster}}; del raw
    inner = read(plan['inner_source_results']); fitted_q = read(plan['frozen_q_results'])
    rules, records, branches = [], [], []; fit_count = 0
    for case in spec['cases']:
        layout, query = case['layout'], case['query_task']; training = [i for i in roster if i != query]
        query_name = f'related_inventory_{query:02d}'
        folder = out/case['folder']; folder.mkdir(parents=True, exist_ok=True)
        parent_layout = coverage['parent_layouts'][layout]; sites = [parent['layouts'][parent_layout][i] for i in training]
        assert all(set(s).issubset(coverage['layouts'][layout][i]) and len(s) == 8 for i, s in zip(training, sites))
        selected_data = {'bank': data['bank'], 'tasks': {i: data['tasks'][i] for i in training}}
        coordinate, risk_tasks, center, scale = encode_diverse_source(selected_data, training, sites, descriptors, public['plan']['task_distribution']['L'])
        names = [t['name'] for t in risk_tasks]
        assert set(names) == {f'related_inventory_{i:02d}' for i in training} and query_name not in names
        reference, held_psi = coordinate.transform(descriptors[training[0]]), coordinate.transform(descriptors[query])
        risk_basis = OrthogonalLowOrderBasis(reference); batches = head_batches(risk_tasks, 3); noise = [t['noise'][:, 3] for t in risk_tasks]
        risk_inputs = {'source_tasks': [{**t, **{k: t[k].tolist() for k in ('psi', 'features', 'response', 'noise')}} for t in risk_tasks],
            'training_task_ids': training, 'source_site_indices_by_task': sites, 'public_reference_psi': reference.tolist(),
            'heldout_public_psi': held_psi.tolist(), 'coordinate': coordinate.diagnostics(),
            'source_feature_center': center.tolist(), 'source_feature_scale': scale.tolist(), 'query_task': query,
            'both_deleted_tasks_excluded_from_response_fits': True}
        save(folder/'risk_inputs.json', risk_inputs)
        risk_fits = {}; settings = plan['risk_optimizer']
        regularizer = fit_orthogonal_function_prior(batches, noise, risk_basis, max_iterations=settings['regularizer_max_iterations'])
        risk_fits['regularizer'] = regularizer.diagnostics; fit_count += 1
        fixed = fit_joint_source_population(regularizer, batches, noise, max_iterations=settings['fixed_population_max_iterations'])
        risk_fits['fixed_structural'] = fixed.diagnostics; fit_count += 1
        joint = fit_joint_common_source_population(regularizer, batches, noise, fixed.diagnostics, optimizer=settings['joint_optimizer'])
        risk_fits['joint_structural'] = joint.diagnostics; fit_count += 1
        save(folder/'risk_source_fit.json', {'source_task_names': names, 'head': 3, 'fits': risk_fits})
        converged = all(f['optimizer_converged'] for f in risk_fits.values())
        print('source4 risk fitted', layout, {k: (f['optimizer_converged'], f['optimizer_iterations']) for k, f in risk_fits.items()}, flush=True)
        if not converged:
            save(out/'results.json', {'execution': spec, 'all_source_laws_complete': False, 'failed_risk_layout': layout,
                'actual_risk_objective_fits': fit_count, 'raw_forecasts_scored': False, 'cpu_seconds': time.process_time()-started})
            return
        eta_mean, eta_cov = source_prior_moments(restore_joint(risk_basis, joint.diagnostics), held_psi)
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f"task{plan['outer_deleted_task']}"/'inputs.json')
        kept = {**inputs, 'source_tasks': [t for t in inputs['source_tasks'] if t['name'] in names]}
        row = next(r for r in inner['rows'] if r['layout'] == layout and r['inner_query_task'] == query_name)
        source = read(row['source_fit']); assert source['inner_training_task_names'] == names
        head = {'regularizer': source['fits']['regularizer'], 'joint_structural': source['fits']['structural'], 'joint_gp': source['fits']['gp']}
        selected_q = next(r['q'] for r in fitted_q['rows'] if r['layout'] == layout)
        for arm, q in [('native', 0.), ('shape', selected_q)]:
            native = outer_law(kept, head, q)
            record, law = finite_reference(kept, native, plan); trials = []; selected = None
            for size in plan['quadrature']['positive_rule_nodes_per_interval']:
                nodes, weights = rule_on_intervals(law, record['adaptive_u_intervals'], size)
                error = finite_rule_errors(nodes, weights, law); passed = all(error[k] <= t for k, t in spec['finite_moment_tolerances'].items())
                trials.append({'nodes_per_interval': size, 'component_count': len(nodes), 'errors': error, 'passed': passed})
                if passed:
                    selected = {'layout': layout, 'query_task': query, 'arm': arm, 'variance_nodes': nodes.tolist(), 'posterior_weights': (weights/weights.sum()).tolist(),
                        'source_reference_law': law, 'adaptive_u_intervals': record['adaptive_u_intervals'], 'nodes_per_interval': size}
                    rules.append(selected); break
            records.append({'layout': layout, 'query_task': query, 'arm': arm, **record, 'rule_trials': trials, 'rule_complete': selected is not None})
            print('source4 finite variance rule', layout, query, arm, 'mean', record['reference_posterior']['mean'], 'nodes', len(selected['variance_nodes']) if selected else None, flush=True)
        branch = {'layout': layout, 'query_task': query, 'source_task_names': names, 'eta_mean': eta_mean.tolist(), 'eta_covariance': eta_cov.tolist(),
            'public_mean_reference': inputs['public_mean_reference'], 'query_mean_psi': next(t['psi'] for t in inputs['source_tasks'] if t['name'] == query_name),
            'paid_prefix': plan['source_anchors'], 'scoring_sites': coverage['layouts'][layout][query][4:]}
        branches.append(branch)
    complete = len(rules) == 2*len(spec['cases']) and all(r['rule_complete'] for r in records)
    save(out/'source_quadrature.json', {'execution': spec, 'all_source_rules_frozen_before_raw_query_values': complete, 'rows': rules})
    save(out/'source_branches.json', {'execution': spec, 'all_honest_risk_laws_complete': True, 'rows': branches})
    result = {'execution': spec, 'rows': records, 'all_source_laws_complete': complete,
        'cost': {**plan['cost'], **spec['cost'], 'actual_risk_coordinate_refits': len(branches), 'actual_risk_objective_fits': fit_count, 'actual_source_variance_integrations': len(records),
            'actual_inner_raw_forecasts': 0, 'actual_numerical_reference_forecasts': 0},
        'raw_query_values_used': False, 'no_model_adopted': True, 'source6_7_evaluated': False, 'target_promoted': False,
        'independent_confirmation': False, 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)


if __name__ == '__main__':
    run()
