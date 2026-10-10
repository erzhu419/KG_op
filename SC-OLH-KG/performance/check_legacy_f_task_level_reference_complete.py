"""Frozen nested f variance/g correlation mixture under complete source gates."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_counterbalanced_complete_mean_uncertainty import contrast, load_group
from performance.check_legacy_crossfit_variance_noise import risk_models
from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_public_mean_risk_uncertainty import forecast, read
from performance.check_legacy_task_bank_centering_holdout import ARM, centered_model
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.run_legacy_related_inventory_source import save, training_view
from performance.vectorized_joint_batch_density import component_batch_log_evidence
from representation.joint_function_risk import local_joint_moments
from representation.joint_response_mixture import JointResponseMixture
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.task_bank_centering import centered_g_field

EXECUTION = 'performance/manifests/legacy_joint_population_f_task_level_reference_complete_execution_v1.json'


def build_model(group, means, psi, risk_queries, g_law, source, nodes, weights):
    basis = OrthogonalLowOrderBasis(group[0][0]['public_mean_reference'])
    design = np.column_stack([np.ones(len(psi)), basis.transform(psi)]); center = design[:, 1:].mean(axis=0)
    template = risk_models(group, means, psi, risk_queries, [ARM])[0][1]
    pairs = []
    for f, g in template.components:
        mean, covariance = f.predict(full_cov=True)
        eta = mean[:, 1], covariance[:, 1, :, 1]
        pairs.append((mixture_field(basis, psi, source, nodes, weights, eta), centered_g_field(g, design, g_law, center)))
    return JointResponseMixture(template.names, pairs)


def model_forecast(query, model):
    mean, covariance = model.predict_joint(); n = len(mean)
    variance = np.diag(covariance.reshape(4*n, 4*n)).reshape(n, 4)
    episode = np.zeros((n, 2))
    for identity_weight, pair in zip(model.weights, model.components):
        for h, field in enumerate(pair):
            diagonal = np.diagonal(field.covariances, axis1=1, axis2=2)
            episode[:, h] += identity_weight*(np.exp(field.log_weights)@np.exp(field.means[:, 1::2]+.5*diagonal[:, 1::2]))
    row = {key: query[key] for key in ('task_id', 'task_index', 'kind')}
    row.update(mean_arm=ARM, risk_case='f_variance_mixture_g_correlation', head_order=['f', 'g', 'eta_g', 'eta_f'],
        mean=mean[:, [0, 2, 3, 1]].tolist(), variance=variance[:, [0, 2, 3, 1]].tolist(),
        expected_episode_variance=episode.tolist(), identity_names=model.names, identity_weights=model.weights.tolist(),
        paid_predictive_log_score=float(sum(r['mixture_log_score'] for r in model.history)), paid_history=list(model.history))
    return row, mean, covariance


def density_snapshot(model):
    pairs = []
    for pair in model.components:
        heads = []
        for field in pair:
            indices = np.arange(2*field.candidate_count).reshape(-1, 2)
            heads.append({'log_weights': field.log_weights.copy(), 'means': field.means.reshape(-1, field.candidate_count, 2).copy(),
                'covariances': field.covariances[:, indices[:, :, None], indices[:, None, :]].copy()})
        pairs.append(heads)
    return {'log_weights': model.log_weights.copy(), 'components': pairs}


def site_density(snapshot, site, value, replicates):
    scores = []; worst_error = 0.
    for pair in snapshot['components']:
        heads = []
        for h, field in enumerate(pair):
            density, error = component_batch_log_evidence(field['means'][:, site], field['covariances'][:, site],
                value[h], value[3-h], replicates)
            heads.append(logsumexp(field['log_weights']+density)); worst_error = max(worst_error, error)
        scores.append(heads)
    scores = np.asarray(scores); weights = snapshot['log_weights']
    return {'joint': float(logsumexp(weights+scores.sum(axis=1))),
        'heads': logsumexp(weights[:, None]+scores, axis=0), 'maximum_component_integration_error_bound': worst_error}


def validate_training(spec, layout, group, means, g_law, rule, raw, public):
    started = time.process_time(); psi = np.asarray(group[0][0]['public_mean_reference'])
    risk_queries = [np.asarray(g[0]['public_risk_reference']) for g in group]
    basis = OrthogonalLowOrderBasis(group[0][0]['public_mean_reference'])
    design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
    source = rule['source_reference_law']
    native = centered_model(risk_models(group, means, psi, risk_queries, [ARM])[0][1], design, g_law, design[:, 1:].mean(axis=0))
    one = build_model(group, means, psi, risk_queries, g_law, source,
        np.array([source['native_task_intercept_variance']]), np.ones(1))
    mixed = build_model(group, means, psi, risk_queries, g_law, source,
        np.asarray(rule['variance_nodes']), np.asarray(rule['posterior_weights']))
    labels = training_view(raw, [0], public['plan']['task_distribution']['L'])[0]
    query = {'task_id': raw['tasks'][0]['task_id'], 'task_index': 0, 'kind': 'source'}
    max_difference = max_error_bound = 0.; reference_count = 0; fast_cpu = reference_cpu = 0.
    for stage in ('prior', 'raw_paid'):
        actual, am, ac = model_forecast(query, one); expected, em, ec = model_forecast(query, native)
        np.testing.assert_allclose(am, em, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(ac, ec, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(actual['paid_predictive_log_score'], expected['paid_predictive_log_score'], atol=1e-8, rtol=1e-8)
        old = forecast(query, ARM, 'original_interface', native)
        for key in ('mean', 'variance', 'expected_episode_variance'):
            np.testing.assert_allclose(expected[key], old[key], atol=1e-12, rtol=1e-12)
        for identity, pair in enumerate(mixed.components):
            for site in ([0, 35, 72] if identity == 0 else [35]):
                value = labels['response'][site]
                for h, field in enumerate(pair):
                    local = field.local_components(site)
                    start = time.process_time()
                    fast, bound = component_batch_log_evidence([c[1] for c in local], [c[2] for c in local], value[h], value[3-h], 3)
                    fast_cpu += time.process_time()-start; start = time.process_time()
                    reference = [local_joint_moments(m, c, value[h], value[3-h], 3)['log_evidence'] for _, m, c in local]
                    reference_cpu += time.process_time()-start; reference_count += len(reference)
                    max_difference = max(max_difference, float(np.max(np.abs(fast-reference)))); max_error_bound = max(max_error_bound, bound)
        if stage == 'prior':
            for j in spec['paid_candidates']:
                values = raw['tasks'][0]['Y_replicates'][j][0]; values = [values[0], values[1]-raw['tasks'][0]['tau']]
                for model in (native, one, mixed): model.update_raw(j, values)
    passed = max_difference <= spec['numerical_tolerances']['training_component_log_density_absolute']
    return {'layout': layout, 'training_task': 0, 'native_interface_verified': True,
        'component_log_density_max_absolute_difference': max_difference, 'maximum_component_integration_error_bound': max_error_bound,
        'reference_component_count': reference_count, 'vectorized_density_cpu_seconds': fast_cpu,
        'original_density_cpu_seconds': reference_cpu, 'passed': passed, 'cpu_seconds': time.process_time()-started}


def run():
    started = time.process_time(); spec = read(EXECUTION); training = read(spec['training_results'])
    assert training['passes_both_layouts'] and training['numerical_density_verified']
    assert read(spec['source_rule_results'])['all14_source_rules_complete']
    frozen = read(spec['source_rules']); assert frozen['all14_rules_frozen_without_query_data']
    assert read(spec['g_results'])['same_information_source_gate_passed']
    g_laws = read(spec['g_source_laws']); assert g_laws['both_laws_frozen_before_heldout_paid_observations']
    coverage = read(spec['coverage_execution']); parent = read(coverage['source_execution'])
    native_spec = read(spec['native_execution']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    groups, means, rules, laws = {}, {}, {}, {}
    for layout in spec['layouts']:
        groups[layout] = load_group(spec, coverage, parent, layout)
        means[layout] = read(Path(native_spec['output'])/f'layout{layout}'/'selected_source_fit.json')
        rules[layout] = next(r for r in frozen['rows'] if r['layout'] == layout and r['deleted_task'] is None)
        laws[layout] = next(r['law'] for r in g_laws['rows'] if r['layout'] == layout)
    save(out/'frozen_model_inputs.json', {'execution': spec, 'all_complete_inputs_frozen_before_heldout_outcomes': True,
        'risk_identity_names': ['full144']+[f'drop{i}' for i in spec['training_tasks']],
        'rows': [{'layout': l, 'f_rule': rules[l], 'g_law': laws[l]} for l in spec['layouts']]})
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); raw = read(coverage['source_data'])
    validation = []
    for layout in spec['layouts']:
        checked = validate_training(spec, layout, groups[layout], means[layout], laws[layout], rules[layout], raw, public)
        validation.append(checked); print('training interface/density', checked, flush=True)
    save(out/'training_validation.json', {'execution': spec, 'source6_7_outcomes_indexed': False, 'rows': validation})
    if not all(r['passed'] for r in validation):
        save(out/'results.json', {'execution': spec, 'training_validation_passed': False, 'forecasts_scored': False,
            'no_model_adopted': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started}); return
    rows, densities, refined_densities, numerical = [], {}, {}, []
    t = spec['numerical_tolerances']
    for layout in spec['layouts']:
        group, rule = groups[layout], rules[layout]; source = rule['source_reference_law']
        fine_nodes, fine_weights = rule_on_intervals(source, rule['adaptive_u_intervals'], 32)
        for i in spec['heldout_tasks']:
            task = raw['tasks'][i]; query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source'}
            psi = np.asarray(group[0][0]['heldout_mean_psi'][str(i)])
            risk_queries = [np.asarray(g[0]['heldout_risk_psi'][str(i)]) for g in group]
            forecast_pair = []; errors = {'mean_max_absolute': 0., 'covariance_relative_frobenius': 0., 'paid_log_evidence_absolute': 0.}
            for nodes, weights in ((np.asarray(rule['variance_nodes']), np.asarray(rule['posterior_weights'])), (fine_nodes, fine_weights)):
                model = build_model(group, means[layout], psi, risk_queries, laws[layout], source, nodes, weights)
                prior, pm, pc = model_forecast(query, model)
                for j in spec['paid_candidates']:
                    value = task['Y_replicates'][j][0]; model.update_raw(j, [value[0], value[1]-task['tau']])
                paid, mm, mc = model_forecast(query, model)
                paid.update(layout=layout, parent_layout=spec['parent_layouts'][layout], prior=prior, variance_node_count=len(nodes),
                    paid_mean_task_intercept_variance_by_identity=[float(np.exp(f.log_weights)@nodes) for f, _ in model.components])
                forecast_pair.append((paid, (pm, pc, mm, mc), density_snapshot(model))); del model
            row, actual, compact = forecast_pair[0]; reference, fine, refined = forecast_pair[1]
            for am, ac, fm, fc in ((actual[0], actual[1], fine[0], fine[1]), (actual[2], actual[3], fine[2], fine[3])):
                errors['mean_max_absolute'] = max(errors['mean_max_absolute'], float(np.max(np.abs(am-fm))))
                errors['covariance_relative_frobenius'] = max(errors['covariance_relative_frobenius'], float(np.linalg.norm(ac-fc)/np.linalg.norm(fc)))
            errors['paid_log_evidence_absolute'] = abs(row['paid_predictive_log_score']-reference['paid_predictive_log_score'])
            passed = all(errors[key] <= t[key] for key in errors)
            numerical.append({'layout': layout, 'task_index': i, 'errors': errors, 'passed': passed})
            row['refined_forecast'] = reference; rows.append(row); densities[(layout, i)] = compact; refined_densities[(layout, i)] = refined
            print('complete forecast', layout, i, len(rule['variance_nodes']), passed, errors, flush=True)
    save(out/'source_predictions.json', {'execution': spec, 'all4_selected_refined_forecasts_frozen_before_scored_labels': True,
        'training_validation': validation, 'numerical_checks': numerical, 'rows': rows})
    if not all(r['passed'] for r in numerical):
        save(out/'results.json', {'execution': spec, 'training_validation_passed': True, 'numerical_forecasts_verified': False,
            'forecasts_scored': False, 'numerical_checks': numerical, 'scientific_gate_interpreted': False,
            'no_model_adopted': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started}); return
    labels = {i: training_view(raw, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}; del raw
    density_checks = []; integration_error = 0.
    for row in rows:
        key = row['layout'], row['task_index']; label = labels[key[1]]
        mean, variance = np.asarray(row['mean']), np.asarray(row['variance']); error = mean-label['response']
        observation_variance = variance.copy(); observation_variance[:, :2] += np.asarray(row['expected_episode_variance'])/3
        observation_variance[:, 2:] += label['noise'][:, 2:]; standardized = error/np.sqrt(observation_variance)
        scores = []; max_difference = 0.
        for j in range(2, len(mean)):
            score = site_density(densities[key], j, label['response'][j], 3); scores.append(score)
            integration_error = max(integration_error, score['maximum_component_integration_error_bound'])
            if j in (2, 36, 72):
                fine_score = site_density(refined_densities[key], j, label['response'][j], 3)
                max_difference = max(max_difference, abs(score['joint']-fine_score['joint']),
                    float(np.max(np.abs(score['heads']-fine_score['heads']))))
        row['scores'] = {'observed_head_mse': (error[2:]**2).mean(axis=0).tolist(),
            'joint_batch_log_score': float(np.mean([s['joint'] for s in scores])),
            'response_batch_log_scores': np.mean([s['heads'] for s in scores], axis=0).tolist(),
            'predictive_moment_band_95_observed_coverage': (abs(standardized[2:]) <= 1.96).mean(axis=0).tolist(),
            'predictive_standardized_residual_rms': np.sqrt((standardized[2:]**2).mean(axis=0)).tolist()}
        row['scored_site_densities'] = [{'site': j, 'joint': s['joint'], 'heads': s['heads'].tolist()} for j, s in enumerate(scores, 2)]
        density_checks.append({'layout': key[0], 'task_index': key[1], 'sites': [2, 36, 72],
            'maximum_log_density_difference': max_difference, 'passed': max_difference <= t['raw_density_refinement_absolute']})
        print('complete scoring', key, row['scores']['response_batch_log_scores'], max_difference, flush=True)
    references, parents, g_profiles = (read(spec[k])['rows'] for k in ('reference_profiles', 'parent_reference_profiles', 'g_profiles'))
    contrasts, parent_contrasts, previous = [], [], []
    for row in rows:
        layout, i = row['layout'], row['task_index']
        same = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == ARM)
        old = next(r for r in parents if r['layout'] == row['parent_layout'] and r['task_index'] == i and r['mean_arm'] == ARM)
        g_only = next(r for r in g_profiles if r['layout'] == layout and r['task_index'] == i)
        contrasts.append(contrast(row, same, 'same216_call_native_structural'))
        parent_contrasts.append(contrast(row, old, 'original144_call_structural'))
        previous.append({'layout': layout, 'task_index': i,
            'joint_batch_density_difference': row['scores']['joint_batch_log_score']-g_only['scores']['joint_batch_log_score'],
            'head_batch_density_differences': (np.asarray(row['scores']['response_batch_log_scores'])-g_only['scores']['response_batch_log_scores']).tolist(),
            'observed_head_mse_differences': (np.asarray(row['scores']['observed_head_mse'])-g_only['scores']['observed_head_mse']).tolist()})
    verified = all(r['passed'] for r in density_checks)
    same_pass, old_pass = all(c['passed'] for c in contrasts), all(c['passed'] for c in parent_contrasts)
    result = {'execution': spec, 'training_validation': validation, 'numerical_checks': numerical, 'density_checks': density_checks,
        'training_validation_passed': True, 'numerical_forecasts_verified': True, 'numerical_density_verified': verified,
        'maximum_component_density_integration_error_bound': integration_error, 'forecasts_scored': True,
        'scientific_gate_interpreted': verified, 'contrasts': contrasts, 'parent_contrasts': parent_contrasts,
        'stage104_contrasts': previous, 'same_information_source_gate_passed': same_pass if verified else None,
        'original144_source_gate_passed': old_pass if verified else None, 'source_gate_passed': (same_pass and old_pass) if verified else None,
        'candidate_stopped_after_complete_gate_failure': verified and not (same_pass and old_pass),
        'no_model_adopted': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': True, 'independent_confirmation': False, 'target_promoted': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'source_gate_passed': result['source_gate_passed'], 'same216': result['same_information_source_gate_passed'],
        'original144': result['original144_source_gate_passed'], 'numerical_density_verified': verified, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
