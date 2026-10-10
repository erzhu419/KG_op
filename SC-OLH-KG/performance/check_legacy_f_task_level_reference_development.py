"""Frozen conditional variance mixtures under the unchanged f development gate."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_data_scaled_f_tail import NATIVE, forecast_field, score_f
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments

EXECUTION = 'performance/manifests/legacy_joint_population_f_task_level_reference_development_execution_v1.json'
CANDIDATE = 'conditional_task_intercept_variance_mixture'


def mixture_field(basis, psi, source, variances, weights, eta):
    design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
    law = source['native_law']; diagonal = np.asarray(law['residual_coefficient_diagonal']).copy(); diagonal[0] = 0.
    beta0 = np.asarray(source['coefficient_mean_at_zero_task_intercept_variance'])
    h0 = np.asarray(source['common_covariance_at_zero_task_intercept_variance'])
    update = design@np.asarray(source['common_coefficient_update_columns'])
    eigenvalues, rhs = np.asarray(source['task_variance_eigenvalues']), np.asarray(source['task_variance_rhs'])
    conditional = variances[:, None]/(1+variances[:, None]*eigenvalues)
    base = design@h0@design.T+(design*diagonal)@design.T+law['discrepancy_variance']*np.eye(len(psi))
    fmean = design@beta0-(conditional*rhs)@update.T
    n, k = len(psi), len(variances)
    means = np.empty((k, n, 2)); means[:, :, 0] = fmean; means[:, :, 1] = eta[0]
    covariance = np.zeros((k, n, 2, n, 2)); covariance[:, :, 1, :, 1] = eta[1]
    covariance[:, :, 0, :, 0] = (base+np.einsum('ki,ai,bi->kab', conditional, update, update)
        +variances[:, None, None]*np.outer(design[:, 0], design[:, 0]))
    result = JointFunctionRiskPosterior.from_moments(means[0], covariance[0])
    result.means, result.covariances = means.reshape(k, 2*n), covariance.reshape(k, 2*n, 2*n)
    result.log_weights = np.log(weights/weights.sum()); result.component_pairs = [(0, 0)]*k
    return result


def forecast_errors(actual, reference):
    errors = {'mean_max_absolute': 0., 'covariance_relative_frobenius': 0.,
        'paid_log_evidence_absolute': abs(actual['raw_paid']['paid_predictive_log_score']-reference['raw_paid']['paid_predictive_log_score'])}
    for stage in ('prior', 'raw_paid'):
        errors['mean_max_absolute'] = max(errors['mean_max_absolute'], float(np.max(np.abs(
            np.asarray(actual[stage]['mean'])-reference[stage]['mean']))))
        ac, rc = np.asarray(actual[stage]['covariance']), np.asarray(reference[stage]['covariance'])
        errors['covariance_relative_frobenius'] = max(errors['covariance_relative_frobenius'], float(np.linalg.norm(ac-rc)/np.linalg.norm(rc)))
    return errors


def mixture_mean_density(field, sites, labels):
    observed, noise = np.asarray(labels['response'])[:, 0], np.asarray(labels['noise'])[:, 0]
    indices = 2*np.asarray(sites)
    return float(logsumexp([weight+gaussian_score(mean[indices], covariance[np.ix_(indices, indices)]+np.diag(noise), observed)
        for weight, mean, covariance in zip(field.log_weights, field.means, field.covariances)])/len(sites))


def run():
    started = time.process_time(); spec = read(EXECUTION)
    checked = read(spec['source_results']); assert checked['all14_source_rules_complete']
    frozen = read(spec['source_laws']); assert frozen['all14_rules_frozen_without_query_data']
    centering = read(read(spec['source_execution'])['centering_execution'])
    coverage = read(centering['coverage_execution']); adaptation = read(spec['adaptation_execution'])
    risk_source = read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    coordinates = {i: public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank)
        for i in spec['training_tasks']}
    raw = read(coverage['source_data'])
    paid = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']] for i in spec['training_tasks']}; del raw
    controls = [r for r in read(spec['native_f_profiles'])['rows'] if r['case'] == NATIVE]
    rows, fields, refined_fields, numerical = [], {}, {}, []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            risk_fit = read(Path(risk_source['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            basis, risk_basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']), OrthogonalLowOrderBasis(inputs['public_risk_reference'])
            eta_law = next(r['joint_structural'] for r in risk_fit['heads'] if r['head'] == 3)
            eta = source_prior_moments(restore_joint(risk_basis, eta_law), np.asarray(inputs['public_risk_reference']))
            psi = (coordinates[deleted]-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            sites = coverage['layouts'][layout][deleted]
            control = next(r for r in controls if r['layout'] == layout and r['deleted_task'] == deleted)
            np.testing.assert_array_equal(sites, control['scoring_sites'])
            rule = next(r for r in frozen['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
            source = rule['source_reference_law']
            if deleted == 1:
                native, _ = forecast_field(mixture_field(basis, psi, source,
                    np.array([source['native_task_intercept_variance']]), np.ones(1), eta), sites, spec['paid_candidates'], paid[deleted])
                for stage in ('prior', 'raw_paid'):
                    for key in ('mean', 'covariance'):
                        np.testing.assert_allclose(native[stage][key], control['predictions'][stage][key], atol=1e-8, rtol=1e-8)
                np.testing.assert_allclose(native['raw_paid']['paid_predictive_log_score'],
                    control['predictions']['raw_paid']['paid_predictive_log_score'], atol=1e-8, rtol=1e-8)
            field = mixture_field(basis, psi, source, np.asarray(rule['variance_nodes']), np.asarray(rule['posterior_weights']), eta)
            mean, covariance = field.predict(full_cov=True)
            np.testing.assert_allclose(mean[:, 1], eta[0], atol=1e-13, rtol=1e-13)
            np.testing.assert_allclose(covariance[:, 1, :, 1], eta[1], atol=1e-13, rtol=1e-13)
            predictions, stored = forecast_field(field, sites, spec['paid_candidates'], paid[deleted])
            nodes, weights = rule_on_intervals(source, rule['adaptive_u_intervals'], 32)
            refined, refined_stored = forecast_field(mixture_field(basis, psi, source, nodes, weights, eta), sites, spec['paid_candidates'], paid[deleted])
            errors = forecast_errors(predictions, refined)
            passed = errors['mean_max_absolute'] <= 1e-7 and errors['covariance_relative_frobenius'] <= 1e-5 and errors['paid_log_evidence_absolute'] <= 1e-6
            numerical.append({'layout': layout, 'deleted_task': deleted, 'errors': errors, 'passed': passed})
            rows.append({'layout': layout, 'deleted_task': deleted, 'case': CANDIDATE, 'scoring_sites': sites,
                'predictions': predictions, 'refined_predictions': refined, 'component_count': len(rule['variance_nodes']),
                'paid_component_weights': np.exp(stored[1].log_weights).tolist()})
            fields[(layout, deleted)] = stored
            if deleted == 1: refined_fields[layout] = refined_stored
            print('mixture forecast', layout, deleted, len(rule['variance_nodes']), passed, flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'predictions.json', {'execution': spec, 'all12_selected_and_refined_forecasts_frozen_before_scored_labels': True,
        'source_rules_frozen_before_paid_queries': True, 'eta_f_prior_unchanged': True, 'numerical_checks': numerical, 'rows': rows})
    if not all(r['passed'] for r in numerical):
        result = {'execution': spec, 'numerical_forecasts_verified': False, 'forecasts_scored': False,
            'scientific_gate_interpreted': False, 'numerical_checks': numerical, 'no_model_adopted': True,
            'cost': spec['cost'], 'cpu_seconds': time.process_time()-started}
        save(out/'results.json', result); print('Numerical forecast mismatch retained; no scoring', flush=True); return
    labels = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks'] for layout in spec['layouts']}
    contrasts, density_checks = [], []
    for row in rows:
        layout, deleted = row['layout'], row['deleted_task']; label = labels[layout][deleted]
        row['scores'] = score_f(row, fields[(layout, deleted)], label)
        row['mixture_mean_densities'] = {stage: mixture_mean_density(field, row['scoring_sites'], label)
            for stage, field in zip(('prior', 'raw_paid'), fields[(layout, deleted)])}
        row['mixture_transcript_density'] = (row['scores']['raw_paid']['raw_paid_log_evidence']+12*row['mixture_mean_densities']['raw_paid'])/14
        if deleted == 1:
            refrow = {**row, 'predictions': row['refined_predictions']}
            refscores = score_f(refrow, refined_fields[layout], label)
            errors = {key: max(abs(row['scores'][stage][key]-refscores[stage][key]) for stage in ('prior', 'raw_paid') if key in refscores[stage])
                for key in ('f_raw_batch_density', 'sequential_transcript_density')}
            density_checks.append({'layout': layout, 'deleted_task': deleted, 'errors': errors, 'passed': max(errors.values()) <= 1e-5})
        old = next(r for r in controls if r['layout'] == layout and r['deleted_task'] == deleted)
        contrasts.append({'layout': layout, 'deleted_task': deleted, 'mixture_minus_native': {stage:
            {key: row['scores'][stage][key]-old['scores'][stage][key] for key in row['scores'][stage]} for stage in ('prior', 'raw_paid')},
            'actual_mixture_transcript_minus_native': row['mixture_transcript_density']-old['scores']['raw_paid']['sequential_transcript_density']})
        print('mixture scoring', layout, deleted, flush=True)
    summary = {layout: {case: {stage: {key: float(np.mean([r['scores'][stage][key] for r in members if r['layout'] == layout]))
        for key in members[0]['scores'][stage]} for stage in ('prior', 'raw_paid')}
        for case, members in ((NATIVE, controls), (CANDIDATE, rows))} for layout in spec['layouts']}
    gates = {}
    for layout in spec['layouts']:
        new, old = (summary[layout][case]['raw_paid'] for case in (CANDIDATE, NATIVE))
        gates[layout] = {'paid_f_mse_nonincrease': new['f_mse'] <= old['f_mse'],
            'raw_batch_density_improved': new['f_raw_batch_density'] > old['f_raw_batch_density'],
            'sequential_transcript_density_nondecrease': new['sequential_transcript_density'] >= old['sequential_transcript_density']}
    verified = all(r['passed'] for r in density_checks); passed = all(all(g.values()) for g in gates.values())
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'gates_by_layout': gates,
        'numerical_forecasts_verified': True, 'numerical_density_verified': verified, 'numerical_checks': numerical,
        'density_checks': density_checks, 'forecasts_scored': True, 'scientific_gate_interpreted': verified,
        'passes_both_layouts': passed if verified else None, 'candidate_stopped_after_development_failure': verified and not passed,
        'no_model_adopted': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'passes_both_layouts': result['passes_both_layouts'], 'numerical_density_verified': verified,
        'gates': gates, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
