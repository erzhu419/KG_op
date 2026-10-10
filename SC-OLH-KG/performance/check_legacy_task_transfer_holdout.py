"""Frozen two-stage source-prior calibration at the original source gate."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_counterbalanced_complete_mean_uncertainty import load_group, contrast
from performance.check_legacy_crossfit_variance_noise import risk_models
from performance.check_legacy_gp_profile_source_holdout import reproduce
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.run_legacy_related_inventory_source import save, training_view
from representation.joint_response_mixture import JointResponseMixture
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.task_transfer_calibration import calibrated_g_field, scaled_task_diagonal

EXECUTION = 'performance/manifests/legacy_joint_population_task_transfer_holdout_execution_v1.json'
ARM = 'joint_structural'


def calibrated_model(template, design, native, multipliers):
    return JointResponseMixture(template.names, [(deepcopy(f), calibrated_g_field(g, design, native, multipliers))
        for f, g in template.components])


def run(execution=EXECUTION):
    started = time.process_time(); spec = read(execution); calibration = read(spec['calibration_results'])
    assert calibration['calibration_preferred']
    for layout in spec['layouts']:
        np.testing.assert_array_equal(spec['fixed_multipliers'][layout], calibration['fits'][layout]['multipliers'])
    native_execution = read(spec['native_execution']); coverage = read(spec['coverage_execution'])
    parent = read(coverage['source_execution']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    groups, fits, laws = {}, {}, []
    for layout in spec['layouts']:
        group = load_group(spec, coverage, parent, layout)
        means = read(Path(native_execution['output'])/f'layout{layout}'/'selected_source_fit.json')
        native = means['heads'][1][ARM]; basis = OrthogonalLowOrderBasis(group[0][0]['public_mean_reference'])
        groups[layout], fits[layout] = group, means
        residual = scaled_task_diagonal(native, basis.feature_dim, spec['fixed_multipliers'][layout])
        laws.append({'layout': layout, 'native_fit': str(Path(native_execution['output'])/f'layout{layout}'/'selected_source_fit.json'),
            'multipliers': spec['fixed_multipliers'][layout], 'coefficient_mean': native['coefficient_mean'],
            'population_mean_covariance': native['population_mean_covariance'],
            'residual_coefficient_diagonal': residual.tolist(), 'discrepancy_variance': native['discrepancy_variance'],
            'native_task_intercept_variance': native['residual_coefficient_diagonal'][0],
            'calibrated_task_intercept_variance': float(residual[0]), 'source_signal_scale': native['source_signal_scale']})
    save(out/'calibrated_prior_laws.json', {'execution': spec, 'both_laws_frozen_before_paid_observations': True, 'rows': laws})
    cached = read(Path(native_execution['output'])/'calibration_profiles.json')['rows']
    raw = read(coverage['source_data']); rows, models = [], {}
    for layout, group in groups.items():
        native = fits[layout]['heads'][1][ARM]; basis = OrthogonalLowOrderBasis(group[0][0]['public_mean_reference'])
        for task_index in spec['heldout_tasks']:
            task = raw['tasks'][task_index]
            query = {'task_id': task['task_id'], 'task_index': task_index, 'kind': 'source', 'tau': task['tau'],
                'candidate_indices': spec['paid_candidates'],
                'paired_observations': [task['Y_replicates'][j][0] for j in spec['paid_candidates']]}
            mq = np.asarray(group[0][0]['heldout_mean_psi'][str(task_index)])
            rq = [np.asarray(g[0]['heldout_risk_psi'][str(task_index)]) for g in group]
            design = np.column_stack([np.ones(len(mq)), basis.transform(mq)])
            template = risk_models(group, fits[layout], mq, rq, [ARM])[0][1]
            original = calibrated_model(template, design, native, [1., 1.])
            model = calibrated_model(template, design, native, spec['fixed_multipliers'][layout])
            old = next(r for r in cached if r['layout'] == layout and r['task_index'] == task_index)
            reproduce(forecast(query, ARM, 'native_control', original), old['prior'])
            prior = forecast(query, ARM, 'source_transfer_calibration', model)
            for j, value in zip(query['candidate_indices'], query['paired_observations']):
                centered = [value[0], value[1]-query['tau']]
                original.update_raw(j, centered); model.update_raw(j, centered)
            reproduce(forecast(query, ARM, 'native_control', original), old)
            row = forecast(query, ARM, 'source_transfer_calibration', model)
            row.update(layout=layout, parent_layout=spec['parent_layouts'][layout], prior=prior)
            rows.append(row); models[(layout, task_index)] = model
            print('fixed source transfer forecasts frozen', layout, task_index, flush=True)
    save(out/'source_predictions.json', {'execution': spec, 'all4_prior_paid_forecasts_frozen': True,
        'native_all73_prior_paid_reproduction_passed': True, 'rows': rows})
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    labels = {i: training_view(raw, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    references = read(spec['reference_profiles'])['rows']; parents = read(spec['parent_reference_profiles'])['rows']
    contrasts, parent_contrasts, selected_contrasts, gp_contrasts = [], [], [], []
    for row in rows:
        layout, i = row['layout'], row['task_index']; score_source(row, labels[i], models[(layout, i)], 3)
        same = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == ARM)
        previous = next(r for r in parents if r['layout'] == row['parent_layout'] and r['task_index'] == i and r['mean_arm'] == ARM)
        selected = next(r for r in cached if r['layout'] == layout and r['task_index'] == i)
        gp = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == 'joint_gp')
        contrasts.append(contrast(row, same, 'same216_call_native_structural'))
        parent_contrasts.append(contrast(row, previous, 'original144_call_structural'))
        selected_contrasts.append(contrast(row, selected, 'same216_call_source_nll_selection'))
        gp_contrasts.append(contrast(row, gp, 'same216_call_gp'))
    metrics = ('observed_head_mse', 'joint_batch_log_score', 'response_batch_log_scores')
    summary = {layout: {case: {k: np.mean([r['scores'][k] for r in members if r['layout'] == layout], axis=0).tolist()
        for k in metrics} for case, members in (('calibrated', rows), ('native', cached),
            ('gp', [r for r in references if r['mean_arm'] == 'joint_gp']))} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'parent_contrasts': parent_contrasts,
        'selected_contrasts': selected_contrasts, 'gp_contrasts': gp_contrasts,
        'same_information_source_gate_passed': all(c['passed'] for c in contrasts),
        'original144_source_gate_passed': all(c['passed'] for c in parent_contrasts),
        'source_gate_passed': all(c['passed'] for c in contrasts+parent_contrasts),
        'native_all73_prior_paid_reproduction_passed': True, 'cost': spec['cost'],
        'cpu_seconds': time.process_time()-started, 'independent_confirmation': False, 'target_promoted': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'source_gate_passed': result['source_gate_passed'], 'summary': summary, 'contrasts': contrasts,
        'parent_contrasts': parent_contrasts, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
