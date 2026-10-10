"""Training-only f budget comparison with fixed old covariance parameters."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_data_scaled_f_tail import NATIVE, forecast_field, score_f
from performance.check_legacy_f_common_scale_correlation_development import f_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_nested_source_conditioning import condition_fixed_law
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments

EXECUTION = 'performance/manifests/legacy_joint_population_f_nested_budget_development_execution_v1.json'
OLD, EXTENDED, MIXTURE = 'old120', 'fixed_old180', 'integrated180'


def descriptive_groups(rows, controls, integrated, layouts):
    members = rows+controls+integrated
    result = {}
    for layout in layouts:
        result[layout] = {}
        for group, tasks in (('all6', range(6)), ('task1', [1]), ('other5', [0, 2, 3, 4, 5])):
            result[layout][group] = {}
            for case in (NATIVE, OLD, EXTENDED, MIXTURE):
                selected = [r['scores']['raw_paid'] for r in members
                    if r['layout'] == layout and r['deleted_task'] in tasks and r['case'] == case]
                assert len(selected) == len(tasks)
                result[layout][group][case] = {key: float(np.mean([s[key] for s in selected]))
                    for key in selected[0]}
    return result


def run():
    started = time.process_time(); spec = read(EXECUTION)
    source = read(spec['source_execution']); coverage = read(source['coverage_execution']); parent = read(coverage['source_execution'])
    laws, inputs_by_fold, rows = [], {}, []
    for layout in spec['layouts']:
        parent_layout = coverage['parent_layouts'][layout]
        full_old = read(Path(parent['output'])/f'layout{parent_layout}'/'inputs.json')
        for deleted in spec['training_tasks']:
            suffix = Path(f'layout{layout}')/f'task{deleted}'; old_suffix = Path(f'layout{parent_layout}')/f'task{deleted}'
            before = read(Path(parent['output'])/old_suffix/'inputs.json'); after = read(Path(coverage['output'])/suffix/'inputs.json')
            head = read(Path(spec['old_fold_fits'])/old_suffix/'source_fit.json')['heads'][0]
            native = head['joint_structural']; assert native['optimizer_converged'] and native['source_task_count'] == 5
            keep = [i for i in spec['training_tasks'] if i != deleted]
            assert [t['name'] for t in before['source_tasks']] == [t['name'] for t in after['source_tasks']]
            assert f'related_inventory_{deleted:02d}' not in [t['name'] for t in before['source_tasks']]
            old_sites = [full_old['execution']['source_site_indices_by_task'][i] for i in keep]
            new_sites = [coverage['layouts'][layout][i] for i in keep]
            np.testing.assert_array_equal(before['public_mean_reference'], after['public_mean_reference'])
            for old_task, new_task, old_indices, new_indices in zip(before['source_tasks'], after['source_tasks'], old_sites, new_sites):
                positions = [new_indices.index(j) for j in old_indices]
                for key in ('response', 'noise'):
                    np.testing.assert_array_equal(old_task[key], np.asarray(new_task[key])[positions])
                np.testing.assert_allclose(new_task['psi'], np.asarray(before['public_mean_reference'])[new_indices], atol=1e-12, rtol=1e-12)
            reconstructed = condition_fixed_law(head, before['public_mean_reference'], before['source_tasks'], old_sites)
            for key in ('coefficient_mean', 'population_mean_covariance', 'source_negative_log_likelihood_per_label'):
                np.testing.assert_allclose(reconstructed[key], native[key], atol=1e-8, rtol=1e-8)
            extended = condition_fixed_law(head, before['public_mean_reference'], after['source_tasks'], new_sites)
            law = {**native, **{key: extended[key] for key in ('coefficient_mean', 'population_mean_covariance', 'source_negative_log_likelihood_per_label')},
                'source_label_count': 60, 'fixed_covariance_parameter_source_labels': 40, 'no_new_covariance_fit': True,
                'population_model': 'original40-record source parameters coherently conditioned on60 records'}
            laws.append({'layout': layout, 'deleted_task': deleted, 'old120_law': native, 'fixed_old180_law': law})
            inputs_by_fold[(layout, deleted)] = before
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'source_laws.json', {'execution': spec, 'all12_laws_frozen_before_paid_records': True,
        'all_old_fits_converged_and_exclude_query_task': True, 'rows': laws})
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(x) for x in public['bank']]
    coordinates = {i: public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank) for i in spec['training_tasks']}
    raw = read(coverage['source_data']); paid = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in spec['paid_candidates']] for i in spec['training_tasks']}; del raw
    controls = [r for r in read(spec['native_f_profiles'])['rows'] if r['case'] == NATIVE]
    integrated = [{**r, 'case': MIXTURE} for r in read(spec['integrated_profiles'])['rows']]
    fields = {}
    for fitted in laws:
        layout, deleted = fitted['layout'], fitted['deleted_task']; inputs = inputs_by_fold[(layout, deleted)]
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); risk_basis = OrthogonalLowOrderBasis(inputs['public_risk_reference'])
        risk_fit = read(Path(parent['output'])/f"layout{coverage['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
        eta_law = next(h['joint_structural'] for h in risk_fit['heads'] if h['head'] == 3)
        eta = source_prior_moments(restore_joint(risk_basis, eta_law), np.asarray(inputs['public_risk_reference']))
        psi = (coordinates[deleted]-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
        sites = coverage['layouts'][layout][deleted]
        control = next(r for r in controls if r['layout'] == layout and r['deleted_task'] == deleted)
        np.testing.assert_array_equal(sites, control['scoring_sites'])
        for case, key in ((OLD, 'old120_law'), (EXTENDED, 'fixed_old180_law')):
            predictions, stored = forecast_field(f_field(basis, psi, fitted[key], eta), sites, spec['paid_candidates'], paid[deleted])
            row = {'layout': layout, 'deleted_task': deleted, 'case': case, 'scoring_sites': sites, 'predictions': predictions}
            rows.append(row); fields[(layout, deleted, case)] = stored
        print('nested budget forecasts frozen', layout, deleted, flush=True)
    save(out/'predictions.json', {'execution': spec, 'all24_forecasts_frozen_before_scored_labels': True,
        'eta_f_prior_and_paid_data_identical_to_native180': True, 'rows': rows})
    labels = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks'] for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        layout, deleted, case = row['layout'], row['deleted_task'], row['case']
        row['scores'] = score_f(row, fields[(layout, deleted, case)], labels[layout][deleted])
        old = next(r for r in controls if r['layout'] == layout and r['deleted_task'] == deleted)
        contrasts.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'minus_native180': {stage:
            {key: row['scores'][stage][key]-old['scores'][stage][key] for key in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    summary = {layout: {case: {stage: {key: float(np.mean([r['scores'][stage][key] for r in members if r['layout'] == layout]))
        for key in members[0]['scores'][stage]} for stage in ('prior', 'raw_paid')}
        for case, members in ((NATIVE, controls), (OLD, [r for r in rows if r['case'] == OLD]),
            (EXTENDED, [r for r in rows if r['case'] == EXTENDED]), (MIXTURE, integrated))} for layout in spec['layouts']}
    gates = {}
    for layout in spec['layouts']:
        actual, old = (summary[layout][case]['raw_paid'] for case in (EXTENDED, NATIVE))
        gates[layout] = {'paid_f_mse_nonincrease': actual['f_mse'] <= old['f_mse'],
            'raw_batch_density_improved': actual['f_raw_batch_density'] > old['f_raw_batch_density'],
            'sequential_transcript_density_nondecrease': actual['sequential_transcript_density'] >= old['sequential_transcript_density']}
    passed = all(all(g.values()) for g in gates.values())
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'gates_by_layout': gates, 'passes_both_layouts': passed,
        'descriptive_raw_paid_groups': descriptive_groups(rows, controls, integrated, spec['layouts']),
        'fixed_old_candidate_stopped_after_development_failure': not passed, 'no_model_adopted': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'passes_both_layouts': passed, 'gates': gates, 'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
