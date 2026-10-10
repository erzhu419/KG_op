"""Full-source g common correlation under the existing complete-model gates."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_common_correlation import verify_dense
from performance.check_legacy_counterbalanced_complete_mean_uncertainty import contrast, load_group
from performance.check_legacy_crossfit_variance_noise import risk_models
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import forecast, read, score_source
from performance.check_legacy_task_bank_centering_holdout import ARM, centered_model
from performance.diagnose_legacy_common_prior_geometry import diagnose
from performance.run_legacy_related_inventory_source import save, training_view
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_common_correlation import CommonPriorCorrelationLikelihood, fit_common_correlation
from representation.source_function_prior import head_batches
from representation.task_bank_centering import CenteredTaskPopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_common_correlation_holdout_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); training = read(spec['training_results'])
    assert training['all_fits_complete'] and training['head_passes_both_layouts']['g']
    training_spec = read(spec['training_execution']); geometry_spec = read(spec['geometry_execution'])
    centered_spec = read(spec['centered_holdout_execution']); native_spec = read(spec['native_execution'])
    coverage = read(spec['coverage_execution']); parent = read(coverage['source_execution'])
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    groups, means, laws, geometry, verification = {}, {}, {}, [], []
    for layout in spec['layouts']:
        group = load_group(spec, coverage, parent, layout); inputs = group[0][0]
        native_fit = read(Path(native_spec['output'])/f'layout{layout}'/'selected_source_fit.json')
        native = read(Path(centered_spec['output'])/f'layout{layout}'/'source_fit.json')
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
        prior = restore_regularizer(basis, native_fit['heads'][1]['regularizer'], tasks, 1)
        source = CenteredTaskPopulationLikelihood(prior, head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks],
            [np.asarray(native['source_bank_centers'][t['name']]) for t in tasks])
        diagnostic = diagnose(source, native, group[0][1]['heads'][1]['joint_gp'], np.asarray(inputs['public_mean_reference']), geometry_spec)
        geometry.append({'layout': layout, 'head': 'g', **diagnostic})
        if not diagnostic['direction_defined']:
            save(out/'results.json', {'execution': spec, 'source_gate_passed': False, 'forecasts_scored': False,
                'geometry': geometry, 'failure': diagnostic['failure'], 'independent_confirmation': False, 'target_promoted': False})
            print('actual full-source common direction undefined; stop before holdout forecasts', flush=True); return
        likelihood = CommonPriorCorrelationLikelihood(source, native, diagnostic['coefficient_covariance_direction'])
        verification.append({'layout': layout, 'point': 'interior',
            **verify_dense(likelihood, training_spec['gradient_verification_p'], training_spec['gradient_verification_epsilon'])})
        law = fit_common_correlation(likelihood, training_spec['optimizer'])
        verification.append({'layout': layout, 'point': 'fitted', **verify_dense(likelihood, law['common_correlation_strength'])})
        folder = out/f'layout{layout}'; folder.mkdir(parents=True, exist_ok=True); save(folder/'source_fit.json', law)
        groups[layout], means[layout], laws[layout] = group, native_fit, law
        print('full-six-source g correlation fit frozen', layout, law['optimizer_converged'], law['common_correlation_strength'], flush=True)
    save(out/'source_laws.json', {'execution': spec, 'both_laws_frozen_before_heldout_paid_observations': True,
        'rows': [{'layout': l, 'law': law} for l, law in laws.items()], 'geometry': geometry, 'verification': verification})
    cost = {**spec['cost'], 'actual_new_g_source_fits': len(laws)}
    if not all(law['optimizer_converged'] for law in laws.values()):
        save(out/'results.json', {'execution': spec, 'all_fits_complete': False, 'forecasts_scored': False,
            'source_gate_passed': False, 'cost': cost, 'cpu_seconds': time.process_time()-started,
            'independent_confirmation': False, 'target_promoted': False})
        print('incomplete scalar fits; no holdout scoring', flush=True); return
    raw = read(coverage['source_data']); rows, jobs = [], {}
    for layout, group in groups.items():
        basis = OrthogonalLowOrderBasis(group[0][0]['public_mean_reference'])
        for i in spec['heldout_tasks']:
            task = raw['tasks'][i]
            query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source', 'tau': task['tau'],
                'candidate_indices': spec['paid_candidates'], 'paired_observations': [task['Y_replicates'][j][0] for j in spec['paid_candidates']]}
            psi = np.asarray(group[0][0]['heldout_mean_psi'][str(i)])
            design = np.column_stack([np.ones(len(psi)), basis.transform(psi)]); center = design[:, 1:].mean(axis=0)
            risk_queries = [np.asarray(g[0]['heldout_risk_psi'][str(i)]) for g in group]
            template = risk_models(group, means[layout], psi, risk_queries, [ARM])[0][1]
            model = centered_model(template, design, laws[layout], center)
            prior = forecast(query, ARM, 'source_common_correlation', model)
            for j, value in zip(query['candidate_indices'], query['paired_observations']):
                model.update_raw(j, [value[0], value[1]-query['tau']])
            row = forecast(query, ARM, 'source_common_correlation', model)
            row.update(layout=layout, parent_layout=spec['parent_layouts'][layout], prior=prior, query_bank_center=center.tolist())
            rows.append(row); jobs[(layout, i)] = model
    save(out/'source_predictions.json', {'execution': spec, 'all4_forecasts_frozen_before_scoring': True,
        'f_eta_priors_and_risk_maps_unchanged': True, 'rows': rows})
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    labels = {i: training_view(raw, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    del raw
    references = read(spec['reference_profiles'])['rows']; parents = read(spec['parent_reference_profiles'])['rows']
    selected = read(Path(native_spec['output'])/'calibration_profiles.json')['rows']
    centered = read(spec['centered_holdout_profiles'])['rows']
    contrasts, parent_contrasts, selected_contrasts, centered_contrasts, gp_contrasts = [], [], [], [], []
    for row in rows:
        layout, i = row['layout'], row['task_index']; score_source(row, labels[i], jobs[(layout, i)], 3)
        same = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == ARM)
        old = next(r for r in parents if r['layout'] == row['parent_layout'] and r['task_index'] == i and r['mean_arm'] == ARM)
        prior_selected = next(r for r in selected if r['layout'] == layout and r['task_index'] == i)
        prior_centered = next(r for r in centered if r['layout'] == layout and r['task_index'] == i)
        gp = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == 'joint_gp')
        contrasts.append(contrast(row, same, 'same216_call_native_structural'))
        parent_contrasts.append(contrast(row, old, 'original144_call_structural'))
        selected_contrasts.append(contrast(row, prior_selected, 'same216_call_selected_native'))
        centered_contrasts.append(contrast(row, prior_centered, 'same216_call_stage89_centered'))
        gp_contrasts.append(contrast(row, gp, 'same216_call_gp'))
    metrics = ('observed_head_mse', 'joint_batch_log_score', 'response_batch_log_scores')
    summary = {layout: {case: {key: np.mean([r['scores'][key] for r in members if r['layout'] == layout], axis=0).tolist()
        for key in metrics} for case, members in (('correlated', rows), ('centered', centered), ('native', selected),
        ('gp', [r for r in references if r['mean_arm'] == 'joint_gp']))} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'parent_contrasts': parent_contrasts,
        'selected_contrasts': selected_contrasts, 'centered_contrasts': centered_contrasts, 'gp_contrasts': gp_contrasts,
        'same_information_source_gate_passed': all(c['passed'] for c in contrasts),
        'original144_source_gate_passed': all(c['passed'] for c in parent_contrasts),
        'source_gate_passed': all(c['passed'] for c in contrasts+parent_contrasts),
        'all_fits_complete': True, 'forecasts_scored': True, 'no_model_adopted': True,
        'geometry': geometry, 'verification': verification, 'cost': cost, 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': True, 'independent_confirmation': False, 'target_promoted': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'source_gate_passed': result['source_gate_passed'], 'summary': summary, 'contrasts': contrasts,
        'parent_contrasts': parent_contrasts, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
