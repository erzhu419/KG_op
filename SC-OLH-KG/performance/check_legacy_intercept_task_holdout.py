"""Full-source development holdouts for the training-selected task restriction."""

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
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.run_legacy_related_inventory_source import save, training_view
from representation.intercept_task_population import InterceptTaskPopulationLikelihood, fit_intercept_task_population
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import head_batches

EXECUTION = 'performance/manifests/legacy_joint_population_intercept_task_holdout_execution_v1.json'
ARM = 'joint_structural'


def run():
    started = time.process_time(); spec = read(EXECUTION)
    if not read(spec['validation_results'])['restricted_family_preferred']:
        raise RuntimeError('the registered training selection did not prefer the restricted family')
    coverage = read(spec['coverage_execution']); parent = read(coverage['source_execution'])
    selected = read(spec['profile_holdout_execution']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    groups, bases, fits, attribution, fit_count = {}, {}, {}, [], 0
    for layout in spec['layouts']:
        group = load_group(spec, coverage, parent, layout); inputs = group[0][0]
        means = read(Path(selected['output'])/f'layout{layout}'/'selected_source_fit.json')
        native = means['heads'][1][ARM]; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
        regularizer = restore_regularizer(basis, means['heads'][1]['regularizer'], tasks, 1)
        likelihood = InterceptTaskPopulationLikelihood(regularizer, head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks])
        candidate = fit_intercept_task_population(likelihood, native, optimizer=inputs['execution']['optimizer']).diagnostics
        fit_count += 1; folder = out/f'layout{layout}'; folder.mkdir(parents=True, exist_ok=True)
        if not candidate['optimizer_converged']:
            save(folder/'before_completion.json', candidate)
            candidate = fit_intercept_task_population(likelihood, candidate, optimizer=inputs['execution']['optimizer']).diagnostics
            fit_count += 1
        save(folder/'restricted_g_fit.json', candidate)
        if not candidate['optimizer_converged']:
            raise RuntimeError('full-source restricted fit incomplete; no holdout scoring')
        fitted = deepcopy(means); fitted['heads'][1][ARM] = candidate; save(folder/'source_fit.json', fitted)
        groups[layout], bases[layout], fits[layout] = group, means, fitted
        attribution.append({'layout': layout, 'candidate': candidate,
            'restricted_minus_native_source_nll': candidate['source_negative_log_likelihood_per_label']-native['source_negative_log_likelihood_per_label']})
        print('full-source restricted task fit frozen', layout, flush=True)
    profile_rows = read(Path(selected['output'])/'calibration_profiles.json')['rows']
    data = read(coverage['source_data']); public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    rows, models = [], {}
    for layout, group in groups.items():
        for i in spec['heldout_tasks']:
            task = data['tasks'][i]
            query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source', 'tau': task['tau'],
                'candidate_indices': spec['paid_candidates'],
                'paired_observations': [task['Y_replicates'][j][0] for j in spec['paid_candidates']]}
            mq = np.asarray(group[0][0]['heldout_mean_psi'][str(i)])
            rq = [np.asarray(g[0]['heldout_risk_psi'][str(i)]) for g in group]
            original = risk_models(group, bases[layout], mq, rq, [ARM])[0][1]
            model = risk_models(group, fits[layout], mq, rq, [ARM])[0][1]
            cached = next(r for r in profile_rows if r['layout'] == layout and r['task_index'] == i)
            reproduce(forecast(query, ARM, 'native_source_nll_control', original), cached['prior'])
            prior = forecast(query, ARM, 'intercept_task_source_validation', model)
            for j, value in zip(query['candidate_indices'], query['paired_observations']):
                centered = [value[0], value[1]-query['tau']]
                model.update_raw(j, centered); original.update_raw(j, centered)
            reproduce(forecast(query, ARM, 'native_source_nll_control', original), cached)
            row = forecast(query, ARM, 'intercept_task_source_validation', model)
            row.update(layout=layout, parent_layout=spec['parent_layouts'][layout], prior=prior)
            rows.append(row); models[(layout, i)] = model
    save(out/'source_predictions.json', {'execution': spec, 'all4_candidate_forecasts_frozen': True,
        'selected_native_all73_prior_paid_reproduction_passed': True, 'rows': rows})
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    references = read(spec['reference_profiles'])['rows']; parents = read(spec['parent_reference_profiles'])['rows']
    contrasts, parent_contrasts, selected_contrasts, gp_contrasts = [], [], [], []
    for row in rows:
        layout, i = row['layout'], row['task_index']; score_source(row, labels[i], models[(layout, i)], 3)
        same = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == ARM)
        original = next(r for r in parents if r['layout'] == row['parent_layout'] and r['task_index'] == i and r['mean_arm'] == ARM)
        baseline = next(r for r in profile_rows if r['layout'] == layout and r['task_index'] == i)
        gp = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == 'joint_gp')
        contrasts.append(contrast(row, same, 'same216_call_native_structural'))
        parent_contrasts.append(contrast(row, original, 'original144_call_structural'))
        selected_contrasts.append(contrast(row, baseline, 'same216_call_source_nll_selection'))
        gp_contrasts.append(contrast(row, gp, 'same216_call_gp'))
    summary = {layout: {k: np.mean([r['scores'][k] for r in rows if r['layout'] == layout], axis=0).tolist()
        for k in rows[0]['scores']} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'parent_contrasts': parent_contrasts,
        'selected_contrasts': selected_contrasts, 'gp_contrasts': gp_contrasts, 'fit_attribution': attribution,
        'same_information_source_gate_passed': all(c['passed'] for c in contrasts),
        'original144_source_gate_passed': all(c['passed'] for c in parent_contrasts),
        'source_gate_passed': all(c['passed'] for c in contrasts+parent_contrasts),
        'cost': {**spec['cost'], 'actual_new_g_source_fits': fit_count}, 'cpu_seconds': time.process_time()-started,
        'independent_confirmation': False, 'target_promoted': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'source_gate_passed': result['source_gate_passed'], 'summary': summary,
        'contrasts': contrasts, 'parent_contrasts': parent_contrasts, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
