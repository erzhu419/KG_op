"""Separate source common estimation from new-task residual covariance."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_counterbalanced_complete_mean_uncertainty import load_group
from performance.check_legacy_crossfit_variance_noise import risk_models
from performance.check_legacy_gp_profile_source_holdout import reproduce
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.run_legacy_related_inventory_source import save, training_view
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

EXECUTION = 'performance/manifests/legacy_joint_population_common_task_attribution_execution_v1.json'
ARM = 'joint_structural'
CASES = {
    'native_common_native_task': ('native', 'native'),
    'native_common_restricted_task': ('native', 'restricted'),
    'restricted_common_native_task': ('restricted', 'native'),
    'restricted_common_restricted_task': ('restricted', 'restricted'),
}


def assemble_common_task(template, design, common, task):
    mean = design@np.asarray(common['coefficient_mean'])
    covariance = design@(np.asarray(common['population_mean_covariance'])+
        np.diag(task['residual_coefficient_diagonal']))@design.T+task['discrepancy_variance']*np.eye(len(design))
    pairs = []
    for f, g in template.components:
        gm, gc = g.predict(full_cov=True)
        gm[:, 0], gc[:, 0, :, 0] = mean, covariance
        pairs.append((deepcopy(f), JointFunctionRiskPosterior.from_moments(gm, gc)))
    return JointResponseMixture(template.names, pairs)


def metrics(scores):
    return {'g_mse': scores['observed_head_mse'][1], 'g_raw_batch_density': scores['response_batch_log_scores'][1],
        'joint_raw_batch_density': scores['joint_batch_log_score']}


def run():
    started = time.process_time(); spec = read(EXECUTION)
    native_execution, restricted_execution = read(spec['native_execution']), read(spec['restricted_execution'])
    coverage = read(native_execution['coverage_execution']); parent = read(coverage['source_execution'])
    cached = {name: read(Path(execution['output'])/'calibration_profiles.json')['rows']
        for name, execution in (('native', native_execution), ('restricted', restricted_execution))}
    groups, fits = {}, {}
    for layout in spec['layouts']:
        group = load_group(native_execution, coverage, parent, layout)
        fits[layout] = {'native': read(Path(native_execution['output'])/f'layout{layout}'/'selected_source_fit.json'),
            'restricted': read(Path(restricted_execution['output'])/f'layout{layout}'/'source_fit.json')}
        groups[layout] = group
    raw = read(coverage['source_data']); rows, models, prefixes = [], {}, []
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    for layout, group in groups.items():
        full = group[0][0]; basis = OrthogonalLowOrderBasis(full['public_mean_reference'])
        laws = {name: fit['heads'][1][ARM] for name, fit in fits[layout].items()}
        for task_index in spec['heldout_tasks']:
            task = raw['tasks'][task_index]
            query = {'task_id': task['task_id'], 'task_index': task_index, 'kind': 'source', 'tau': task['tau'],
                'candidate_indices': spec['paid_candidates'],
                'paired_observations': [task['Y_replicates'][j][0] for j in spec['paid_candidates']]}
            mq = np.asarray(full['heldout_mean_psi'][str(task_index)])
            rq = [np.asarray(g[0]['heldout_risk_psi'][str(task_index)]) for g in group]
            design = np.column_stack([np.ones(len(mq)), basis.transform(mq)])
            template = risk_models(group, fits[layout]['native'], mq, rq, [ARM])[0][1]
            for case, (common_name, task_name) in CASES.items():
                prior_model = assemble_common_task(template, design, laws[common_name], laws[task_name])
                prior = forecast(query, ARM, case, prior_model); model = deepcopy(prior_model)
                endpoint = common_name if common_name == task_name else None
                if endpoint:
                    old = next(r for r in cached[endpoint] if r['layout'] == layout and r['task_index'] == task_index)
                    reproduce(prior, old['prior'])
                for prefix, (j, value) in enumerate(zip(query['candidate_indices'], query['paired_observations']), 1):
                    model.update_raw(j, [value[0], value[1]-query['tau']])
                    # Retain the actual prefix means for later paid-update attribution; no extra action or score.
                    mean, _ = model.predict_joint()
                    prefixes.append({'layout': layout, 'task_index': task_index, 'case': case,
                        'paid_count': prefix, 'g_mean': mean[:, 2].tolist()})
                row = forecast(query, ARM, case, model)
                if endpoint:
                    reproduce(row, old)
                row.update(layout=layout, case=case, common_block=common_name, task_block=task_name, prior=prior)
                rows.append(row); models[(layout, task_index, case)] = prior_model, model
            print('common/task forecasts frozen', layout, task_index, flush=True)
    save(out/'source_predictions.json', {'execution': spec, 'all16_prior_paid_forecasts_frozen': True,
        'both_endpoints_all73_prior_paid_reproduced': True, 'rows': rows})
    save(out/'paid_prefix_means.json', {'execution': spec, 'all_prefixes_frozen_before_scoring': True, 'rows': prefixes})
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    labels = {i: training_view(raw, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    for row in rows:
        layout, task_index, case = row['layout'], row['task_index'], row['case']
        prior_model, model = models[(layout, task_index, case)]
        score_source(row['prior'], labels[task_index], prior_model, 3)
        if row['common_block'] == row['task_block']:
            old = next(r for r in cached[row['common_block']] if r['layout'] == layout and r['task_index'] == task_index)
            row['scores'] = old['scores']
        else:
            score_source(row, labels[task_index], model, 3)
        row['stage_metrics'] = {'prior': metrics(row['prior']['scores']), 'raw_paid': metrics(row['scores'])}
    summary = {layout: {case: {stage: {key: np.mean([r['stage_metrics'][stage][key] for r in rows
        if r['layout'] == layout and r['case'] == case]).item() for key in rows[0]['stage_metrics'][stage]}
        for stage in ('prior', 'raw_paid')} for case in CASES} for layout in spec['layouts']}
    contrasts, interactions = [], []
    for row in rows:
        baseline = next(r for r in rows if r['layout'] == row['layout'] and r['task_index'] == row['task_index']
            and r['case'] == 'native_common_native_task')
        contrasts.append({'layout': row['layout'], 'task_index': row['task_index'], 'case': row['case'],
            'minus_native': {stage: {key: row['stage_metrics'][stage][key]-baseline['stage_metrics'][stage][key]
                for key in row['stage_metrics'][stage]} for stage in ('prior', 'raw_paid')}})
    for layout in spec['layouts']:
        for task_index in spec['heldout_tasks']:
            local = {r['case']: r['stage_metrics'] for r in rows if r['layout'] == layout and r['task_index'] == task_index}
            interactions.append({'layout': layout, 'task_index': task_index,
                'factorial_interaction': {stage: {key:
                    local['restricted_common_restricted_task'][stage][key]-local['restricted_common_native_task'][stage][key]
                    -local['native_common_restricted_task'][stage][key]+local['native_common_native_task'][stage][key]
                    for key in local['native_common_native_task'][stage]} for stage in ('prior', 'raw_paid')}})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'interactions': interactions,
        'both_endpoints_reproduction_passed': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'no_model_selected': True, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
