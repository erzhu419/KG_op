"""Fixed-prior training-task raw information curve on cached records."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_common_scale_correlation_development import f_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_source_mean_covariance_swap import g_prediction
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.joint_function_risk import joint_batch_log_score
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments

EXECUTION = 'performance/manifests/legacy_joint_population_f_training_information_budget_execution_v1.json'


def public_prefix(features, excluded, budget):
    prefix = [0, 1]
    candidates = [j for j in range(len(features)) if j not in excluded and j not in prefix]
    while len(prefix) < budget:
        distances = ((features[candidates, None]-features[prefix][None])**2).sum(axis=2).min(axis=1)
        j = candidates.pop(int(np.argmax(distances))); prefix.append(j)
    return prefix


def score_new(row, field, labels):
    prediction = row['prediction']; observed = np.asarray(labels['response']); n = len(observed)
    scores = {'f_mse': float(np.mean((prediction['mean']-observed[:, 0])**2)),
        'f_gaussian_mean_density': gaussian_score(np.asarray(prediction['mean']),
            np.asarray(prediction['covariance'])+np.diag(np.asarray(labels['noise'])[:, 0]), observed[:, 0])/n,
        'f_raw_batch_density': float(np.mean([joint_batch_log_score(field.local_components(j), value[0], value[3], 3)
            for j, value in zip(row['scoring_sites'], observed)])),
        'raw_paid_log_evidence': prediction['paid_predictive_log_score']}
    scores['sequential_transcript_density'] = (scores['raw_paid_log_evidence']+n*scores['f_gaussian_mean_density'])/(n+row['episode_count'])
    return scores


def group_summary(scored, spec):
    return {layout: {group: {str(count): {metric: float(np.mean([r['scores'][metric] for r in scored
        if r['layout'] == layout and r['episode_count'] == count and r['deleted_task'] in tasks]))
        for metric in scored[0]['scores']} for count in spec['raw_episode_budgets']}
        for group, tasks in (('all6', range(6)), ('task1', [1]), ('other5', [0, 2, 3, 4, 5]))}
        for layout in spec['layouts']}


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution'])
    centering = read(spec['centering_execution']); profile = read(centering['profile_execution'])
    adaptation = read(spec['adaptation_execution']); parent = read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    controls = [r for r in read(spec['native_f_profiles'])['rows'] if r['case'] == 'native_f']
    excluded = {j for layout in spec['layouts'] for i in spec['training_tasks'] for j in coverage['layouts'][layout][i]}
    branches, prefixes = {}, []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            source_names = [t['name'] for t in inputs['source_tasks']]
            assert public['tasks'][deleted]['task_id'] not in source_names
            assert source_names == [t['name'] for t in inputs['risk_source_tasks']]
            law = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]['joint_structural']
            assert law['optimizer_converged'] and law['source_task_count'] == 5 and law['source_label_count'] == 60
            basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            psi = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][deleted])), bank)
                -inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            prefix = public_prefix(basis.transform(psi), excluded, max(spec['raw_episode_budgets']))
            assert not set(prefix).intersection(excluded)
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            risk_basis = OrthogonalLowOrderBasis(inputs['public_risk_reference'])
            eta_law = next(h['joint_structural'] for h in risk['heads'] if h['head'] == 3)
            eta = source_prior_moments(restore_joint(risk_basis, eta_law), np.asarray(inputs['public_risk_reference']))
            branches[(layout, deleted)] = {'field': f_field(basis, psi, law, eta), 'prefix': prefix}
            prefixes.append({'layout': layout, 'deleted_task': deleted, 'public_prefix': prefix})
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'public_prefixes.json', {'execution': spec, 'all12_prefixes_frozen_before_paid_values': True,
        'excluded_score_site_union': sorted(excluded), 'rows': prefixes})
    raw = read(coverage['source_data']); paid = {key: [raw['tasks'][key[1]]['Y_replicates'][j][0][0]
        for j in b['prefix']] for key, b in branches.items()}; del raw
    rows, fields = [], {}
    for key, b in branches.items():
        layout, deleted = key; field = b['field']; evidence = 0.; sites = coverage['layouts'][layout][deleted]
        baseline = next(r for r in controls if r['layout'] == layout and r['deleted_task'] == deleted)
        for count, (j, value) in enumerate(zip(b['prefix'], paid[key]), 1):
            evidence += field.update_batch(j, float(value), None, 1)
            if count not in spec['raw_episode_budgets']: continue
            mean, covariance = field.predict(full_cov=True)
            prediction = g_prediction(mean[:, 0], covariance[:, 0, :, 0], sites)
            prediction['paid_predictive_log_score'] = float(evidence)
            if count == 2:
                saved = baseline['predictions']['raw_paid']
                for metric in ('mean', 'covariance', 'paid_predictive_log_score'):
                    np.testing.assert_allclose(prediction[metric], saved[metric], atol=1e-8, rtol=1e-8)
            else:
                rows.append({'layout': layout, 'deleted_task': deleted, 'episode_count': count,
                    'scoring_sites': sites, 'paid_prefix': b['prefix'][:count], 'prediction': prediction})
                fields[(*key, count)] = deepcopy(field)
        print('raw information forecasts frozen', layout, deleted, b['prefix'], flush=True)
    save(out/'predictions.json', {'execution': spec, 'all24_new_forecasts_frozen_before_score_labels': True,
        'cached_two_episode_moments_and_evidence_reproduced': True, 'rows': rows})
    labels = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks'] for layout in spec['layouts']}
    for row in rows:
        key = (row['layout'], row['deleted_task'], row['episode_count'])
        row['scores'] = score_new(row, fields[key], labels[key[0]][key[1]])
    scored = rows+[{'layout': r['layout'], 'deleted_task': r['deleted_task'], 'episode_count': 2,
        'scores': r['scores']['raw_paid']} for r in controls]
    summary = {layout: {str(count): {metric: float(np.mean([r['scores'][metric] for r in scored
        if r['layout'] == layout and r['episode_count'] == count])) for metric in scored[0]['scores']}
        for count in spec['raw_episode_budgets']} for layout in spec['layouts']}
    signals = {str(count): all(summary[layout][str(count)]['f_mse'] <= summary[layout]['2']['f_mse']
        and summary[layout][str(count)]['f_raw_batch_density'] > summary[layout]['2']['f_raw_batch_density']
        for layout in spec['layouts']) for count in spec['raw_episode_budgets'] if count != 2}
    result = {'execution': spec, 'summary': summary, 'descriptive_groups': group_summary(scored, spec),
        'information_signal_by_episode_count': signals,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started, 'no_model_adopted': True,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False,
        'original_two_episode_complete_gate_unchanged': True}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': scored}); save(out/'results.json', result)
    print({'summary': summary, 'information_signals': signals, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
