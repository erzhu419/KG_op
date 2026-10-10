"""Locate paid adaptation gains/losses in existing whole-training-task folds."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read, forecast
from performance.check_legacy_public_mean_source_half import components
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.public_control_mean import public_control_mean_coordinates

EXECUTION = 'performance/manifests/legacy_joint_population_training_paid_adaptation_execution_v1.json'


def score_sites(row, labels, sites, model):
    mean, variance = np.asarray(row['mean'])[sites], np.asarray(row['variance'])[sites]
    error = mean-np.asarray(labels['response'])
    variance[:, :2] += np.asarray(row['expected_episode_variance'])[sites]/3
    variance[:, 2:] += np.asarray(labels['noise'])[:, 2:]
    standardized = error/np.sqrt(variance)
    scores = [model.batch_log_scores(j, value[:2], value[[3, 2]], 3)
        for j, value in zip(sites, np.asarray(labels['response']))]
    return {'observed_head_mse': (error**2).mean(axis=0).tolist(),
        'joint_batch_log_score': float(np.mean([s['joint'] for s in scores])),
        'response_batch_log_scores': np.mean([s['heads'] for s in scores], axis=0).tolist(),
        'predictive_moment_band_95_observed_coverage': (abs(standardized) <= 1.96).mean(axis=0).tolist(),
        'predictive_standardized_residual_rms': np.sqrt((standardized**2).mean(axis=0)).tolist()}


def matched_contrasts(rows):
    contrasts = []
    for row in rows:
        if row['mean_arm'] != 'joint_structural':
            continue
        for arm in ('flat', 'joint_gp'):
            reference = next(r for r in rows if r['layout'] == row['layout']
                and r['deleted_task'] == row['deleted_task'] and r['mean_arm'] == arm)
            for stage in ('prior_scores', 'scores'):
                actual, control = row[stage], reference[stage]
                contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'],
                    'reference': arm, 'stage': stage,
                    'structural_minus_control_g_mse': actual['observed_head_mse'][1]-control['observed_head_mse'][1],
                    'structural_minus_control_joint_density': actual['joint_batch_log_score']-control['joint_batch_log_score'],
                    'structural_minus_control_g_density': actual['response_batch_log_scores'][1]-control['response_batch_log_scores'][1]})
    return contrasts


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution'])
    deletion = read(spec['deleted_mean_execution']); parent = read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    bank = [tuple(p) for p in public['bank']]
    providers = {i: no_outcomes(problem_for(public['plan'], public['tasks'][i])) for i in spec['training_tasks']}
    descriptors = {i: legacy_exposure_descriptors(p, bank, 50) for i, p in providers.items()}
    for value in descriptors.values():
        np.testing.assert_array_equal(value, descriptors[spec['training_tasks'][0]])
    frozen = read(Path(read(spec['correlation_validation'])['execution']['output'])/'validation_predictions.json')['rows']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    # Select only adaptation observations here; all other deleted labels enter below.
    raw = read(coverage['source_data'])
    paid = {i: {'tau': raw['tasks'][i]['tau'],
        'values': [raw['tasks'][i]['Y_replicates'][j][0] for j in spec['paid_candidates']]}
        for i in spec['training_tasks']}
    del raw
    rows, jobs = [], {}
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            branch = Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'
            inputs = read(branch/'inputs.json'); gp = read(branch/'gp_source_fit.json')
            means = read(Path(deletion['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
            means = dict(means, heads=[dict(h, joint_gp=g['joint_gp']) for h, g in zip(means['heads'], gp['heads'])])
            risk = read(Path(parent['output'])/f"layout{spec['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            names = [t['name'] for t in inputs['source_tasks']]
            assert public['tasks'][deleted]['task_id'] not in names
            assert names == [t['name'] for t in inputs['risk_source_tasks']]
            mean_psi = (public_control_mean_coordinates(providers[deleted], bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            risk_psi = np.asarray(inputs['public_risk_reference'])
            for value in inputs['heldout_risk_psi'].values():
                np.testing.assert_array_equal(risk_psi, value)
            fields = components(inputs, means, risk, mean_psi, risk_psi)
            query = {'task_id': public['tasks'][deleted]['task_id'], 'task_index': deleted, 'kind': 'source'}
            for arm, pair in zip(spec['mean_arms'], fields):
                prior_model = JointResponseMixture([f'drop{deleted}'], [pair])
                prior = forecast(query, arm, 'single_deleted_source_risk', prior_model)
                if arm == 'joint_structural':
                    reference = next(r for r in frozen if r['layout'] == layout and r['deleted_task'] == deleted)
                    sites = reference['sites']; old = reference['predictions']['independent']; n = len(sites)
                    np.testing.assert_allclose(np.asarray(prior['mean'])[sites, :2].T.reshape(-1), old['mean'], atol=1e-8)
                    np.testing.assert_allclose(np.asarray(prior['variance'])[sites, :2].T.reshape(-1), np.diag(old['covariance']), atol=1e-8)
                # Construct independent copies through native moment fields, preserving the prior for scoring.
                model = deepcopy(prior_model)
                for j, value in zip(spec['paid_candidates'], paid[deleted]['values']):
                    model.update_raw(j, [value[0], value[1]-paid[deleted]['tau']])
                assert model.paid_episodes == len(spec['paid_candidates'])
                row = forecast(query, arm, 'single_deleted_source_risk', model)
                row.update(layout=layout, deleted_task=deleted, scoring_sites=coverage['layouts'][layout][deleted], prior=prior)
                assert not set(row['scoring_sites']).intersection(spec['paid_candidates'])
                rows.append(row); jobs[(layout, deleted, arm)] = prior_model, model
        print('training-task adaptation forecasts frozen', layout, flush=True)
    save(out/'predictions.json', {'execution': spec, 'all36_prior_and36_paid_forecasts_frozen': True, 'rows': rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        layout, deleted, arm = row['layout'], row['deleted_task'], row['mean_arm']
        labels = source[layout]['source_tasks'][deleted]
        prior_model, model = jobs[(layout, deleted, arm)]
        row['prior_scores'] = score_sites(row['prior'], labels, row['scoring_sites'], prior_model)
        row['scores'] = score_sites(row, labels, row['scoring_sites'], model)
    summary = {layout: {arm: {stage: {k: np.mean([r[stage][k] for r in rows if r['layout'] == layout and r['mean_arm'] == arm], axis=0).tolist()
        for k in rows[0][stage]} for stage in ('prior_scores', 'scores')} for arm in spec['mean_arms']} for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        old, new = row['prior_scores'], row['scores']
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'mean_arm': row['mean_arm'],
            'g_mse_paid_minus_prior': new['observed_head_mse'][1]-old['observed_head_mse'][1],
            'joint_density_paid_minus_prior': new['joint_batch_log_score']-old['joint_batch_log_score'],
            'g_density_paid_minus_prior': new['response_batch_log_scores'][1]-old['response_batch_log_scores'][1]})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts,
        'matched_contrasts': matched_contrasts(rows), 'cost': spec['cost'],
        'prior_structural_reproduction_passed': True, 'source_task_exclusion_passed': True,
        'no_model_selected': True, 'cpu_seconds': time.process_time()-started}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({k: result[k] for k in ('summary', 'contrasts', 'cpu_seconds')}, flush=True)


if __name__ == '__main__':
    run()
