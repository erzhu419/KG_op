"""Frozen-noise attribution of existing two-raw-observation source adaptation."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.joint_response_mixture import JointResponseMixture
from representation.public_control_mean import public_control_mean_coordinates

EXECUTION = 'performance/manifests/legacy_joint_population_paid_covariance_gain_execution_v1.json'
STAGES = ('prior', 'latent_risk_paid', 'fixed_noise_paid')


def fixed_noise_condition(mean, covariance, paid_sites, observations, episode_variance):
    """Two individual episodes use episode noise, without a replicate divisor."""
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    paid_sites = np.asarray(paid_sites); noise = np.asarray(episode_variance)
    factor = cho_factor(covariance[np.ix_(paid_sites, paid_sites)]+np.diag(noise), lower=True)
    gain = cho_solve(factor, covariance[paid_sites, :]).T
    updated_mean = mean+gain@(np.asarray(observations)-mean[paid_sites])
    updated_covariance = covariance-gain@covariance[paid_sites, :]
    return updated_mean, .5*(updated_covariance+updated_covariance.T), gain


def subset_prediction(moments, sites):
    return {'mean': np.column_stack([m[0][sites] for m in moments]).tolist(),
        'head_covariances': [m[1][np.ix_(sites, sites)].tolist() for m in moments]}


def score_prediction(prediction, labels):
    observed, noise = np.asarray(labels['response'])[:, :2], np.asarray(labels['noise'])[:, :2]
    mean = np.asarray(prediction['mean']); n = len(mean)
    covariances = [np.asarray(c)+np.diag(noise[:, h]) for h, c in enumerate(prediction['head_covariances'])]
    scores = [gaussian_score(mean[:, h], covariances[h], observed[:, h])/n for h in (0, 1)]
    return {'mean_mse_by_head': np.mean((mean-observed)**2, axis=0).tolist(),
        'head_gaussian_mean_density': scores, 'joint_gaussian_mean_density_per_pair': float(sum(scores))}


def run():
    started = time.process_time(); spec = read(EXECUTION); adaptation = read(spec['adaptation_execution'])
    coverage = read(adaptation['coverage_execution']); deletion = read(adaptation['deleted_mean_execution'])
    parent = read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    bank = [tuple(p) for p in public['bank']]; paid_sites = spec['paid_candidates']
    old_rows = read(Path(read(spec['adaptation_results'])['execution']['output'])/'predictions.json')['rows']
    raw = read(coverage['source_data'])
    paid_values = {i: np.asarray([raw['tasks'][i]['Y_replicates'][j][0] for j in paid_sites])
        -np.array([0., raw['tasks'][i]['tau']]) for i in spec['training_tasks']}
    del raw
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); rows = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            branch = Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'
            inputs = read(branch/'inputs.json'); gp = read(branch/'gp_source_fit.json')
            means = read(Path(deletion['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
            means = dict(means, heads=[dict(h, joint_gp=g['joint_gp']) for h, g in zip(means['heads'], gp['heads'])])
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            fields = components(inputs, means, risk, psi, np.asarray(inputs['public_risk_reference']))
            sites = coverage['layouts'][layout][deleted]
            for arm, pair in zip(spec['mean_arms'], fields):
                old = next(r for r in old_rows if r['layout'] == layout and r['deleted_task'] == deleted and r['mean_arm'] == arm)
                prior, fixed, diagnostics = [], [], []
                for h, field in enumerate(pair):
                    m, c = field.predict(full_cov=True); mean, covariance = m[:, 0], c[:, 0, :, 0]
                    old_slots = [h, 3-h]
                    np.testing.assert_allclose(m, np.asarray(old['prior']['mean'])[:, old_slots], atol=1e-8)
                    np.testing.assert_allclose(np.diag(c.reshape(2*len(m), 2*len(m))).reshape(-1, 2),
                        np.asarray(old['prior']['variance'])[:, old_slots], atol=1e-8)
                    v = np.exp(m[:, 1]+.5*np.diag(c[:, 1, :, 1]))
                    np.testing.assert_allclose(v, np.asarray(old['prior']['expected_episode_variance'])[:, h], atol=1e-8)
                    fm, fc, gain = fixed_noise_condition(mean, covariance, paid_sites, paid_values[deleted][:, h], v[paid_sites])
                    prior.append((mean, covariance)); fixed.append((fm, fc))
                    correlation = covariance[np.ix_(sites, paid_sites)]/np.sqrt(np.outer(np.diag(covariance)[sites], np.diag(covariance)[paid_sites]))
                    diagnostics.append({'paid_prior_mean': mean[paid_sites].tolist(),
                        'paid_innovation': (paid_values[deleted][:, h]-mean[paid_sites]).tolist(),
                        'paid_prior_latent_variance': np.diag(covariance)[paid_sites].tolist(),
                        'paid_fixed_episode_variance': v[paid_sites].tolist(),
                        'paid_noise_fraction_of_marginal_variance': (v[paid_sites]/(v[paid_sites]+np.diag(covariance)[paid_sites])).tolist(),
                        'scoring_site_gain': gain[sites].tolist(), 'scoring_to_paid_prior_correlation': correlation.tolist(),
                        'mean_fixed_variance_reduction_fraction': float(np.mean(1-np.diag(fc)[sites]/np.diag(covariance)[sites])),
                        'fixed_prediction_shift_rms': float(np.sqrt(np.mean((fm[sites]-mean[sites])**2)))})
                model = JointResponseMixture([f'drop{deleted}'], [pair])
                for j, value in zip(paid_sites, paid_values[deleted]):
                    model.update_raw(j, value)
                latent = []
                for h, field in enumerate(model.components[0]):
                    m, c = field.predict(full_cov=True); old_slots = [h, 3-h]
                    np.testing.assert_allclose(m, np.asarray(old['mean'])[:, old_slots], atol=1e-8)
                    np.testing.assert_allclose(np.diag(c.reshape(2*len(m), 2*len(m))).reshape(-1, 2),
                        np.asarray(old['variance'])[:, old_slots], atol=1e-8)
                    latent.append((m[:, 0], c[:, 0, :, 0]))
                    diagnostics[h]['mean_latent_variance_reduction_fraction'] = float(np.mean(1-np.diag(latent[-1][1])[sites]/np.diag(prior[h][1])[sites]))
                    diagnostics[h]['fixed_minus_latent_prediction_rms'] = float(np.sqrt(np.mean((fixed[h][0][sites]-latent[-1][0][sites])**2)))
                rows.append({'layout': layout, 'deleted_task': deleted, 'mean_arm': arm, 'scoring_sites': sites,
                    'predictions': {stage: subset_prediction(moments, sites) for stage, moments in zip(STAGES, (prior, latent, fixed))},
                    'gain_diagnostics_by_head': diagnostics})
        print('fixed-noise forecasts frozen', layout, flush=True)
    save(out/'predictions.json', {'execution': spec, 'all36_branch_forecasts_frozen': True,
        'stage66_all73_site_prior_and_paid_moments_reproduced': True, 'rows': rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        labels = source[row['layout']]['source_tasks'][row['deleted_task']]
        row['scores'] = {stage: score_prediction(prediction, labels) for stage, prediction in row['predictions'].items()}
    summary = {layout: {arm: {stage: {k: np.mean([r['scores'][stage][k] for r in rows if r['layout'] == layout and r['mean_arm'] == arm], axis=0).tolist()
        for k in rows[0]['scores'][stage]} for stage in STAGES} for arm in spec['mean_arms']} for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        for reference in ('prior', 'latent_risk_paid'):
            fixed, original = row['scores']['fixed_noise_paid'], row['scores'][reference]
            contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'mean_arm': row['mean_arm'], 'reference': reference,
                'fixed_minus_reference_g_mse': fixed['mean_mse_by_head'][1]-original['mean_mse_by_head'][1],
                'fixed_minus_reference_g_density': fixed['head_gaussian_mean_density'][1]-original['head_gaussian_mean_density'][1],
                'fixed_minus_reference_joint_density': fixed['joint_gaussian_mean_density_per_pair']-original['joint_gaussian_mean_density_per_pair']})
    matched = []
    for row in rows:
        if row['mean_arm'] != 'joint_structural':
            continue
        for arm in ('flat', 'joint_gp'):
            reference = next(r for r in rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'] and r['mean_arm'] == arm)
            for stage in STAGES:
                actual, control = row['scores'][stage], reference['scores'][stage]
                matched.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'reference': arm, 'stage': stage,
                    'structural_minus_control_g_mse': actual['mean_mse_by_head'][1]-control['mean_mse_by_head'][1],
                    'structural_minus_control_g_density': actual['head_gaussian_mean_density'][1]-control['head_gaussian_mean_density'][1]})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'matched_contrasts': matched,
        'gain_rows': [{k: r[k] for k in ('layout', 'deleted_task', 'mean_arm', 'gain_diagnostics_by_head')} for r in rows],
        'stage66_moment_reproduction_passed': True, 'no_model_selected': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
