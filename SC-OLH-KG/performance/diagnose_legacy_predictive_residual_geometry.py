"""Frozen training-task Gaussian score and offset/shape residual attribution."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.calibrate_legacy_task_transfer import NATIVE
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_task_spectral_moments import INTEGRATED
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_predictive_residual_geometry_execution_v1.json'
GP = 'same_source_gp'


def geometry(mean, covariance, observed, noise):
    n = len(observed); sigma = np.asarray(covariance)+np.diag(noise)
    lower = np.linalg.cholesky(sigma)
    residual = observed-np.asarray(mean)
    white = solve_triangular(lower, residual, lower=True)
    constant = solve_triangular(lower, np.ones(n), lower=True)
    offset = float(constant@white/np.linalg.norm(constant))
    q = float(white@white); logdet = float(2*np.log(np.diag(lower)).sum())
    return {'label_count': n, 'mahalanobis_energy': q, 'energy_per_label': q/n,
        'signed_offset_z': offset, 'offset_energy': offset**2,
        'shape_energy': q-offset**2, 'shape_energy_per_dimension': (q-offset**2)/(n-1),
        'predictive_log_determinant': logdet,
        'reconstructed_gaussian_mean_density': -.5*(n*np.log(2*np.pi)+logdet+q)/n,
        'latent_variance_mean': float(np.mean(np.diag(covariance))),
        'observation_noise_variance_mean': float(np.mean(noise))}


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution'])
    native = read(spec['moments_profiles'])['rows']; gp = read(spec['gp_profiles'])['rows']
    gp_scores = read(spec['gp_scores']); rows, contrasts = [], []
    for layout in spec['layouts']:
        source = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')
        for deleted in spec['training_tasks']:
            label = source['source_tasks'][deleted]
            observed, noise = np.asarray(label['response'])[:, 1], np.asarray(label['noise'])[:, 1]
            saved_gp = next(r for r in gp if r['layout'] == layout and r['deleted_task'] == deleted and r['mean_arm'] == 'joint_gp')
            for case in spec['cases']:
                saved = saved_gp if case == GP else next(r for r in native if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == case)
                np.testing.assert_array_equal(saved['scoring_sites'], coverage['layouts'][layout][deleted])
                stages = {}
                for stage in spec['stages']:
                    if case == GP:
                        p = saved['predictions']['prior' if stage == 'prior' else 'latent_risk_paid']
                        mean, covariance = np.asarray(p['mean'])[:, 1], np.asarray(p['head_covariances'])[1]
                    else:
                        p = saved['predictions'][stage]; mean, covariance = p['mean'], p['covariance']
                    stages[stage] = geometry(mean, covariance, observed, noise)
                    if case != GP:
                        np.testing.assert_allclose(stages[stage]['reconstructed_gaussian_mean_density'],
                            saved['scores'][stage]['g_gaussian_mean_density'], atol=1e-10, rtol=1e-10)
                rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'stages': stages})
            old, new = (next(r for r in rows if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == case)
                for case in (NATIVE, INTEGRATED))
            contributions = {}
            for stage in spec['stages']:
                a, b = new['stages'][stage], old['stages'][stage]; n = a['label_count']
                q_term = -.5*(a['mahalanobis_energy']-b['mahalanobis_energy'])/n
                volume_term = -.5*(a['predictive_log_determinant']-b['predictive_log_determinant'])/n
                actual = a['reconstructed_gaussian_mean_density']-b['reconstructed_gaussian_mean_density']
                np.testing.assert_allclose(q_term+volume_term, actual, atol=1e-12, rtol=1e-12)
                contributions[stage] = {'mahalanobis_contribution': q_term,
                    'log_volume_contribution': volume_term, 'density_difference': actual}
            paid_native, paid_new = (next(r for r in native if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == case)
                for case in (NATIVE, INTEGRATED))
            n = len(observed)
            raw_term = (paid_new['scores']['raw_paid']['raw_paid_log_evidence']-paid_native['scores']['raw_paid']['raw_paid_log_evidence'])/(n+2)
            conditional_term = n*contributions['raw_paid']['density_difference']/(n+2)
            actual = paid_new['scores']['raw_paid']['sequential_transcript_density']-paid_native['scores']['raw_paid']['sequential_transcript_density']
            np.testing.assert_allclose(raw_term+conditional_term, actual, atol=1e-12, rtol=1e-12)
            contrasts.append({'layout': layout, 'deleted_task': deleted, 'stages': contributions,
                'transcript_raw_paid_contribution': raw_term, 'transcript_conditional_contribution': conditional_term,
                'transcript_difference': actual})
        print('cached Gaussian density and transcript components reproduced', layout, flush=True)
    metrics = ('energy_per_label', 'offset_energy', 'shape_energy_per_dimension',
        'predictive_log_determinant', 'reconstructed_gaussian_mean_density', 'latent_variance_mean')
    summary = {layout: {case: {stage: {k: float(np.mean([r['stages'][stage][k] for r in rows if r['layout'] == layout and r['case'] == case]))
        for k in metrics} for stage in spec['stages']} for case in spec['cases']} for layout in spec['layouts']}
    for layout in spec['layouts']:
        for stage, original in (('prior', 'prior'), ('raw_paid', 'latent_risk_paid')):
            np.testing.assert_allclose(summary[layout][GP][stage]['reconstructed_gaussian_mean_density'],
                gp_scores['summary'][layout]['joint_gp'][original]['head_gaussian_mean_density'][1], atol=1e-10, rtol=1e-10)
    contribution_summary = {layout: {
        'paid_mahalanobis_contribution': float(np.mean([r['stages']['raw_paid']['mahalanobis_contribution'] for r in contrasts if r['layout'] == layout])),
        'paid_log_volume_contribution': float(np.mean([r['stages']['raw_paid']['log_volume_contribution'] for r in contrasts if r['layout'] == layout])),
        'paid_raw_evidence_transcript_contribution': float(np.mean([r['transcript_raw_paid_contribution'] for r in contrasts if r['layout'] == layout])),
        'paid_conditional_transcript_contribution': float(np.mean([r['transcript_conditional_contribution'] for r in contrasts if r['layout'] == layout]))}
        for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'contribution_summary': contribution_summary,
        'rows': rows, 'contrasts': contrasts, 'saved_scores_and_transcript_decomposition_reproduced': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'summary': summary, 'contribution_summary': contribution_summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
