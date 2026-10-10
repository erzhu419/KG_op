"""Frozen f tail contrast: constant/shape geometry and raw-density chain rule."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import digamma, logsumexp, roots_hermitenorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_data_scaled_f_tail import DATA, NATIVE, forecast_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_task_bank_centering import prepare
from performance.diagnose_legacy_predictive_residual_geometry import geometry
from performance.run_legacy_related_inventory_source import save
from representation.physical_risk_prior import log_sample_variance_density
from representation.task_bank_centering import centered_g_field

EXECUTION = 'performance/manifests/legacy_joint_population_data_scaled_f_tail_endpoint_execution_v1.json'


def variance_score(components, eta_label, order=None):
    """Marginal p(log S2) for the same Gaussian eta law and three repeats."""
    k = 1.
    sample_variance = np.exp(eta_label+digamma(k)-np.log(k))
    terms = []
    for weight, mean, covariance in components:
        em, ev = mean[1], covariance[1, 1]
        if order is not None:
            nodes, weights = roots_hermitenorm(order)
            positive = weights > 0.
            nodes, weights = nodes[positive], weights[positive]
            score = logsumexp(np.log(weights)+log_sample_variance_density(
                em+np.sqrt(ev)*nodes, sample_variance, 3))-.5*np.log(2*np.pi)
        else:
            def log_integrand(eta):
                return (-.5*(np.log(2*np.pi*ev)+(eta-em)**2/ev)
                    +log_sample_variance_density(eta, sample_variance, 3))
            lo, hi = em-12*np.sqrt(ev), em+12*np.sqrt(ev)
            mode = minimize_scalar(lambda e: -log_integrand(e), bounds=(lo, hi), method='bounded').x
            shift = log_integrand(mode)
            integral, _ = quad(lambda e: np.exp(log_integrand(e)-shift), lo, hi,
                points=[mode], epsabs=1e-10, epsrel=1e-10)
            score = shift+np.log(integral)
        terms.append(np.log(weight)+score)
    return float(logsumexp(terms))


def run():
    started = time.process_time(); spec = read(EXECUTION); tail = read(spec['tail_execution'])
    centering = read(tail['centering_execution']); coverage, branches = prepare(centering)
    profiles = read(spec['tail_profiles'])['rows']; gp = read(spec['gp_profiles'])['rows']
    laws = read(Path(tail['output'])/'source_laws.json')['rows']
    raw = read(coverage['source_data'])
    paid_values = {i: [raw['tasks'][i]['Y_replicates'][j][0][0] for j in tail['paid_candidates']]
        for i in spec['training_tasks']}; del raw
    rows, verification = [], []
    for (layout, deleted), branch in branches.items():
        labels = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks'][deleted]
        observed, noise = np.asarray(labels['response']), np.asarray(labels['noise'])[:, 0]
        template = components(branch['inputs'], branch['means'], read(branch['risk_path']),
            branch['mean_psi'], np.asarray(branch['inputs']['public_risk_reference']))[0][0]
        law = next(r['law'] for r in laws if r['layout'] == layout and r['deleted_task'] == deleted)
        candidate = centered_g_field(template, branch['designs'][branch['task_name']], law,
            np.zeros(branch['basis'].feature_dim))
        for case, field in ((NATIVE, template), (DATA, candidate)):
            cached = next(r for r in profiles if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == case)
            prediction, (_, paid) = forecast_field(field, branch['sites'], tail['paid_candidates'], paid_values[deleted])
            for key in ('mean', 'covariance'):
                np.testing.assert_allclose(prediction['raw_paid'][key], cached['predictions']['raw_paid'][key], atol=1e-10, rtol=1e-10)
            stages = {stage: geometry(p['mean'], p['covariance'], observed[:, 0], noise)
                for stage, p in cached['predictions'].items()}
            for stage, values in stages.items():
                np.testing.assert_allclose(values['reconstructed_gaussian_mean_density'],
                    cached['scores'][stage]['f_gaussian_mean_density'], atol=1e-10, rtol=1e-10)
            marginal = []
            for j, value in zip(branch['sites'], observed):
                local = paid.local_components(j); score = variance_score(local, value[3]); marginal.append(score)
                if deleted == 1:
                    q256, q512 = (variance_score(local, value[3], order) for order in (256, 512))
                    verification.append({'layout': layout, 'case': case, 'candidate': j,
                        'quadrature_256_512_difference': abs(q256-q512), 'adaptive_512_difference': abs(score-q512)})
                    np.testing.assert_allclose([q256, score], q512, atol=1e-7, rtol=0.)
            mean, covariance = paid.predict(full_cov=True); sites = branch['sites']
            chosen = law if case == DATA else branch['means']['heads'][0]['joint_structural']
            rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'stages': stages,
                'raw_batch_density': cached['scores']['raw_paid']['f_raw_batch_density'],
                'marginal_log_variance_density': float(np.mean(marginal)),
                'conditional_sample_mean_density': cached['scores']['raw_paid']['f_raw_batch_density']-float(np.mean(marginal)),
                'eta_mean': mean[sites, 1].tolist(), 'eta_variance_mean': float(np.mean(covariance[sites, 1, sites, 1])),
                'task_intercept_variance': chosen['residual_coefficient_diagonal'][0]})
        saved_gp = next(r for r in gp if r['layout'] == layout and r['deleted_task'] == deleted and r['mean_arm'] == 'joint_gp')
        rows.append({'layout': layout, 'deleted_task': deleted, 'case': 'same_source_gp', 'stages': {stage:
            geometry(np.asarray(saved_gp['predictions'][key]['mean'])[:, 0], saved_gp['predictions'][key]['head_covariances'][0], observed[:, 0], noise)
            for stage, key in (('prior', 'prior'), ('raw_paid', 'latent_risk_paid'))}})
    contrasts = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            old, new = (next(r for r in rows if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == case) for case in (NATIVE, DATA))
            n = old['stages']['raw_paid']['label_count']; a, b = new['stages']['raw_paid'], old['stages']['raw_paid']
            contrasts.append({'layout': layout, 'deleted_task': deleted,
                'raw_density_difference': new['raw_batch_density']-old['raw_batch_density'],
                'log_variance_density_difference': new['marginal_log_variance_density']-old['marginal_log_variance_density'],
                'conditional_mean_density_difference': new['conditional_sample_mean_density']-old['conditional_sample_mean_density'],
                'gaussian_energy_contribution': -.5*(a['mahalanobis_energy']-b['mahalanobis_energy'])/n,
                'gaussian_log_volume_contribution': -.5*(a['predictive_log_determinant']-b['predictive_log_determinant'])/n,
                'eta_mean_shift_rms': float(np.sqrt(np.mean((np.asarray(new['eta_mean'])-old['eta_mean'])**2))),
                'eta_variance_ratio': new['eta_variance_mean']/old['eta_variance_mean']})
    metrics = ('raw_density_difference', 'log_variance_density_difference', 'conditional_mean_density_difference')
    summary = {layout: {group: {metric: float(np.mean([r[metric] for r in contrasts
        if r['layout'] == layout and (group == 'all6' or r['deleted_task'] != 1)])) for metric in metrics}
        for group in ('all6', 'other5')} for layout in spec['layouts']}
    result = {'execution': spec, 'rows': rows, 'contrasts': contrasts, 'summary': summary,
        'marginal_quadrature_verification': verification, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'summary': summary, 'task1_contrasts': [r for r in contrasts if r['deleted_task'] == 1], 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
