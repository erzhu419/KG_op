"""Variance-informed Gaussian proposal with exact density compensation."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import digamma, polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_source_incremental_calibration import log_density
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_variance_reference_plan_v1.json'


def variance_reference(mean, covariance, sample_variances):
    """Existing Gaussian log-variance reference; not a scientific prior."""
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    labels = np.log(sample_variances) - digamma(1.)
    noise = float(polygamma(1, 1.))
    factor = cho_factor(covariance + noise*np.eye(len(mean)), lower=True)
    reference_mean = mean + covariance @ cho_solve(factor, labels-mean)
    reference_covariance = covariance - covariance @ cho_solve(factor, covariance)
    return reference_mean, .5*(reference_covariance+reference_covariance.T)


class VarianceReferencePosterior(CollapsedRawSourcePosterior):
    def __init__(self, mean, covariance, labels, sample_variances):
        super().__init__(mean, covariance, labels, sample_variances)
        self.approximate_labels = np.log(self.s2) - digamma(1.)
        self.approximate_noise = float(polygamma(1, 1.))

    def evaluate(self, eta):
        score, factor, alpha = super().evaluate(eta)
        noise = self.approximate_noise
        approximate = -.5 * np.sum(np.log(2 * np.pi * noise) + (self.approximate_labels - eta) ** 2 / noise)
        return float(score - approximate), factor, alpha


def run():
    started = time.process_time(); spec = read(PLAN); previous = read(spec['original_source_results'])
    assert previous['all_four_source_updates_complete'] and not previous['all_monte_carlo_gates_passed']
    fields = read(spec['source_inputs'])['rows']; references, checks = [], []
    for field in fields:
        mean = np.asarray(field['eta_mean']); covariance = np.asarray(field['eta_covariance'])
        labels = np.log(field['sample_variances']) - digamma(1.); noise = float(polygamma(1, 1.))
        posterior_mean, posterior_covariance = variance_reference(mean, covariance, field['sample_variances'])
        np.linalg.cholesky(posterior_covariance)
        evidence = log_density(labels - mean, covariance + noise * np.eye(48))
        errors = []
        for index in range(2):
            original = CollapsedRawSourcePosterior(field['f_means'][index], field['f_covariances'][index],
                field['batch_means'], field['sample_variances'])
            corrected = VarianceReferencePosterior(field['f_means'][index], field['f_covariances'][index],
                field['batch_means'], field['sample_variances'])
            for point in (mean, posterior_mean, posterior_mean + .1 * np.sqrt(np.diag(posterior_covariance))):
                left = log_density(point - posterior_mean, posterior_covariance) + corrected.evaluate(point)[0]
                right = log_density(point - mean, covariance) + original.evaluate(point)[0] - evidence
                error = abs(left - right); assert error <= 1e-8; errors.append(error)
        row = next(r for r in previous['rows'] if r['layout'] == field['layout'] and r['arm'] == 'source_gp')
        j = int(np.argmin(row['diagnostics']['autocorrelation_ess']))
        references.append({'layout': field['layout'], 'reference_mean': posterior_mean.tolist(),
            'reference_covariance': posterior_covariance.tolist(), 'approximate_Gaussian_variance_evidence': evidence,
            'reference_is_not_population_prior': True})
        checks.append({'layout': field['layout'], 'all_six_density_identities_passed': True,
            'maximum_density_identity_error': max(errors), 'old_GP_minimum_ESS': row['minimum_autocorrelation_ess'],
            'old_GP_minimum_ESS_group': ['eta', 'conditional_f_mean', 'conditional_f_second_moment'][j // 48],
            'old_GP_minimum_ESS_site': field['sites'][j % 48]})
        assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'references.json', {'execution': spec, 'rows': references, 'all_reference_density_identities_verified': True})
    save(out / 'results.json', {'execution': spec, 'checks': checks, 'diagnostic_interpreted': True,
        'cpu_seconds': time.process_time() - started, 'cost': spec['cost'], 'new_MCMC_sweeps': 0,
        'original_population_priors_and_exact_likelihood_unchanged': True,
        'interpretation': 'Variance-informed elliptical reference with exact density compensation;original scientific prior unchanged. All Gaussian log-variance factors cancel from the target law. No sampling or predictive claims.'})
    print('variance reference identities', checks, flush=True)


if __name__ == '__main__':
    run()
