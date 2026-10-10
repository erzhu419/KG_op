"""Adapt historical corrected-reference source conditioning to restored maps."""

import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import digamma, polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_collapsed_source_raw_update import sample
from performance.check_legacy_f_learned_covariance_source_update import retained_mixture, verify_learned
from performance.check_legacy_f_original_source_raw_update import common_moments
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_source_incremental_calibration import log_density
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.refine_historical_raw_lookahead import cpu_timeout
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/historical_source_reconditioning_v1.json'
IDENTITY = ('layout', 'selector', 'support', 'treatment')


def source_case(f, eta, source):
    """Existing sampler fields, with the NEW unconditioned shared prior."""
    design = np.asarray(f['source_design']); common = np.asarray(f['unconditioned_common_covariance'])
    name = '/'.join(f[k] for k in IDENTITY[1:])
    return {'layout': f['layout'], 'arms': [name], 'sites': source['sites'],
        'f_means': [f['source_mean']], 'f_covariances': [f['source_covariance']],
        'eta_mean': eta['source_mean'], 'eta_covariance': eta['source_covariance'],
        'batch_means': source['batch_means'], 'sample_variances': source['sample_variances'],
        'common_latent_prior_means': [f['unconditioned_common_mean']],
        'common_latent_prior_covariances': [common.tolist()],
        'common_latent_to_source_f_covariances': [(common @ design.T).tolist()],
        'source_common_coordinate_systems': [f"{f['selector']}/{f['support']}:intercept+restored_features"]}


def run():
    started = time.process_time(); spec = read(PLAN); parent = read(spec['covariance_plan'])
    result = read(spec['covariance_results']); failed = read(spec['failed_transport'])
    assert result['all_registered_cases_complete'] and failed['all_registered_cases_complete']
    assert failed['eligible_raw_importance_cases'] == 0
    laws = read(spec['source_laws'])['rows']; sources = read(parent['source_fields'])['rows']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    (out/'cases').mkdir(exist_ok=True)
    cases, rows, checks = [], [], []
    stopped = False; phase = 'preparation'; likelihood_evaluations = 0
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        for f in (r for r in laws if r['head'] == 'f'):
            eta = next(r for r in laws if r['head'] == 'eta_f' and
                all(r[k] == f[k] for k in IDENTITY))
            source = next(r for r in sources if
                (r['layout'], r['active_arm']) == (f['layout'], parent['proposal_arm']))
            field = source_case(f, eta, source)
            mean, covariance = variance_reference(field['eta_mean'], field['eta_covariance'], field['sample_variances'])
            labels = np.log(field['sample_variances'])-digamma(1.)
            reference = {'reference_mean': mean, 'reference_covariance': covariance,
                'approximate_Gaussian_variance_evidence': log_density(labels-np.asarray(field['eta_mean']),
                    np.asarray(field['eta_covariance'])+float(polygamma(1, 1.))*np.eye(48))}
            check = verify_learned(field, 0, reference)
            checks.append({**{k: f[k] for k in IDENTITY}, **check})
            cases.append((f, field, reference))
        assert len(cases) == 16
        save(out/'verification.json', {'execution': spec, 'checks': checks,
            'all_new_law_density_and_common_projection_checks_passed_before_sampling': True})
        phase = 'sampling'
        for f, field, reference in cases:
            paired = {**spec, 'sampler': {**spec['sampler'], 'seed': spec['sampler']['seed']
                +1000*(f['selector']=='additive')+10*(f['support']=='joint')}}
            model = VarianceReferencePosterior(field['f_means'][0], field['f_covariances'][0],
                field['batch_means'], field['sample_variances'])
            retained = []
            posterior = sample(field, 0, paired, started, model,
                reference['reference_mean'], reference['reference_covariance'], retained)
            mixture = retained_mixture(field, 0, model, retained[0], posterior, spec)
            cm, cc = common_moments(field, 0, np.asarray(posterior['posterior_mean']), np.asarray(posterior['posterior_covariance']))
            # The reference stays a computation aid. Source posterior eta
            # uncertainty is represented by retained positive raw-posterior
            # states, never by this reference's Gaussian moments.
            identity = {k: f[k] for k in IDENTITY}
            filename = 'layout'+'_'.join(f[k] for k in IDENTITY)+'.json'
            row = {**identity, **{k: posterior[k] for k in (
                'monte_carlo_gate_passed', 'maximum_split_rhat', 'minimum_autocorrelation_ess',
                'maximum_first_second_half_f_variance_relative_difference',
                'elliptical_likelihood_evaluations_by_chain')},
                'mixture_gate_passed': mixture['mixture_gate_passed'],
                'retention_variance_relative_error': mixture['maximum_source_f_variance_relative_difference'],
                'retention_mean_rms_error_in_posterior_sd': mixture['source_f_mean_rms_difference_in_full_posterior_sd'],
                'source_posterior_usable': bool(posterior['monte_carlo_gate_passed'] and mixture['mixture_gate_passed']),
                'component_file': str((out/'cases'/filename).relative_to(ROOT))}
            save(out/'cases'/filename, {'execution_plan': PLAN, **identity, 'gates': row,
                'source_laws': spec['source_laws'], 'positive_eta_mixture': mixture,
                'common_coefficient_posterior_mean': cm.tolist(), 'common_coefficient_posterior_covariance': cc.tolist(),
                'source_f_posterior_mean': posterior['posterior_mean'],
                'source_f_posterior_variance': np.diag(posterior['posterior_covariance']).tolist(),
                'reference_mean': reference['reference_mean'].tolist(), 'reference_covariance': reference['reference_covariance'].tolist(),
                'reference_is_not_scientific_prior_or_posterior': True})
            rows.append(row); likelihood_evaluations += sum(posterior['elliptical_likelihood_evaluations_by_chain'])
            print('historical source reuse', *identity.values(), 'posterior', row['monte_carlo_gate_passed'],
                'Rhat', row['maximum_split_rhat'], 'ESS', row['minimum_autocorrelation_ess'],
                'retention', row['mixture_gate_passed'], flush=True)
            save(out/'results.json', {'execution': spec, 'rows': rows, 'all_registered_cases_complete': len(rows)==16,
                'cpu_seconds': time.process_time()-started, 'no_model_adopted': True})
    except TimeoutError:
        signal.setitimer(signal.ITIMER_PROF, 0.)
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    complete = len(rows)==16
    usable = complete and all(r['source_posterior_usable'] for r in rows)
    save(out/'results.json', {'execution': spec, 'rows': rows, 'all_registered_cases_complete': complete,
        'all_source_posterior_gates_passed': complete and all(r['monte_carlo_gate_passed'] for r in rows),
        'all_fixed_mixture_retention_gates_passed': complete and all(r['mixture_gate_passed'] for r in rows),
        'all_source_posteriors_usable': usable, 'usable_cases': sum(r['source_posterior_usable'] for r in rows),
        'stopped_at_cpu_budget': stopped, 'stop_phase': phase if stopped else None,
        'cpu_seconds': time.process_time()-started, 'cost': {**spec['cost'],
            'completed_chains': 4*len(rows), 'completed_sweeps': 6144*len(rows),
            'completed_chain_likelihood_evaluations': likelihood_evaluations,
            'retained_positive_components_per_completed_case': 512},
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Historical source sampler reconditioned the new frozen f/eta_f laws. Original cached-proposal failures remain. Source posterior precision supplies no evidence normalizing constants,hyperparameter convergence,physical factor recovery or target decision advantage. Full g/eta_g conditioning remains outstanding.'})


if __name__ == '__main__':
    run()
