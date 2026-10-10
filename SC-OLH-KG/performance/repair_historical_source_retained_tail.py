"""Historical inverse-h sampler for the actual round4 site8 retention failure."""

import signal
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.resolve_historical_source_variance import (
    VarianceTiltedPosterior, original_posterior_summary, retained_weighted_mixture,
)
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.check_legacy_f_collapsed_source_raw_update import sample
from performance.historical_source_block_transition import source_block_transition
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN = 'performance/manifests/historical_source_retained_tail_v1.json'
KEY = (*IDENTITY, 'head')


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']
    selection = read(source['selection_plan']); learning = read(spec['learning_results'])
    case = next(r for r in learning['evidence_rows'] if all(r[k] == spec['case'][k] for k in KEY))
    reference = next(r for r in read(spec['functional_results'])['functional_repairs']
        if all(r[k] == case[k] for k in KEY))
    one = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    row = {k: case[k] for k in (*KEY, 'source_case')}
    try:
        _, relative, _, field, gaussian = candidate_inputs(case, laws, selection)
        model = VarianceTiltedPosterior(relative.mean, relative.covariance, relative.labels, relative.s2,
            spec['tilt_sites'], spec['tilt_center'], spec['tilt_scale'])
        rm, rc = gaussian['reference_mean'], gaussian['reference_covariance']
        transition, check = source_block_transition(rm, rc, spec['tilt_sites'])
        paired = {**spec, 'sampler': {**spec['sampler'], 'seed': spec['sampler']['seed'] + 1000}}
        eta = []; moments = []
        sample(field, 0, paired, one, model, rm, rc, eta, transition, moments)
        states = eta[0]; ds, cs = moments[0]; inverse = model.inverse_weights(ds, cs)
        raw = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
        for c in range(4):
            np.testing.assert_allclose(model.evaluate(states[c, 0])[0] + np.log(inverse[c, 0]),
                raw.evaluate(states[c, 0])[0], atol=1e-9, rtol=0.)
        summary = original_posterior_summary(model, states, ds, cs, inverse, spec)
        variances = np.diag(summary['posterior_covariance']).copy()
        splits = np.asarray(summary['half_variance_relative_differences']).copy()
        # The already-certified conditional integral belongs to this same fixed law.
        for site, scalar in zip(reference['sites'], reference['site_summaries']):
            variances[site] = scalar['variance']; splits[site] = scalar['half_variance_relative_difference']
        functional_reference = {'posterior_mean': summary['posterior_mean'],
            'posterior_covariance': np.diag(variances).tolist()}
        mixture = retained_weighted_mixture(model, states, ds, cs, inverse, functional_reference, spec)
        rhat = max(summary['maximum_split_rhat'], reference['maximum_split_rhat'])
        ess = min(summary['minimum_autocorrelation_ess'], reference['minimum_autocorrelation_ess'])
        gate = spec['monte_carlo_gate']
        functional_pass = bool(rhat <= gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
            and ess >= gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments']
            and max(splits) <= gate['maximum_first_second_half_f_variance_relative_difference'])
        passed = bool(functional_pass and mixture['mixture_gate_passed'])
        component = out / ('candidate_' + '_'.join(case[k] for k in KEY) + '.json')
        save(component, {'execution_plan': plan, 'fixed_candidate': {k: case[k] for k in (*KEY,
            'f_common_task_scales', 'eta_common_task_scales')}, 'posterior': summary,
            'positive_eta_mixture': mixture})
        row.update(summary)
        row.update(tilted_full_chain_precision_passed=summary['monte_carlo_gate_passed'],
            monte_carlo_gate_passed=functional_pass, posterior_functional_precision_passed=functional_pass,
            point_mixture_retention_passed=mixture['mixture_gate_passed'], new_source_posterior_precision_passed=passed,
            maximum_split_rhat=rhat, minimum_autocorrelation_ess=ess,
            maximum_first_second_half_f_variance_relative_difference=float(max(splits)),
            posterior_variance_reference=variances.tolist(), functional_precision_reference=spec['functional_results'],
            retained_mixture_variance_error=mixture['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws'],
            retained_mixture_mean_error=mixture['source_f_mean_rms_difference_in_full_posterior_sd'],
            candidate_component_file=str(component.relative_to(ROOT)), source_block_check=check,
            new_candidate_MCMC_sweeps=4 * (spec['sampler']['warmup_sweeps_per_chain'] + spec['sampler']['retained_sweeps_per_chain']))
    except TimeoutError:
        row.update(new_source_posterior_precision_passed=False, failure='registered retained-tail CPU cap')
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    row['cpu_seconds'] = time.process_time() - one
    result = {'execution_plan': plan, 'rows': [row], 'all_registered_cases_complete': True,
        'cpu_seconds': row['cpu_seconds'], 'new_candidate_MCMC_sweeps': row.get('new_candidate_MCMC_sweeps', 0),
        'source_posterior_precision_cases_passed': int(row['new_source_posterior_precision_passed']),
        'original_source_cases_unchanged': True, 'cost': spec['cost'], 'no_model_adopted': True,
        'independent_confirmation': False}
    save(path, result)
    print('fixed-law retained tail', row['new_source_posterior_precision_passed'],
        'retention', row.get('retained_mixture_variance_error'), 'CPU', round(row['cpu_seconds'], 2), flush=True)
    return result


if __name__ == '__main__':
    run()
