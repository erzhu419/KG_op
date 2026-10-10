"""Preserve the site8 certificate and integrate the actual site16 mismatch."""

import signal
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.complete_historical_source_learning_round2 import integrate_sites
from performance.finalize_historical_changed_candidate import site_summary
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN = 'performance/manifests/historical_source_retained_tail_completion_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    parent = read(spec['tail_plan']); source = read(parent['source_plan'])
    laws = read(source['source_laws'])['rows']; selection = read(source['selection_plan'])
    key = (*IDENTITY, 'head'); case = next(r for r in read(parent['learning_results'])['evidence_rows']
        if all(r[k] == parent['case'][k] for k in key))
    original = next(r for r in read(spec['unweighted_results'])['rows'] if all(r[k] == case[k] for k in key))
    states = np.asarray(read(original['candidate_component_file'])['positive_eta_mixture']['eta_states_by_chain'])
    saved = read(spec['tail_component']); tail = saved['posterior']
    reference = read(parent['functional_results'])['functional_repairs'][0]
    point = read(spec['point_comparison'])
    _, relative, _, _, gaussian = candidate_inputs(case, laws, selection)
    model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    started = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds']); result = {'execution_plan': plan,
        **{k: case[k] for k in key}, 'functional_precision_passed': False}
    try:
        flat = states.reshape(-1, 48); chunks = []
        for begin in range(0, len(flat), spec['integration_batch_size']):
            chunks.append(integrate_sites(model, flat[begin:begin + spec['integration_batch_size']],
                gaussian['reference_mean'], gaussian['reference_covariance'], spec['site'], [spec['site']]))
        ds, second = [np.concatenate([r[name] for r in chunks]).reshape(states.shape[:2])
            for name in ('displacements', 'second_moments')]
        scalar = site_summary(ds, second, np.ones(states.shape[:2]))
        variances = np.asarray(reference['posterior_variance_reference']).copy()
        differences = np.asarray(original['half_variance_relative_differences']).copy()
        for site, summary in zip(reference['sites'], reference['site_summaries']):
            differences[site] = summary['half_variance_relative_difference']
        variances[spec['site']] = scalar['variance']
        differences[spec['site']] = scalar['half_variance_relative_difference']
        rhat = max(tail['maximum_split_rhat'], reference['maximum_split_rhat'], scalar['maximum_split_rhat'])
        ess = min(tail['minimum_autocorrelation_ess'], reference['minimum_autocorrelation_ess'], scalar['minimum_autocorrelation_ess'])
        ve = float(np.max(abs(np.asarray(point['source_f_variances']) - variances) / variances))
        me = float(np.sqrt(np.mean((np.asarray(point['source_f_mean']) - original['posterior_mean']) ** 2 / variances)))
        gate, retain = parent['monte_carlo_gate'], parent['retained_mixture']
        passed = bool(rhat <= gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
            and ess >= gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments']
            and differences.max() <= gate['maximum_first_second_half_f_variance_relative_difference']
            and ve <= retain['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws']
            and me <= retain['maximum_source_f_mean_rms_difference_in_full_posterior_sd'])
        result.update(functional_precision_passed=passed, site_summary=scalar,
            posterior_variance_reference=variances.tolist(), maximum_split_rhat=rhat, minimum_autocorrelation_ess=ess,
            maximum_first_second_half_f_variance_relative_difference=float(differences.max()),
            retained_mixture_variance_error=ve, retained_mixture_mean_error=me,
            candidate_component_file=spec['tail_component'], previous_site8_reference_preserved=True,
            maximum_quadrature_relative_error=max(r['maximum_quadrature_relative_error'] for r in chunks),
            maximum_endpoint_log_density=max(r['maximum_endpoint_log_density'] for r in chunks))
    except (TimeoutError, RuntimeError) as exc:
        result['failure'] = 'registered retained-tail functional CPU cap' if isinstance(exc, TimeoutError) else str(exc)
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result.update(cpu_seconds=time.process_time() - started, new_candidate_MCMC_sweeps=0,
        new_source_labels=0, server_jobs=0, no_model_adopted=True, independent_confirmation=False)
    save(path, result)
    print('retained source tail completion', result['functional_precision_passed'],
        'retention', result.get('retained_mixture_variance_error'), 'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


if __name__ == '__main__':
    run()
