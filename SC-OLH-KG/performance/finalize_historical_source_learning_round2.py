"""Preserve the passed site42 integral and resolve the remaining site4."""

import signal
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.complete_historical_source_learning_round2 import integrate_sites
from performance.finalize_historical_changed_candidate import site_summary
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.resolve_historical_source_variance import weighted_function_moments
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN = 'performance/manifests/historical_source_learning_round2_completion_v2.json'


def run():
    spec = read(PLAN); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists(): return read(str(path.relative_to(ROOT)))
    previous = read(spec['previous_functional_results'])
    repair = next(r for r in previous['functional_repairs'] if not r['functional_precision_passed'])
    fresh = read(spec['fresh_results'])
    row = next(r for r in fresh['rows'] if all(r[k] == repair[k] for k in (*IDENTITY, 'head')))
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']; selection = read(source['selection_plan'])
    case = next(r for r in read(spec['learning_results'])['evidence_rows'] if all(r[k] == row[k] for k in (*IDENTITY, 'head')))
    _, relative, _, _, reference = candidate_inputs(case, laws, selection)
    model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    mixture = read(row['candidate_component_file'])['positive_eta_mixture']
    states = np.asarray(mixture['eta_states_by_chain']); flat = states.reshape(-1, 48); site = spec['integrated_source_site']
    started = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds']); result = {'functional_precision_passed': False}
    try:
        chunks = [integrate_sites(model, flat[i:i + spec['integration_batch_size']],
            reference['reference_mean'], reference['reference_covariance'], site, [site])
            for i in range(0, len(flat), spec['integration_batch_size'])]
        ds, second = [np.concatenate([b[k] for b in chunks]).reshape(states.shape[:2])
            for k in ('displacements', 'second_moments')]
        summary = site_summary(ds, second, np.ones(states.shape[:2]))
        variance = np.asarray(repair['posterior_variance_reference']); variance[site] = summary['variance']
        gaps = np.asarray(row['half_variance_relative_differences'])
        for s, value in zip(repair['sites'], repair['site_summaries']): gaps[s] = value['half_variance_relative_difference']
        gaps[site] = summary['half_variance_relative_difference']
        points = [model.moments(model.evaluate(p)) for p in flat]
        delta, covariance = weighted_function_moments(np.stack([p[0] for p in points]).reshape(states.shape),
            np.stack([p[1] for p in points]).reshape(*states.shape[:2], 48, 48), np.ones(states.shape[:2]))
        ve = float(np.max(abs(np.diag(covariance) - variance) / variance))
        me = float(np.sqrt(np.mean((model.mean + delta - row['posterior_mean']) ** 2 / variance)))
        rhat = max(repair['maximum_split_rhat'], summary['maximum_split_rhat'])
        ess = min(repair['minimum_autocorrelation_ess'], summary['minimum_autocorrelation_ess'])
        gate = source['monte_carlo_gate']; retain = source['retained_mixture']
        passed = bool(rhat <= gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
            and ess >= gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments']
            and gaps.max() <= gate['maximum_first_second_half_f_variance_relative_difference']
            and ve <= retain['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws']
            and me <= retain['maximum_source_f_mean_rms_difference_in_full_posterior_sd'])
        result.update(**{k: case[k] for k in (*IDENTITY, 'head')}, functional_precision_passed=passed,
            site4_summary=summary, previous_site42_reference_preserved=True,
            posterior_variance_reference=variance.tolist(), maximum_split_rhat=rhat, minimum_autocorrelation_ess=ess,
            maximum_first_second_half_f_variance_relative_difference=float(gaps.max()),
            retained_mixture_variance_error=ve, retained_mixture_mean_error=me,
            candidate_component_file=row['candidate_component_file'],
            maximum_quadrature_relative_error=max(b['maximum_quadrature_relative_error'] for b in chunks),
            maximum_endpoint_log_density=max(b['maximum_endpoint_log_density'] for b in chunks))
    except (TimeoutError, RuntimeError, AssertionError) as exc:
        result['failure'] = 'registered site4 CPU cap' if isinstance(exc, TimeoutError) else str(exc)
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result.update(execution_plan=PLAN, cpu_seconds=time.process_time() - started, new_MCMC_sweeps=0,
        new_source_labels=0, server_jobs=0, no_model_adopted=True, independent_confirmation=False)
    save(path, result); print('round2 remaining site4', result['functional_precision_passed'], flush=True)
    return result


if __name__ == '__main__': run()
