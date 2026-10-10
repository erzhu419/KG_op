"""Adjacent-source evidence and local variance repair from saved candidate chains."""

import signal
import time

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import logsumexp

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.continue_historical_raw_source_learning import load_mixture
from performance.learn_historical_raw_source_amplitudes import covariance_builder, normal_logs
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.finalize_historical_changed_candidate import site_summary
from performance.diagnose_legacy_f_inner_conditional_calibration import integrate
from performance.resolve_historical_source_variance import weighted_function_moments
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN = 'performance/manifests/historical_source_learning_round2_completion_v1.json'


def integrate_sites(model, states, reference_mean, reference_covariance, axis, sites, bound=32.):
    """Unweighted new-law residuals; retain only the registered f functionals."""
    states = np.asarray(states); sd = np.sqrt(reference_covariance[axis, axis])
    direction = reference_covariance[:, axis] / sd
    origins = states - ((states[:, axis] - reference_mean[axis]) / sd)[:, None] * direction
    baseline = np.asarray([model.evaluate(p)[0] for p in states]); rows = model.covariance[sites]; cache = {}

    def at(t, index=slice(None)):
        points = np.atleast_2d(origins[index]) + np.asarray(t)[..., None] * direction
        evaluations = [model.evaluate(p) for p in points]
        logs = np.asarray([e[0] for e in evaluations]) - baseline[index] - .5 * (np.log(2 * np.pi) + np.asarray(t) ** 2)
        if isinstance(index, slice):
            ds = np.asarray([rows @ e[2] for e in evaluations])
            vs = np.asarray([np.diag(model.covariance)[sites]
                - np.sum(rows * cho_solve(e[1], rows.T).T, axis=1) for e in evaluations])
            cache['statistics'] = np.column_stack([np.ones(len(points)), ds, vs + ds ** 2])
        return logs if isinstance(index, slice) else float(logs[0])

    values, logs, error = integrate(at, lambda t: cache['statistics'], coordinate_bound=bound)
    endpoint = max(float(np.max(at(t) - logs)) for t in (-bound, bound))
    if endpoint >= np.log(1e-10):
        raise RuntimeError('Source functional integral reaches the registered coordinate boundary')
    return {'displacements': values[:, 1:1 + len(sites)], 'second_moments': values[:, 1 + len(sites):],
        'maximum_quadrature_relative_error': error, 'maximum_endpoint_log_density': endpoint}


def reverse_adjacent(old, new, risk_mean, old_risk, new_risk, states, weights, gate, evidence_gate):
    """Z(new)/Z(previous) from the actual new posterior, not original laws."""
    flat = states.reshape(-1, len(risk_mean))
    reverse = np.asarray([old.evaluate(p)[0] - new.evaluate(p)[0] for p in flat])
    reverse += normal_logs(flat, risk_mean, old_risk) - normal_logs(flat, risk_mean, new_risk)
    reverse = reverse.reshape(weights.shape); scaled = np.exp(reverse - reverse.max())
    total = weights * scaled; influence = total / total.mean() - weights / weights.mean()
    diagnostic = diagnostics(influence[..., None]); count = states.shape[1]
    logratio = float(-logsumexp(np.log(weights / weights.sum()) + reverse))
    halves = [float(-logsumexp(np.log(weights[:, s] / weights[:, s].sum()) + reverse[:, s]))
        for s in (slice(0, count // 2), slice(count // 2, None))]
    ess = float(total.sum() ** 2 / np.sum(total ** 2)); gap = abs(halves[0] - halves[1])
    passed = bool(ess >= evidence_gate['minimum_importance_ESS']
        and gap <= evidence_gate['maximum_split_log_ratio_difference']
        and max(diagnostic['split_rhat']) <= gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
        and min(diagnostic['autocorrelation_ess']) >= gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments'])
    return {'evidence_ratio_precision_passed': passed, 'log_source_evidence_ratio_new_vs_previous': logratio,
        'split_log_ratio_difference': gap, 'reverse_importance_ESS': ess,
        'reverse_ratio_rhat': max(diagnostic['split_rhat']),
        'reverse_ratio_autocorrelation_ESS': min(diagnostic['autocorrelation_ess'])}


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']
    selection = read(source['selection_plan']); prior = read(spec['previous_learning'])['evidence_rows']
    step = read(spec['learning_results']); fresh = read(spec['fresh_results'])
    candidates = step['evidence_rows']; evidence_rows = []; repairs = []
    started = time.process_time()
    for row in fresh['rows']:
        case = next(r for r in candidates if all(r[k] == row[k] for k in (*IDENTITY, 'head')))
        old_case = next(r for r in prior if all(r[k] == case[k] for k in (*IDENTITY, 'head')))
        saved = read(row['candidate_component_file']); mixture = saved['positive_eta_mixture']
        states = np.asarray(mixture['eta_states_by_chain'])
        weights = np.full(states.shape[:2], mixture['equal_positive_component_weight'])
        law = next(r for r in laws if all(r[k] == case[k] for k in (*IDENTITY, 'head')))
        risk = next(r for r in laws if r['head'] == ('eta_g' if case['head'] == 'g' else 'eta_f')
            and all(r[k] == case[k] for k in IDENTITY))
        _, relative, _, _, reference = candidate_inputs(case, laws, selection)
        old = CollapsedRawSourcePosterior(relative.mean, covariance_builder(law)(old_case['f_common_task_scales'])[0],
            relative.labels, relative.s2)
        new = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
        # Reverse density uses uncompensated raw likelihood for both laws.
        raw_new = CollapsedRawSourcePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
        evidence = reverse_adjacent(old, raw_new, np.asarray(risk['source_mean']),
            covariance_builder(risk)(old_case['eta_common_task_scales'])[0],
            covariance_builder(risk)(case['eta_common_task_scales'])[0], states, weights,
            source['monte_carlo_gate'], spec['evidence_precision'])
        evidence_rows.append({**{k: case[k] for k in (*IDENTITY, 'head')}, **evidence})
        registered = next((r for r in spec['functional_cases']
            if all(r[k] == case[k] for k in (*IDENTITY, 'head'))), None)
        if registered is None:
            continue
        one = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
        signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds_per_case'])
        repair = {**registered, 'functional_precision_passed': False}
        try:
            flat = states.reshape(-1, 48); chunks = []
            for begin in range(0, len(flat), spec['integration_batch_size']):
                chunks.append(integrate_sites(new, flat[begin:begin + spec['integration_batch_size']],
                    reference['reference_mean'], reference['reference_covariance'], registered['axis'], registered['sites']))
            ds, second = [np.concatenate([b[k] for b in chunks]).reshape(*states.shape[:2], len(registered['sites']))
                for k in ('displacements', 'second_moments')]
            summaries = [site_summary(ds[..., i], second[..., i], np.ones(states.shape[:2]))
                for i in range(len(registered['sites']))]
            corrected = np.diag(row['posterior_covariance']).copy()
            differences = np.asarray(row['half_variance_relative_differences']).copy()
            for site, summary in zip(registered['sites'], summaries):
                corrected[site] = summary['variance']; differences[site] = summary['half_variance_relative_difference']
            point = [new.moments(new.evaluate(p)) for p in flat]
            delta, covariance = weighted_function_moments(np.stack([m[0] for m in point]).reshape(states.shape),
                np.stack([m[1] for m in point]).reshape(*states.shape[:2], 48, 48), weights)
            ve = float(np.max(abs(np.diag(covariance) - corrected) / corrected))
            me = float(np.sqrt(np.mean((new.mean + delta - row['posterior_mean']) ** 2 / corrected)))
            gate = source['monte_carlo_gate']; retain = source['retained_mixture']
            rhat = max(row['maximum_split_rhat'], *(s['maximum_split_rhat'] for s in summaries))
            ess = min(row['minimum_autocorrelation_ess'], *(s['minimum_autocorrelation_ess'] for s in summaries))
            passed = bool(rhat <= gate['maximum_split_rhat_eta_and_rao_blackwell_f_moments']
                and ess >= gate['minimum_autocorrelation_ess_eta_and_rao_blackwell_f_moments']
                and differences.max() <= gate['maximum_first_second_half_f_variance_relative_difference']
                and ve <= retain['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws']
                and me <= retain['maximum_source_f_mean_rms_difference_in_full_posterior_sd'])
            repair.update(functional_precision_passed=passed, site_summaries=summaries,
                maximum_split_rhat=rhat, minimum_autocorrelation_ess=ess,
                maximum_first_second_half_f_variance_relative_difference=float(differences.max()),
                posterior_variance_reference=corrected.tolist(), retained_mixture_variance_error=ve,
                retained_mixture_mean_error=me,
                maximum_quadrature_relative_error=max(b['maximum_quadrature_relative_error'] for b in chunks),
                maximum_endpoint_log_density=max(b['maximum_endpoint_log_density'] for b in chunks),
                candidate_component_file=row['candidate_component_file'])
        except (TimeoutError, RuntimeError, AssertionError) as exc:
            repair['failure'] = 'registered source functional CPU cap' if isinstance(exc, TimeoutError) else str(exc)
        finally:
            signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
        repair['cpu_seconds'] = time.process_time() - one; repairs.append(repair)
        print('source local variance', *[case[k] for k in (*IDENTITY, 'head')],
            'pass', repair['functional_precision_passed'], 'cpu', round(repair['cpu_seconds'], 2), flush=True)
    result = {'execution_plan': plan, 'evidence_rows': evidence_rows, 'functional_repairs': repairs,
        'all_registered_cases_complete': len(evidence_rows) == len(fresh['rows']) and len(repairs) == len(spec['functional_cases']),
        'adjacent_evidence_cases_passed': sum(r['evidence_ratio_precision_passed'] for r in evidence_rows),
        'functional_repairs_passed': sum(r['functional_precision_passed'] for r in repairs),
        'cpu_seconds': time.process_time() - started, 'new_MCMC_sweeps': 0, 'new_source_labels': 0,
        'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'original_source_cases_unchanged': True, 'no_model_adopted': True, 'independent_confirmation': False}
    save(path, result)
    return result


if __name__ == '__main__':
    run()
