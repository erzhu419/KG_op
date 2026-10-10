"""Continue raw-source MCEM from certified covariance candidates."""

import signal
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, logsumexp

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.learn_historical_raw_source_amplitudes import covariance_builder, normal_logs
from performance.check_legacy_f_raw_covariance_mstep import expected_objective
from performance.diagnose_legacy_f_raw_source_covariance_score import score, projected
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.resolve_historical_source_variance import (
    weighted_moments, weighted_function_moments, original_posterior_summary,
)
from performance.refine_historical_raw_lookahead import cpu_timeout

PLAN = 'performance/manifests/historical_raw_source_amplitude_learning_v2.json'


def load_mixture(row, case):
    """Use candidate inverse-h weights or the registered new-law importance weights."""
    saved = read(row['positive_mixture_component_file'])
    mixture = saved['positive_eta_mixture']
    eta = np.asarray(mixture['eta_states_by_chain'])
    if row['positive_weights_results'] == row['positive_mixture_component_file']:
        for name in ('f_common_task_scales', 'eta_common_task_scales'):
            np.testing.assert_allclose(saved['fixed_candidate'][name], case[name], atol=0., rtol=0.)
        weights = (mixture['positive_component_weights'] if 'positive_component_weights' in mixture
            else np.full(eta.shape[:2], mixture['equal_positive_component_weight']))
    else:
        weights = case['new_positive_eta_weights']
    weights = np.asarray(weights).reshape(eta.shape[:2])
    return eta, weights / weights.sum()


def sufficient_moments(model, eta, displacements, covariances, weights, part):
    delta, covariance = weighted_function_moments(displacements[:, part], covariances[:, part], weights[:, part])
    em, ec = weighted_moments(eta[:, part], weights[:, part])
    return covariance + np.outer(delta, delta), em, ec


def risk_bank_moments(bank, case, prior_mean):
    """Use full/split source-risk moments from the same current covariance law."""
    assert bank['functional_precision_passed'] and bank['score_95_MC_halfwidth_below_stationarity_threshold']
    for name in (*IDENTITY, 'head', 'f_common_task_scales', 'eta_common_task_scales'):
        assert bank['current_candidate_parameters'][name] == case[name]
    values = [bank['risk_sufficient_moments'][name] for name in ('full', 'first_half', 'second_half')]
    seconds = []
    for value in values:
        pm = np.asarray(value['eta_posterior_mean']); pc = np.asarray(value['eta_posterior_covariance'])
        delta = pm - prior_mean
        seconds.append(pc + np.outer(delta, delta))
    return np.asarray(values[0]['eta_posterior_mean']), np.asarray(values[0]['eta_posterior_covariance']), seconds


def run(plan=PLAN):
    spec = read(plan)
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists() and read(str(path.relative_to(ROOT))).get('all_registered_cases_complete'):
        return read(str(path.relative_to(ROOT)))
    parent = read(spec['source_plan'])
    laws = read(parent['source_laws'])['rows']; selection = read(parent['selection_plan'])
    previous = read(spec['previous_learning']); closure = read(spec['previous_precision'])
    assert len(closure['rows']) == 32 and all(r['positive_mixture_precision_passed'] for r in closure['rows'])
    prior_cases = previous['evidence_rows']; rows = []; evidence = []; posterior = []
    banks = {}
    if 'risk_sufficient_statistics' in spec:
        registry = read(spec['risk_sufficient_statistics'])
        assert registry['all_registered_cases_complete'] and registry['risk_score_precision_cases_passed'] == 32
        banks = {tuple(r[k] for k in (*IDENTITY, 'head')): r['results'] for r in registry['rows']}
    started = time.process_time(); handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds']); stopped = False
    try:
        for record in closure['rows']:
            case = next(r for r in prior_cases if all(r[k] == record[k] for k in (*IDENTITY, 'head')))
            law = next(r for r in laws if all(r[k] == case[k] for k in (*IDENTITY, 'head')))
            risk = next(r for r in laws if r['head'] == ('eta_g' if case['head'] == 'g' else 'eta_f')
                and all(r[k] == case[k] for k in IDENTITY))
            source = read(selection['source_inputs_by_layout'][case['layout']])
            response = np.vstack([t['response'] for t in source['source_tasks'] if t['name'] in selection['source_tasks']])
            head = int(case['head'] == 'g'); labels = response[:, head]
            s2 = np.exp(response[:, 2 if head else 3] + digamma(1.))
            fb, eb = covariance_builder(law), covariance_builder(risk)
            ft, et = np.asarray(case['f_common_task_scales']), np.asarray(case['eta_common_task_scales'])
            model = CollapsedRawSourcePosterior(law['source_mean'], fb(ft)[0], labels, s2)
            eta, weights = load_mixture(record, case); count = eta.shape[1]
            evaluations = [model.evaluate(p) for p in eta.reshape(-1, 48)]
            moments = [model.moments(e) for e in evaluations]
            ds = np.stack([m[0] for m in moments]).reshape(eta.shape)
            cs = np.stack([m[1] for m in moments]).reshape(*eta.shape[:2], 48, 48)
            sufficient = [sufficient_moments(model, eta, ds, cs, weights, part)
                for part in (slice(None), slice(0, count // 2), slice(count // 2, None))]
            learned = []; pair_gates = []
            for current, builder, theta, index in ((law, fb, ft, 0), (risk, eb, et, 1)):
                prior_mean = np.asarray(current['source_mean'])
                seconds = [value[0] if index == 0 else value[2] + np.outer(value[1] - prior_mean, value[1] - prior_mean)
                    for value in sufficient]
                delta, cov = weighted_function_moments(ds, cs, weights)
                pm, pc = ((model.mean + delta, cov) if index == 0 else sufficient[0][1:])
                if index == 1 and banks:
                    bank_path = banks[tuple(case[k] for k in (*IDENTITY, 'head'))]
                    pm, pc, seconds = risk_bank_moments(read(bank_path), case, prior_mean)
                old_value, gradient = expected_objective(theta, builder, seconds[0])
                direct = score(prior_mean, builder(theta)[0], builder(theta)[1], pm, pc)
                np.testing.assert_allclose(gradient, direct, atol=1e-8, rtol=1e-8)
                scale = np.where(theta > 0., theta, 1.)
                fit = minimize(lambda z: (lambda v, g: (v, g * scale))(
                    *expected_objective(z * scale, builder, seconds[0])), theta / scale,
                    method='L-BFGS-B', jac=True, bounds=[(0., None)] * 2,
                    options={'maxiter': spec['optimizer']['max_iterations'],
                        'ftol': spec['optimizer']['ftol'], 'gtol': spec['optimizer']['gtol']})
                new_theta = fit.x * scale
                value, new_gradient = expected_objective(new_theta, builder, seconds[0])
                improvements = [expected_objective(theta, builder, s)[0] - expected_objective(new_theta, builder, s)[0]
                    for s in seconds]
                step_pass = bool(fit.success and min(improvements) > 0.)
                relative_change = float(np.linalg.norm(builder(new_theta)[0] - builder(theta)[0])
                    / np.linalg.norm(builder(theta)[0]))
                pg = projected(theta, gradient, np.full(2, np.inf)) * scale
                stationary = bool(relative_change <= spec['convergence']['maximum_relative_covariance_change']
                    and np.max(abs(pg)) <= spec['convergence']['maximum_scaled_projected_raw_score'])
                previous_head = next(r for r in previous['rows'] if r['head'] == current['head']
                    and all(r[k] == case[k] for k in IDENTITY))
                previous_streak = previous_head.get('consecutive_source_stationarity_iterations',
                    int(previous_head.get('source_stationarity_diagnostic_passed', False)))
                streak = previous_streak + 1 if stationary else 0
                rows.append({**{k: case[k] for k in IDENTITY}, 'head': current['head'],
                    'risk_sufficient_statistics_results': bank_path if index == 1 and banks else None,
                    'old_common_task_covariance_scales': theta.tolist(), 'common_task_covariance_scales': new_theta.tolist(),
                    'old_expected_complete_NLL_per_batch': old_value, 'learned_expected_complete_NLL_per_batch': value,
                    'full_and_split_Q_improvements': improvements, 'source_Mstep_gate_passed': step_pass,
                    'optimizer_converged': bool(fit.success), 'optimizer_iterations': int(fit.nit),
                    'initial_raw_score_identity_max_error': float(np.max(abs(gradient - direct))),
                    'scaled_projected_raw_score_at_old_parameters': pg.tolist(),
                    'projected_Q_gradient_at_new_parameters': projected(new_theta, new_gradient, np.full(2, np.inf)).tolist(),
                    'relative_source_covariance_change': relative_change, 'source_stationarity_diagnostic_passed': stationary,
                    'consecutive_source_stationarity_iterations': streak})
                learned.append(new_theta); pair_gates.append(step_pass)
            new = CollapsedRawSourcePosterior(law['source_mean'], fb(learned[0])[0], labels, s2)
            flat = eta.reshape(-1, 48); next_evaluations = [new.evaluate(p) for p in flat]
            logratio = np.asarray([a[0] - b[0] for a, b in zip(next_evaluations, evaluations)])
            logratio += normal_logs(flat, np.asarray(risk['source_mean']), eb(learned[1])[0])
            logratio -= normal_logs(flat, np.asarray(risk['source_mean']), eb(et)[0])
            lw = np.log(weights.reshape(-1)) + logratio
            logZ = float(logsumexp(lw)); new_weights = np.exp(lw - logZ).reshape(weights.shape)
            splits = [float(logsumexp(np.log(weights[:, s] / weights[:, s].sum())
                + logratio.reshape(weights.shape)[:, s])) for s in (slice(0, count // 2), slice(count // 2, None))]
            ess = float(1. / np.sum(new_weights ** 2)); gap = abs(splits[0] - splits[1])
            evidence_pass = bool(ess >= spec['evidence_precision']['minimum_importance_ESS']
                and gap <= spec['evidence_precision']['maximum_split_log_ratio_difference'])
            updated = [new.moments(e) for e in next_evaluations]
            summary = original_posterior_summary(new, eta,
                np.stack([m[0] for m in updated]).reshape(eta.shape),
                np.stack([m[1] for m in updated]).reshape(cs.shape), new_weights, parent)
            case_pass = bool(all(pair_gates) and evidence_pass and summary['monte_carlo_gate_passed'])
            component = out / ('candidate_' + '_'.join(case[k] for k in (*IDENTITY, 'head')) + '.json')
            save(component, {'fixed_candidate': {**{k: case[k] for k in (*IDENTITY, 'head')},
                'f_common_task_scales': learned[0].tolist(), 'eta_common_task_scales': learned[1].tolist()},
                'posterior': summary, 'positive_eta_mixture': {'eta_states_by_chain': eta.tolist(),
                    'positive_component_weights': new_weights.reshape(-1).tolist()},
                'previous_candidate_component_file': record['positive_mixture_component_file']})
            evidence.append({**{k: case[k] for k in (*IDENTITY, 'head')}, 'source_case': case['source_case'],
                'f_common_task_scales': learned[0].tolist(), 'eta_common_task_scales': learned[1].tolist(),
                'source_Mstep_pair_passed': all(pair_gates), 'evidence_ratio_precision_passed': evidence_pass,
                'log_source_evidence_ratio_new_vs_previous': logZ,
                'previous_within_identity_evidence_results': record['within_identity_evidence_results'],
                'split_log_evidence_ratios': splits, 'split_log_ratio_difference': gap, 'importance_ESS': ess,
                'candidate_component_file': str(component.relative_to(ROOT)),
                'new_positive_eta_weights': new_weights.reshape(-1).tolist()})
            posterior.append({**{k: case[k] for k in (*IDENTITY, 'head')}, **summary,
                'source_case': case['source_case'], 'candidate_component_file': str(component.relative_to(ROOT)),
                'new_source_posterior_precision_passed': case_pass, 'evidence_ratio_precision_passed': evidence_pass})
            print('source MCEM', *[case[k] for k in (*IDENTITY, 'head')],
                'Q', all(pair_gates), 'evidence', evidence_pass, 'posterior', summary['monte_carlo_gate_passed'], flush=True)
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    convergence_pass = bool(len(evidence) == 32 and len(rows) == 64
        and all(r['consecutive_source_stationarity_iterations'] >= spec['convergence']['required_consecutive_iterations'] for r in rows)
        and all(r['new_source_posterior_precision_passed'] for r in posterior))
    result = {'execution_plan': plan, 'rows': rows, 'evidence_rows': evidence,
        'all_registered_cases_complete': len(evidence) == 32, 'source_Mstep_gates_passed': sum(r['source_Mstep_gate_passed'] for r in rows),
        'source_stationarity_heads': sum(r['source_stationarity_diagnostic_passed'] for r in rows),
        'evidence_ratio_precision_cases_passed': sum(r['evidence_ratio_precision_passed'] for r in evidence),
        'posterior_precision_cases_passed': sum(r['new_source_posterior_precision_passed'] for r in posterior),
        'source_learning_converged': convergence_pass, 'stopped_at_cpu_budget': stopped, 'cpu_seconds': time.process_time() - started,
        'risk_sufficient_statistics_registry': spec.get('risk_sufficient_statistics'),
        'cost': spec['cost'], 'no_model_adopted': True, 'independent_confirmation': False}
    save(path, result)
    save(out / 'posterior_precision.json', {'rows': posterior, 'cases_passed': result['posterior_precision_cases_passed'],
        'all_registered_cases_complete': len(posterior) == 32})
    return result


if __name__ == '__main__':
    run()
