"""Keep stationary mean parameters when split-Q proposals fail; continue risk updates."""

import copy
import time

import numpy as np
from scipy.special import digamma, logsumexp

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.continue_historical_raw_source_learning import load_mixture
from performance.learn_historical_raw_source_amplitudes import covariance_builder, normal_logs
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.resolve_historical_source_variance import original_posterior_summary

PLAN = 'performance/manifests/historical_stationary_source_continuation_v1.json'
KEY = (*IDENTITY, 'head')


def same(a, b):
    return all(a[k] == b[k] for k in KEY)


def retain_stationary(row, required):
    """A zero update requires prior registered stationarity, not a Q tolerance."""
    chosen = copy.deepcopy(row)
    hold = not row['source_Mstep_gate_passed']
    if hold:
        assert row['head'] in ('f', 'g')
        assert row['source_stationarity_diagnostic_passed']
        assert row['consecutive_source_stationarity_iterations'] >= required
        chosen['proposed_common_task_covariance_scales'] = row['common_task_covariance_scales']
        chosen['proposed_full_and_split_Q_improvements'] = row['full_and_split_Q_improvements']
        chosen['proposed_projected_Q_gradient_at_new_parameters'] = row['projected_Q_gradient_at_new_parameters']
        chosen['common_task_covariance_scales'] = row['old_common_task_covariance_scales']
        chosen['learned_expected_complete_NLL_per_batch'] = row['old_expected_complete_NLL_per_batch']
        chosen['full_and_split_Q_improvements'] = [0., 0., 0.]
        chosen['relative_source_covariance_change'] = 0.
        theta = np.asarray(row['old_common_task_covariance_scales'])
        scale = np.where(theta > 0., theta, 1.)
        chosen['projected_Q_gradient_at_new_parameters'] = (
            np.asarray(row['scaled_projected_raw_score_at_old_parameters']) / scale).tolist()
    chosen['stationary_parameter_held'] = hold
    chosen['source_update_accepted'] = bool(row['source_Mstep_gate_passed'] or hold)
    return chosen


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    proposal = read(spec['proposed_learning']); previous = read(spec['previous_learning'])
    precision = read(spec['previous_precision']); parent = read(spec['source_plan'])
    laws = read(parent['source_laws'])['rows']; selection = read(parent['selection_plan'])
    source_posteriors = read(spec['proposed_posterior_precision'])['rows']
    rows = [retain_stationary(row, spec['required_consecutive_stationarity_iterations']) for row in proposal['rows']]
    assert sum(r['stationary_parameter_held'] for r in rows) == len(spec['stationary_cases'])
    for registered in spec['stationary_cases']:
        assert next(r for r in rows if same(r, registered))['stationary_parameter_held']
    started = time.process_time(); evidence = []; posterior = []
    for proposed in proposal['evidence_rows']:
        mean_row = next(r for r in rows if same(r, proposed))
        risk_row = next(r for r in rows if r['head'] == ('eta_g' if proposed['head'] == 'g' else 'eta_f')
            and all(r[k] == proposed[k] for k in IDENTITY))
        case = copy.deepcopy(proposed); post = copy.deepcopy(next(r for r in source_posteriors if same(r, case)))
        joint = np.asarray(mean_row['full_and_split_Q_improvements']) + risk_row['full_and_split_Q_improvements']
        accepted = bool(mean_row['source_update_accepted'] and risk_row['source_update_accepted'] and min(joint) > 0.)
        case['source_Mstep_pair_passed'] = accepted; case['joint_full_and_split_Q_improvements'] = joint.tolist()
        case['stationary_mean_parameter_held'] = mean_row['stationary_parameter_held']
        if mean_row['stationary_parameter_held']:
            assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
            old = next(r for r in previous['evidence_rows'] if same(r, case))
            record = next(r for r in precision['rows'] if same(r, case))
            assert record['positive_mixture_precision_passed']
            law = next(r for r in laws if same(r, case))
            risk = next(r for r in laws if r['head'] == risk_row['head'] and all(r[k] == case[k] for k in IDENTITY))
            source = read(selection['source_inputs_by_layout'][case['layout']])
            response = np.vstack([t['response'] for t in source['source_tasks'] if t['name'] in selection['source_tasks']])
            head = int(case['head'] == 'g')
            model = CollapsedRawSourcePosterior(law['source_mean'], covariance_builder(law)(old['f_common_task_scales'])[0],
                response[:, head], np.exp(response[:, 2 if head else 3] + digamma(1.)))
            eta, weights = load_mixture(record, old); flat = eta.reshape(-1, 48)
            # The raw likelihood cancels exactly because the mean law is held.
            risk_mean = np.asarray(risk['source_mean']); builder = covariance_builder(risk)
            logratio = normal_logs(flat, risk_mean, builder(case['eta_common_task_scales'])[0])
            logratio -= normal_logs(flat, risk_mean, builder(old['eta_common_task_scales'])[0])
            lw = np.log(weights.ravel()) + logratio; logZ = float(logsumexp(lw))
            updated_weights = np.exp(lw - logZ).reshape(weights.shape); count = eta.shape[1]
            halves = [float(logsumexp(np.log(weights[:, s] / weights[:, s].sum()) + logratio.reshape(weights.shape)[:, s]))
                for s in (slice(0, count // 2), slice(count // 2, None))]
            ess = float(1. / np.sum(updated_weights ** 2)); gap = abs(halves[0] - halves[1])
            evidence_pass = bool(ess >= spec['evidence_precision']['minimum_importance_ESS']
                and gap <= spec['evidence_precision']['maximum_split_log_ratio_difference'])
            moments = [model.moments(model.evaluate(p)) for p in flat]
            summary = original_posterior_summary(model, eta,
                np.stack([m[0] for m in moments]).reshape(eta.shape),
                np.stack([m[1] for m in moments]).reshape(*eta.shape[:2], 48, 48), updated_weights, parent)
            case['f_common_task_scales'] = old['f_common_task_scales']
            component = out / ('candidate_' + '_'.join(case[k] for k in KEY) + '.json')
            save(component, {'fixed_candidate': {**{k: case[k] for k in KEY},
                'f_common_task_scales': case['f_common_task_scales'], 'eta_common_task_scales': case['eta_common_task_scales']},
                'posterior': summary, 'positive_eta_mixture': {'eta_states_by_chain': eta.tolist(),
                    'positive_component_weights': updated_weights.ravel().tolist()},
                'previous_candidate_component_file': record['positive_mixture_component_file']})
            component_file = str(component.relative_to(ROOT))
            case.update(evidence_ratio_precision_passed=evidence_pass,
                log_source_evidence_ratio_new_vs_previous=logZ, split_log_evidence_ratios=halves,
                split_log_ratio_difference=gap, importance_ESS=ess, candidate_component_file=component_file,
                new_positive_eta_weights=updated_weights.ravel().tolist())
            post = {**{k: case[k] for k in KEY}, **summary, 'source_case': case['source_case'],
                'candidate_component_file': component_file, 'evidence_ratio_precision_passed': evidence_pass}
        post['new_source_posterior_precision_passed'] = bool(accepted and case['evidence_ratio_precision_passed']
            and post['monte_carlo_gate_passed'])
        evidence.append(case); posterior.append(post)
    result = {**{k: copy.deepcopy(v) for k, v in proposal.items() if k not in ('rows', 'evidence_rows', 'cost')},
        'execution_plan': plan, 'rows': rows, 'evidence_rows': evidence,
        'source_updates_accepted': sum(r['source_update_accepted'] for r in rows),
        'stationary_parameters_held': sum(r['stationary_parameter_held'] for r in rows),
        'source_Mstep_Q_heads_improved': sum(r['source_Mstep_gate_passed'] for r in rows),
        'joint_full_and_split_Q_pairs_improved': sum(r['source_Mstep_pair_passed'] for r in evidence),
        'evidence_ratio_precision_cases_passed': sum(r['evidence_ratio_precision_passed'] for r in evidence),
        'posterior_precision_cases_passed': sum(r['new_source_posterior_precision_passed'] for r in posterior),
        'cpu_seconds': time.process_time() - started, 'cost': spec['cost'], 'source_learning_converged': False}
    save(path, result)
    save(out / 'posterior_precision.json', {'rows': posterior,
        'cases_passed': result['posterior_precision_cases_passed'], 'all_registered_cases_complete': len(posterior) == 32})
    print('qualified source-step pairs', result['joint_full_and_split_Q_pairs_improved'],
        'stationary heads held', result['stationary_parameters_held'],
        'cached new-law cases passed', result['posterior_precision_cases_passed'], flush=True)
    return result


if __name__ == '__main__':
    run()
