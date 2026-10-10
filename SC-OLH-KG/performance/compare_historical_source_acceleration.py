"""Compare registered risk acceleration with the certified seventh source law."""

import time

import numpy as np

from core.gpr import posterior_mixture_weights
from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.continue_historical_raw_source_learning import load_mixture
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.probe_historical_source_risk_plateau import ratio_summary
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.resolve_historical_source_variance import original_posterior_summary

PLAN = 'performance/manifests/historical_source_acceleration_comparison_v1.json'
KEY = (*IDENTITY, 'head')


def raw_source_log_ratios(old, new, states, mean, old_covariance, new_covariance):
    """Both the collapsed response likelihood and normalized risk prior change."""
    return np.asarray([new.evaluate(p)[0] - old.evaluate(p)[0] for p in states]) + log_ratios(
        states - mean, old_covariance, new_covariance)


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); probe = read(spec['probe']); previous = read(spec['previous_learning'])
    selected = read(spec['selected_learning']); precision = read(spec['selected_precision'])
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']; selection = read(source['selection_plan'])
    assert len(precision['rows']) == 32 and all(r['positive_mixture_precision_passed'] for r in precision['rows'])
    rows = []
    for proposal in probe['rows']:
        if not proposal['joint_acceleration_probe_passed']:
            continue
        record = next(r for r in precision['rows'] if all(r[k] == proposal[k] for k in KEY))
        current = next(r for r in selected['evidence_rows'] if all(r[k] == proposal[k] for k in KEY))
        parent = next(r for r in previous['evidence_rows'] if all(r[k] == proposal[k] for k in KEY))
        case = {**parent, 'eta_common_task_scales': proposal['proposed_eta_common_task_scales']}
        law = next(r for r in laws if r['head'] == case['head'] and all(r[k] == case[k] for k in IDENTITY))
        risk = next(r for r in laws if r['head'] == ('eta_g' if case['head'] == 'g' else 'eta_f')
            and all(r[k] == case[k] for k in IDENTITY))
        _, relative, _, _, _ = candidate_inputs(current, laws, selection)
        old = CollapsedRawSourcePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
        new = CollapsedRawSourcePosterior(relative.mean, covariance_builder(law)(case['f_common_task_scales'])[0],
            relative.labels, relative.s2)
        eta, weights = load_mixture(record, current); flat = eta.reshape(-1, 48)
        lr = raw_source_log_ratios(old, new, flat, np.asarray(risk['source_mean']),
            covariance_builder(risk)(current['eta_common_task_scales'])[0],
            covariance_builder(risk)(case['eta_common_task_scales'])[0]).reshape(weights.shape)
        evidence = ratio_summary(lr, weights, spec['precision_gates'])
        row = {**{k: case[k] for k in KEY}, **evidence,
            'f_common_task_scales': case['f_common_task_scales'], 'eta_common_task_scales': case['eta_common_task_scales'],
            'source6_full_and_split_Q_improvements': proposal['full_and_split_Q_improvements'],
            'posterior_evaluated': False, 'reference_learning': spec['selected_learning'],
            'normalizer_scope': 'Z(accelerated fixed source6 response law)/Z(actual source7 law) within one identity.'}
        if evidence['eligible_improvement_excludes_zero']:
            _, nw = posterior_mixture_weights(weights.ravel(), lr.ravel()); nw = nw.reshape(weights.shape)
            moments = [new.moments(new.evaluate(p)) for p in flat]
            summary = original_posterior_summary(new, eta, np.stack([m[0] for m in moments]).reshape(eta.shape),
                np.stack([m[1] for m in moments]).reshape(*eta.shape[:2], 48, 48), nw, source)
            component = out / ('candidate_' + '_'.join(case[k] for k in KEY) + '.json')
            save(component, {'execution_plan': plan, 'fixed_candidate': {k: case[k] for k in
                (*KEY, 'f_common_task_scales', 'eta_common_task_scales')}, 'posterior': summary,
                'positive_eta_mixture': {'eta_states_by_chain': eta.tolist(), 'positive_component_weights': nw.ravel().tolist()},
                'previous_candidate_component_file': record['positive_mixture_component_file']})
            row.update(posterior_evaluated=True, source_case=case['source_case'],
                candidate_component_file=str(component.relative_to(ROOT)), posterior=summary,
                direct_changed_law_posterior_precision_passed=summary['monte_carlo_gate_passed'])
        rows.append(row)
        print('compare source7', *[case[k] for k in KEY], 'evidence improvement',
            evidence['eligible_improvement_excludes_zero'], 'posterior', row.get('direct_changed_law_posterior_precision_passed'), flush=True)
    result = {'execution_plan': plan, 'rows': rows,
        'all_registered_cases_complete': len(rows) == sum(r['joint_acceleration_probe_passed'] for r in probe['rows']),
        'registered_candidates': len(rows), 'source7_evidence_precision_cases_passed': sum(r['precision_gate_passed'] for r in rows),
        'source7_evidence_improvement_cases_passed': sum(r['eligible_improvement_excludes_zero'] for r in rows),
        'direct_changed_law_posterior_cases_passed': sum(r.get('direct_changed_law_posterior_precision_passed', False) for r in rows),
        'cpu_seconds': time.process_time() - started, 'new_candidate_MCMC_sweeps': 0,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'no_model_adopted': True, 'independent_confirmation': False}
    save(path, result); return result


if __name__ == '__main__':
    run()
