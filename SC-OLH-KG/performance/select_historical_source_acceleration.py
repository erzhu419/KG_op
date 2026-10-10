"""Connect certified acceleration candidates to a complete source registry."""

import copy
import time

import numpy as np

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.continue_historical_raw_source_learning import risk_bank_moments
from performance.check_legacy_f_raw_covariance_mstep import expected_objective
from performance.diagnose_legacy_f_raw_source_covariance_score import projected

PLAN = 'performance/manifests/historical_source_acceleration_selection_v1.json'
KEY = (*IDENTITY, 'head')


def same(a, b):
    return all(a[k] == b[k] for k in KEY)


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    started = time.process_time(); comparisons = read(spec['comparisons'])
    previous = read(spec['source7_learning']); precision = read(spec['source7_precision'])
    parent = read(spec['source6_learning']); source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']
    banks = read(spec['risk_statistics']); rows = copy.deepcopy(previous['rows']); cases = []; posterior = []; certified = []
    selected = [r for r in comparisons['rows'] if r['eligible_improvement_excludes_zero']
        and r.get('direct_changed_law_posterior_precision_passed', False)]
    for record in precision['rows']:
        assert record['positive_mixture_precision_passed']
        old = next(r for r in previous['evidence_rows'] if same(r, record))
        chosen = next((r for r in selected if same(r, record)), None)
        case = copy.deepcopy(old); certificate = copy.deepcopy(record)
        component = record['positive_mixture_component_file']; evidence = 0.
        if chosen is not None:
            original = next(r for r in parent['evidence_rows'] if same(r, record))
            case.update(f_common_task_scales=chosen['f_common_task_scales'],
                eta_common_task_scales=chosen['eta_common_task_scales'])
            component = chosen['candidate_component_file']; evidence = chosen['log_source_evidence_ratio']
            mean_row = next(r for r in rows if same(r, record))
            assert mean_row['consecutive_source_stationarity_iterations'] >= 2
            mean_row['parent_selected_source_row'] = copy.deepcopy(mean_row)
            mean_row.update(common_task_covariance_scales=case['f_common_task_scales'],
                learned_expected_complete_NLL_per_batch=mean_row['old_expected_complete_NLL_per_batch'],
                full_and_split_Q_improvements=[0., 0., 0.], relative_source_covariance_change=0.,
                source_Mstep_gate_passed=False, source_update_accepted=True, stationary_parameter_held=True,
                selection_reason='Source6 response law fixed in the registered acceleration proposal after certified stationarity.')
            theta = np.asarray(mean_row['old_common_task_covariance_scales']); scale = np.where(theta > 0., theta, 1.)
            mean_row['projected_Q_gradient_at_new_parameters'] = (
                np.asarray(mean_row['scaled_projected_raw_score_at_old_parameters']) / scale).tolist()
            risk = next(r for r in laws if r['head'] == ('eta_g' if case['head'] == 'g' else 'eta_f')
                and all(r[k] == case[k] for k in IDENTITY))
            risk_row = next(r for r in rows if r['head'] == risk['head'] and all(r[k] == case[k] for k in IDENTITY))
            risk_row['parent_selected_source_row'] = copy.deepcopy(risk_row)
            bank = read(next(r['results'] for r in banks['rows'] if same(r, record)))
            _, _, seconds = risk_bank_moments(bank, original, np.asarray(risk['source_mean']))
            builder = covariance_builder(risk); theta = np.asarray(case['eta_common_task_scales'])
            before = np.asarray(original['eta_common_task_scales'])
            value, gradient = expected_objective(theta, builder, seconds[0])
            improvements = [expected_objective(before, builder, s)[0] - expected_objective(theta, builder, s)[0]
                for s in seconds]
            np.testing.assert_allclose(improvements, chosen['source6_full_and_split_Q_improvements'], atol=1e-12)
            assert min(improvements) > 0.
            risk_row.update(common_task_covariance_scales=theta.tolist(), learned_expected_complete_NLL_per_batch=value,
                full_and_split_Q_improvements=improvements, projected_Q_gradient_at_new_parameters=projected(
                    theta, gradient, np.full(2, np.inf)).tolist(),
                relative_source_covariance_change=float(np.linalg.norm(builder(theta)[0] - builder(before)[0])
                    / np.linalg.norm(builder(before)[0])),
                source_Mstep_gate_passed=True, source_update_accepted=True, stationary_parameter_held=False,
                optimizer_converged=None, optimizer_iterations=0, update_rule='generalized_EM_fixed1.5_step',
                source_stationarity_diagnostic_passed=False, consecutive_source_stationarity_iterations=0)
            certificate.update(maximum_split_rhat=chosen['posterior']['maximum_split_rhat'],
                minimum_autocorrelation_ess=chosen['posterior']['minimum_autocorrelation_ess'],
                maximum_first_second_half_f_variance_relative_difference=chosen['posterior'][
                    'maximum_first_second_half_f_variance_relative_difference'],
                point_mixture_retention_comparison_passed=None, functional_precision_reference=None,
                retention_scope='Directly certified reweighted retained-state estimate;no subsequent component reduction.')
        saved = read(component); mixture = saved['positive_eta_mixture']
        for name in ('f_common_task_scales', 'eta_common_task_scales'):
            assert saved['fixed_candidate'][name] == case[name]
        shape = np.asarray(mixture['eta_states_by_chain']).shape[:2]
        weights = np.asarray(mixture.get('positive_component_weights',
            np.full(shape, mixture.get('equal_positive_component_weight', 0.))))
        assert weights.size == np.prod(shape) and np.min(weights) > 0.
        weights = weights.ravel() / weights.sum()
        origin = spec['comparisons'] if chosen is not None else str((out / 'results.json').relative_to(ROOT))
        case.update(candidate_component_file=component, new_positive_eta_weights=weights.tolist(),
            source_Mstep_pair_passed=True, evidence_ratio_precision_passed=True,
            log_source_evidence_ratio_new_vs_previous=evidence, previous_within_identity_evidence_results=record['within_identity_evidence_results'],
            accelerated_source_candidate_selected=chosen is not None,
            source_Q_reference_learning=spec['source6_learning'] if chosen is not None else spec['source7_learning'])
        case['split_log_ratio_difference'] = chosen['split_log_ratio_difference'] if chosen is not None else 0.
        case['importance_ESS'] = float(1. / np.sum(weights ** 2))
        if chosen is not None:
            case.pop('split_log_evidence_ratios', None)
            case['joint_full_and_split_Q_improvements'] = chosen['source6_full_and_split_Q_improvements']
            case['stationary_mean_parameter_held'] = True
        else:
            case['split_log_evidence_ratios'] = [0., 0.]
        summary = saved['posterior']; assert summary['monte_carlo_gate_passed']
        posterior.append({**{k: case[k] for k in KEY}, **summary, 'source_case': case['source_case'],
            'candidate_component_file': component, 'evidence_ratio_precision_passed': True,
            'new_source_posterior_precision_passed': True})
        certificate.update(positive_mixture_component_file=component, positive_weights_results=component,
            within_identity_evidence_results=origin, log_source_evidence_ratio_new_vs_previous=evidence,
            normalizer_scope='Z(selected accelerated branch)/Z(source7);unchanged laws have exact ratio1.',
            accelerated_source_candidate_selected=chosen is not None, source_Q_reference_learning=case['source_Q_reference_learning'],
            positive_mixture_component_count=int(weights.size))
        cases.append(case); certified.append(certificate)
    assert len(cases) == len(certified) == 32 and len(rows) == 64
    result = {'execution_plan': plan, 'rows': rows, 'evidence_rows': cases, 'all_registered_cases_complete': True,
        'accelerated_source_cases_selected': len(selected), 'source7_cases_retained': 32 - len(selected),
        'posterior_precision_cases_passed': 32, 'cpu_seconds': time.process_time() - started,
        'source_learning_converged': False, 'risk_sufficient_statistics_registry': spec['risk_statistics'],
        'no_model_adopted': True, 'independent_confirmation': False,
        'scope': 'Certified alternative to iteration7,not another MCEM iteration. Selected generalized-EM candidates have positive source6 full/split Q and source evidence improvement over source7;all other source7 laws are retained.'}
    save(out / 'results.json', result)
    save(out / 'posterior_precision.json', {'rows': posterior, 'cases_passed': 32, 'all_registered_cases_complete': True})
    phase = {'status': 'Accelerated source7 alternative certified;source-risk convergence remains open',
        'rows': certified, 'source_posterior_precision_cases_passed': 32, 'complete_learned_source_identities': 16,
        'accelerated_source_cases_selected': len(selected), 'source7_cases_retained': 32 - len(selected),
        'source_learning_converged': False, 'learning_results': str((out / 'results.json').relative_to(ROOT)),
        'new_candidate_MCMC_sweeps_executed': 0, 'original_source_cases_resampled': 0, 'KG_components_changed': 0,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'source_mixture_component_counts': {str(n): sum(r['positive_mixture_component_count'] == n for r in certified)
            for n in sorted({r['positive_mixture_component_count'] for r in certified})},
        'no_model_adopted': True, 'independent_confirmation': False,
        'limitations': ['Current source7 and original source laws remain archived. This is a source-parameter registry selection;it has not entered target KG or independent evaluation.',
            'The source6 risk-statistics bank supplies the Q reference,not posterior score precision at these selected parameters. No extra stationary iteration is counted.']}
    save(out / 'phase_results.json', phase)
    print('certified acceleration registry', len(selected), 'updated', 32-len(selected), 'retained;32 cases/16 identities', flush=True)
    return phase


if __name__ == '__main__':
    run()
