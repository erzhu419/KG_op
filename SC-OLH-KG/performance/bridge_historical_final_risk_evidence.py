"""Compose actual source-risk sampling anchors without assuming independent errors."""

import time

import numpy as np
from scipy.stats import norm

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior, variance_reference
from performance.extend_historical_source_risk_score_bank import draw_risk_states
from performance.reuse_historical_source_risk_states import risk_statistics
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.check_legacy_f_risk_amplitude_importance import log_ratios
from performance.probe_historical_source_risk_plateau import ratio_summary

KEY = (*IDENTITY, 'head')


def compose_intervals(segments):
    precise = all(s['source_evidence']['precision_gate_passed'] for s in segments)
    total = sum(s['source_evidence']['log_source_evidence_ratio'] for s in segments)
    if not precise:
        return {'log_source_evidence_ratio': total, 'conditional_MC_interval_95': None,
            'precision_gate_passed': False, 'eligible_improvement_excludes_zero': False}
    z = float(norm.ppf(1. - .05 / (2 * len(segments))))
    width = sum((s['source_evidence']['conditional_MC_interval_95'][1]
        - s['source_evidence']['conditional_MC_interval_95'][0]) / (2 * 1.96) * z for s in segments)
    interval = [total - width, total + width]
    return {'log_source_evidence_ratio': total, 'conditional_MC_interval_95': interval,
        'precision_gate_passed': True, 'eligible_improvement_excludes_zero': interval[0] > 0.,
        'per_segment_normal_quantile': z, 'composition': 'Bonferroni allocation of0.05 across conditional normal-MC intervals;no error independence assumed.'}


def run(plan):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); candidate = read(spec['candidate'])
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']
    selection = read(source['selection_plan'])
    parent = next(r for r in read(spec['parent_learning'])['evidence_rows']
        if all(r[k] == candidate[k] for k in KEY))
    _, relative, _, _, _ = candidate_inputs(parent, laws, selection)
    model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    risk = next(r for r in laws if r['head'] == ('eta_g' if candidate['head'] == 'g' else 'eta_f')
        and all(r[k] == candidate[k] for k in IDENTITY))
    builder = covariance_builder(risk); mean = np.asarray(risk['source_mean'])
    cursor = parent['eta_common_task_scales']; segments = []; sweeps = 0
    for entry in spec['chronological_history']:
        data = read(entry['results'])
        assert data['fixed_response_scales'] == candidate['fixed_response_scales'] == parent['f_common_task_scales']
        if entry['kind'] == 'fixed_update':
            assert data['accepted_fixed_risk_update'] and data['checked_current_scales'] == cursor
            segments.append({'kind': 'certified fixed update', 'reference_results': entry['results'],
                'initial_scales': cursor, 'final_scales': data['final_risk_scales'],
                'source_evidence': data['source_evidence']})
            cursor = data['final_risk_scales']; continue
        assert data['initial_risk_scales'] == cursor
        original = read(data['execution_plan'])
        for number, bank in enumerate(data['sampling_batches'], 1):
            assert bank['anchor_theta'] == cursor
            attempts = [r for r in data['attempts'] if r['bank'] == number]
            accepted = [r for r in attempts if r['update_accepted']]
            endpoint = accepted[-1]['proposed_theta'] if accepted else cursor
            if endpoint == cursor:
                continue
            rm, rc = variance_reference(mean, builder(cursor)[0], relative.s2)
            sampler = {**original['sampler'], 'seed': bank['seed']}
            one = time.process_time()
            states = draw_risk_states(model, rm, rc, candidate, sampler, one, spec['maximum_single_thread_cpu_seconds_per_bank'])
            errors = states - mean
            _, summary, _ = risk_statistics(errors, np.ones(states.shape[:2]), builder, np.asarray(cursor), spec)
            prefix_error = float(np.max(abs(np.asarray(summary[0]['scaled_raw_score'])
                - attempts[0]['old_score_summaries'][0]['scaled_raw_score'])))
            assert prefix_error < 1e-9
            lr = log_ratios(errors.reshape(-1, 48), builder(cursor)[0], builder(endpoint)[0]).reshape(states.shape[:2])
            evidence = ratio_summary(lr, np.ones(states.shape[:2]), spec['evidence_precision'])
            segments.append({'kind': 'actual bank endpoints', 'reference_results': entry['results'],
                'bank': number, 'seed': bank['seed'], 'initial_scales': cursor, 'final_scales': endpoint,
                'source_evidence': evidence, 'prefix_score_maximum_error': prefix_error})
            sweeps += sampler['chains'] * (sampler['warmup_sweeps_per_chain'] + sampler['retained_sweeps_per_chain'])
            cursor = endpoint
        assert cursor == data['final_risk_scales']
    assert cursor == candidate['final_risk_scales']
    evidence = compose_intervals(segments)
    result = {'execution_plan': plan, **{k: candidate[k] for k in KEY}, 'segments': segments,
        'source_evidence': evidence, 'final_vs_parent_evidence_certified': evidence['eligible_improvement_excludes_zero'],
        'completed_candidate_MCMC_sweeps': sweeps, 'cpu_seconds': time.process_time() - started,
        'source_registry_replaced': False, 'joint_source_learning_converged': False,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'independent_confirmation': False,
        'scope': 'Actual chronological source-risk anchors and already qualified fixed updates,at fixed response law. Joint conditional numerical interval uses a union bound. No cross-identity absolute evidence,source-convergence or decision claim.'}
    save(path, result)
    print('final risk anchor bridge', *[candidate[k] for k in KEY], result['final_vs_parent_evidence_certified'],
        'segments', len(segments), 'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result
