"""Compare actual source structures using normalized raw-batch density ratios."""

import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.continue_historical_raw_source_learning import load_mixture
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.adapt_historical_source_covariance import gaussian_rows_log_density
from performance.probe_historical_source_risk_plateau import ratio_summary

PLAN = 'performance/manifests/historical_cross_structure_evidence_v1.json'
KEY = (*IDENTITY, 'head')


def key(row):
    return tuple(row[k] for k in KEY)


def scientific_log_density(model, risk_mean, risk_covariance, eta):
    """Scientific prior times raw likelihood;sampling references are absent."""
    eta = np.asarray(eta)
    flat = eta.reshape(-1, eta.shape[-1])
    values = np.asarray([model.evaluate(p)[0] for p in flat])
    values += gaussian_rows_log_density(flat, np.asarray(risk_mean), risk_covariance)
    return values.reshape(eta.shape[:2])


def combine_directions(forward, reverse):
    """Two estimates of log Z(candidate)/Z(reference),from distinct banks."""
    positive = forward['log_source_evidence_ratio']
    negative = -reverse['log_source_evidence_ratio']
    precise = forward['precision_gate_passed'] and reverse['precision_gate_passed']
    left = forward['conditional_MC_interval_95']
    right = ([-reverse['conditional_MC_interval_95'][1], -reverse['conditional_MC_interval_95'][0]]
        if reverse['conditional_MC_interval_95'] is not None else None)
    consistent = bool(precise and max(left[0], right[0]) <= min(left[1], right[1]))
    return {'forward_log_evidence_ratio': positive, 'reverse_log_evidence_ratio': negative,
        'directional_log_ratio_difference': abs(positive - negative),
        'forward_and_reverse_precision_passed': bool(precise), 'directional_intervals_overlap': consistent,
        'cross_structure_evidence_certified': consistent,
        'conditional_MC_interval_95_envelope': [min(left[0], right[0]), max(left[1], right[1])] if consistent else None,
        'forward': forward, 'reverse': reverse}


def prepare(spec):
    source = read(spec['source_plan'])
    laws = read(source['source_laws'])['rows']
    selection = read(source['selection_plan'])
    learning = read(spec['source_learning'])
    precision = read(spec['source_precision'])
    assert learning['source_learning_converged'] and len(learning['evidence_rows']) == 32
    records = {key(r): r for r in precision['rows']}
    models = {}
    for case in learning['evidence_rows']:
        record = records[key(case)]
        assert record['positive_mixture_precision_passed']
        _, relative, _, _, _ = candidate_inputs(case, laws, selection)
        model = CollapsedRawSourcePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
        risk = next(r for r in laws if key(r)[:-1] == key(case)[:-1] and r['head'] == 'eta_' + case['head'])
        covariance = covariance_builder(risk)(case['eta_common_task_scales'])[0]
        eta, weights = load_mixture(record, case)
        risk_mean = np.asarray(risk['source_mean'])
        models[key(case)] = {'model': model, 'risk_mean': risk_mean, 'risk_covariance': covariance,
            'eta': eta, 'weights': weights, 'case': case,
            'own_density': scientific_log_density(model, risk_mean, covariance, eta)}
    for layout, reference_identity in spec['reference_identities_by_layout'].items():
        for head in ('f', 'g'):
            reference = models[(*reference_identity, head)]
            for k, entry in models.items():
                if k[0] != layout or k[-1] != head:
                    continue
                # A mismatch would prevent a common-data evidence comparison.
                np.testing.assert_array_equal(entry['model'].labels, reference['model'].labels)
                np.testing.assert_array_equal(entry['model'].s2, reference['model'].s2)
                assert entry['case']['source_case'] and len(entry['model'].labels) == 48
    return models


def assemble(spec, rows, cpu_seconds):
    out = ROOT / spec['output']
    by_identity = {}
    for row in rows:
        by_identity.setdefault(key(row)[:-1], {})[row['head']] = row
    joint = []
    # Four directional intervals per identity share source data/banks.
    # The union bound does not assume independent numerical errors.
    quantile = float(norm.ppf(1. - .05 / 8.))
    for identity in spec['candidate_identities']:
        pair = by_identity.get(tuple(identity), {})
        certified = len(pair) == 2 and all(r['cross_structure_evidence_certified'] for r in pair.values())
        interval = None
        if certified:
            lower = upper = 0.
            for row in pair.values():
                intervals = []
                for direction in ('forward', 'reverse'):
                    summary = row[direction]
                    center = summary['log_source_evidence_ratio']
                    width = (summary['conditional_MC_interval_95'][1] - summary['conditional_MC_interval_95'][0]) / (2 * 1.96) * quantile
                    bounds = [center - width, center + width]
                    intervals.append(bounds if direction == 'forward' else [-bounds[1], -bounds[0]])
                lower += min(v[0] for v in intervals)
                upper += max(v[1] for v in intervals)
            interval = [lower, upper]
        joint.append({**dict(zip(IDENTITY, identity)), 'joint_source_evidence_certified': certified,
            'joint_forward_log_evidence_ratio': sum(r['forward_log_evidence_ratio'] for r in pair.values()) if len(pair) == 2 else None,
            'joint_reverse_log_evidence_ratio': sum(r['reverse_log_evidence_ratio'] for r in pair.values()) if len(pair) == 2 else None,
            'conditional_joint_MC_interval_95': interval, 'directional_interval_quantile': quantile})
    all_passed = len(rows) == len(spec['candidate_identities']) * 2 and all(r['joint_source_evidence_certified'] for r in joint)
    weights = {}
    for layout, reference in spec['reference_identities_by_layout'].items():
        comparisons = [r for r in joint if r['layout'] == layout]
        if len(comparisons) == 7 and all(r['joint_source_evidence_certified'] for r in comparisons):
            logs = np.asarray([0.] + [r['joint_forward_log_evidence_ratio'] for r in comparisons])
            weights[layout] = {'identities': [reference] + [[r[k] for k in IDENTITY] for r in comparisons],
                'equal_prior_source_posterior_weights': np.exp(logs - logsumexp(logs)).tolist()}
    result = {'execution_plan': PLAN, 'rows': rows, 'joint_rows': joint,
        'reference_identities_by_layout': spec['reference_identities_by_layout'], 'all_registered_cases_complete': len(rows) == 28,
        'head_evidence_cases_certified': sum(r['cross_structure_evidence_certified'] for r in rows),
        'joint_source_identities_certified_including_reference': 2 + sum(r['joint_source_evidence_certified'] for r in joint),
        'all_cross_structure_evidence_certified': all_passed, 'source_structure_weights_by_layout': weights,
        'same_48_raw_source_batches_within_each_layout_verified': True, 'source_parameters_changed': False,
        'cross_layout_observation_contract': spec['cross_layout_observation_contract'],
        'source_row_alignment': spec['source_row_alignment'],
        'execution_incident': spec['execution_incident'],
        'new_source_MCMC_sweeps': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0,
        'new_verifier_calls': 0, 'server_jobs': 0, 'cpu_seconds': cpu_seconds,
        'absolute_reference_log_evidence_evaluated': False, 'source_registry_replaced': False,
        'independent_confirmation': False,
        'limitations': ['Ratios compare empirical source-fitted priors on the same training data,not held-source predictive validation.',
            'Intervals describe conditional Monte Carlo error on retained historical banks;they do not cover model misspecification or adaptive source fitting.',
            'Independent target decision benefit remains untested for these final learned priors.'],
        'next_work': 'Use certified common-reference ratios for source structure weights. Resolve failed overlaps with the existing exact source sampler before target decisions;never normalize only the passing subset.'}
    save(out / 'results.json', result)
    return result


def run(plan=PLAN):
    spec = read(plan)
    out = ROOT / spec['output']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists() and read(spec['output'] + '/results.json')['all_registered_cases_complete']:
        return read(spec['output'] + '/results.json')
    started = time.process_time()
    models = prepare(spec)
    rows = []
    for identity in spec['candidate_identities']:
        for head in ('f', 'g'):
            if time.process_time() - started > spec['maximum_single_thread_cpu_seconds']:
                raise TimeoutError('registered cached comparison CPU cap')
            k = (*identity, head)
            target = models[k]
            reference = models[(*spec['reference_identities_by_layout'][identity[0]], head)]
            def at(entry, states):
                return scientific_log_density(entry['model'], entry['risk_mean'], entry['risk_covariance'], states)
            forward = ratio_summary(at(target, reference['eta']) - reference['own_density'], reference['weights'], spec['evidence_precision'])
            reverse = ratio_summary(at(reference, target['eta']) - target['own_density'], target['weights'], spec['evidence_precision'])
            row = {**dict(zip(KEY, k)), **combine_directions(forward, reverse)}
            rows.append(row)
            print('source structure', '/'.join(k), 'certified', row['cross_structure_evidence_certified'],
                'directional gap', round(row['directional_log_ratio_difference'], 4), flush=True)
            assemble(spec, rows, time.process_time() - started)
    result = assemble(spec, rows, time.process_time() - started)
    print('Cross-structure evidence', result['head_evidence_cases_certified'], '/28;identities',
        result['joint_source_identities_certified_including_reference'], '/16;CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


if __name__ == '__main__':
    run()
