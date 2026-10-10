"""Estimate each actual source normalizer against a normalized local Gaussian."""

import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from performance.complete_historical_source_heads import ROOT, read, save
from performance.compare_historical_source_structures import KEY, key, prepare, scientific_log_density
from performance.adapt_historical_source_covariance import gaussian_rows_log_density
from performance.resolve_historical_source_variance import weighted_moments
from performance.check_legacy_f_correlated_source_raw_update import diagnostics

PLAN = 'performance/manifests/historical_source_normalizers_v1.json'


def bridge_solution(proposal_delta, posterior_delta, proposal_weights, posterior_weights, tolerance=1e-10):
    """Solve Z=E_G[q/(G+q/Z)] / E_P[G/(G+q/Z)],with equal bridge shares.

    G is a normalized Gaussian,q is the scientific unnormalized source
    density,and P=q/Z. Delta=log(q/G);normalized mixture masses are retained.
    The common factor2 from the equal shares cancels in the ratio.
    """
    wp = np.asarray(proposal_weights); wp = wp / wp.sum()
    wt = np.asarray(posterior_weights); wt = wt / wt.sum()
    dp, dt = np.asarray(proposal_delta), np.asarray(posterior_delta)
    logz = float(logsumexp(np.log(wp) + dp))
    converged = False
    for iteration in range(200):
        denp = np.logaddexp(0., dp - logz)
        dent = np.logaddexp(0., dt - logz)
        top = float(logsumexp(np.log(wp) + dp - denp))
        bottom = float(logsumexp(np.log(wt) - dent))
        updated = top - bottom
        if abs(updated - logz) <= tolerance:
            converged = True
            logz = updated
            break
        logz = updated
    return logz, converged, iteration + 1


def bridge_summary(dp, dt, wp, wt, gate):
    logz, converged, iterations = bridge_solution(dp, dt, wp, wt)
    wp, wt = wp / wp.sum(), wt / wt.sum()
    denp, dent = np.logaddexp(0., dp - logz), np.logaddexp(0., dt - logz)
    lp, lt = np.log(wp) + dp - denp, np.log(wt) - dent
    cp, ct = np.exp(lp - logsumexp(lp)), np.exp(lt - logsumexp(lt))
    influences = (wp.size * (cp - wp), wt.size * (ct - wt))
    diags = [diagnostics(v[..., None]) for v in influences]
    esses = [min(d['autocorrelation_ess']) for d in diags]
    rhats = [max(d['split_rhat']) for d in diags]
    weight_esses = [float(1. / np.sum(c ** 2)) for c in (cp, ct)]
    derivative = float(np.sum(cp * np.exp(dp - logz - denp)) - np.sum(ct * np.exp(dt - logz - dent)))
    root_scale = abs(1. - derivative)
    mcse = [float(v.std(ddof=1) / np.sqrt(ess) / root_scale) for v, ess in zip(influences, esses)]
    halves = []
    for part in (0, 1):
        sp = slice(part * dp.shape[1] // 2, (part + 1) * dp.shape[1] // 2)
        st = slice(part * dt.shape[1] // 2, (part + 1) * dt.shape[1] // 2)
        value, passed, _ = bridge_solution(dp[:, sp], dt[:, st], wp[:, sp], wt[:, st])
        halves.append(value)
        converged = converged and passed
    gap = abs(halves[0] - halves[1])
    passed = bool(converged and min(weight_esses) >= gate['minimum_importance_ESS']
        and min(esses) >= gate['minimum_chain_functional_ESS']
        and max(rhats) <= gate['maximum_split_Rhat']
        and gap <= gate['maximum_split_log_ratio_difference'])
    z = float(norm.ppf(1. - .05 / 4.))
    width = z * sum(mcse)
    return {'log_source_evidence': logz, 'conditional_MC_interval_95': [logz - width, logz + width] if passed else None,
        'precision_gate_passed': passed, 'bridge_solver_converged': converged, 'bridge_solver_iterations': iterations,
        'proposal_and_posterior_weight_ESS': weight_esses, 'proposal_and_posterior_functional_ESS': esses,
        'proposal_and_posterior_split_Rhat': rhats, 'proposal_and_posterior_log_evidence_MCSE': mcse,
        'split_log_evidence_estimates': halves, 'split_log_evidence_difference': gap,
        'bridge_fixed_point_derivative': derivative,
        'interval_contract': 'Bonferroni conditional normal-MC interval for two ratio influences;no error independence assumed.'}


def case_run(entry, spec):
    case = entry['case']; suffix = '_'.join(key(case))
    out = ROOT / spec['output'] / 'cases' / suffix
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        return read(str((out / 'results.json').relative_to(ROOT)))
    started = time.process_time()
    eta, mass = entry['eta'], entry['weights']
    mean, covariance = weighted_moments(eta, mass)
    factor = np.linalg.cholesky(covariance)
    seed = spec['seed'] + 100 * int(case['layout']) + 1000 * (case['selector'] == 'additive')
    seed += 10 * (case['support'] == 'joint') + 10000 * (case['head'] == 'g')
    draws = np.random.default_rng(seed).normal(size=(4, spec['Gaussian_draws_per_chain'], 48))
    draws = mean + draws @ factor.T
    gaussian_proposal = gaussian_rows_log_density(draws.reshape(-1, 48), mean, covariance).reshape(draws.shape[:2])
    gaussian_posterior = gaussian_rows_log_density(eta.reshape(-1, 48), mean, covariance).reshape(eta.shape[:2])
    raw_proposal = scientific_log_density(entry['model'], entry['risk_mean'], entry['risk_covariance'], draws)
    summary = bridge_summary(raw_proposal - gaussian_proposal, entry['own_density'] - gaussian_posterior,
        np.ones(draws.shape[:2]), mass, spec['evidence_precision'])
    result = {**{k: case[k] for k in KEY}, 'execution_plan': PLAN, **summary,
        'fixed_response_scales': case['f_common_task_scales'], 'fixed_risk_scales': case['eta_common_task_scales'],
        'posterior_component_file': case['candidate_component_file'], 'posterior_components': int(mass.size),
        'Gaussian_reference_mean': mean.tolist(), 'Gaussian_reference_covariance': covariance.tolist(),
        'Gaussian_reference_seed': seed, 'Gaussian_reference_draws': int(np.prod(draws.shape[:2])),
        'cpu_seconds': time.process_time() - started, 'source_parameters_changed': False}
    save(out / 'results.json', result)
    print('source normalizer', suffix, 'certified', result['precision_gate_passed'],
        'split gap', round(summary['split_log_evidence_difference'], 4), flush=True)
    return result


def assemble(spec, rows, elapsed):
    by_key = {key(r): r for r in rows}; joint = []; weights = {}
    # Four influence terms (two per response) give a joint source interval.
    quantile = float(norm.ppf(1. - .05 / 8.))
    for identity in spec['all_identities']:
        pair = [by_key.get((*identity, head)) for head in ('f', 'g')]
        passed = all(r is not None and r['precision_gate_passed'] for r in pair)
        center = sum(r['log_source_evidence'] for r in pair) if all(r is not None for r in pair) else None
        width = quantile * sum(sum(r['proposal_and_posterior_log_evidence_MCSE']) for r in pair) if passed else None
        joint.append({**dict(zip(KEY[:-1], identity)), 'joint_source_evidence_certified': passed,
            'joint_log_source_evidence': center,
            'conditional_joint_MC_interval_95': [center - width, center + width] if passed else None})
    for layout in ('5', '6'):
        block = [r for r in joint if r['layout'] == layout]
        if all(r['joint_source_evidence_certified'] for r in block):
            values = np.asarray([r['joint_log_source_evidence'] for r in block])
            weights[layout] = {'identities': [[r[k] for k in KEY[:-1]] for r in block],
                'equal_prior_source_posterior_weights': np.exp(values - logsumexp(values)).tolist()}
    result = {'execution_plan': PLAN, 'rows': rows, 'joint_rows': joint, 'all_registered_cases_complete': len(rows) == 32,
        'absolute_source_normalizers_certified': sum(r['precision_gate_passed'] for r in rows),
        'joint_source_identities_certified': sum(r['joint_source_evidence_certified'] for r in joint),
        'all_cross_structure_evidence_certified': len(rows) == 32 and all(r['precision_gate_passed'] for r in rows),
        'source_structure_weights_by_layout': weights, 'source_learning': spec['source_learning'],
        'source_precision': spec['source_precision'], 'cross_layout_observation_contract': spec['cross_layout_observation_contract'],
        'new_Gaussian_integration_draws': sum(r['Gaussian_reference_draws'] for r in rows), 'new_source_MCMC_sweeps': 0,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0, 'server_jobs': 0,
        'source_parameters_changed': False, 'source_registry_replaced': False, 'independent_confirmation': False,
        'cpu_seconds': elapsed, 'recorded_case_CPU_seconds': sum(r['cpu_seconds'] for r in rows),
        'limitations': ['Source empirical-Bayes structures and local Gaussian references were fitted to the same source data/posterior banks;intervals are conditional numerical errors.',
            'Layout source data differ;each eight-structure posterior is normalized within its own layout.',
            'Independent target decision benefit remains open.'],
        'next_work': 'Resolve any failed normalizer precision before structure weighting;then reconnect these actual learned priors to matched cached target KG/GP decisions.'}
    save(ROOT / spec['output'] / 'results.json', result)
    return result


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists() and read(spec['output'] + '/results.json')['all_registered_cases_complete']:
        return read(spec['output'] + '/results.json')
    started = time.process_time(); models = prepare(spec); rows = []
    for case in spec['cases']:
        if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
            break
        rows.append(case_run(models[key(case)], spec))
        assemble(spec, rows, time.process_time() - started)
    result = assemble(spec, rows, time.process_time() - started)
    print('Source normalizers', result['absolute_source_normalizers_certified'], '/32;identities',
        result['joint_source_identities_certified'], '/16;CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


if __name__ == '__main__':
    run()
