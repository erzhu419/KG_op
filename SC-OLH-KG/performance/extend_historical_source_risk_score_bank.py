"""Risk-only same-prefix source sampling;retain compact sufficient statistics."""

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from performance.complete_historical_source_heads import ROOT, IDENTITY, read, save
from performance.recondition_historical_changed_candidates import candidate_inputs
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.check_legacy_f_collapsed_source_raw_update import elliptical_step
from performance.learn_historical_raw_source_amplitudes import covariance_builder
from performance.diagnose_historical_source_risk_score_precision import summarize
from performance.check_legacy_f_raw_covariance_mstep import expected_objective

PLAN = 'performance/manifests/historical_source_risk_score_bank_v1.json'
KEY = (*IDENTITY, 'head')


def draw_risk_states(model, reference_mean, reference_covariance, key, sampler, started, cpu_cap):
    """Historical compensated elliptical chains,kept only in the caller's memory."""
    root = np.linalg.cholesky(reference_covariance); count = sampler['retained_sweeps_per_chain']
    draws = []
    seed = sampler['seed'] + 1000 * (key['selector'] == 'additive') + 10 * (key['support'] == 'joint') + 10000 * (key['head'] == 'g')
    for chain in range(sampler['chains']):
        rng = np.random.default_rng(seed + 100 * int(key['layout']) + chain)
        eta = reference_mean + root @ rng.normal(size=len(model.mean)); evaluation = model.evaluate(eta)
        retained = np.empty((count, len(model.mean)))
        for iteration in range(sampler['warmup_sweeps_per_chain'] + count):
            if time.process_time() - started >= cpu_cap:
                raise TimeoutError('registered risk-only bank CPU cap')
            eta, evaluation, _ = elliptical_step(eta, reference_mean, root, evaluation, model, rng)
            if iteration >= sampler['warmup_sweeps_per_chain']:
                retained[iteration - sampler['warmup_sweeps_per_chain']] = eta
        draws.append(retained)
    return np.asarray(draws)


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists():
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); key = spec['case']
    source = read(spec['source_plan']); laws = read(source['source_laws'])['rows']
    selection = read(source['selection_plan'])
    case = next(r for r in read(spec['source_learning'])['evidence_rows'] if all(r[k] == key[k] for k in KEY))
    fixed = {k: case[k] for k in (*KEY, 'f_common_task_scales', 'eta_common_task_scales')}
    prefix = None
    if 'fresh_results' in spec:
        prior = next(r for r in read(spec['fresh_results'])['rows'] if all(r[k] == key[k] for k in KEY))
        saved = read(prior['candidate_component_file'])
        for name in ('f_common_task_scales', 'eta_common_task_scales'):
            np.testing.assert_allclose(saved['fixed_candidate'][name], case[name], atol=0., rtol=0.)
        prefix = np.asarray(saved['positive_eta_mixture']['eta_states_by_chain'])
    _, relative, _, field, gaussian = candidate_inputs(case, laws, selection)
    model = VarianceReferencePosterior(relative.mean, relative.covariance, relative.labels, relative.s2)
    rm, rc = gaussian['reference_mean'], gaussian['reference_covariance']
    sampler = spec['sampler']; count = sampler['retained_sweeps_per_chain']; prefix_error = 0.
    states = draw_risk_states(model, rm, rc, key, sampler, started, spec['maximum_single_thread_cpu_seconds'])
    if prefix is not None:
        expected = states[:, :spec['prefix_sweeps_per_chain']:spec['prefix_stride']]
        np.testing.assert_allclose(expected, prefix, atol=1e-10, rtol=0.)
        prefix_error = float(np.max(abs(expected - prefix)))
    risk = next(r for r in laws if r['head'] == ('eta_g' if key['head'] == 'g' else 'eta_f')
        and all(r[k] == key[k] for k in IDENTITY))
    theta = np.asarray(case['eta_common_task_scales']); covariance, directions = covariance_builder(risk)(theta)[:2]
    factor = cho_factor(covariance, lower=True); error = states.reshape(-1, 48) - np.asarray(risk['source_mean'])
    pe = cho_solve(factor, error.T).T
    traces = [np.trace(cho_solve(factor, d)) for d in directions]
    series = np.column_stack([.5 * (traces[i] - np.einsum('ni,ij,nj->n', pe, d, pe)) / 48
        for i, d in enumerate(directions)]).reshape(*states.shape[:2], 2)
    moments = {}; summaries = {}
    for name, sl in [('full', slice(None)), ('first_half', slice(0, count // 2)),
            ('second_half', slice(count // 2, None)), ('original_prefix', slice(0, spec['prefix_sweeps_per_chain']))]:
        bank = states[:, sl].reshape(-1, 48)
        moments[name] = {'eta_posterior_mean': bank.mean(axis=0).tolist(),
            'eta_posterior_covariance': np.cov(bank, rowvar=False, ddof=0).tolist()}
        score = series[:, sl]; mean, se, diag = summarize(score, np.ones(score.shape[:2]))
        summaries[name] = {'scaled_scores': (mean * theta).tolist(), 'scaled_MCSE': (se * theta).tolist(),
            'scaled_95_MC_halfwidth': (1.96 * se * theta).tolist(), 'diagnostics': diag}
    full = summaries['full']; diag = full['diagnostics']
    precise = all(all(v <= 1.05 for v in summaries[name]['diagnostics']['split_rhat'])
        and all(v >= 100 for v in summaries[name]['diagnostics']['autocorrelation_ess'])
        for name in ('full', 'first_half', 'second_half'))
    pm = np.asarray(moments['full']['eta_posterior_mean']) - np.asarray(risk['source_mean'])
    second = np.asarray(moments['full']['eta_posterior_covariance']) + np.outer(pm, pm)
    identity_error = float(np.max(abs(expected_objective(theta, covariance_builder(risk), second)[1]
        * theta - np.asarray(full['scaled_scores']))))
    assert identity_error < 1e-10
    result = {'execution_plan': plan, **key, 'current_candidate_parameters': fixed,
        'same_seed_prefix_matched': True if prefix is not None else None, 'maximum_prefix_error': prefix_error,
        'score_summaries': summaries, 'risk_sufficient_moments': moments,
        'functional_precision_passed': precise,
        'score_95_MC_halfwidth_below_stationarity_threshold': bool(precise
            and max(full['scaled_95_MC_halfwidth']) <= spec['stationarity_threshold']),
        'new_candidate_MCMC_sweeps_executed': sampler['chains'] * (sampler['warmup_sweeps_per_chain'] + count),
        'extension_sweeps_beyond_existing_prefix': sampler['chains'] * (count - spec['prefix_sweeps_per_chain']) if prefix is not None else 0,
        'risk_sufficient_statistics_score_identity_max_error': identity_error,
        'cpu_seconds': time.process_time() - started, 'stored_full_f_covariance_arrays': 0,
        'KG_components_changed': 0, 'new_source_labels': 0, 'new_target_simulator_calls': 0,
        'new_verifier_calls': 0, 'server_jobs': 0, 'no_model_adopted': True, 'independent_confirmation': False,
        'interval_scope': 'Conditional source-score numerical error,not task generalization.'}
    save(path, result)
    print('risk-only source bank', result['score_95_MC_halfwidth_below_stationarity_threshold'],
        'MC halfwidth', full['scaled_95_MC_halfwidth'], 'CPU', round(result['cpu_seconds'], 2), flush=True)
    return result


if __name__ == '__main__':
    run()
