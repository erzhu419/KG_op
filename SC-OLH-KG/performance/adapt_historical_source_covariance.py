"""V47 fit-noise subtraction and common/task source laws on frozen maps."""

import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.gpr import posterior_mixture_weights
from performance.check_legacy_f_collapsed_source_raw_update import CollapsedRawSourcePosterior
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_source_incremental_calibration import log_density
from performance.prepare_legacy_f_original_source_raw_contract import common_replay
from performance.refine_historical_raw_lookahead import cpu_timeout
from performance.run_legacy_related_inventory_source import save
from representation.exchangeable_mean import (
    ExchangeableBoundaryMeanCoordinate, centered_fit_covariance_parts)
from representation.physical_risk_prior import log_sample_variance_density

PLAN = 'performance/manifests/historical_source_covariance_adaptation_v1.json'
HEADS = ('f', 'g', 'eta_g', 'eta_f')


def coefficient_fits(features, tasks, head, units, ridge):
    """Existing weighted ridge estimator; sampling covariance uses label noise.

    The ridge posterior inverse is not the estimator's sampling covariance.
    Keep the actual noise pushforward H diag(noise) H' for subtraction.
    """
    means, covariances = [], []
    for X, task in zip(np.split(features, len(tasks)), tasks):
        y = (np.asarray(task['response'])[:, head] - units['center']) / units['scale']
        noise = np.asarray(task['noise'])[:, head] / units['scale']**2
        weights = 1 / np.maximum(noise, 1e-8)
        beta, precision = ExchangeableBoundaryMeanCoordinate._solve(X, y, weights, ridge)
        design = np.column_stack([np.ones(len(X)), X])
        influence = np.linalg.pinv(precision) @ (design.T * weights)
        covariance = (influence * noise) @ influence.T * units['scale']**2
        beta *= units['scale']; beta[0] += units['center']
        means.append(beta); covariances.append(.5 * (covariance + covariance.T))
    return np.asarray(means), np.asarray(covariances)


def population_law(design, mean, common, task, white):
    residual = block_diag(*[d @ task @ d.T + white*np.eye(len(d)) for d in np.split(design, 4)])
    return design @ mean, design @ common @ design.T + residual, residual


def importance_summary(logs, spec):
    _, weights = posterior_mixture_weights(np.full(len(logs), 1/len(logs)), logs)
    ess = float(1/(weights @ weights))
    row = {'weight_ESS': ess, 'maximum_weight': float(weights.max()),
        'weights_by_chain': weights.reshape(4, 128).sum(axis=1).tolist(),
        'raw_source_log_evidence_ratio_estimate': float(logsumexp(logs)-np.log(len(logs))),
        'chain_functional_ESS': None, 'split_Rhat': None, 'half_NLL_differences': None,
        'precision_gate_passed': False, 'conditional_MC_interval_95': None}
    if ess < spec['minimum_weight_ESS']:
        row['failure'] = 'insufficient_cached_proposal_overlap'
        return row
    values = np.exp(logs-logs.max()).reshape(4, 128, 1)
    diagnostic = diagnostics(values)
    chain_ess, rhat = diagnostic['autocorrelation_ess'][0], diagnostic['split_rhat'][0]
    halves = [float(-(logsumexp(p)-np.log(p.size))/48) for p in np.split(logs.reshape(4, 128), 2, axis=1)]
    delta = -row['raw_source_log_evidence_ratio_estimate']/48
    passed = (chain_ess >= spec['minimum_chain_functional_ESS'] and rhat <= spec['maximum_Rhat']
        and abs(halves[0]-halves[1]) <= spec['maximum_half_NLL_difference'])
    row.update(chain_functional_ESS=chain_ess, split_Rhat=rhat, half_NLL_differences=halves,
        precision_gate_passed=bool(passed))
    if passed:
        se = float(values.std(ddof=1)/np.sqrt(chain_ess)/values.mean()/48)
        row['conditional_MC_interval_95'] = [delta-1.96*se, delta+1.96*se]
    else:
        row['failure'] = 'cached_functional_precision_failed'
    return row


def gaussian_rows_log_density(values, mean, covariance):
    factor = cho_factor(covariance, lower=True)
    error = values - mean
    return -.5*(len(mean)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()
        + np.einsum('ij,ji->i', error, cho_solve(factor, error.T)))


def run():
    started = time.process_time(); spec = read(PLAN)
    maps = read(spec['maps'])['rows']; priors = read(spec['coefficient_priors'])['rows']
    selection = read(spec['selection_plan']); source_fields = read(spec['source_fields'])['rows']
    mixture_file = read(spec['source_components'])
    assert mixture_file['all_source_posterior_precision_gates_passed'] and mixture_file['all_mixture_retention_gates_passed']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows, raw_rows, laws, checks, completed = [], [], [], [], 0
    # The same CPU interrupt already used by the cached lookahead refinement.
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds'])
    try:
        for layout, path in selection['source_inputs_by_layout'].items():
            inputs = read(path); tasks = [t for t in inputs['source_tasks'] if t['name'] in selection['source_tasks']]
            assert [t['name'] for t in tasks] == selection['source_tasks']
            source = next(r for r in source_fields if (r['layout'], r['active_arm']) == (layout, spec['proposal_arm']))
            mixture = next(r for r in mixture_file['rows'] if (r['layout'], r['arm']) == (layout, spec['proposal_arm']))
            draws = np.asarray(mixture['eta_states_by_chain']).reshape(512, 48)
            old = CollapsedRawSourcePosterior(source['f_means'][source['active_arm_index']],
                source['f_covariances'][source['active_arm_index']], source['batch_means'], source['sample_variances'])
            old_logs = np.array([old.evaluate(e)[0] for e in draws])
            old_prior_logs = gaussian_rows_log_density(draws, np.asarray(source['eta_mean']), np.asarray(source['eta_covariance']))
            density_error = abs(old_prior_logs[0]-log_density(draws[0]-np.asarray(source['eta_mean']), np.asarray(source['eta_covariance'])))
            independent_raw = log_density(np.asarray(source['batch_means'])-old.mean,
                old.covariance+np.diag(np.exp(draws[0])/3))
            independent_raw += np.sum(log_sample_variance_density(draws[0], source['sample_variances'], 3))
            assert density_error < 1e-9 and abs(independent_raw-old_logs[0]) < 1e-9
            checks.append({'layout': layout, 'Gaussian_prior_density_normalization_error': float(density_error),
                'raw_mean_and_variance_density_error': abs(float(independent_raw)-old_logs[0]),
                'unchanged_raw_law_ratio': 0., 'passed': True})
            for kind in ('spectral', 'additive'):
                for support in ('head_specific', 'joint'):
                    family = {}
                    for head, name in enumerate(HEADS):
                        record = next(r for r in maps if (r['layout'],r['selector'],r['head']) ==
                            (layout, kind, 'joint_f_g_eta_g_eta_f' if support == 'joint' else name))
                        anchor = next(r for r in priors if (r['layout'],r['selector'],r['support'],r['head']) ==
                            (layout, kind, support, name))
                        features = np.asarray(record['source_features'])
                        coefficients, estimation = coefficient_fits(features, tasks, head, anchor['response_units'], spec['ridge'])
                        parts = centered_fit_covariance_parts(coefficients, estimation)
                        design = np.column_stack([np.ones(48), features])
                        labels = np.concatenate([np.asarray(t['response'])[:,head] for t in tasks])
                        noise = np.concatenate([np.asarray(t['noise'])[:,head] for t in tasks])
                        variants = {}
                        for treatment in ('observed_task', 'deconvolved_task'):
                            task = parts['observed'] if treatment == 'observed_task' else parts['deconvolved']
                            mean, covariance, residual = population_law(design, np.asarray(anchor['mean']),
                                np.asarray(anchor['covariance']), task, anchor['discrepancy_variance'])
                            beta, H, nll = common_replay(np.asarray(anchor['mean']), np.asarray(anchor['covariance']),
                                design, residual, labels, noise)
                            if kind == 'spectral' and support == 'head_specific' and head == 0:
                                joint_design = block_diag(*np.split(design, 4))
                                joint_beta_covariance = (np.kron(np.ones((4, 4)), np.asarray(anchor['covariance']))
                                    + np.kron(np.eye(4), task))
                                dense = joint_design @ joint_beta_covariance @ joint_design.T
                                dense += anchor['discrepancy_variance']*np.eye(48)
                                np.testing.assert_allclose(dense, covariance, atol=1e-12, rtol=1e-10)
                                factor = cho_factor(dense+np.diag(noise), lower=True)
                                cross = np.asarray(anchor['covariance']) @ design.T
                                np.testing.assert_allclose(H, np.asarray(anchor['covariance'])-cross@cho_solve(factor,cross.T),
                                    atol=1e-12,rtol=1e-10)
                                checks.append({'layout':layout,'treatment':treatment,
                                    'independent_common_task_block_covariance_error':float(np.max(abs(dense-covariance))),
                                    'new_task_inherits_common_uncertainty_and_an_independent_task_residual': True,'passed':True})
                            variants[treatment] = {'source_mean': mean, 'source_covariance': covariance}
                            laws.append({'layout': layout, 'selector': kind, 'support': support, 'head': name,
                                'treatment': treatment, 'task_names': selection['source_tasks'], 'source_family': 'InventorySupplyChain',
                                'unconditioned_common_mean': anchor['mean'], 'unconditioned_common_covariance': anchor['covariance'],
                                'task_covariance': task.tolist(), 'point_discrepancy_variance': anchor['discrepancy_variance'],
                                'source_design': design.tolist(), 'source_mean': mean.tolist(), 'source_covariance': covariance.tolist(),
                                'Gaussian_source_posterior_common_mean': beta.tolist(), 'Gaussian_source_posterior_common_covariance': H.tolist(),
                                'Gaussian_new_task_coefficient_covariance': (H+task).tolist(),
                                'Gaussian_source_NLL_per_label': nll})
                        family[name] = variants
                        eigen = np.linalg.eigvalsh(parts['deconvolved'])
                        rank = int(np.count_nonzero(eigen > max(float(eigen.max())*1e-10, 1e-14)))
                        assert rank <= 3
                        assert np.trace(parts['deconvolved']) <= np.trace(parts['observed'])+1e-9
                        rows.append({'layout': layout, 'selector': kind, 'support': support, 'head': name,
                            'observed_task_trace': float(np.trace(parts['observed'])), 'fit_noise_trace': float(np.trace(parts['noise'])),
                            'deconvolved_task_trace': float(np.trace(parts['deconvolved'])), 'shared_mean_fit_noise_trace': float(np.trace(parts['shared_fit_covariance'])),
                            'removed_trace_fraction': float(1-np.trace(parts['deconvolved'])/np.trace(parts['observed'])),
                            'deconvolved_rank': rank, 'negative_eigenvalues_before_projection': np.linalg.eigvalsh(parts['observed']-parts['noise']).tolist(),
                            'source_coefficients': coefficients.tolist(), 'coefficient_estimation_covariances': estimation.tolist()})
                    for treatment in ('observed_task', 'deconvolved_task'):
                        f, eta = family['f'][treatment], family['eta_f'][treatment]
                        model = CollapsedRawSourcePosterior(f['source_mean'], f['source_covariance'], source['batch_means'], source['sample_variances'])
                        # Prior populations are unconditioned. The Gaussian
                        # source posterior above is NEVER fed to the raw update.
                        logs = np.array([model.evaluate(e)[0] for e in draws])-old_logs
                        logs += gaussian_rows_log_density(draws, eta['source_mean'], eta['source_covariance'])-old_prior_logs
                        row = {'layout': layout, 'selector': kind, 'support': support, 'treatment': treatment,
                            **importance_summary(logs, spec['precision_gates'])}
                        raw_rows.append(row)
                        print('raw covariance reference',layout,kind,support,treatment,'ESS',row['weight_ESS'],
                            'eligible',row['precision_gate_passed'],flush=True)
                    completed += 1
                    save(out/'coefficient_parts.json', {'execution': spec, 'rows': rows})
                    save(out/'source_laws.json', {'execution': spec, 'rows': laws})
                    save(out/'raw_importance.json', {'execution': spec, 'rows': raw_rows})
    except TimeoutError:
        signal.setitimer(signal.ITIMER_PROF, 0.)
        save(out/'stopped_partial.json', {'coefficient_rows': rows, 'law_rows': laws, 'raw_rows': raw_rows})
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.)
        signal.signal(signal.SIGPROF, handler)
    save(out/'results.json', {'execution': spec, 'coefficient_heads': len(rows), 'source_laws': len(laws),
        'complete_map_families': completed, 'all_registered_cases_complete': completed==8,
        'raw_importance_cases': len(raw_rows), 'eligible_raw_importance_cases': sum(r['precision_gate_passed'] for r in raw_rows),
        'checks': checks, 'cpu_seconds': time.process_time()-started,
        'cost': {'new_selector_fits': 0, 'source_coefficient_fits': 4*len(rows), 'Gaussian_source_conditionings': len(laws),
            'cached_raw_source_likelihood_evaluations': 1024+512*len(raw_rows), 'new_source_chains': 0,
            'new_simulator_calls': 0, 'new_verifier_calls': 0, 'new_target_predictions': 0},
        'no_model_adopted': True, 'independent_confirmation': False,
        'limitations': 'Four related Inventory tasks,not four business families. Frozen source maps and previous finite coefficient regularizers remain empirical-Bayes estimates. Ridge-fit noise subtraction concerns fitted coefficients and does not remove ridge bias. Source Gaussian label conditioning is a reference,including the eta-label approximation. Raw importance uses f/eta_f ordinary batches once,not an additional Gaussian eta-label likelihood;proposal failures do not rank laws or supply a new posterior. No g/eta_g raw-source proposal or full joint-four-head posterior is claimed.'})


if __name__ == '__main__':
    run()
