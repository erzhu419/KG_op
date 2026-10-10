"""Bounded MCEM of the frozen four-parameter source learning subproblem."""

import copy
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_collapsed_source_raw_update import sample
from performance.check_legacy_f_learned_covariance_source_update import retained_mixture, verify_learned
from performance.check_legacy_f_level_risk_source_update import variance_reference
from performance.check_legacy_f_original_source_raw_update import common_moments
from performance.check_legacy_f_raw_level_risk_mstep import risk_system, expected, source_moments
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_raw_source_level_risk_score import source_systems
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_bounded_level_risk_learning_plan_v1.json'


def project(theta, gradient):
    result = np.asarray(gradient).copy()
    for j in (1, 3):
        if theta[j] == 0:
            result[..., j] = np.minimum(result[..., j], 0.)
    return result


def make_systems(spec):
    step = read(spec['Mstep_plan']); level = read(step['level_risk_plan'])
    initial_fits = read(spec['initial_Mstep_results'])['rows']
    resumed = read(spec['resume_accepted_results'])['final_parameters'] if 'resume_accepted_results' in spec else []
    initial_laws = read(spec['initial_source_laws'])['rows']; result = []
    for field in read(step['source_inputs'])['rows']:
        systems, _, risk = source_systems(level, field)
        rb, strong, fixed, tail = risk_system(field, risk, step['risk_original_shape_tail_budget'])
        for index, (full_builder, old_f, _) in enumerate(systems):
            key = (field['layout'], field['arms'][index])
            def fb(theta, build=full_builder, fixed_common=old_f[1]):
                m, k, ds = build(np.asarray([theta[0], fixed_common, theta[1]])); return m, k, [ds[1]]
            sd = np.sqrt(np.mean(np.diag(field['f_covariances'][index])))
            theta = next((r['parameters'] for r in resumed if (r['layout'], r['arm']) == key),
                next(r['learned_parameters'] for r in initial_fits if (r['layout'], r['arm']) == key))
            result.append({'key': key, 'field': field, 'index': index, 'f_builder': fb, 'risk_builder': rb,
                'theta': np.asarray(theta),
                'scale': np.asarray([sd, 1. if index == 0 else sd ** 2, 1., 1.]),
                'law': next(r for r in initial_laws if (r['layout'], r['arm']) == key),
                'risk': risk, 'risk_strong': strong, 'risk_fixed': fixed, 'risk_tail': tail})
    return result


def objective(system, theta, moments, half=0):
    fm, rm = moments
    f, fg = expected(theta[:2], system['f_builder'], *fm[half])
    r, rg = expected(theta[2:], system['risk_builder'], *rm[half])
    return f + r, np.r_[fg, rg], np.asarray([f, r])


def materialize(system, theta):
    field = copy.deepcopy(system['field']); index = system['index']; layout, arm = system['key']
    m, k, _ = system['f_builder'](theta[:2]); em, ek, _ = system['risk_builder'](theta[2:])
    shift = theta[0] - field['f_means'][index][0]
    common = np.asarray(field['common_latent_prior_means'][index])
    if index == 0:
        common[0] += shift
    else:
        common += shift
    field['common_latent_prior_means'][index] = common.tolist()
    field['f_means'][index] = m.tolist(); field['f_covariances'][index] = k.tolist()
    field.update(active_arm=arm, active_arm_index=index, eta_mean=em.tolist(), eta_covariance=ek.tolist())
    law = copy.deepcopy(system['law']); law.update(source_f_population_covariance=k.tolist(),
        learned_f_center=float(theta[0]), level_risk_Mstep_parameters=theta.tolist(), no_new_posterior_attached=True)
    if index == 0:
        # Undo Stage213's task-root scaling,then apply the current root to Stage205's fixed shape law.
        initial = law['task_intercept_root_multiplier']; b = np.eye(len(law['task_coefficient_covariance']))
        b[0, 0] = theta[1] / initial
        r = b @ np.asarray(law['task_coefficient_covariance']) @ b
        law.update(task_coefficient_covariance=r.tolist(), task_intercept_root_multiplier=float(theta[1]))
    else:
        # The GP task constant stayed at zero in Stage213 and has a positive boundary score.
        assert theta[1] == 0 and law['learned_parameters'][2] == 0
    risk = copy.deepcopy(system['risk']); a = system['risk_fixed'] + theta[3] * system['risk_strong']
    mean = np.asarray(risk['prior_mean']); mean[0] += theta[2]; d = np.asarray(risk['source_design'])
    cross = a @ d.T; gain = cho_solve(cho_factor(ek, lower=True), cross.T).T
    conditional = a - gain @ cross.T
    risk.update(arm=arm, prior_mean=mean.tolist(), common_prior_covariance=a.tolist(),
        source_eta_prior_mean=em.tolist(), source_eta_prior_covariance=ek.tolist(), source_eta_gain=gain.tolist(),
        conditional_covariance=(.5 * (conditional + conditional.T)).tolist(),
        common_strong_shape_covariance_multiplier=float(theta[3]), no_new_posterior_attached=True)
    return field, law, risk


def run(plan=PLAN):
    started = time.process_time(); spec = read(plan); systems = make_systems(spec)
    initial_source = read(spec['initial_source_results']); assert initial_source['all_monte_carlo_gates_passed'] and initial_source['all_mixture_retention_gates_passed']
    posterior = initial_source['rows']; partitions = read(spec['initial_source_partitions'])['rows']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    rounds = []; stop_reason = 'additional_Mstep_limit'; total_fits = total_chains = total_sweeps = 0

    def current_scores():
        scores = []
        for system in systems:
            key = system['key']; post = next(r for r in posterior if (r['layout'], r['arm']) == key)
            part = next(r for r in partitions if (r['layout'], r['arm']) == key)
            moments = source_moments(post, part)
            raw = np.asarray([objective(system, system['theta'], moments, h)[1] for h in range(3)])
            scaled = project(system['theta'], raw) * system['scale']
            scores.append({'layout': key[0], 'arm': key[1], 'raw_scores_full_first_second': raw.tolist(),
                'scaled_projected_scores_full_first_second': scaled.tolist(), 'maximum_absolute_scaled_score': float(np.abs(scaled).max())})
        return scores

    try:
        for iteration in range(1, spec['maximum_additional_Msteps'] + 1):
            round_number = iteration + spec.get('accepted_round_number_offset', 0)
            scores = current_scores(); tolerance = spec['scaled_projected_source_score_tolerance']
            if all(r['maximum_absolute_scaled_score'] <= tolerance for r in scores):
                stop_reason = 'approximate_subproblem_stationarity'; break
            stable = True
            for row in scores:
                g = np.asarray(row['scaled_projected_scores_full_first_second']); active = np.abs(g).max(axis=0) > tolerance
                stable &= bool(np.all(((g[0] * g[1] > 0) & (g[0] * g[2] > 0))[active]))
            if not stable:
                stop_reason = 'source_direction_split_sign_uncertainty'; break
            candidates, fits = [], []
            for system in systems:
                key = system['key']; post = next(r for r in posterior if (r['layout'], r['arm']) == key)
                part = next(r for r in partitions if (r['layout'], r['arm']) == key)
                moments = source_moments(post, part); old = system['theta']; scale = system['scale']
                def q(z):
                    if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                        raise TimeoutError('registered CPU budget reached')
                    value, gradient, _ = objective(system, z * scale, moments); return value, gradient * scale
                opt = spec['optimizer']
                result = minimize(q, old / scale, method=opt['method'], jac=True,
                    bounds=[(None, None), (0., None), (None, None), (0., None)],
                    options={'maxiter': opt['max_iterations'], 'ftol': opt['ftol'], 'gtol': opt['gtol']})
                theta = result.x * scale; total_fits += 1
                old_q = [objective(system, old, moments, h)[2] for h in range(3)]
                new_q = [objective(system, theta, moments, h)[2] for h in range(3)]
                improvements = np.asarray(old_q) - new_q
                passed = bool(result.success and np.all(improvements.sum(axis=1) > 0) and improvements.min() >= -1e-10)
                fits.append({'layout': key[0], 'arm': key[1], 'old_parameters': old.tolist(), 'candidate_parameters': theta.tolist(),
                    'optimizer_converged': bool(result.success), 'optimizer_iterations': int(result.nit),
                    'optimizer_message': str(result.message), 'Q_component_improvements_full_first_second': improvements.tolist(), 'Mstep_gate_passed': passed})
                candidates.append(theta)
            entry = {'round': round_number, 'source_scores_before_Mstep': scores, 'fits': fits, 'round_accepted': False}
            rounds.append(entry); save(out / 'rounds.json', {'execution': spec, 'rounds': rounds})
            if not all(r['Mstep_gate_passed'] for r in fits):
                stop_reason = 'Mstep_optimizer_or_component_Q_failure'; break
            sample_spec = copy.deepcopy(spec); sample_spec['sampler']['seed'] += round_number * spec['sampler']['round_seed_stride']
            new_post, new_parts, mixtures, fields, laws, risks, checks = [], [], [], [], [], [], []
            for system, theta in zip(systems, candidates):
                field, law, risk = materialize(system, theta); index = system['index']
                reference = variance_reference(field); checks.append(verify_learned(field, index, reference))
                model = VarianceReferencePosterior(field['f_means'][index], field['f_covariances'][index], field['batch_means'], field['sample_variances'])
                retained = []; row = sample(field, index, sample_spec, started, model, reference['reference_mean'], reference['reference_covariance'], retained)
                total_chains += 4; total_sweeps += 6144
                cm, cc = common_moments(field, index, np.asarray(row['posterior_mean']), np.asarray(row['posterior_covariance']))
                row.update(common_posterior_mean=cm.tolist(), common_posterior_covariance=cc.tolist())
                states = retained[0]; mixture = retained_mixture(field, index, model, states, row, spec)
                selected = states[:, ::8].reshape(-1, 48); diagonal = np.diag(row['eta_posterior_covariance'])
                mean_error = float(np.sqrt(np.mean((selected.mean(axis=0) - row['eta_posterior_mean']) ** 2 / diagonal)))
                variance_error = float(np.max(np.abs(selected.var(axis=0, ddof=1) / diagonal - 1)))
                mixture.update(source_eta_mean_rms_difference_in_full_posterior_sd=mean_error,
                    maximum_source_eta_variance_relative_difference=variance_error)
                mixture['mixture_gate_passed'] &= bool(mean_error <= .1 and variance_error <= .15)
                part = {'layout': field['layout'], 'arm': field['active_arm']}
                for name, block in [('first_half', states[:, :512]), ('second_half', states[:, 512:])]:
                    flat = block.reshape(-1, 48); part[name + '_eta_mean'] = flat.mean(axis=0).tolist()
                    part[name + '_eta_covariance'] = np.cov(flat, rowvar=False).tolist()
                new_post.append(row); new_parts.append(part); mixtures.append(mixture); fields.append(field); laws.append(law); risks.append(risk)
            save(out / f'round{round_number}_source_moments.json', {'execution': sample_spec, 'rows': new_post,
                'partitions': new_parts, 'verification_checks': checks})
            entry['posterior_gate_results'] = [{k: r[k] for k in ('layout', 'arm', 'monte_carlo_gate_passed',
                'maximum_split_rhat', 'minimum_autocorrelation_ess')} for r in new_post]
            entry['mixture_gate_results'] = [{k: r[k] for k in ('layout', 'arm', 'mixture_gate_passed')} for r in mixtures]
            if not all(r['monte_carlo_gate_passed'] for r in new_post) or not all(r['mixture_gate_passed'] for r in mixtures):
                stop_reason = 'fresh_posterior_or_retention_precision_failure'; break
            for system, theta in zip(systems, candidates):
                system['theta'] = theta
            posterior, partitions = new_post, new_parts; entry['round_accepted'] = True
            for filename, rows in [('source_laws.json', laws), ('risk_reconstruction.json', risks), ('frozen_inputs.json', fields),
                ('mixtures.json', mixtures), ('source_results.json', posterior), ('source_eta_partitions.json', partitions)]:
                document = {'execution': sample_spec, 'rows': rows, 'accepted_source_round': round_number,
                    'all_source_and_retention_gates_passed': True}
                if filename == 'source_results.json':
                    document.update(all_four_source_updates_complete=True, all_monte_carlo_gates_passed=True,
                        all_mixture_retention_gates_passed=True)
                elif filename == 'mixtures.json':
                    document.update(all_mixture_retention_gates_passed=True, all_source_posterior_precision_gates_passed=True)
                save(out / filename, document)
            save(out / 'rounds.json', {'execution': spec, 'rounds': rounds})
            print('bounded source round', round_number, 'accepted', 'parameters', [s['theta'].tolist() for s in systems], flush=True)
    except TimeoutError:
        stop_reason = 'registered_CPU_budget'; save(out / 'rounds.json', {'execution': spec, 'rounds': rounds})
    final_scores = current_scores(); stationary = all(r['maximum_absolute_scaled_score'] <= spec['scaled_projected_source_score_tolerance'] for r in final_scores)
    save(out / 'rounds.json', {'execution': spec, 'rounds': rounds})
    save(out / 'results.json', {'execution': spec, 'rounds': rounds, 'stop_reason': stop_reason,
        'accepted_additional_source_rounds': sum(r['round_accepted'] for r in rounds), 'final_source_scores': final_scores,
        'approximately_stationary_four_parameter_subproblem': stationary,
        'final_parameters': [{'layout': s['key'][0], 'arm': s['key'][1], 'parameters': s['theta'].tolist()} for s in systems],
        'cpu_seconds': time.process_time() - started, 'cost': {**spec['cost'], 'actual_joint_Mstep_optimizations': total_fits,
            'actual_new_source_chains': total_chains, 'actual_source_sweeps': total_sweeps},
        'task0_outcomes_indexed': False, 'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Bounded source-only MCEM in the unchanged four-parameter subproblem. Approximate stationarity does not cover function shape/support hyperparameters,hyperparameter uncertainty,independent predictive performance or decision/KG advantage. Stop failures and intermediate moments retained.'})
    print('bounded source learning stop', stop_reason, 'accepted', sum(r['round_accepted'] for r in rounds),
        'stationary', stationary, 'final maximum scaled scores', [r['maximum_absolute_scaled_score'] for r in final_scores], flush=True)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else PLAN)
