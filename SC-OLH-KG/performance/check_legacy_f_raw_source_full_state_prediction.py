"""Restore the original retained chains and diagnose predictive functionals."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_collapsed_source_raw_update import sample
from performance.check_legacy_f_raw_source_predictive_mixture import chain_half_indices, source_projection, split_prediction_checks
from performance.check_legacy_f_reference_training_budget import prediction
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.run_legacy_related_inventory_source import save
from performance.vectorized_joint_batch_density import component_batch_log_evidence

PLAN = 'performance/manifests/legacy_joint_population_f_raw_source_full_state_prediction_plan_v1.json'


def run():
    started = time.process_time(); spec = read(PLAN); pilot = read(spec['failed_pilot_results'])
    assert pilot['all_eight_zero_shot_predictions_scored'] and not pilot['all_numerical_gates_passed']
    pspec = pilot['execution']; original = read(spec['original_source_execution']); old = read(spec['original_source_results'])
    fields = read(spec['frozen_source_inputs'])['rows']; references = read(original['references'])['rows']
    thin = read(Path(original['output']) / 'mixtures.json')['rows']
    frozen = read(spec['frozen_pilot_predictions'])['rows']; retained, replay_checks = {}, []
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    for field in fields:
        reference = next(r for r in references if r['layout'] == field['layout'])
        for index, arm in enumerate(field['arms']):
            model = VarianceReferencePosterior(field['f_means'][index], field['f_covariances'][index], field['batch_means'], field['sample_variances'])
            captured = []
            row = sample(field, index, original, started, model, reference['reference_mean'], reference['reference_covariance'], captured)
            prior = next(r for r in old['rows'] if (r['layout'], r['arm']) == (field['layout'], arm))
            for key in ('posterior_mean', 'posterior_covariance', 'eta_posterior_mean', 'eta_posterior_covariance'):
                np.testing.assert_allclose(row[key], prior[key], atol=1e-10, rtol=0)
            previous_thin = next(r for r in thin if (r['layout'], r['arm']) == (field['layout'], arm))
            np.testing.assert_allclose(captured[0][:, ::8], previous_thin['eta_states_by_chain'], atol=1e-10, rtol=0)
            assert row['monte_carlo_gate_passed']
            retained[(field['layout'], arm)] = captured[0]
            replay_checks.append({'layout': field['layout'], 'arm': arm, 'original_moments_and512_states_reproduced': True,
                'original4096_retained_states_recovered': True, 'additional_independent_chain': False})
    save(out / 'replay.json', {'execution': spec, 'checks': replay_checks, 'all_original_replays_matched': True})
    risks = read(pspec['reconstruction'])['rows']; laws = read(pspec['source_laws'])['rows']
    mean_spec = read(pspec['source_mean_plan']); registry = read(mean_spec['source_registry'])['rows']; originals = read(mean_spec['source_laws'])['rows']
    coverage, _ = source_paths(read(mean_spec['source_context_execution']))
    packages = read(pspec['retained_control_results'])['execution']['source_packages']
    branches = sum([read(Path(p) / 'source_branches.json')['rows'] for p in packages], [])
    known = read(read(pspec['source_contract_plan'])['known_source_plan']); execution = read(known['risk_source_execution'])
    work = []
    for field in fields:
        layout = field['layout']; branch = next(b for b in branches if (b['layout'], b['query_task']) == (layout, 0))
        risk = next(r for r in risks if r['layout'] == layout); sites = coverage['layouts'][layout][0]
        case = next(c for c in execution['cases'] if (c['layout'], c['query_task']) == (layout, 0))
        risk_inputs = read(Path(execution['output']) / case['folder'] / 'risk_inputs.json')
        psi = np.asarray(branch['query_mean_psi']); risk_psi = np.asarray(risk_inputs['heldout_public_psi'])[sites]
        gp = read(next(r['source_fit'] for r in registry if (r['layout'], r['inner_query_task']) == (layout, 'related_inventory_00')))['fits']['gp']
        for index, arm in enumerate(field['arms']):
            law = next(r for r in laws if (r['layout'], r['arm']) == (layout, arm))
            if index == 0: law = {**law, 'white_discrepancy': next(r['white_discrepancy'] for r in originals if (r['layout'], r['query_task']) == (layout, 0))}
            state = source_projection(field, index, law, risk, psi, risk_psi, branch['public_mean_reference'],
                {'eta_states_by_chain': retained[(layout, arm)]}, gp)
            selected = np.concatenate([np.arange(1024 * c, 1024 * (c + 1), 8) for c in range(4)])
            previous = next(r for r in frozen if (r['layout'], r['arm']) == (layout, arm))
            for key in ('component_means', 'component_local_covariances'):
                np.testing.assert_allclose(state[key][selected], previous[key], atol=1e-10, rtol=0)
            row = {'layout': layout, 'arm': arm, 'source_validation_task': 0, 'sites': sites,
                'original_component_projection_reproduced': True, 'prediction': prediction(state),
                'split_prediction_errors': split_prediction_checks(state)}
            work.append((row, state))
    save(out / 'predictions.json', {'execution': spec, 'rows': [r for r, _ in work], 'all_four_predictions_frozen_before_scores': True,
        'components_per_arm': 4096, 'joint_mixture_not_replaced_by_Gaussian': True})
    rows, functional_records = [], []
    try:
        for row, state in work:
            inputs = read(Path(coverage['output']) / f"layout{row['layout']}" / 'task1' / 'inputs.json')
            labels = np.asarray(next(t['response'] for t in inputs['source_tasks'] if t['name'] == 'related_inventory_00'))
            full_logs, density_values, split_errors, errors = [], [], [], []
            for j, label in enumerate(labels):
                if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']: raise TimeoutError('registered CPU budget reached')
                logs, error = component_batch_log_evidence(state['component_means'][:, j], state['component_local_covariances'][:, j], label[0], label[3], 3)
                full = float(logsumexp(logs) - np.log(len(logs))); density_values.append(full); full_logs.append(logs); errors.append(error)
                split_errors.extend(abs(float(logsumexp(logs[k]) - np.log(len(k))) - full) for k in chain_half_indices(len(logs)))
            log_values = np.stack(full_logs, axis=1).reshape(4, 1024, 12)
            shifts = log_values.max(axis=(0, 1)); functional = diagnostics(np.exp(log_values - shifts))
            maximum_rhat = max(functional['split_rhat']); minimum_ess = min(functional['autocorrelation_ess'])
            passed = (maximum_rhat <= 1.05 and minimum_ess >= 100 and max(split_errors) <= .02
                and all(e['mean_rms_difference_in_full_sd'] <= .1 and e['maximum_variance_relative_difference'] <= .15 for e in row['split_prediction_errors']))
            residual = np.asarray(row['prediction']['mean']) - labels[:, 0]
            row.update(scores={'f_mse': float(np.mean(residual ** 2)), 'f_raw_batch_density': float(np.mean(density_values))},
                per_site_raw_batch_density=density_values, maximum_split_raw_density_absolute_difference=max(split_errors),
                maximum_density_integration_relative_error_bound=max(errors), predictive_functional_diagnostics=functional,
                maximum_predictive_functional_rhat=maximum_rhat, minimum_predictive_functional_ess=minimum_ess,
                all_future_functional_gates_passed=bool(passed))
            rows.append(row); functional_records.append({'layout': row['layout'], 'arm': row['arm'],
                'conditional_future_batch_log_densities_by_chain': log_values.tolist(), 'density_numerical_shifts': shifts.tolist()})
            print('full ORIGINAL state prediction', row['layout'], row['arm'], 'gate', passed, 'ESS', minimum_ess,
                'Rhat', maximum_rhat, 'split density', max(split_errors), 'scores', row['scores'], flush=True)
            save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False, 'cpu_seconds': time.process_time() - started})
    except TimeoutError:
        save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
            'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started})
        save(out / 'functional_records.json', {'execution': spec, 'rows': functional_records}); return
    numerical = all(r['all_future_functional_gates_passed'] for r in rows); comparisons, gates = [], []
    for layout in ('5', '6'):
        structure = next(r for r in rows if r['layout'] == layout and r['arm'] == 'joint_spectral_geometry')
        gp = next(r for r in rows if r['layout'] == layout and r['arm'] == 'source_gp')
        native = next(r for r in pilot['rows'] if r['layout'] == layout and r['arm'] == 'native_mixture' and r['resolution'] == 'selected')
        for arm, reference in [('source_gp', gp), ('native_mixture', native)]:
            delta = {k: structure['scores'][k] - reference['scores'][k] for k in ('f_mse', 'f_raw_batch_density')}
            comparisons.append({'layout': layout, 'reference': arm, 'structure_minus_reference': delta,
                'mse_relative_change': structure['scores']['f_mse'] / reference['scores']['f_mse'] - 1})
            if arm == 'source_gp': gates.append({'layout': layout, 'mse_nonincrease': delta['f_mse'] <= 0, 'raw_density_strict_improvement': delta['f_raw_batch_density'] > 0})
    predictive = all(g['mse_nonincrease'] and g['raw_density_strict_improvement'] for g in gates) if numerical else None
    save(out / 'functional_records.json', {'execution': spec, 'rows': functional_records})
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_full_state_predictions_complete': True,
        'all_original_source_replays_matched': True, 'all_future_functional_gates_passed': numerical,
        'source_structure_predictive_gate_passed': predictive, 'comparisons': comparisons, 'matched_GP_predictive_gates': gates,
        'diagnostic_interpreted': numerical, 'cpu_seconds': time.process_time() - started, 'cost': spec['cost'],
        'no_model_adopted': True, 'independent_confirmation': False, 'prior512_state_failures_retained': True,
        'interpretation': 'Same original source sampler/states and learned law;all4096 original retained states used for future batch functional diagnostics. No additional independent chain,data,paid update or native quadrature. Predictive functional precision remains separate from source moments and predictive advantage.'})


if __name__ == '__main__':
    run()
