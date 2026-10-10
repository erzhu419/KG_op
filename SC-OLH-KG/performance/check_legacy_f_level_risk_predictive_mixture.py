"""Source-development prediction after learning function and risk levels."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import digamma, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_raw_source_predictive_mixture import source_projection, split_prediction_checks, chain_half_indices
from performance.check_legacy_f_reference_training_budget import prediction
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from performance.vectorized_joint_batch_density import component_batch_log_evidence

PLAN = 'performance/manifests/legacy_joint_population_f_level_risk_predictive_mixture_plan_v1.json'


def run():
    started = time.process_time(); spec = read(PLAN); source = read(spec['source_update_results'])
    assert source['all_monte_carlo_gates_passed'] and source['all_mixture_retention_gates_passed']
    old_plan = read(spec['previous_prediction_plan']); mean_spec = read(old_plan['source_mean_plan'])
    coverage, _ = source_paths(read(mean_spec['source_context_execution']))
    controls = read(old_plan['retained_control_results']); packages = controls['execution']['source_packages']
    branches = sum([read(Path(p) / 'source_branches.json')['rows'] for p in packages], [])
    registry = read(mean_spec['source_registry'])['rows']; original_laws = read(mean_spec['source_laws'])['rows']
    known = read(read(old_plan['source_contract_plan'])['known_source_plan']); risk_execution = read(known['risk_source_execution'])
    fields = read(spec['source_inputs'])['rows']; laws = read(spec['source_laws'])['rows']
    risks = read(spec['risk_reconstruction'])['rows']; mixtures = read(spec['eta_components'])['rows']
    native = read(spec['native_cached_results']); assert all(r['passed'] for r in native['native_refinement_checks'])
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True); work, frozen = [], []
    query = spec['source_validation_task']; query_name = f'related_inventory_{query:02d}'
    for field in fields:
        layout = field['layout']; arm = field['active_arm']; index = field['active_arm_index']; key = (layout, arm)
        branch = next(r for r in branches if (r['layout'], r['query_task']) == (layout, query))
        assert branch['source_task_names'] == [f'related_inventory_{i:02d}' for i in (2, 3, 4, 5)]
        sites = coverage['layouts'][layout][query]; assert len(sites) == 12
        case = next(c for c in risk_execution['cases'] if (c['layout'], c['query_task']) == (layout, query))
        ri = read(Path(risk_execution['output']) / case['folder'] / 'risk_inputs.json')
        risk = next(r for r in risks if (r['layout'], r['arm']) == key)
        np.testing.assert_allclose(risk['source_eta_prior_mean'], field['eta_mean'], atol=0, rtol=0)
        np.testing.assert_allclose(risk['source_eta_prior_covariance'], field['eta_covariance'], atol=0, rtol=0)
        np.testing.assert_allclose(risk['public_reference_psi'], ri['public_reference_psi'], atol=1e-8, rtol=1e-8)
        law = next(r for r in laws if (r['layout'], r['arm']) == key)
        if index == 0:
            original = next(r for r in original_laws if (r['layout'], r['query_task']) == (layout, query))
            law = {**law, 'white_discrepancy': original['white_discrepancy']}
        gp = read(next(r['source_fit'] for r in registry if (r['layout'], r['inner_query_task']) ==
            (layout, query_name)))['fits']['gp']
        component = next(r for r in mixtures if (r['layout'], r['arm']) == key)
        state = source_projection(field, index, law, risk, np.asarray(branch['query_mean_psi']),
            np.asarray(ri['heldout_public_psi'])[sites], branch['public_mean_reference'], component, gp)
        row = {'layout': layout, 'arm': arm, 'sites': sites, 'source_validation_task': query,
            'prediction': prediction(state), 'split_prediction_errors': split_prediction_checks(state)}
        work.append((row, state))
        frozen.append({**row, 'component_means': state['component_means'].tolist(),
            'component_local_covariances': state['component_local_covariances'].tolist(), 'log_weights': state['log_weights'].tolist()})
    save(out / 'predictions.json', {'execution': spec, 'rows': frozen, 'all_four_forecasts_frozen_before_scores': True,
        'paid_target_observations': 0, 'positive_joint_mixture_not_replaced_by_Gaussian': True})
    rows = []; limits = spec['precision_gates']
    for row, state in work:
        inputs = read(Path(coverage['output']) / f"layout{row['layout']}" / 'task1' / 'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == query_name)
        labels = np.asarray(task['response'])
        np.testing.assert_allclose(np.exp(labels[:, 3] + digamma(1.)) / 3,
            np.asarray(task['noise'])[:, 0], atol=1e-12, rtol=1e-10)
        densities, splits, errors = [], [], []
        for j, label in enumerate(labels):
            if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_forecasts_scored': False,
                    'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started})
                return
            logs, error = component_batch_log_evidence(state['component_means'][:, j],
                state['component_local_covariances'][:, j], label[0], label[3], 3)
            full = float(logsumexp(state['log_weights'] + logs)); densities.append(full); errors.append(error)
            halves = [float(logsumexp(logs[h]) - np.log(len(h))) for h in chain_half_indices(len(logs))]
            splits.append(halves)
        maximum_density_error = float(np.max(np.abs(np.asarray(splits) - np.asarray(densities)[:, None])))
        residual = np.asarray(row['prediction']['mean']) - labels[:, 0]
        numerical = (maximum_density_error <= limits['maximum_split_raw_density_absolute_difference']
            and all(e['mean_rms_difference_in_full_sd'] <= limits['maximum_split_prediction_mean_rms_difference_in_full_sd']
                and e['maximum_variance_relative_difference'] <= limits['maximum_split_prediction_variance_relative_difference']
                for e in row['split_prediction_errors']))
        row.update(scores={'f_mse': float(np.mean(residual ** 2)), 'f_raw_batch_density': float(np.mean(densities))},
            per_site_raw_batch_densities=densities, per_site_half_raw_batch_densities=splits,
            maximum_density_integration_relative_error_bound=max(errors),
            maximum_split_raw_density_absolute_difference=maximum_density_error, mixture_precision_gate_passed=bool(numerical))
        rows.append(row)
        print('learned level/risk zero-shot', row['layout'], row['arm'], 'numerical pass', numerical,
            'density half error', maximum_density_error, 'scores', row['scores'], flush=True)
    comparisons = []
    for layout in spec['layouts']:
        structure, gp = [next(r for r in rows if (r['layout'], r['arm']) == (layout, arm))
            for arm in ('joint_spectral_geometry', 'source_gp')]
        eligible = structure['mixture_precision_gate_passed'] and gp['mixture_precision_gate_passed']
        delta = {k: structure['scores'][k] - gp['scores'][k] for k in spec['metrics']}
        historical = next(r for r in native['rows'] if (r['layout'], r['arm'], r.get('resolution')) == (layout, 'native_mixture', 'selected'))
        comparisons.append({'layout': layout, 'primary_comparison_numerically_eligible': eligible,
            'structure_minus_learned_GP': delta, 'mse_relative_change': structure['scores']['f_mse'] / gp['scores']['f_mse'] - 1,
            'source_structure_predictive_gate_passed': bool(delta['f_mse'] <= 0 and delta['f_raw_batch_density'] > 0) if eligible else None,
            'cached_native_scores': historical['scores'], 'native_information_contract_matches_raw_arms': False})
    numerical = all(r['mixture_precision_gate_passed'] for r in rows)
    predictive = all(r['source_structure_predictive_gate_passed'] for r in comparisons) if numerical else None
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'comparisons': comparisons,
        'all_four_forecasts_scored': True, 'all_numerical_gates_passed': numerical,
        'source_structure_predictive_gate_passed': predictive, 'cpu_seconds': time.process_time() - started,
        'cost': spec['cost'], 'native_reference_recomputed': False, 'no_model_adopted': True,
        'independent_confirmation': False, 'outer_task1_outcomes_indexed': False,
        'interpretation': 'Zero-shot task0 development forecast under the source-only learned level/risk law. Both matched arms use the same original raw-source information and training step. Numerical failures retained;native historical reference has a different source contract. No decision/KG or independent confirmation claim.'})
    print('source-development comparison', comparisons, flush=True)


if __name__ == '__main__':
    run()
