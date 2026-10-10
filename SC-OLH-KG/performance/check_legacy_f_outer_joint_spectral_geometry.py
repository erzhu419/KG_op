"""Frozen source5/task1 test of the source4 joint spectral learning rule."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_exact_raw_filter import prepare_cases
from performance.check_legacy_f_reference_training_budget import compact, prediction, score
from performance.check_legacy_f_source_information_design import gaussian_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_joint_compression import source_laws
from performance.diagnose_legacy_f_inner_joint_spectral_compatibility import fit_spectrum
from performance.diagnose_legacy_f_level_shape_mean_error import decompose
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.fit_legacy_f_inner_joint_spectral_geometry import verify
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_joint_spectral_geometry import JointSpectralGeometryLikelihood
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_outer_joint_spectral_geometry_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); plan = read(spec['plan'])
    inner = read(plan['inner_source_results']); assert inner['all_ten_source_fits_complete'] and inner['gradient_check']['passed']
    assert read(plan['inner_raw_results'])['diagnostic_interpreted']
    cases, _, coverage = prepare_cases(spec); _, profile = source_paths(read(plan['source_context_execution']))
    controls = read(plan['control_results']); out = ROOT / plan['output']; out.mkdir(parents=True, exist_ok=True)
    base = {'execution': spec, 'cost': plan['cost'], 'no_model_adopted': True, 'training_diagnostic_only': True,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    laws, fields, frozen = [], [], []
    for case in cases:
        layout = case['identity']['layout']; inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        tasks = inputs['source_tasks']; names = [t['name'] for t in tasks]
        assert names == [f'related_inventory_{i:02d}' for i in (0, 2, 3, 4, 5)]
        assert sum(len(t['response']) for t in tasks) == 60
        head = read(Path(profile['output']) / f'layout{layout}' / 'task1' / 'selected_source_fit.json')['heads'][0]
        reference = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(reference)
        design = np.column_stack([np.ones(len(reference)), basis.transform(reference)])
        identity = case['identity']; prefix, sites = identity['paid_prefix'], identity['scoring_sites']; selection = prefix + sites
        assert prefix == plan['paid_prefix'] and len(sites) == 12 and not set(prefix).intersection(sites)
        templates, gp = source_laws(tasks, reference, reference[selection],
            {'fits': {'gp': head['joint_gp'], 'structural': head['joint_structural']}}, 0.)
        np.testing.assert_allclose(gp[0], case['prior']['f_mean'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(gp[1], case['prior']['f_covariance'], atol=1e-8, rtol=1e-8)
        old = next(r for r in templates if r['arm'] == 'joint_native_support')
        init_p, diagnostics = [], []
        for key, exponent_key in [('common_prior_coefficient_covariance', 'common_spectral_exponent'),
                                  ('task_coefficient_covariance', 'spectral_exponent')]:
            _, diagnostic = fit_spectrum(basis, np.asarray(old[key]), old['columns'], head['joint_structural'][exponent_key], design)
            init_p.append(diagnostic['fitted_exponent']); diagnostics.append(diagnostic)
        anchor = head['joint_gp']['anchor']; center = anchor['normalization']['mean'] + anchor['normalization']['scale'] * anchor['constant']
        model = JointSpectralGeometryLikelihood(basis, tasks, old['common_prior_coefficient_covariance'],
            old['task_coefficient_covariance'], old['columns'], center, old['white_discrepancy'], inner['execution']['positive_tail_fraction'])
        initial = np.asarray([1., init_p[0], 1., init_p[1]]); initial_value = model.evaluate(initial)[0]
        def objective(parameters):
            if time.process_time() - started >= plan['maximum_single_thread_cpu_seconds']: raise TimeoutError('outer pilot CPU cap')
            return model.evaluate(parameters)[:2]
        optimizer = plan['optimizer']
        try:
            fit = minimize(objective, initial, jac=True, method=optimizer['method'], bounds=[(0., None), (0., 4.)] * 2,
                options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
        except TimeoutError:
            save(out / 'results.json', {**base, 'source_rows': laws, 'diagnostic_interpreted': False,
                'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started}); return
        value, _, state = model.evaluate(fit.x)
        law = {'layout': layout, 'query_task': 1, 'source_task_names': names, 'columns': old['columns'],
            'parameters': fit.x.tolist(), 'initial_parameters': initial.tolist(), 'initial_nll_per_label': initial_value,
            'source_nll_per_label': value, 'nll_change_vs_gp_per_label': value - old['source_nll_per_label'] + old['source_total_nll_change_vs_gp'] / 60,
            'common_tail_per_coefficient': float(model.tails[0]), 'task_tail_per_coefficient': float(model.tails[1]),
            'white_discrepancy': old['white_discrepancy'], 'optimizer_converged': bool(fit.success),
            'optimizer_iterations': int(fit.nit), 'optimizer_message': str(fit.message),
            'initial_spectrum_diagnostics': diagnostics, **{k: v.tolist() for k, v in state.items()}}
        laws.append(law)
        if not fit.success:
            save(out / 'results.json', {**base, 'source_rows': laws, 'diagnostic_interpreted': False,
                'all_source_fits_complete': False, 'cpu_seconds': time.process_time() - started}); return
        law['verification'] = verify(model, fit.x, state, value)
        dq = design[selection]
        f = dq @ state['source_posterior_coefficient_mean'], dq @ (state['source_posterior_common_covariance'] + state['task_coefficient_covariance']) @ dq.T + law['white_discrepancy'] * np.eye(16)
        eta = case['prior']['eta_mean'], case['prior']['eta_covariance']
        row = {**identity, 'arm': 'joint_spectral_geometry', 'query_task': 1, 'episode_count': 4}
        fields.append({'row': row, 'field': gaussian_field(f, eta)})
        frozen.append({**row, 'source_task_names': names, 'parameters': law['parameters'], 'same_source5_eta': True})
        print('outer source law frozen', layout, law['parameters'], 'source NLL change vs GP', law['nll_change_vs_gp_per_label'], flush=True)
    assert len(laws) == len(fields) == 2
    save(out / 'source_laws.json', {**base, 'rows': laws, 'both_source5_laws_frozen_before_query_values': True})
    save(out / 'frozen_inputs.json', {'execution': spec, 'rows': frozen, 'both_sixteen_site_priors_frozen_before_paid_values': True})
    raw = read(coverage['source_data']); paid = [raw['tasks'][1]['Y_replicates'][j][0][0] for j in plan['paid_prefix']]; del raw
    rows = []
    for work in fields:
        row, field = work['row'], work.pop('field')
        evidence = sum(update_raw_observation(field, j, float(y)) for j, y in enumerate(paid))
        state = compact(field, list(range(4, 16)), evidence); filename = f"layout{row['layout']}_task1.npz"
        np.savez_compressed(out / filename, **state); row.update(prediction=prediction(state), component_file=filename); rows.append(row)
    save(out / 'predictions.json', {'execution': spec, 'rows': rows, 'both_forecasts_frozen_before_score_labels': True})
    for row in rows:
        if time.process_time() - started >= plan['maximum_single_thread_cpu_seconds']:
            save(out / 'results.json', {**base, 'source_rows': laws, 'rows': rows, 'diagnostic_interpreted': False,
                'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started}); return
        inputs = read(Path(coverage['output']) / f"layout{row['layout']}" / 'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == 'related_inventory_01')
        labels = {k: np.asarray(task[k]) for k in ('response', 'noise')}
        with np.load(out / row['component_file']) as state: row['scores'], row['density_integration_error_bound'] = score(row, state, labels)
        row['mean_error'] = decompose(row['prediction']['mean'], labels['response'][:, 0]); row['comparisons'] = {}
        for control in plan['controls']:
            old = next(r for r in controls['rows'] if r['layout'] == row['layout'] and r['design'] == row['design'] and r['arm'] == control)
            assert (old['paid_prefix'], old['scoring_sites']) == (row['paid_prefix'], row['scoring_sites'])
            row['comparisons'][control] = {'differences': {m: row['scores'][m] - old['scores'][m] for m in plan['metrics']},
                'mse_relative_change': row['scores']['f_mse'] / old['scores']['f_mse'] - 1}
    advance = all(c['differences']['f_mse'] <= 0 and c['differences']['f_raw_batch_density'] > 0
        and c['differences']['sequential_transcript_density'] > 0 for row in rows
        for name, c in row['comparisons'].items() if name in ('source_variance_mixture', 'source_gp'))
    save(out / 'results.json', {**base, 'source_rows': laws, 'rows': rows, 'diagnostic_interpreted': True,
        'all_source_fits_complete': True, 'matched_outer_development_advances': advance, 'candidate_stopped': not advance,
        'cpu_seconds': time.process_time() - started, 'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz'))})
    print('outer raw comparisons', [{k: r[k] for k in ('layout', 'comparisons', 'mean_error')} for r in rows], 'advance', advance, flush=True)


if __name__ == '__main__':
    run()
