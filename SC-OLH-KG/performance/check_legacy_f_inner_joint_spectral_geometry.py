"""Matched raw-prefix development test of the source-fitted spectral law."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, score
from performance.check_legacy_f_source_information_design import gaussian_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_level_shape_mean_error import decompose
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_joint_spectral_geometry_raw_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION)
    fitted = read(spec['source_results']); assert fitted['all_ten_source_fits_complete']
    laws = read(spec['source_laws']); assert laws['all_ten_source_laws_frozen_without_query_values']
    controls = read(spec['control_results']); assert controls['diagnostic_interpreted']
    branches = sum([read(Path(p) / 'source_branches.json')['rows'] for p in controls['execution']['source_packages']], [])
    coverage, _ = source_paths(read(spec['source_context_execution']))
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True); fields, frozen = [], []
    for law in laws['rows']:
        layout, query = law['layout'], law['query_task']
        assert law['optimizer_converged'] and law['verification']['independent_coefficient_conditioning_verified']
        branch = next(b for b in branches if b['layout'] == layout and b['query_task'] == query)
        assert branch['source_task_names'] == law['source_task_names']
        inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        psi = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][query])), bank)
            - inputs['mean_coordinate_center']) / inputs['mean_coordinate_scale']
        np.testing.assert_allclose(psi[coverage['layouts'][layout][query]], branch['query_mean_psi'], atol=1e-12, rtol=1e-12)
        sites = branch['scoring_sites']; selection = spec['paid_prefix'] + sites
        basis = OrthogonalLowOrderBasis(branch['public_mean_reference'])
        design = np.column_stack([np.ones(len(selection)), basis.transform(psi[selection])])
        f = (design @ np.asarray(law['source_posterior_coefficient_mean']),
            design @ (np.asarray(law['source_posterior_common_covariance']) + np.asarray(law['task_coefficient_covariance'])) @ design.T
            + law['white_discrepancy'] * np.eye(len(selection)))
        eta = np.asarray(branch['eta_mean'])[selection], np.asarray(branch['eta_covariance'])[np.ix_(selection, selection)]
        row = {'layout': layout, 'query_task': query, 'arm': 'joint_spectral_geometry', 'design': 'fixed_public_prefix',
            'episode_count': 4, 'paid_prefix': spec['paid_prefix'], 'scoring_sites': sites}
        fields.append({'row': row, 'field': gaussian_field(f, eta)})
        frozen.append({**row, 'source_task_names': law['source_task_names'], 'parameters': law['parameters'], 'positive_component_tails':
            [law['common_tail_per_coefficient'], law['task_tail_per_coefficient']]})
    assert len(fields) == 10
    save(out / 'frozen_inputs.json', {'execution': spec, 'rows': frozen, 'all_ten_priors_frozen_before_paid_values': True})
    raw = read(coverage['source_data'])
    paid = {b['query_task']: [raw['tasks'][b['query_task']]['Y_replicates'][j][0][0] for j in spec['paid_prefix']] for b in branches}; del raw
    rows = []
    for work in fields:
        row, field = work['row'], work.pop('field')
        evidence = sum(update_raw_observation(field, j, float(y)) for j, y in enumerate(paid[row['query_task']]))
        state = compact(field, list(range(4, 12)), evidence); filename = f"layout{row['layout']}_task{row['query_task']}.npz"
        np.savez_compressed(out / filename, **state); row.update(prediction=prediction(state), component_file=filename); rows.append(row)
    save(out / 'predictions.json', {'execution': spec, 'rows': rows, 'all_ten_forecasts_frozen_before_score_labels': True})
    base = {'execution': spec, 'cost': spec['cost'], 'no_model_adopted': True, 'training_diagnostic_only': True,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    for row in rows:
        if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
            save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False, 'stopped_at_cpu_budget': True,
                'cpu_seconds': time.process_time() - started}); return
        inputs = read(Path(coverage['output']) / f"layout{row['layout']}" / 'task1' / 'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == f"related_inventory_{row['query_task']:02d}")
        labels = {k: np.asarray(task[k])[4:] for k in ('response', 'noise')}
        with np.load(out / row['component_file']) as state: row['scores'], row['density_integration_error_bound'] = score(row, state, labels)
        row['mean_error'] = decompose(row['prediction']['mean'], labels['response'][:, 0]); row['comparisons'] = {}
        for control in spec['controls']:
            reference = next(r for r in controls['rows'] if r['layout'] == row['layout'] and r['query_task'] == row['query_task']
                and r['arm'] == control and r['design'] == row['design'])
            assert (reference['paid_prefix'], reference['scoring_sites']) == (row['paid_prefix'], row['scoring_sites'])
            row['comparisons'][control] = {k: row['scores'][k] - reference['scores'][k] for k in spec['metrics']}
    summary = {layout: {control: {'mean_differences': {m: float(np.mean([r['comparisons'][control][m] for r in rows if r['layout'] == layout])) for m in spec['metrics']},
        'mse_relative_mean_change': float(np.mean([r['scores']['f_mse'] for r in rows if r['layout'] == layout]) /
            np.mean([r['scores']['f_mse'] for r in controls['rows'] if r['layout'] == layout and r['arm'] == control and r['design'] == 'fixed_public_prefix']) - 1),
        'improved_tasks': {m: sum(r['comparisons'][control][m] < 0 if m == 'f_mse' else r['comparisons'][control][m] > 0 for r in rows if r['layout'] == layout) for m in spec['metrics']}}
        for control in spec['controls']} for layout in spec['layouts']}
    save(out / 'results.json', {**base, 'rows': rows, 'summary': summary, 'diagnostic_interpreted': True,
        'cpu_seconds': time.process_time() - started, 'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz'))})
    print('joint spectral raw development', summary, 'CPU', time.process_time() - started, flush=True)


if __name__ == '__main__':
    run()
