"""Align source4 few-shot designs and add the same-source GP prior control."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, errors, score, raw_density
from performance.check_legacy_f_source_information_design import gaussian_field
from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_kernel_population import predict_unequal_site_kernel_population
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_query_design_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); previous = read(spec['inner_results'])
    assert previous['all_ten_inner_folds_complete'] and previous['all_twenty_mixture_numerical_references_verified']
    context = read(spec['plan']); coverage, _ = source_paths(read(context['source_context_execution']))
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    source_branches, rules = [], []
    for directory in spec['source_packages']:
        source_branches += read(Path(directory)/'source_branches.json')['rows']
        rules += read(Path(directory)/'source_quadrature.json')['rows']
    inner = read(context['inner_source_results']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows = [{**r, 'design': 'source_anchors', 'reused_existing_forecast_and_scores': True} for r in previous['rows']]
    fields, frozen = [], []
    for branch in source_branches:
        layout, query, sites = branch['layout'], branch['query_task'], branch['scoring_sites']
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f"task{context['outer_deleted_task']}"/'inputs.json')
        provider = no_outcomes(problem_for(public['plan'], public['tasks'][query]))
        psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
        np.testing.assert_allclose(psi[coverage['layouts'][layout][query]], branch['query_mean_psi'], atol=1e-12, rtol=1e-12)
        basis = OrthogonalLowOrderBasis(branch['public_mean_reference'])
        fitting = next(r for r in inner['rows'] if r['layout'] == layout and r['inner_query_task'] == f'related_inventory_{query:02d}')
        gp = read(fitting['source_fit']); assert gp['inner_training_task_names'] == branch['source_task_names']
        for design in spec['designs']:
            prefix = spec['fixed_public_prefix'] if design == 'fixed_public_prefix' else context['source_anchors']
            assert len(prefix) == 4 and not set(prefix).intersection(sites)
            selection = prefix+sites; eta = np.asarray(branch['eta_mean'])[selection], np.asarray(branch['eta_covariance'])[np.ix_(selection, selection)]
            for arm in spec['arms']:
                if design == 'source_anchors' and arm != 'source_gp': continue
                record = {'layout': layout, 'query_task': query, 'design': design, 'arm': arm, 'episode_count': 4,
                    'paid_prefix': prefix, 'scoring_sites': sites}
                if arm == 'source_gp':
                    field = gaussian_field(predict_unequal_site_kernel_population(psi[selection], gp['fits']['gp']), eta)
                    work = {'row': record, 'selected': field}
                else:
                    allocation = arm.split('_')[0]
                    rule = next(r for r in rules if r['layout'] == layout and r['arm'] == allocation
                        and r['source_reference_law']['source_task_names'] == branch['source_task_names'])
                    law = rule['source_reference_law']
                    nodes, weights = (np.asarray([law['native_task_intercept_variance']]), np.ones(1)) if arm.endswith('point') else (
                        np.asarray(rule['variance_nodes']), np.asarray(rule['posterior_weights']))
                    work = {'row': record, 'selected': mixture_field(basis, psi[selection], law, nodes, weights, eta)}
                    if arm.endswith('mixture'):
                        rn, rw = rule_on_intervals(law, rule['adaptive_u_intervals'], 32)
                        work['refined'] = mixture_field(basis, psi[selection], law, rn, rw, eta)
                fields.append(work); frozen.append({**record, 'source_task_names': branch['source_task_names'],
                    'source_label_count': 48, 'source_calls': 144, 'risk_calls_subset': 96, 'component_count': len(work['selected'].log_weights),
                    'eta_law_shared': True, 'source_q_unchanged': True})
    save(out/'frozen_inputs.json', {'execution': spec, 'all_sixty_priors_and_prefixes_frozen_before_new_paid_values': True, 'rows': frozen})
    raw = read(coverage['source_data'])
    paid = {(r['query_task'], r['design']): [raw['tasks'][r['query_task']]['Y_replicates'][j][0][0] for j in r['paid_prefix']] for r in frozen}; del raw
    numerical = []; new_rows = []
    for work in fields:
        row = work['row']
        for name in ('selected', 'refined'):
            if name not in work: continue
            field = work.pop(name); evidence = sum(update_raw_observation(field, j, float(y)) for j, y in enumerate(paid[(row['query_task'], row['design'])]))
            state = compact(field, list(range(4, 12)), evidence)
            filename = f"layout{row['layout']}_task{row['query_task']}_{row['design']}_{row['arm']}_{name}.npz"
            np.savez_compressed(out/filename, **state); row[f'{name}_component_file'], row[f'{name}_prediction'] = filename, prediction(state)
            del field, state
        row['prediction'] = row.pop('selected_prediction')
        if 'refined_prediction' in row:
            error = errors(row['prediction'], row['refined_prediction'])
            numerical.append({'layout': row['layout'], 'query_task': row['query_task'], 'arm': row['arm'], 'errors': error,
                'passed': all(error[k] <= t for k, t in spec['prediction_tolerances'].items())})
        new_rows.append(row)
    save(out/'predictions.json', {'execution': spec, 'all_sixty_new_forecasts_frozen_before_score_labels': True, 'rows': new_rows, 'numerical_checks': numerical})
    base = {'execution': spec, 'numerical_checks': numerical, 'source6_7_evaluated': False, 'target_promoted': False,
        'independent_confirmation': False, 'no_model_adopted': True, 'training_diagnostic_only': True}
    if not all(r['passed'] for r in numerical):
        save(out/'results.json', {**base, 'forecasts_scored': False, 'diagnostic_interpreted': False, 'cpu_seconds': time.process_time()-started}); return
    density_checks = []
    for row in new_rows:
        inputs = read(Path(coverage['output'])/f"layout{row['layout']}"/f"task{context['outer_deleted_task']}"/'inputs.json')
        labels = next(t for t in inputs['source_tasks'] if t['name'] == f"related_inventory_{row['query_task']:02d}")
        labels = {key: np.asarray(labels[key])[4:] for key in ('response', 'noise')}
        with np.load(out/row['selected_component_file']) as state: row['scores'], row['density_integration_error_bound'] = score(row, state, labels)
        error = np.asarray(row['prediction']['mean'])-labels['response'][:, 0]
        row['scores'].update(level_mse=float(error.mean()**2), centered_shape_mse=float(np.mean((error-error.mean())**2)))
        if 'refined_component_file' in row:
            with np.load(out/row['refined_component_file']) as state: density, bound = raw_density(state, labels)
            difference = abs(density-row['scores']['f_raw_batch_density'])
            density_checks.append({'layout': row['layout'], 'query_task': row['query_task'], 'arm': row['arm'],
                'raw_density_refinement_error': difference, 'reference_integration_error_bound': bound, 'passed': difference <= spec['raw_density_tolerance']})
    rows += new_rows; summary = {}
    for layout in ('5', '6'):
        summary[layout] = {}
        for design in spec['designs']:
            groups = {arm: [r for r in rows if r['layout'] == layout and r['design'] == design and r['arm'] == arm] for arm in spec['arms']}
            assert all(len(members) == 5 for members in groups.values())
            means = {arm: {k: float(np.mean([r['scores'][k] for r in members])) for k in members[0]['scores']} for arm, members in groups.items()}
            contrasts = {}
            for a, b in [('shape_point', 'native_point'), ('shape_mixture', 'native_mixture'), ('shape_point', 'source_gp'), ('shape_mixture', 'source_gp')]:
                contrasts[f'{a}_minus_{b}'] = {'differences': {k: means[a][k]-means[b][k] for k in means[a]},
                    'mse_relative_change': means[a]['f_mse']/means[b]['f_mse']-1,
                    'mse_improved_folds': sum(r['scores']['f_mse'] < next(s['scores']['f_mse'] for s in groups[b] if s['query_task'] == r['query_task']) for r in groups[a])}
            summary[layout][design] = {'means': means, 'contrasts': contrasts}
    verified = all(r['passed'] for r in density_checks)
    result = {**base, 'rows': rows, 'summary': summary, 'density_checks': density_checks,
        'numerical_density_verified': verified, 'forecasts_scored': True, 'diagnostic_interpreted': verified,
        'cost': {'new_source_fits': 0, 'new_source_integrations': 0, 'new_raw_forecasts': len(new_rows), 'new_numerical_reference_forecasts': len(numerical),
            'reused_raw_forecasts': 40, 'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started, 'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz'))}
    save(out/'results.json', result)
    print('source4 query-design contrasts', {l: {d: {name: c['mse_relative_change'] for name, c in g['contrasts'].items()}
        for d, g in ds.items()} for l, ds in summary.items()}, flush=True)


if __name__ == '__main__':
    run()
