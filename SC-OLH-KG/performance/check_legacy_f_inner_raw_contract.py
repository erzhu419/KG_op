"""Fixed source4 raw-contract diagnosis at the canonical inner task0."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, errors, score, raw_density
from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_raw_contract_execution_v1.json'


def run(execution=EXECUTION):
    started = time.process_time(); spec = read(execution); plan = read(spec['plan'])
    source = read(spec['source_results']); assert source['all_source_laws_complete']
    package, branches = read(spec['source_rules']), read(spec['source_branches'])
    assert package['all_source_rules_frozen_before_raw_query_values']
    coverage, _ = source_paths(read(plan['source_context_execution']))
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); fields, frozen = [], []
    for branch in branches['rows']:
        layout, query = branch['layout'], branch['query_task']; prefix, sites = branch['paid_prefix'], branch['scoring_sites']
        assert prefix == plan['source_anchors'] == coverage['layouts'][layout][query][:4]
        assert len(sites) == 8 and not set(prefix).intersection(sites)
        names = [f'related_inventory_{i:02d}' for i in range(6) if i not in (plan['outer_deleted_task'], query)]
        assert branch['source_task_names'] == names
        basis = OrthogonalLowOrderBasis(branch['public_mean_reference']); psi = np.asarray(branch['query_mean_psi'])
        assert len(psi) == len(prefix)+len(sites) == 12
        selection = prefix+sites; eta = np.asarray(branch['eta_mean'])[selection], np.asarray(branch['eta_covariance'])[np.ix_(selection, selection)]
        for allocation in ('native', 'shape'):
            rule = next(r for r in package['rows'] if r['layout'] == layout and r['arm'] == allocation
                and r['source_reference_law']['source_task_names'] == names)
            law = rule['source_reference_law']
            for contract in ('point', 'mixture'):
                if contract == 'point': nodes, weights = np.asarray([law['native_task_intercept_variance']]), np.ones(1)
                else: nodes, weights = np.asarray(rule['variance_nodes']), np.asarray(rule['posterior_weights'])
                field = mixture_field(basis, psi, law, nodes, weights, eta)
                arm = f'{allocation}_{contract}'; record = {'layout': layout, 'query_task': query, 'arm': arm,
                    'episode_count': 4, 'paid_prefix': prefix, 'scoring_sites': sites, 'component_count': len(nodes)}
                work = {'row': record, 'selected': field}
                if contract == 'mixture':
                    rn, rw = rule_on_intervals(law, rule['adaptive_u_intervals'], plan['quadrature']['refinement_nodes_per_interval'])
                    work['refined'] = mixture_field(basis, psi, law, rn, rw, eta)
                    record['reference_component_count'] = len(rn)
                fields.append(work); frozen.append({**record, 'source_task_names': branch['source_task_names'],
                    'shape_q': law['native_law']['fitted_shape_allocation_q'], 'eta_law_shared_between_arms': True})
    save(out/'frozen_inputs.json', {'execution': spec, 'all_priors_frozen_before_raw_query_values': True, 'rows': frozen})
    raw = read(coverage['source_data'])
    paid = {b['query_task']: [raw['tasks'][b['query_task']]['Y_replicates'][j][0][0] for j in plan['source_anchors']] for b in branches['rows']}; del raw
    rows, numerical = [], []
    for work in fields:
        row = work['row']
        for name in ('selected', 'refined'):
            if name not in work: continue
            field = work.pop(name); evidence = sum(update_raw_observation(field, j, float(y)) for j, y in enumerate(paid[row['query_task']]))
            state = compact(field, list(range(4, 12)), evidence); p = prediction(state)
            filename = f"layout{row['layout']}_task{row['query_task']}_{row['arm']}_{name}.npz"; np.savez_compressed(out/filename, **state)
            row[f'{name}_prediction'], row[f'{name}_component_file'] = p, filename
            del field, state
        row['prediction'] = row.pop('selected_prediction')
        if 'refined_prediction' in row:
            difference = errors(row['prediction'], row['refined_prediction'])
            passed = all(difference[k] <= t for k, t in spec['prediction_tolerances'].items())
            numerical.append({'layout': row['layout'], 'query_task': row['query_task'], 'arm': row['arm'], 'errors': difference, 'passed': passed})
        rows.append(row)
    save(out/'predictions.json', {'execution': spec, 'all_forecasts_frozen_before_score_labels': True, 'rows': rows, 'numerical_checks': numerical})
    base = {'execution': spec, 'numerical_checks': numerical, 'no_model_adopted': True,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False,
        'cost': {**plan['cost'], **spec['cost'], 'actual_new_source_fits_at_raw_step': 0, 'actual_inner_raw_forecasts': len(rows), 'actual_numerical_reference_forecasts': len(numerical)}}
    if not all(r['passed'] for r in numerical):
        save(out/'results.json', {**base, 'forecasts_scored': False, 'diagnostic_interpreted': False, 'cpu_seconds': time.process_time()-started}); return
    density_checks = []
    for row in rows:
        inputs = read(Path(coverage['output'])/f"layout{row['layout']}"/f"task{plan['outer_deleted_task']}"/'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == f"related_inventory_{row['query_task']:02d}")
        labels = {key: np.asarray(task[key])[4:] for key in ('response', 'noise')}
        with np.load(out/row['selected_component_file']) as state:
            row['scores'], row['density_integration_error_bound'] = score(row, state, labels)
        error = np.asarray(row['prediction']['mean'])-labels['response'][:, 0]
        row['scores'].update(level_mse=float(error.mean()**2), centered_shape_mse=float(np.mean((error-error.mean())**2)))
        if 'refined_component_file' in row:
            with np.load(out/row['refined_component_file']) as state: density, bound = raw_density(state, labels)
            difference = abs(density-row['scores']['f_raw_batch_density'])
            density_checks.append({'layout': row['layout'], 'query_task': row['query_task'], 'arm': row['arm'], 'raw_density_refinement_error': difference,
                'reference_density_integration_error_bound': bound, 'passed': difference <= spec['raw_density_tolerance']})
        print('inner source raw forecast', row['layout'], row['arm'], {k: row['scores'][k] for k in ('f_mse', 'f_raw_batch_density', 'sequential_transcript_density')}, flush=True)
    contrasts = []
    for branch in branches['rows']:
        layout, query = branch['layout'], branch['query_task']
        selected = {r['arm']: r for r in rows if r['layout'] == layout and r['query_task'] == query}
        pairs = [(f'shape_{c}', f'native_{c}', f'shape_minus_native_{c}') for c in ('point', 'mixture')]
        pairs += [(f'{a}_mixture', f'{a}_point', f'mixture_minus_point_{a}') for a in ('native', 'shape')]
        for candidate, reference, name in pairs:
            a, b = selected[candidate]['scores'], selected[reference]['scores']
            contrasts.append({'layout': layout, 'query_task': query, 'contrast': name, 'differences': {key: a[key]-b[key] for key in a},
                'mse_relative_change': a['f_mse']/b['f_mse']-1})
    verified = all(r['passed'] for r in density_checks)
    result = {**base, 'rows': rows, 'contrasts': contrasts, 'density_checks': density_checks,
        'numerical_density_verified': verified, 'forecasts_scored': True, 'diagnostic_interpreted': verified,
        'cpu_seconds': time.process_time()-started, 'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz')),
        'training_diagnostic_only': True, 'stopped_shape_candidate_not_promoted': True}
    save(out/'results.json', result)
    print('inner raw contrasts', [{k: c[k] for k in ('layout', 'contrast', 'mse_relative_change')} for c in contrasts], flush=True)


if __name__ == '__main__':
    run()
