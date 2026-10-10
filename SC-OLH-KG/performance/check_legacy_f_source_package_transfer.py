"""Replace cached source4 f packages by source5 at identical query/risk sites."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, errors, score
from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_conditional_calibration import local_calibration
from performance.diagnose_legacy_f_outer_conditional_calibration import summarize
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_source_package_transfer_execution_v1.json'


def calibration(state, labels):
    return [local_calibration(state['component_means'][:, j], state['component_local_covariances'][:, j],
        state['log_weights'], label[0], label[3])[0] for j, label in enumerate(labels)]


def run():
    started = time.process_time(); spec = read(EXECUTION)
    assert read(spec['outer_diagnostic_results'])['diagnostic_interpreted']
    old = read(spec['source4_forecasts']); assert old['diagnostic_interpreted']
    rules = read(spec['source5_rules']); assert rules['all14_rules_frozen_without_query_data']
    coverage, _ = source_paths(read(spec['source_context_execution']))
    branches = []
    for folder in old['execution']['source_packages']:
        branches += read(Path(folder) / 'source_branches.json')['rows']
    work, frozen = [], []
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    base = {'execution': spec, 'cost': spec['cost'], 'no_model_adopted': True,
        'no_new_unscored_outcomes_read': True, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False, 'training_diagnostic_only': True}
    for branch in branches:
        layout, query = branch['layout'], branch['query_task']
        rule = next(r for r in rules['rows'] if r['layout'] == layout and r['deleted_task'] == query)
        inputs = read(Path(coverage['output']) / f'layout{layout}' / f'task{query}' / 'inputs.json')
        names = [t['name'] for t in inputs['source_tasks']]
        assert names == rule['source_reference_law']['source_task_names']
        assert set(names) - set(branch['source_task_names']) == {'related_inventory_01'}
        assert f'related_inventory_{query:02d}' not in names
        prior = next(r for r in old['rows'] if r['layout'] == layout and r['query_task'] == query
            and r['arm'] == 'native_mixture' and r['design'] == 'fixed_public_prefix')
        assert prior['paid_prefix'] == spec['paid_prefix'] and prior['scoring_sites'] == branch['scoring_sites']
        indices = spec['paid_prefix'] + branch['scoring_sites']
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        psi = np.asarray(inputs['public_mean_reference'])[indices]
        eta = np.asarray(branch['eta_mean'])[indices], np.asarray(branch['eta_covariance'])[np.ix_(indices, indices)]
        nodes, weights = rule_on_intervals(rule['source_reference_law'], rule['adaptive_u_intervals'], 32)
        work.append({'layout': layout, 'query_task': query, 'baseline': prior,
            'selected': mixture_field(basis, psi, rule['source_reference_law'], np.asarray(rule['variance_nodes']), np.asarray(rule['posterior_weights']), eta),
            'refined': mixture_field(basis, psi, rule['source_reference_law'], nodes, weights, eta)})
        frozen.append({'layout': layout, 'query_task': query, 'source4_names': branch['source_task_names'],
            'source5_names': names, 'paid_prefix': spec['paid_prefix'], 'scoring_sites': branch['scoring_sites'],
            'eta_source4_unchanged': True})
    assert len(work) == 10
    save(out / 'frozen_inputs.json', {'execution': spec, 'all_ten_source5_query_priors_frozen_before_paid_values': True, 'rows': frozen})
    raw = read(coverage['source_data'])
    paid = {b['query_task']: [raw['tasks'][b['query_task']]['Y_replicates'][j][0][0] for j in spec['paid_prefix']] for b in work}
    del raw
    rows, numerical = [], []
    for branch in work:
        row = {'layout': branch['layout'], 'query_task': branch['query_task'], 'episode_count': 4,
            'paid_prefix': spec['paid_prefix'], 'scoring_sites': branch['baseline']['scoring_sites']}
        for resolution in ('selected', 'refined'):
            field = branch.pop(resolution)
            evidence = sum(update_raw_observation(field, j, float(y)) for j, y in enumerate(paid[row['query_task']]))
            state = compact(field, list(range(4, 12)), evidence)
            filename = f"layout{row['layout']}_task{row['query_task']}_{resolution}.npz"
            np.savez_compressed(out / filename, **state)
            row[f'{resolution}_component_file'], row[f'{resolution}_prediction'] = filename, prediction(state)
        row['prediction'] = row.pop('selected_prediction')
        difference = errors(row['prediction'], row['refined_prediction'])
        numerical.append({'layout': row['layout'], 'query_task': row['query_task'], 'errors': difference,
            'passed': all(difference[k] <= v for k, v in spec['prediction_tolerances'].items())})
        rows.append(row)
    save(out / 'predictions.json', {'execution': spec, 'all_ten_forecasts_frozen_before_score_labels': True, 'rows': rows, 'numerical_checks': numerical})
    if not all(r['passed'] for r in numerical):
        save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False,
            'numerical_checks': numerical, 'cpu_seconds': time.process_time() - started}); return
    checks = []
    for row, branch in zip(rows, work):
        if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
            save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False,
                'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started}); return
        inputs = read(Path(coverage['output']) / f"layout{row['layout']}" / 'task1' / 'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == f"related_inventory_{row['query_task']:02d}")
        labels = {k: np.asarray(task[k])[4:] for k in ('response', 'noise')}
        old_file = ROOT / old['execution']['output'] / branch['baseline']['selected_component_file']
        with np.load(old_file) as state:
            baseline_sites = calibration(state, labels['response'])
        sites_by_rule = []
        for resolution in ('selected', 'refined'):
            with np.load(out / row[f'{resolution}_component_file']) as state:
                values, _ = score(row, state, labels)
                sites = calibration(state, labels['response'])
            row[f'{resolution}_scores'] = values; sites_by_rule.append(sites)
            assert abs(summarize(sites)['joint_batch_density'] - values['f_raw_batch_density']) <= 1e-8
        prior = branch['baseline']; before = summarize(baseline_sites)
        assert abs(before['joint_batch_density'] - prior['scores']['f_raw_batch_density']) <= 1e-8
        selected, refined = sites_by_rule
        error = {
            'conditional_mean_max_absolute': max(abs(a['conditional_mean'] - b['conditional_mean']) for a, b in zip(selected, refined)),
            'variance_relative': max(abs(a['conditional_batch_mean_variance'] / b['conditional_batch_mean_variance'] - 1) for a, b in zip(selected, refined)),
            'cdf_max_absolute': max(abs(a['conditional_cdf'] - b['conditional_cdf']) for a, b in zip(selected, refined)),
            'gradient_max_absolute': max(abs(a['log_conditional_f_variance_score_gradient'] - b['log_conditional_f_variance_score_gradient']) for a, b in zip(selected, refined))}
        checks.append({'layout': row['layout'], 'query_task': row['query_task'], 'errors': error,
            'density_error': abs(row['selected_scores']['f_raw_batch_density'] - row['refined_scores']['f_raw_batch_density']),
            'passed': all(error[k] <= v for k, v in spec['refinement_tolerances'].items()) and
                abs(row['selected_scores']['f_raw_batch_density'] - row['refined_scores']['f_raw_batch_density']) <= spec['density_refinement_tolerance']})
        row.update(source4_scores=prior['scores'], source4_calibration=before, source5_calibration=summarize(selected))
        row['source5_minus_source4'] = {k: row['selected_scores'][k] - prior['scores'][k] for k in row['selected_scores']}
        save(out / 'partial_results.json', {**base, 'rows': rows, 'checks': checks})
        print('source-package contrast', row['layout'], row['query_task'], row['source5_minus_source4']['f_mse'],
            row['source5_minus_source4']['f_raw_batch_density'], before['log_conditional_f_variance_score_gradient'],
            row['source5_calibration']['log_conditional_f_variance_score_gradient'], flush=True)
    summary = {layout: {
        'source5_minus_source4': {k: float(np.mean([r['source5_minus_source4'][k] for r in rows if r['layout'] == layout])) for k in rows[0]['source5_minus_source4']},
        'source4_calibration': {k: float(np.mean([r['source4_calibration'][k] for r in rows if r['layout'] == layout])) for k in rows[0]['source4_calibration']},
        'source5_calibration': {k: float(np.mean([r['source5_calibration'][k] for r in rows if r['layout'] == layout])) for k in rows[0]['source5_calibration']}}
        for layout in spec['layouts']}
    save(out / 'results.json', {**base, 'rows': rows, 'summary': summary, 'numerical_checks': numerical,
        'calibration_checks': checks, 'diagnostic_interpreted': all(c['passed'] for c in checks),
        'cpu_seconds': time.process_time() - started, 'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz')),
        'interpretation': 'Matched query task,physical sites,paid values and source4 eta law. F source package adds already-used task1 and changes its fitted public map/basis/hyperparameters;source-count,task inclusion and representation effects remain bundled. Unequal historical source budgets144/180 calls;not a same-budget algorithm comparison or confirmation.'})


if __name__ == '__main__':
    run()
