"""Complete mean/risk source-deletion mixtures on the 216-call design."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_covered_complete_mean_uncertainty import fit_means, complete_models
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.check_legacy_source_layout_robustness import models_at_query
from performance.run_legacy_related_inventory_source import save, training_view

EXECUTION = 'performance/manifests/legacy_joint_population_counterbalanced_complete_mean_uncertainty_execution_v1.json'


def load_group(spec, coverage, source, layout):
    current = Path(coverage['output'])/f'layout{layout}'
    parent = Path(source['output'])/f"layout{spec['parent_layouts'][layout]}"
    group = [(read(current/'inputs.json'), read(current/'source_fit.json'), read(parent/'risk_source_fit.json'))]
    group += [(read(current/f'task{i}'/'inputs.json'), read(current/f'task{i}'/'gp_source_fit.json'),
               read(parent/f'task{i}'/'risk_source_fit.json')) for i in spec['training_tasks']]
    for j, (inputs, _, _) in enumerate(group):
        mean_tasks, risk_tasks = inputs['source_tasks'], inputs['risk_source_tasks']
        assert [t['name'] for t in mean_tasks] == [t['name'] for t in risk_tasks]
        assert sum(len(t['psi']) for t in mean_tasks) == (72 if j == 0 else 60)
        assert sum(len(t['psi']) for t in risk_tasks) == (48 if j == 0 else 40)
        assert inputs['mean_source_design_rank'] == 27
        if j:
            assert inputs['deleted_task'] == spec['training_tasks'][j-1]
    return group


def contrast(row, reference, kind):
    actual, previous = row['scores'], reference['scores']
    result = {'layout': row['layout'], 'parent_layout': row['parent_layout'],
        'task_id': row['task_id'], 'mean_arm': row['mean_arm'], 'reference': kind,
        'g_mse_difference': actual['observed_head_mse'][1]-previous['observed_head_mse'][1],
        'joint_batch_score_difference': actual['joint_batch_log_score']-previous['joint_batch_log_score'],
        'g_batch_score_difference': actual['response_batch_log_scores'][1]-previous['response_batch_log_scores'][1]}
    result['passed'] = (result['g_mse_difference'] <= 0 and result['joint_batch_score_difference'] >= 0
                        and result['g_batch_score_difference'] > 0)
    return result


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution'])
    source = read(coverage['source_execution']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    groups, fits = {}, {}
    for layout in spec['layouts']:
        group = load_group(spec, coverage, source, layout)
        means = [group[0][1]]+[fit_means(g[0]['execution'], g[0], out/f'layout{layout}'/f'task{i}')
            for i, g in zip(spec['training_tasks'], group[1:])]
        groups[layout], fits[layout] = group, means
        print('all counterbalanced deletion mean fits frozen', layout, flush=True)

    data = read(coverage['source_data'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    rows, jobs = [], {}
    for layout, group in groups.items():
        for i in spec['heldout_tasks']:
            task = data['tasks'][i]
            query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source', 'tau': task['tau'],
                'candidate_indices': [0, 1], 'paired_observations': [task['Y_replicates'][j][0] for j in (0, 1)]}
            mq = [g[0]['heldout_mean_psi'][str(i)] for g in group]
            rq = [g[0]['heldout_risk_psi'][str(i)] for g in group]
            controls = dict(models_at_query(group, mq, rq))
            for arm, model in complete_models(group, fits[layout], mq, rq, spec['mean_arms']):
                prior = forecast(query, arm, 'counterbalanced_complete_mean_risk_deletions', model)
                for j, value in zip(query['candidate_indices'], query['paired_observations']):
                    centered = [value[0], value[1]-query['tau']]
                    model.update_raw(j, centered); controls[arm].update_raw(j, centered)
                for field, original in zip(model.components[0], controls[arm].components[0]):
                    for actual, expected in zip(field.predict(full_cov=True), original.predict(full_cov=True)):
                        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
                row = forecast(query, arm, 'counterbalanced_complete_mean_risk_deletions', model)
                row.update(layout=layout, parent_layout=spec['parent_layouts'][layout], prior=prior)
                rows.append(row); jobs[(layout, i, arm)] = model
    save(out/'source_predictions.json', {'execution': spec, 'all_forecasts_frozen': True, 'rows': rows})

    references = read(spec['reference_profiles'])['rows']
    parents = read(spec['parent_reference_profiles'])['rows']
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    contrasts, parent_contrasts = [], []
    for row in rows:
        layout, i, arm = row['layout'], row['task_index'], row['mean_arm']
        score_source(row, labels[i], jobs[(layout, i, arm)], 3)
        reference = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == arm)
        parent = next(r for r in parents if r['layout'] == row['parent_layout'] and r['task_index'] == i and r['mean_arm'] == arm)
        contrasts.append(contrast(row, reference, 'same216_call_risk_only_mixture'))
        parent_contrasts.append(contrast(row, parent, 'original144_call_risk_only_mixture'))
    gates = {a: all(c['passed'] for c in contrasts if c['mean_arm'] == a) for a in spec['mean_arms']}
    parent_gates = {a: all(c['passed'] for c in parent_contrasts if c['mean_arm'] == a) for a in spec['mean_arms']}
    summary = {layout: {a: {k: np.mean([r['scores'][k] for r in rows if r['layout'] == layout and r['mean_arm'] == a], axis=0).tolist()
        for k in rows[0]['scores']} for a in spec['mean_arms']} for layout in spec['layouts']}
    controls = {layout: {a: {k: np.mean([r['scores'][k] for r in references if r['layout'] == layout and r['mean_arm'] == a], axis=0).tolist()
        for k in references[0]['scores']} for a in ('joint_structural', 'flat', 'joint_gp')} for layout in spec['layouts']}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows})
    continuations = sum('continued_from_optimizer_iterations' in h['joint_structural']
        for means in fits.values() for fit in means[1:] for h in fit['heads'])
    result = {'execution': spec, 'summary': summary, 'same216_call_control_summary': controls,
        'contrasts': contrasts, 'parent_contrasts': parent_contrasts, 'source_gates': gates,
        'parent_source_gates': parent_gates,
        'structural_advance_gate_passed': gates['joint_structural'] and parent_gates['joint_structural'],
        'full_identity_reproduction_passed': True, 'source_identity_pairing_passed': True,
        'native_mean_fits': 96+continuations,
        'source_mean_fit_cpu_seconds': sum(f['cpu_seconds'] for means in fits.values() for f in means[1:]),
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print({k: result[k] for k in ('summary', 'contrasts', 'parent_contrasts', 'source_gates',
        'parent_source_gates', 'structural_advance_gate_passed', 'native_mean_fits', 'cpu_seconds')}, flush=True)


if __name__ == '__main__':
    run()
