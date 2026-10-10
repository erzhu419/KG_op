"""216-call cross-task coverage under frozen mean bases and risk laws."""

import copy
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_coordinates import fit as fit_mean
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.check_legacy_source_layout_robustness import fit_gp, models_at_query, ARMS, CASES
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.legacy_function_prior import replicated_source_labels
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates

EXECUTION = 'performance/manifests/legacy_joint_population_counterbalanced_source_mean_execution_v1.json'


def mean_view(data, public, source, parent, deleted=None):
    """Slice the paid raw labels; keep the parent's mean/risk representation."""
    result = copy.deepcopy(parent); bank = [tuple(p) for p in public['bank']]
    tasks = []
    for i, sites in zip(source['training_tasks'], source['source_site_indices_by_task']):
        if i == deleted:
            continue
        task = data['tasks'][i]
        labels = replicated_source_labels(task['task_id'], np.asarray(data['bank'])[sites]/
            public['plan']['task_distribution']['L'], [task['Y_replicates'][j] for j in sites], task['tau'])
        raw = public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank)
        psi = (raw[sites]-parent['mean_coordinate_center'])/parent['mean_coordinate_scale']
        tasks.append({**labels, 'psi': psi, 'features': psi})
    result['source_tasks'] = [{**t, **{k: t[k].tolist() for k in ('psi', 'features', 'response', 'noise')}} for t in tasks]
    result['execution'] = source
    basis = OrthogonalLowOrderBasis(result['public_mean_reference']); psi = np.vstack([t['psi'] for t in tasks])
    design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
    result['mean_source_design_rank'] = int(np.linalg.matrix_rank(design))
    result['mean_source_record_count'] = len(psi)
    # Dropping a prefix or relabeling its site would invalidate this information contrast.
    for old, new in zip(parent['source_tasks'], result['source_tasks']):
        assert old['name'] == new['name']
        for key in ('psi', 'features', 'response', 'noise'):
            np.testing.assert_allclose(np.asarray(new[key])[:len(old[key])], old[key], rtol=1e-12, atol=1e-12)
    return result


def prepare_layout(spec, source_execution, data, public, layout):
    parent = spec['parent_layouts'][layout]; old_folder = Path(source_execution['output'])/f'layout{parent}'
    old = read(old_folder/'inputs.json'); folder = ROOT/spec['output']/f'layout{layout}'; folder.mkdir(parents=True, exist_ok=True)
    source = dict(old['execution'], source_site_indices_by_task=spec['layouts'][layout],
        source_response_calls_per_arm=216, selected_source_records=72, source_risk_calls_per_arm=144,
        source_budget_per_model=216, records_per_layout=72, output=str(folder.relative_to(ROOT)),
        mean_coordinates=f'unchanged parent layout{parent} public normalization/reference and26 functions',
        risk_coordinates_and_prior=f'frozen parent layout{parent} full48/six40-record maps and risk laws',
        source_fit_details='14 full f/g objectives plus six branches with6 native GP objectives each; risk fits0',
        matched_information='all arms share216 source calls and the frozen parent144-call risk priors',
        source_validation=spec['source_gate'], next_action='source gate before any risk extension or target probe')
    inputs = mean_view(data, public, source, old); save(folder/'inputs.json', inputs)
    means = fit_mean(source, inputs); risk = read(old_folder/'risk_source_fit.json')
    members = [(inputs, means, risk)]; fit_count = 14
    fit_count += sum((folder/f'head{h}_joint_before_completion.json').exists() for h in (0, 1))
    cpu = means['cpu_seconds']; branches = []
    for deleted in spec['training_tasks']:
        branch = folder/f'task{deleted}'; branch.mkdir(parents=True, exist_ok=True)
        branch_source = dict(source, output=str(branch.relative_to(ROOT)))
        view = mean_view(data, public, branch_source, read(old_folder/f'task{deleted}'/'inputs.json'), deleted)
        save(branch/'inputs.json', view)
        gp = fit_gp(branch_source, view); eta = read(old_folder/f'task{deleted}'/'risk_source_fit.json')
        members.append((view, gp, eta)); fit_count += 6; cpu += gp['cpu_seconds']
        branches.append({'deleted_task': deleted, 'records': view['mean_source_record_count'], 'design_rank': view['mean_source_design_rank']})
    counts = Counter(j for sites in spec['layouts'][layout] for j in sites)
    np.testing.assert_equal(sum(counts.values()), 72)
    record = {'layout': layout, 'parent_layout': parent, 'source_calls': 216, 'frozen_risk_source_calls': 144,
        'records': inputs['mean_source_record_count'], 'distinct_policies': len(counts), 'policy_task_counts': dict(counts),
        'mean_design_rank': inputs['mean_source_design_rank'], 'branches': branches,
        'native_mean_fits': fit_count, 'source_fit_cpu_seconds': cpu}
    print('counterbalanced source fits frozen', layout, record, flush=True)
    return members, record


def run():
    started = time.process_time(); spec = read(EXECUTION); source_execution = read(spec['source_execution'])
    data = read(spec['source_data']); public = read(Path(spec['source_data']).with_name('frozen_inputs.json'))
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); groups, fitting = {}, []
    for layout in spec['layouts']:
        members, record = prepare_layout(spec, source_execution, data, public, layout)
        groups[layout] = members; fitting.append(record)
    rows, jobs = [], {}; references = read(spec['reference_profiles'])['rows']
    for layout, members in groups.items():
        parent = spec['parent_layouts'][layout]
        for i in spec['heldout_tasks']:
            task = data['tasks'][i]; query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source',
                'tau': task['tau'], 'candidate_indices': [0, 1], 'paired_observations': [task['Y_replicates'][j][0] for j in (0, 1)]}
            for arm, model in models_at_query(members, [m[0]['heldout_mean_psi'][str(i)] for m in members],
                    [m[0]['heldout_risk_psi'][str(i)] for m in members]):
                prior = forecast(query, arm, CASES[arm], model)
                old = next(r for r in references if r['layout'] == parent and r['task_index'] == i and r['mean_arm'] == arm)
                # The mean-only ablation must keep each arm's marginal risk prior intact.
                for key in ('mean', 'variance'):
                    np.testing.assert_allclose(np.asarray(prior[key])[:, 2:], np.asarray(old['prior'][key])[:, 2:], atol=1e-8)
                for j, value in zip(query['candidate_indices'], query['paired_observations']):
                    model.update_raw(j, [value[0], value[1]-query['tau']])
                row = forecast(query, arm, CASES[arm], model)
                row.update(layout=layout, parent_layout=parent, prior=prior); rows.append(row); jobs[(layout, i, arm)] = model
    save(out/'source_predictions.json', {'execution': spec, 'all_forecasts_frozen': True, 'rows': rows})
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    contrasts = []
    for row in rows:
        layout, i, arm = row['layout'], row['task_index'], row['mean_arm']
        score_source(row, labels[i], jobs[(layout, i, arm)], 3)
        old = next(r for r in references if r['layout'] == row['parent_layout'] and r['task_index'] == i and r['mean_arm'] == arm)
        c = {'layout': layout, 'parent_layout': row['parent_layout'], 'task_id': row['task_id'], 'mean_arm': arm,
            'g_mse_difference': row['scores']['observed_head_mse'][1]-old['scores']['observed_head_mse'][1],
            'joint_batch_score_difference': row['scores']['joint_batch_log_score']-old['scores']['joint_batch_log_score'],
            'g_batch_score_difference': row['scores']['response_batch_log_scores'][1]-old['scores']['response_batch_log_scores'][1]}
        c['passed'] = c['g_mse_difference'] <= 0 and c['joint_batch_score_difference'] >= 0 and c['g_batch_score_difference'] > 0
        contrasts.append(c)
    source_gate = all(c['passed'] for c in contrasts if c['mean_arm'] == 'joint_structural')
    summary = {layout: {arm: {k: np.mean([r['scores'][k] for r in rows if r['layout'] == layout and r['mean_arm'] == arm], axis=0).tolist()
        for k in rows[0]['scores']} for arm in ARMS} for layout in spec['layouts']}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'source_gate_passed': source_gate,
        'layout_fits': fitting, 'cost': {'historical_calls_per_model': 216, 'incremental_historical_calls_per_model': 72,
            'native_mean_fits': sum(f['native_mean_fits'] for f in fitting), 'new_risk_fits': 0, 'new_simulator_calls': 0,
            'new_verification_calls': 0, 'new_terminal_integrations': 0}, 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print({'source_gate_passed': source_gate, 'summary': summary, 'contrasts': contrasts, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
