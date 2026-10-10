"""Post-selection input-transfer attribution, without reranking or integration."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import components, encode_diverse_source
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis


def run():
    started = time.process_time()
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_diverse_source_decision_v1.json').read_text())
    scored = json.loads((ROOT/spec['output']/'results.json').read_text())
    source = json.loads((ROOT/spec['source_results']).read_text())['execution']
    inputs = json.loads((ROOT/spec['source_inputs']).read_text()); fits = json.loads((ROOT/spec['source_fit']).read_text())
    raw = json.loads((ROOT/source['source_data']).read_text())
    public = json.loads((ROOT/source['source_data']).with_name('frozen_inputs.json').read_text())
    bank50 = [tuple(p) for p in public['bank']]
    desc = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank50, 50)
            for i in source['training_tasks']}
    coordinate, _, center, scale = encode_diverse_source(raw, source['training_tasks'],
        source['source_site_indices_by_task'], desc, public['plan']['task_distribution']['L'])
    source_psi = np.asarray(inputs['public_reference_psi']); basis = OrthogonalLowOrderBasis(source_psi)
    target_inputs = json.loads((ROOT/spec['target_inputs']).read_text()); bank = [tuple(p) for p in target_inputs['bank']]
    n = len(bank); rows = []
    for target in target_inputs['targets']:
        provider = no_outcomes(target_problem(target_inputs['target_plan'], target['config']))
        psi = coordinate.transform(legacy_exposure_descriptors(provider, bank, 50))
        model = JointResponseMixture(spec['arms'], components(inputs, fits, np.vstack([psi, source_psi])))
        prior_means = {arm: g.predict()[0].copy() for arm, (f, g) in zip(spec['arms'], model.components)}
        for j, value in zip(target['initialization']['candidate_indices'], target['initialization']['paired_observations']):
            model.update_raw(j, [value[0], value[1]-provider.tau])
        for arm in ('joint_common_structural', 'joint_common_rbf'):
            f, g = model.components[spec['arms'].index(arm)]; fm, fv = f.predict(); gm, gv = g.predict()
            before = next(r for r in scored['rows'] if r['task_id'] == target['task_id'] and r['arm'] == arm)
            points = list(dict.fromkeys([before['action']['primary'], 27, 63]))
            for j in points:
                frozen = next(p for p in before['forecast']['diagnostic_points'] if p['candidate'] == j)
                np.testing.assert_allclose(np.r_[fm[j], gm[j]], frozen['mean'], rtol=1e-8, atol=1e-9)
                np.testing.assert_allclose(np.sqrt(np.r_[fv[j], gv[j]]), frozen['sd'], rtol=1e-8, atol=1e-9)
                truth = np.asarray(frozen['scoring_truth'])
                target_g, source_g = gm[j, 0], gm[n+j, 0]
                rows.append({'task_id': target['task_id'], 'arm': arm, 'candidate': j,
                    'selected_primary': j == before['action']['primary'],
                    'target_input_g_mean': float(target_g), 'source_input_same_paid_state_g_mean': float(source_g),
                    'target_input_prior_g_mean': float(prior_means[arm][j, 0]),
                    'source_input_prior_g_mean': float(prior_means[arm][n+j, 0]),
                    'paid_update_g_mean_shift': float(target_g-prior_means[arm][j, 0]),
                    'source_prior_to_target_truth_bias': float(prior_means[arm][n+j, 0]-truth[2]),
                    'prior_input_transfer_mean_shift': float(prior_means[arm][j, 0]-prior_means[arm][n+j, 0]),
                    'input_transfer_mean_shift': float(target_g-source_g), 'target_true_g': float(truth[2]),
                    'target_g_bias': float(target_g-truth[2]),
                    'remaining_bias_at_source_input': float(source_g-truth[2]),
                    'standardized_input_distance': float(np.linalg.norm((psi[j]-source_psi[j])/scale)),
                    'source_basis_feature_norm': float(np.linalg.norm(basis.transform(source_psi[[j]]))),
                    'target_basis_feature_norm': float(np.linalg.norm(basis.transform(psi[[j]]))),
                    'target_mean_eta_plugin_margin': float(target_g+norm.ppf(1-provider.alpha)*np.exp(gm[j, 1]/2)),
                    'true_chance_margin': float(truth[2]+norm.ppf(1-provider.alpha)*np.exp(truth[3]/2)),
                    'target_eta_g_bias': float(gm[j, 1]-truth[3])})
    summaries = {}
    for arm in ('joint_common_structural', 'joint_common_rbf'):
        selected = [r for r in rows if r['arm'] == arm and r['selected_primary']]
        summaries[arm] = {k: float(np.mean([r[k] for r in selected])) for k in
            ('target_g_bias', 'input_transfer_mean_shift', 'remaining_bias_at_source_input',
             'source_prior_to_target_truth_bias', 'prior_input_transfer_mean_shift', 'paid_update_g_mean_shift',
             'target_mean_eta_plugin_margin', 'true_chance_margin', 'target_eta_g_bias')}
    result = {'rows': rows, 'selected_primary_summary': summaries,
        'interpretation': 'same target paid state, query-only change to the corresponding public source-policy input; diagnostic comparison is not a target correction or new action. Source/target policies may differ after dimension resampling. Mean-eta margins are descriptive.',
        'cost': {'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_native_source_fits': 0,
                 'new_terminal_integrations': 0}, 'cpu_seconds': time.process_time()-started}
    labels = training_view(raw, source['training_tasks'], public['plan']['task_distribution']['L'])
    source_points = []
    for j in sorted({r['candidate'] for r in rows}):
        tasks = [{'task_id': t['name'], 'selected_for_training': j in sites,
            'observed_g_mean': float(t['response'][j, 1]),
            'g_mean_label_standard_error': float(np.sqrt(t['noise'][j, 1]))}
            for t, sites in zip(labels, source['source_site_indices_by_task'])]
        selected = [t for t in tasks if t['selected_for_training']]
        source_points.append({'candidate': j, 'selected_source_task_count': len(selected),
            'observed_g_mean_all_training_tasks': float(np.mean([t['observed_g_mean'] for t in tasks])),
            'observed_g_mean_selected_tasks': float(np.mean([t['observed_g_mean'] for t in selected])), 'tasks': tasks})
    save(ROOT/spec['output']/'source_point_observation_diagnostic.json', {'rows': source_points,
        'interpretation': 'post-freeze observations only; unselected labels never enter source fits. Noisy task labels are not population truth.',
        'new_fits': 0, 'new_simulator_calls': 0})
    save(ROOT/spec['output']/'input_transfer_diagnostic.json', result)
    print(json.dumps({'selected_primary_summary': summaries, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
