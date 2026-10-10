"""Freeze cheap target forecasts for the valid144-call source fit."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import encode_diverse_source
from performance.check_legacy_public_mean_coordinates import forecast_rows
from performance.check_legacy_public_mean_source_half import components
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.public_control_mean import public_control_mean_coordinates


def run():
    started = time.process_time()
    spec = json.loads((ROOT/'performance/manifests/legacy_joint_population_public_mean_source_half_target_v1.json').read_text())
    source = json.loads((ROOT/spec['source_execution']).read_text())
    inputs = json.loads((ROOT/spec['source_inputs']).read_text())
    means = json.loads((ROOT/spec['mean_fit']).read_text()); risks = json.loads((ROOT/spec['risk_fit']).read_text())
    data = json.loads((ROOT/source['source_data']).read_text())
    public = json.loads((ROOT/source['source_data']).with_name('frozen_inputs.json').read_text())
    source_bank = [tuple(p) for p in public['bank']]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], public['tasks'][i])),
        source_bank, 50) for i in source['training_tasks']}
    coordinate, _, center, scale = encode_diverse_source(data, source['training_tasks'],
        source['source_site_indices_by_task'], descriptors, public['plan']['task_distribution']['L'])
    # The target map must be the one used by the cached source risk priors.
    np.testing.assert_array_equal(center, inputs['risk_coordinate_center'])
    np.testing.assert_array_equal(scale, inputs['risk_coordinate_scale'])
    target_inputs = json.loads((ROOT/spec['target_inputs']).read_text())
    bank = [tuple(p) for p in target_inputs['bank']]; rows, targets = [], []
    for i, target in enumerate(target_inputs['targets']):
        provider = no_outcomes(target_problem(target_inputs['target_plan'], target['config']))
        mean_psi = (public_control_mean_coordinates(provider, bank)-np.asarray(inputs['mean_coordinate_center']))/np.asarray(inputs['mean_coordinate_scale'])
        risk_psi = coordinate.transform(legacy_exposure_descriptors(provider, bank, 50))
        model = JointResponseMixture(spec['arms'], components(inputs, means, risks, mean_psi, risk_psi))
        for j, value in zip(target['initialization']['candidate_indices'], target['initialization']['paired_observations']):
            model.update_raw(j, [value[0], value[1]-provider.tau])
        rows.extend(forecast_rows(model, target['task_id'], i))
        targets.append({**target, 'mean_psi': mean_psi.tolist(), 'risk_psi': risk_psi.tolist(), 'alpha': provider.alpha})
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'inputs.json', {**target_inputs, 'execution': spec, 'targets': targets})
    save(out/'predictions.json', {'execution': spec, 'rows': rows})
    # All18 local forecasts above precede cached target-truth scoring.
    scoring = json.loads((ROOT/spec['target_scoring']).read_text())
    reference = json.loads((ROOT/spec['reference_profiles']).read_text()); diagnostics, contrasts = [], []
    for row in rows:
        target = targets[row['task_index']]
        fold = next(f for f in scoring['folds'] if f['task_id'] == row['task_id'])
        truth = np.asarray(fold['scoring_truth'])[:, [0, 2, 3, 1]]
        mean, variance = np.asarray(row['mean']), np.asarray(row['variance']); error = mean-truth
        row['scores'] = {'head_order': ['f', 'g', 'eta_g', 'eta_f'],
            'truth_mse': (error[2:]**2).mean(axis=0).tolist(),
            'moment_interval_95_truth_coverage': (abs(error[2:]) <= 1.96*np.sqrt(variance[2:])).mean(axis=0).tolist()}
        old = next(r for r in reference['rows'] if r['task_id'] == row['task_id'] and r['arm'] == row['arm'])
        contrasts.append({'task_id': row['task_id'], 'arm': row['arm'],
            'scores_144': row['scores'], 'scores_288': old['scores']})
        for j in spec['diagnostic_candidates']:
            diagnostics.append({'task_id': row['task_id'], 'arm': row['arm'], 'candidate': j,
                'g_bias': float(error[j, 1]), 'eta_g_bias': float(error[j, 2]),
                'g_mean': float(mean[j, 1]), 'g_sd': float(np.sqrt(variance[j, 1])),
                'mean_eta_plugin_margin': float(mean[j, 1]+norm.ppf(1-target['alpha'])*np.exp(mean[j, 2]/2)),
                'true_chance_margin': float(truth[j, 1]+norm.ppf(1-target['alpha'])*np.exp(truth[j, 2]/2)),
                'true_feasible': j in fold['feasible_policies']})
    summary = {a: {'truth_mse': np.mean([r['scores']['truth_mse'] for r in rows if r['arm'] == a], axis=0).tolist(),
        'moment_interval_95_truth_coverage': np.mean([r['scores']['moment_interval_95_truth_coverage'] for r in rows if r['arm'] == a], axis=0).tolist(),
        'points': {str(j): {k: float(np.mean([r[k] for r in diagnostics if r['arm'] == a and r['candidate'] == j]))
            for k in ('g_bias', 'eta_g_bias', 'mean_eta_plugin_margin', 'true_chance_margin')}
            for j in spec['diagnostic_candidates']}} for a in spec['arms']}
    result = {'execution': spec, 'summary': summary, 'budget_contrasts': contrasts, 'diagnostic_rows': diagnostics,
        'cost': {'new_source_calls': 0, 'new_target_calls': 0, 'new_verification_calls': 0,
            'new_source_hyperparameter_fits': 0, 'source_encoder_reconstructions': 1, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'calibration_profiles.json', {'rows': [{k: v for k, v in r.items() if k != 'local_joint_components'} for r in rows]})
    save(out/'results.json', result)
    print(json.dumps({'summary': summary, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
