"""Separate source-risk location, input transfer and paid-update budget effects."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments


def read(path):
    return json.loads((ROOT/path).read_text())


def run():
    started = time.process_time()
    scoring = read('paper_artifacts/legacy_joint_population_independent_pilot_v1/results.json')
    old_risk_inputs = read('paper_artifacts/legacy_joint_population_diverse_source_v1/inputs.json')
    old_risk_fit = read('paper_artifacts/legacy_joint_population_diverse_source_v1/source_fit.json')
    new_risk_inputs = read('paper_artifacts/legacy_joint_population_public_mean_source_half_v1/inputs.json')
    new_risk_fit = read('paper_artifacts/legacy_joint_population_public_mean_source_half_v1/risk_source_fit.json')
    budgets = [
        (288, old_risk_inputs['public_reference_psi'], [old_risk_fit['heads'][h] for h in (2, 3)],
         read('paper_artifacts/legacy_joint_population_public_mean_v1/inputs.json'),
         read('paper_artifacts/legacy_joint_population_public_mean_v1/target_calibration_profiles.json')),
        (144, new_risk_inputs['public_risk_reference'], new_risk_fit['heads'],
         read('paper_artifacts/legacy_joint_population_public_mean_source_half_target_v1/inputs.json'),
         read('paper_artifacts/legacy_joint_population_public_mean_source_half_target_v1/calibration_profiles.json'))]
    rows = []
    for calls, reference, heads, target_inputs, predictions in budgets:
        basis = OrthogonalLowOrderBasis(reference)
        for k, (name, forecast_column, truth_column) in enumerate((('eta_g', 2, 3), ('eta_f', 3, 1))):
            prior = restore_joint(basis, heads[k]['joint_structural'])
            source_mean = source_prior_moments(prior, np.asarray(reference))[0]
            for target in target_inputs['targets']:
                target_mean = source_prior_moments(prior, np.asarray(target['risk_psi']))[0]
                row = next(r for r in predictions['rows'] if r['task_id'] == target['task_id']
                           and r['arm'] == 'public_mean_joint_structural')
                posterior = np.asarray(row['mean'])[:len(reference), forecast_column]
                fold = next(f for f in scoring['folds'] if f['task_id'] == target['task_id'])
                truth = np.asarray(fold['scoring_truth'])[:, truth_column]
                for j in (27, 60, 63):
                    rows.append({'source_calls': calls, 'task_id': target['task_id'], 'head': name,
                        'candidate': j, 'source_input_prior_to_target_truth_bias': float(source_mean[j]-truth[j]),
                        'prior_query_input_shift': float(target_mean[j]-source_mean[j]),
                        'paid_update_shift': float(posterior[j]-target_mean[j]),
                        'posterior_bias': float(posterior[j]-truth[j])})
    summary = {str(calls): {name: {str(j): {metric: float(np.mean([r[metric] for r in rows
        if r['source_calls'] == calls and r['head'] == name and r['candidate'] == j])) for metric in
        ('source_input_prior_to_target_truth_bias', 'prior_query_input_shift', 'paid_update_shift', 'posterior_bias')}
        for j in (27, 60, 63)} for name in ('eta_g', 'eta_f')} for calls in (288, 144)}
    result = {'scope': 'opened-target attribution of previously frozen priors and paid forecasts; first term includes task and policy differences, not a pure representation effect',
        'rows': rows, 'summary': summary, 'new_source_fits': 0, 'new_simulator_calls': 0,
        'new_verification_calls': 0, 'new_terminal_integrations': 0,
        'cpu_seconds': time.process_time()-started}
    save(ROOT/'paper_artifacts/legacy_joint_population_public_mean_source_half_target_v1/risk_budget_diagnostic.json', result)
    print(json.dumps({'summary': summary, 'cpu_seconds': result['cpu_seconds']}), flush=True)


if __name__ == '__main__':
    run()
