"""Repair only unresolved quadrature accuracy for the frozen Gaussian controls."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_exact_raw_filter import prepare_cases, prediction
from performance.check_legacy_f_reference_training_budget import errors
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.exact_gaussian_raw_posterior import quadrature, moments, future_batch_density
from performance.run_legacy_related_inventory_source import save
from performance.validate_legacy_paired_taskout import gaussian_score

EXECUTION = sys.argv[1] if len(sys.argv) > 1 else 'performance/manifests/legacy_joint_population_f_exact_raw_density_refinement_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); original = read(spec['original_results'])
    assert not original['diagnostic_interpreted']
    assert all(r['moment_quadrature_verified'] for r in original['rows'])
    cases, frozen, coverage = prepare_cases(original['execution'])
    previous = read(spec['previous_results'])
    retained = [r for r in previous['rows'] if r['density_quadrature_verified']]
    done = {(r['layout'], r['design'], r['arm']) for r in retained}
    cases = [c for c in cases if tuple(c['identity'][k] for k in ('layout', 'design', 'arm')) not in done]
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    base = {k: original[k] for k in ('no_model_adopted', 'source6_7_evaluated', 'target_promoted',
        'independent_confirmation', 'no_new_unscored_outcomes_read', 'training_diagnostic_only')}
    base.update(execution=spec, cost=spec['cost'])
    save(out / 'frozen_inputs.json', {'execution': spec, 'same_eight_prior_design_identities': frozen})
    raw = read(coverage['source_data'])
    paid = {(c['identity']['layout'], c['identity']['design']): [raw['tasks'][1]['Y_replicates'][j][0][0] for j in c['identity']['paid_prefix']] for c in cases}
    del raw
    rows = [{**r, 'reused_verified_density_row': True} for r in retained]; new_rows = []
    for case in cases:
        identity, prior = case['identity'], case['prior']
        values = paid[(identity['layout'], identity['design'])]
        case['grids'] = {n: quadrature(prior, values, n) for n in spec['paid_orders']}
        predictions = {n: prediction(moments(prior, g), g) for n, g in case['grids'].items()}
        moment_errors = errors(predictions[spec['paid_orders'][0]], predictions[spec['paid_orders'][1]])
        assert all(moment_errors[k] <= v for k, v in spec['prediction_tolerances'].items())
        before = next(r for r in original['rows'] if all(r[k] == identity[k] for k in ('layout', 'design', 'arm')))
        reproduced = errors(predictions[spec['paid_orders'][0]], before['refined_prediction'])
        assert all(reproduced[k] <= v for k, v in spec['prediction_tolerances'].items())
        new_rows.append({**identity, 'prediction': predictions[spec['paid_orders'][0]],
            'refined_prediction': predictions[spec['paid_orders'][1]], 'quadrature_errors': moment_errors,
            'original_moment_reproduction_errors': reproduced,
            'single_observation_errors': before['single_observation_errors']})
    rows += new_rows
    assert len(rows) == 8
    save(out / 'predictions.json', {**base, 'all_eight_refined_moments_frozen_before_score_labels': True, 'rows': rows})
    for row, case in zip(new_rows, cases):
        if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']: raise RuntimeError('registered density-refinement CPU budget reached')
        labels = read(Path(coverage['output']) / f"layout{row['layout']}" / 'inputs.json')['source_tasks'][1]
        observed = np.asarray(labels['response']); noise = np.asarray(labels['noise'])[:, 0]
        densities = [future_batch_density(case['prior'], case['grids'][n], observed, f)
            for n, f in zip(spec['paid_orders'], spec['future_orders'])]
        row['density_refinement_error'] = float(np.max(np.abs(densities[0] - densities[1])))
        row['density_quadrature_verified'] = row['density_refinement_error'] <= spec['density_refinement_tolerance']
        p = row['prediction']; moment = gaussian_score(np.asarray(p['mean']), np.asarray(p['covariance']) + np.diag(noise), observed[:, 0]) / 12
        scores = {'f_mse': float(np.mean((np.asarray(p['mean']) - observed[:, 0]) ** 2)),
            'f_raw_batch_density': float(np.mean(densities[0])), 'f_gaussian_mean_density': moment,
            'raw_paid_log_evidence': p['paid_predictive_log_score'],
            'sequential_transcript_density': (p['paid_predictive_log_score'] + 12 * moment) / 16}
        row['scores'] = scores; row['exact_minus_projection'] = {k: v - case['baseline']['scores'][k] for k, v in scores.items()}
        row['mse_relative_change'] = scores['f_mse'] / case['baseline']['scores']['f_mse'] - 1
        row['prediction_difference_from_projection'] = errors(p, case['baseline']['prediction'])
        save(out / 'partial_results.json', {**base, 'rows': rows})
        print('refined exact density', row['layout'], row['design'], row['arm'], 'error', row['density_refinement_error'],
            'passed', row['density_quadrature_verified'], 'MSE change', row['mse_relative_change'], flush=True)
    save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': all(r['density_quadrature_verified'] for r in rows),
        'cpu_seconds': time.process_time() - started, 'interpretation': original['interpretation'],
        'original_unresolved_density_record_retained': True, 'single_observation_checks_reused': True})


if __name__ == '__main__':
    run()
