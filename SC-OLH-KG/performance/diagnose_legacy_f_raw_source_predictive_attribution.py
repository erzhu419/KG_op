"""Cheap mean-error and conditional Monte Carlo attribution from frozen forecasts."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_correlated_source_raw_update import diagnostics
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_raw_source_predictive_attribution_plan_v1.json'


def density_mcse(logs):
    logs = np.asarray(logs); scaled = np.exp(logs - logs.max(axis=(0, 1)))
    influence = (scaled / scaled.mean(axis=(0, 1)) - 1).mean(axis=2)
    diagnostic = diagnostics(influence[:, :, None]); ess = diagnostic['autocorrelation_ess'][0]
    mcse = float(np.sqrt(np.var(influence, ddof=1) / ess))
    return {'average_density_influence_ESS': ess, 'average_density_influence_Rhat': diagnostic['split_rhat'][0],
        'average_log_predictive_density_MCSE': mcse, 'conditional_on_frozen_law_and_scored_batches': True}


def run():
    started = time.process_time(); spec = read(PLAN); source = read(spec['predictions'])
    assert source['all_four_full_state_predictions_complete'] and source['all_original_source_replays_matched']
    records = read(spec['functional_records'])['rows']; rows, comparisons = [], []
    original = read(source['execution']['failed_pilot_results'])['execution']
    coverage, _ = source_paths(read(read(original['source_mean_plan'])['source_context_execution']))
    for row in source['rows']:
        inputs = read(Path(coverage['output']) / f"layout{row['layout']}" / 'task1' / 'inputs.json')
        labels = np.asarray(next(t['response'] for t in inputs['source_tasks'] if t['name'] == 'related_inventory_00'))[:, 0]
        error = np.asarray(row['prediction']['mean']) - labels; level = float(error.mean()); shape = error - level
        record = next(r for r in records if (r['layout'], r['arm']) == (row['layout'], row['arm']))
        rows.append({'layout': row['layout'], 'arm': row['arm'], 'mean_error': level,
            'level_MSE': level ** 2, 'centered_shape_MSE': float(np.mean(shape ** 2)),
            'policy_errors': error.tolist(), 'predictive_functional_gates_passed': row['all_future_functional_gates_passed'],
            'density_MC_error': density_mcse(record['conditional_future_batch_log_densities_by_chain'])})
    for layout in ('5', '6'):
        a, b = [next(r for r in rows if r['layout'] == layout and r['arm'] == arm) for arm in ('joint_spectral_geometry', 'source_gp')]
        score_a, score_b = [next(r for r in source['rows'] if r['layout'] == layout and r['arm'] == arm)['scores'] for arm in ('joint_spectral_geometry', 'source_gp')]
        delta = score_a['f_raw_batch_density'] - score_b['f_raw_batch_density']
        se = float(np.hypot(a['density_MC_error']['average_log_predictive_density_MCSE'], b['density_MC_error']['average_log_predictive_density_MCSE']))
        eligible = a['predictive_functional_gates_passed'] and b['predictive_functional_gates_passed']
        comparisons.append({'layout': layout, 'both_arms_numerically_eligible': eligible,
            'structure_minus_GP_MSE': score_a['f_mse'] - score_b['f_mse'],
            'structure_minus_GP_level_MSE': a['level_MSE'] - b['level_MSE'],
            'structure_minus_GP_centered_shape_MSE': a['centered_shape_MSE'] - b['centered_shape_MSE'],
            'structure_minus_GP_average_raw_density': delta, 'conditional_MCSE_of_density_difference': se,
            'nominal_conditional_MC_interval': [delta - 1.96 * se, delta + 1.96 * se],
            'matched_predictive_gate_passed': bool(score_a['f_mse'] <= score_b['f_mse'] and delta > 0) if eligible else None})
        print('raw source prediction attribution', comparisons[-1], flush=True)
    assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'comparisons': comparisons, 'diagnostic_interpreted': True,
        'at_least_one_numerically_eligible_layout_fails_predictive_gate': any(r['matched_predictive_gate_passed'] is False for r in comparisons),
        'CPU_seconds': time.process_time() - started, 'cost': spec['cost'],
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Level/shape arithmetic and conditional Monte Carlo uncertainty of fixed development density scores. Not independent task confidence intervals,causal attribution,risk factor recovery or a relaxed precision gate. Do not add source draws to rescue a candidate that already loses on eligible layout6.'})


if __name__ == '__main__':
    run()
