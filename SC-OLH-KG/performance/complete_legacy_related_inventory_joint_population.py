"""Numerical completion of only the three iteration-capped source heads."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.run_legacy_related_inventory_population_mean import fit_arms, run
from performance.run_legacy_related_inventory_source import save
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.source_joint_population import continue_joint_source_population, same_source_full_residual_prior


def restore(basis, diagnostics):
    covariance = np.asarray(diagnostics['population_mean_covariance'])+np.diag(diagnostics['residual_coefficient_diagonal'])
    return SourceFunctionPrior(basis, np.asarray(diagnostics['coefficient_mean']), covariance,
                               diagnostics['discrepancy_variance'], dict(diagnostics))


def run_completion():
    protocol_path = 'performance/manifests/legacy_related_inventory_joint_population_completion_v1.json'
    execution = json.loads((ROOT/protocol_path).read_text())
    folder = ROOT/execution['pilot']
    cached_folds = json.loads((folder/'source_preflight.json').read_text())['records']
    cached_final = json.loads((folder/'source_fit.json').read_text())
    counts = {'continued_source_heads': 0, 'reused_converged_joint_heads': 0}

    def fit_completed_arms(tasks, descriptors, spec):
        names = [t['name'] for t in tasks]
        cached = cached_final if names == cached_final['training_task_ids'] else next(
            f for f in cached_folds if f['training_task_ids'] == names)
        coordinate, previous = fit_arms(tasks, descriptors, spec)
        transformed = coordinate.transform(descriptors)
        adapted, offset = [], 0
        for task in tasks:
            n = len(task['response'])
            adapted.append(dict(task, psi=transformed[offset:offset+n])); offset += n
        joint = []
        for h, regularizer in enumerate(previous['zero_mean_structural']):
            d = cached['fit_diagnostics']['joint_population_structural'][h]
            prior = restore(regularizer.basis, d)
            if not d['optimizer_converged']:
                prior = continue_joint_source_population(regularizer, head_batches(adapted, h),
                    [t['noise'][:, h] for t in adapted], prior, max_iterations=4096)
                counts['continued_source_heads'] += 1
                print('source completion', names, h, 'converged', prior.diagnostics['optimizer_converged'],
                    'iterations', prior.diagnostics['optimizer_iterations'], 'projected gradient',
                    prior.diagnostics['optimizer_projected_gradient_max'], flush=True)
            else:
                counts['reused_converged_joint_heads'] += 1
            joint.append(prior)
        return coordinate, {'common_mean_structural': previous['common_mean_structural'],
            'joint_population_structural': joint,
            'joint_population_full': [same_source_full_residual_prior(p) for p in joint]}

    result = run(protocol_path, fit_completed_arms)
    result['cost'].update(counts)
    result['cost']['native_source_head_fits'] += counts['continued_source_heads']
    save(ROOT/execution['output']/'results.json', result)
    print(json.dumps({'completion_cost': result['cost']}), flush=True)


if __name__ == '__main__':
    run_completion()
