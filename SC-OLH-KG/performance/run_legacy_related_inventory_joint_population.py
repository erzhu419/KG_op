"""Joint population fit under the same source-first development contract."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance.run_legacy_related_inventory_population_mean import fit_arms, run
from representation.source_function_prior import head_batches
from representation.source_joint_population import fit_joint_source_population, same_source_full_residual_prior


def fit_joint_arms(tasks, descriptors, spec):
    coordinate, previous = fit_arms(tasks, descriptors, spec)
    transformed = coordinate.transform(descriptors)
    adapted, offset = [], 0
    for task in tasks:
        n = len(task['response'])
        adapted.append(dict(task, psi=transformed[offset:offset+n])); offset += n
    joint = []
    for h, regularizer in enumerate(previous['zero_mean_structural']):
        prior = fit_joint_source_population(regularizer, head_batches(adapted, h),
            [t['noise'][:, h] for t in adapted], max_iterations=spec['source_max_iterations'])
        joint.append(prior)
        print('joint source head', h, 'converged', prior.diagnostics['optimizer_converged'],
              'iterations', prior.diagnostics['optimizer_iterations'],
              'discrepancy', prior.discrepancy_variance, flush=True)
    return coordinate, {'common_mean_structural': previous['common_mean_structural'],
        'joint_population_structural': joint,
        'joint_population_full': [same_source_full_residual_prior(p) for p in joint]}


if __name__ == '__main__':
    run('performance/manifests/legacy_related_inventory_joint_population_execution_v1.json', fit_joint_arms)
