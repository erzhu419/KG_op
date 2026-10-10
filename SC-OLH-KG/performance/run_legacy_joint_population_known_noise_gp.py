"""Measurement-aware generic GP on exactly the same cached target contract."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.run_legacy_joint_population_matched_baselines import evaluate_baselines, gp_kernel
from performance.run_legacy_related_inventory_source import save
from representation.source_known_noise_gp import fit_known_noise_gp


def run():
    started = time.process_time()
    execution = json.loads((ROOT/'performance/manifests/legacy_joint_population_known_noise_gp_v1.json').read_text())
    inputs = json.loads((ROOT/execution['inputs']).read_text())
    out = ROOT/execution['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'source_fit.json'
    if path.exists():
        fit = json.loads(path.read_text())
    else:
        heads = []
        for h in range(4):
            p = fit_known_noise_gp(inputs['source_tasks'], h)
            heads.append(p)
            print('known-noise source head', h, 'converged', p['optimizer_converged'],
                  'signal', p['signal_variance'], 'discrepancy', p['discrepancy_variance'], flush=True)
        fit = {'heads': heads, 'cpu_seconds': time.process_time()-started,
               'source_task_ids': [t['name'] for t in inputs['source_tasks']]}
        save(path, fit)
    def prior(target):
        x = np.asarray(target['source_features']); heads = []
        for d in fit['heads']:
            center, scale = d['normalization']['mean'], d['normalization']['scale']
            mean = center+scale*d['constant']
            covariance = gp_kernel(x, d['lengthscale'], d['signal_variance'])
            covariance += d['constant_variance']*np.ones((len(x), len(x)))
            covariance += d['discrepancy_variance']*np.eye(len(x))
            heads.append((np.full(len(x), mean), scale**2*covariance))
        return heads
    evaluate_baselines(execution, inputs, {execution['arms'][0]: prior}, fit['cpu_seconds'], 4)


if __name__ == '__main__':
    run()
