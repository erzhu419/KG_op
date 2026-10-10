"""Cached common-function-mean GP attribution under the stage20 contract."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.run_legacy_joint_population_matched_baselines import evaluate_baselines
from performance.run_legacy_related_inventory_source import save
from representation.source_kernel_population import fit_kernel_population, predict_kernel_population


def run():
    started = time.process_time()
    execution = json.loads((ROOT/'performance/manifests/legacy_joint_population_kernel_mean_execution_v1.json').read_text())
    inputs = json.loads((ROOT/execution['inputs']).read_text())
    out = ROOT/execution['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'source_fit.json'
    if path.exists():
        fit = json.loads(path.read_text())
    else:
        anchors = json.loads((ROOT/execution['source_kernel_regularizer']).read_text())
        heads = []
        for h, anchor in enumerate(anchors['heads']):
            p = fit_kernel_population(inputs['source_tasks'], h, anchor, optimizer=execution['optimizer'])
            heads.append(p)
            print('joint kernel source head', h, 'converged', p['optimizer_converged'],
                  'residual', p['residual_parameters'], flush=True)
        fit = {'heads': heads, 'cpu_seconds': time.process_time()-started,
               'source_task_ids': [t['name'] for t in inputs['source_tasks']],
               'source_features': inputs['source_tasks'][0]['features']}
        save(path, fit)
    if not all(h['optimizer_converged'] for h in fit['heads']):
        raise RuntimeError('recorded source fits need source-only completion before target comparison')
    def prior(target):
        return [predict_kernel_population(fit['source_features'], target['source_features'], head)
                for head in fit['heads']]
    evaluate_baselines(execution, inputs, {execution['arms'][0]: prior}, fit['cpu_seconds'], 4)


if __name__ == '__main__':
    run()
