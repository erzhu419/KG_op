"""Source-only official GP training in the existing isolated JAX runtime."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_external_models import OfficialHyperBOSurrogate
from baselines.transfer_models import ScalarTaskData


def run():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    started = time.process_time()
    execution = json.loads((ROOT/'performance/manifests/legacy_joint_population_matched_baseline_execution_v1.json').read_text())
    out = ROOT/execution['output']
    data = json.loads((out/'inputs.json').read_text())
    if (out/'source_gp_fit.json').exists():
        print('reuse frozen official source fit', flush=True)
        return
    sources = data['source_tasks']; heads = []
    responses = np.vstack([s['response'] for s in sources])
    for head in range(4):
        head_started = time.process_time()
        center, scale = float(responses[:, head].mean()), float(responses[:, head].std())
        tasks = [ScalarTaskData(s['name'], np.asarray(s['features']),
                    (np.asarray(s['response'])[:, head]-center)/scale,
                    np.asarray(s['noise'])[:, head]/scale**2) for s in sources]
        model = OfficialHyperBOSurrogate(seed=execution['source_gp_seed']+head,
                                        source_steps=execution['source_gp_steps'])
        print('official source head starts', head, 'steps', model.source_steps, flush=True)
        model.meta_fit(tasks)
        _, _, _, gp, _, _, utils = model._modules()
        keys = ['constant', 'lengthscale', 'signal_variance', 'noise_variance']
        warped = gp.retrieve_params(model.params, keys, warp_func=utils.DEFAULT_WARP_FUNC)
        h = {k: np.asarray(v).tolist() for k, v in zip(keys, warped)}
        query = tasks[0].X
        mean, covariance = model.predict(query, full_cov=True)
        delta = (query[:, None]-query[None, :])/np.asarray(h['lengthscale'])
        reconstructed = h['signal_variance']*np.exp(-.5*np.sum(delta**2, axis=2))
        np.testing.assert_allclose(covariance, reconstructed, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(mean, h['constant'], rtol=1e-10, atol=1e-10)
        record = {'head': execution['head_order'][head], 'hyperparameters': h,
            'normalization': {'mean': center, 'scale': scale}, 'diagnostics': model.diagnostics(),
            'kernel_reconstruction_max_error': float(np.max(abs(covariance-reconstructed))),
            'source_label_count': len(responses), 'cpu_seconds': time.process_time()-head_started}
        heads.append(record)
        (out/'source_gp_fit_partial.json').write_text(json.dumps({'heads': heads}, indent=2)+'\n')
        print('official source head complete', head, 'CPU', record['cpu_seconds'], flush=True)
    result = {'heads': heads, 'source_task_ids': [s['name'] for s in sources],
              'new_source_calls': 0, 'cpu_seconds': time.process_time()-started}
    (out/'source_gp_fit.json').write_text(json.dumps(result, indent=2)+'\n')
    print('four official source priors frozen', result['cpu_seconds'], flush=True)


if __name__ == '__main__':
    run()
