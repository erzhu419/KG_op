"""Describe the frozen plug-in source noise;do not label it oracle-known."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save

PLAN = 'performance/manifests/legacy_joint_population_f_source_noise_contract_plan_v1.json'


def run():
    started = time.process_time(); spec = read(PLAN); coverage, _ = source_paths(read(spec['source_context_execution']))
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    n = public['plan']['replicates_per_site']; assert n == 3; rows = []
    names = [f'related_inventory_{q:02d}' for q in spec['source_tasks']]; assert 'related_inventory_01' not in names
    for layout in spec['layouts']:
        inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        assert [t['name'] for t in inputs['source_tasks']] == names
        noise = np.concatenate([np.asarray(t['noise'])[:, 0] for t in inputs['source_tasks']])
        assert len(noise) == 60 and np.all(np.isfinite(noise)) and np.all(noise > 0)
        precision = 1 / noise; share = np.sort(precision / precision.sum())[::-1]
        row = {'layout': layout, 'source_task_names': names, 'source_label_count': len(noise), 'replicates_per_label': n,
            'plug_in_mean_noise_minimum': float(noise.min()), 'plug_in_mean_noise_median': float(np.median(noise)),
            'plug_in_mean_noise_maximum': float(noise.max()), 'maximum_to_minimum_noise_ratio': float(noise.max() / noise.min()),
            'maximum_single_label_diagonal_precision_share': float(share[0]), 'top_five_label_diagonal_precision_share': float(share[:5].sum())}
        rows.append(row); print('source noise contract', row, flush=True)
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started,
        'cost': spec['cost'], 'query_outcomes_indexed': False, 'no_model_adopted': True, 'independent_confirmation': False,
        'gaussian_sample_variance_sampling_law': {'degrees_of_freedom': n - 1, 'relative_standard_deviation': float(np.sqrt(2 / (n - 1))),
            'mean_inverse_sample_variance_finite': False, 'assumption': 'Independent Gaussian replicates at fixed latent mean/variance,as in the current batch likelihood.'},
        'source_noise_implementation': 'representation/legacy_function_prior.py:replicated_source_labels;noise[:,0]=sample_variance_f/n',
        'interpretation': 'Actual source noises are estimated from three repeats. Gaussian conditioning is exact given these frozen plug-in values,not the raw source likelihood. Extreme diagonal estimated precision does not equal full-covariance leverage or prove the cause of target errors.'})


if __name__ == '__main__':
    run()
