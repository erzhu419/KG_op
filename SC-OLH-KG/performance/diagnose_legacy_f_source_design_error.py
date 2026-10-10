"""Attribute remaining error after the frozen source-derived design pilot."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_level_shape_mean_error import decompose
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_source_design_error_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); pilot = read(spec['pilot_results'])
    assert pilot['scientific_gate_interpreted'] and pilot['numerical_density_verified']
    coverage, _ = source_paths(read(pilot['execution']['source_context_execution'])); rows = []
    for layout in pilot['execution']['layouts']:
        deleted = pilot['execution']['deleted_task']
        labels = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks'][deleted]
        observed = np.asarray(labels['response'])[:, 0]
        breakdown = {f"{row['arm']}/{row['design']}": decompose(row['prediction']['mean'], observed)
                     for row in pilot['rows'] if row['layout'] == layout}
        candidate = breakdown['source_variance_mixture/intercept_mixture_greedy']
        contrasts = {}
        for name in ['source_variance_mixture/fixed_public_prefix', 'source_gp/intercept_mixture_greedy', 'source_gp/source_gp_greedy']:
            reference = breakdown[name]
            contrasts[name] = {key: candidate[key]-reference[key] for key in ('mse', 'level_mse', 'centered_shape_mse')}
        rows.append({'layout': layout, 'deleted_task': deleted, 'breakdown': breakdown, 'candidate_minus_reference': contrasts,
            'label_noise_level_variance': float(np.sum(np.asarray(labels['noise'])[:, 0])/len(observed)**2)})
        print({'layout': layout, 'candidate': candidate, 'contrasts': contrasts}, flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'results.json', {'execution': spec, 'rows': rows, 'cpu_seconds': time.process_time()-started, 'cost': spec['cost'],
        'new_query_values_read': False, 'no_model_adopted': True, 'source6_7_evaluated': False, 'target_promoted': False,
        'independent_confirmation': False, 'observed_label_decomposition_not_oracle_recovery': True})


if __name__ == '__main__':
    run()
