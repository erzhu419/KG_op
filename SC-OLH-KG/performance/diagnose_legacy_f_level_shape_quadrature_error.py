"""Attribute source-only errors without reconstructing the full reference again."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_f_task_shape_uncertainty import source_system
from performance.prepare_legacy_f_level_shape_scaled_quadrature import scaled_rule
from performance.run_legacy_related_inventory_source import save
from representation.task_level_shape_reference import TaskLevelShapeReference

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_shape_quadrature_error_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); scaled = read(spec['scaled_execution'])
    reference = read(scaled['reference_execution']); geometry = read(reference['joint_execution'])
    shape = read(geometry['shape_execution']); coverage, profile = source_paths(read(shape['source_context_execution']))
    failed = read(spec['scaled_results'])['rows']; rows = []
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    for fold in spec['folds']:
        layout, deleted = fold['layout'], fold['deleted_task']; suffix = Path(f'layout{layout}')/f'task{deleted}'
        inputs = read(Path(coverage['output'])/suffix/'inputs.json')
        head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
        joint = TaskLevelShapeReference(source_system(inputs, head, deleted)); native = joint.evaluate(joint.native_level, joint.native_shape)
        row = next(r for r in failed if r['layout'] == layout and r['deleted_task'] == deleted)
        with np.load(ROOT/row['reference_moment_vector_file']) as saved:
            context = {'joint': joint, 'native': native, 'vector_size': len(saved['normalized_scaled_moments']),
                'log_scale': float(saved['source_log_scale']), 'reference_normalizer': float(saved['scaled_reference_normalizer']),
                'reference_vector': saved['normalized_scaled_moments'].copy(), 'units': saved['moment_units'].copy(),
                'density_evaluations': 0}
            coordinate_units = saved['source_coordinate_units'].copy()
        rule, errors = scaled_rule(context, coordinate_units, (spec['nodes_per_axis'], spec['nodes_per_axis']))
        difference = rule['normalized_scaled_moments']-context['reference_vector']; d = len(joint.beta00)
        blocks = [('normalization', 0, 1), ('level_first_moment', 1, 2), ('shape_first_moment', 2, 3),
            ('level_second_moment', 3, 4), ('shape_second_moment', 4, 5), ('level_shape_cross_moment', 5, 6),
            ('common_beta', 6, 6+d), ('average_conditional_h', 6+d, 6+d+d*d),
            ('conditional_beta_outer_product', 6+d+d*d, 6+d+2*d*d)]
        block_errors = {name: float(np.max(np.abs(difference[lo:hi]))) for name, lo, hi in blocks}
        index = int(np.argmax(np.abs(difference)))
        largest_block = next(name for name, lo, hi in blocks if lo <= index < hi)
        result = {'layout': layout, 'deleted_task': deleted, 'errors': errors, 'scaled_moment_errors_by_block': block_errors,
            'largest_error_block': largest_block, 'largest_error_index': index,
            'reference_value': float(context['reference_vector'][index]),
            'candidate_value': float(rule['normalized_scaled_moments'][index]),
            'source_density_evaluations': context['density_evaluations']}
        rows.append(result)
        np.savez_compressed(out/f'layout{layout}_task{deleted}_moment_difference.npz', scaled_moment_difference=difference)
        print(result, flush=True)
    save(out/'results.json', {'execution': spec, 'rows': rows, 'cpu_seconds': time.process_time()-started, 'cost': spec['cost'],
        'no_prior_or_model_adopted': True, 'query_paid_outcomes_read': False, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False})


if __name__ == '__main__':
    run()
