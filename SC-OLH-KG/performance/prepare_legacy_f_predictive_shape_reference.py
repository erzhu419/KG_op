"""Rebuild coherent task-intercept uncertainty for the source-predictive shape law."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_f_task_level_reference_integration import diagnose
from performance.prepare_legacy_f_task_level_reference_quadrature import (positive_base_rule, rule_on_intervals,
    approximation_errors, passes)
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_predictive_shape_reference_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); fitted = read(spec['fitted_results'])
    source = read(spec['fitted_laws']); assert fitted['all_fits_complete'] and source['both_laws_frozen_without_outer_query_outcomes']
    coverage, _ = source_paths(read(spec['source_context_execution'])); slices = read(spec['old_slice_results'])['rows']
    records, laws, rules = [], [], []; out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    for native in source['rows']:
        layout, deleted = native['layout'], native['deleted_task']
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
        assert native['source_task_names'] == [t['name'] for t in inputs['source_tasks']]
        assert f'related_inventory_{deleted:02d}' not in native['source_task_names']
        head = {'regularizer': {'coefficient_mean': native['empirical_center_coefficient_mean']}, 'joint_structural': native}
        previous = next(r for r in slices if r['layout'] == layout and r['deleted_task'] == deleted)
        record, law = diagnose(inputs, head, previous, deleted, spec)
        record.update(layout=layout, deleted_task=deleted); law.update(layout=layout, deleted_task=deleted)
        nodes, weights, error, intervals = positive_base_rule(law, record, spec)
        base_errors = approximation_errors(nodes, weights, law, record); base_passed = passes(base_errors, law, record, spec)
        trials, selected = [], None
        if base_passed:
            for size in spec['nodes_per_subinterval_resolutions']:
                variances, probabilities = rule_on_intervals(law, intervals, size)
                errors = approximation_errors(variances, probabilities, law, record); passed = passes(errors, law, record, spec)
                trials.append({'nodes_per_interval': size, 'node_count': len(variances), 'errors': errors, 'passed': passed})
                if passed:
                    selected = {'layout': layout, 'deleted_task': deleted, 'variance_nodes': variances.tolist(),
                        'posterior_weights': (probabilities/probabilities.sum()).tolist(), 'source_reference_law': law,
                        'adaptive_u_intervals': intervals, 'nodes_per_subinterval': size}
                    rules.append(selected); break
        records.append({**record, 'quadrature': {'base_errors': base_errors, 'base_passed': base_passed,
            'adaptive_vector_error': error, 'trials': trials, 'selected_node_count': len(selected['variance_nodes']) if selected else None}})
        laws.append(law); print('predictive shape source reference', layout, 'variance mean', record['reference_posterior']['mean'],
                                  'selected nodes', len(selected['variance_nodes']) if selected else None, flush=True)
    complete = len(rules) == len(source['rows']) == 2
    save(out/'source_laws.json', {'execution': spec, 'both_source_laws_frozen_without_query_data': True, 'rows': laws})
    save(out/'source_quadrature.json', {'execution': spec, 'both_source_rules_frozen_without_query_data': complete, 'rows': rules})
    result = {'execution': spec, 'rows': records, 'both_source_integrals_and_rules_verified': complete,
        'cpu_seconds': time.process_time()-started, 'cost': {**spec['cost'], 'actual_source_variance_integrations': len(records)},
        'no_model_adopted': True, 'outer_query_paid_or_score_outcomes_read': False, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result)


if __name__ == '__main__':
    run()
