"""Post-verification risk bias and propagation decisions on new task geometries."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from performance.run_legacy_related_inventory_source import problem_for

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'paper_artifacts/mainline_d1000_joint_risk_v1'


def read(path):
    return json.loads(path.read_text())


def summarize(output=OUTPUT):
    started = time.process_time()
    output = ROOT/Path(output)
    result = read(output/'results.json')
    assert result['all_recommendations_frozen_before_verification']
    assert result['all_verification_finished_before_truth_scoring']
    frozen = read(output/'frozen_inputs.json')
    original = frozen['dynamic_candidates']['original_count']
    rows = []
    for task_index, task in enumerate(frozen['tasks']):
        problem = problem_for(frozen['problem_plan'], task)
        truth_g = np.asarray([problem.true_constraint_mean(tuple(p))-problem.tau
                              for p in frozen['bank']])
        for arm in frozen['plan']['models']:
            case = read(output/f'task{task_index:02d}'/arm/'case.json')
            observed = set(frozen['plan']['initialization_indices'])
            observed.update(r['decision']['query'] for r in case['rounds'])
            indices = [i for i in range(original, len(truth_g)) if i not in observed]
            stages = [('initial', case['initial']), ('final', case['rounds'][-1]['after'])]
            for stage, snapshot in stages:
                error = np.asarray(snapshot['mean'])[indices, 2]-truth_g[indices]
                rows.append({'task_index': task_index, 'arm': arm, 'phase': stage,
                    'indices': indices, 'point_count': len(indices),
                    'mean_signed_g_error': float(error.mean()),
                    'mean_optimistic_error': float(np.maximum(-error, 0).mean()),
                    'optimistic_point_fraction': float(np.mean(error < 0)),
                    'g_mse': float(np.mean(error**2))})
    intervention = []
    for arm in frozen['plan']['models']:
        for round_index in (1, 2):
            selected = [r for r in result['unsampled_information_decision_diagnostics']
                        if r['arm'] == arm and r['round'] == round_index]
            delta = np.asarray([r['full_conditional_safe_reward']-
                                r['held_unsampled_conditional_safe_reward'] for r in selected])
            intervention.append({'arm': arm, 'round': round_index,
                'mean_conditional_difference': float(delta.mean()),
                'per_task_difference': delta.tolist(),
                'positive_tasks': int(np.sum(delta > 1e-12)),
                'negative_tasks': int(np.sum(delta < -1e-12)),
                'changed_recommendations': sum(r['recommendation_changed_by_unsampled_updates']
                                               for r in selected)})
    report = {'pairing_unit': frozen['plan']['pairing_unit'],
        'risk_error_sign': 'prediction minus truth; negative is optimistic',
        'scope': 'Within each arm, initial/final use identical final-unsampled generated policies.',
        'risk_bias_rows': rows, 'all_unsampled_terminal_interventions': intervention,
        'interpretation': 'Conditional interventions use the actual observation paths; '
            'they are not additional independently verified optimizer arms.',
        'cpu_seconds': time.process_time()-started}
    (output/'post_verification_risk_bias.json').write_text(
        json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(OUTPUT))
    print(json.dumps(summarize(parser.parse_args().output), indent=2, allow_nan=False))
