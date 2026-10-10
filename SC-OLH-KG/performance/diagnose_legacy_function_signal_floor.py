"""Actual source function scale and inherited weak-tail covariance budget."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import head_batches
from representation.task_bank_centering import CenteredTaskPopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_function_signal_floor_audit_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    coverage = read(centering['coverage_execution']); profile = read(centering['profile_execution']); rows = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            means = read(Path(profile['output'])/f'layout{layout}'/f'task{deleted}'/'selected_source_fit.json')
            basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            for h, head in enumerate(spec['heads']):
                law = (read(Path(centering['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
                    if head == 'g' else means['heads'][h]['joint_structural'])
                centers = [np.asarray(law['source_bank_centers'][t['name']]) if head == 'g' else np.zeros(basis.feature_dim) for t in tasks]
                prior = restore_regularizer(basis, means['heads'][h]['regularizer'], tasks, h)
                likelihood = CenteredTaskPopulationLikelihood(prior, head_batches(tasks, h), [t['noise'][:, h] for t in tasks], centers)
                _, _, state = likelihood.evaluate(np.asarray(law['optimizer_parameters']))
                for key, saved in (('mean', 'coefficient_mean'), ('mean_covariance', 'population_mean_covariance'),
                    ('residual_diagonal', 'residual_coefficient_diagonal'), ('discrepancy', 'discrepancy_variance')):
                    np.testing.assert_allclose(state[key], law[saved], atol=1e-8, rtol=1e-8)
                corrected = np.array([np.var(t['response'][:, h])-np.mean(t['noise'][:, h]) for t in tasks])
                original_signal = float(np.mean(np.maximum(corrected, .02)))
                data_signal = float(np.mean(np.maximum(corrected, 0.)))
                np.testing.assert_allclose(original_signal, law['source_signal_scale'], atol=1e-12, rtol=1e-12)
                common, tail, task, white = [], [], [], []
                for (design, _, _), z in zip(likelihood.blocks, likelihood.task_designs):
                    common.extend(np.sum((design@state['mean_covariance'])*design, axis=1))
                    tail.extend(likelihood.tail*np.sum(z[:, 1:]**2, axis=1))
                    task.extend((z**2)@state['residual_diagonal'])
                    white.extend(np.full(len(z), state['discrepancy']))
                common, tail, task, white = map(np.asarray, (common, tail, task, white))
                learned = task-tail; total = common+task+white
                np.testing.assert_allclose(common+tail+learned+white, total, atol=1e-12)
                for (design, _, _), z, offset in zip(likelihood.blocks, likelihood.task_designs,
                    np.cumsum([0]+[len(b[0]) for b in likelihood.blocks])[:-1]):
                    actual = np.sum((design@state['mean_covariance'])*design, axis=1)+(z**2)@state['residual_diagonal']+state['discrepancy']
                    np.testing.assert_allclose(actual, total[offset:offset+len(design)], atol=1e-12)
                rows.append({'layout': layout, 'deleted_task': deleted, 'head': head,
                    'unfloored_corrected_task_variances': corrected.tolist(), 'native_parameter_signal_scale': original_signal,
                    'nonnegative_data_signal_scale': data_signal, 'signal_floor_binding_task_count': int(np.count_nonzero(corrected < .02)),
                    'native_to_data_signal_ratio': original_signal/data_signal if data_signal > 0 else None,
                    'task_weak_tail_variance_per_coefficient': likelihood.tail,
                    'mean_new_task_weak_tail_variance_at_source_coordinates': float(tail.mean()),
                    'mean_learned_task_excess_intercept_variance': float(learned.mean()),
                    'mean_common_posterior_variance': float(common.mean()), 'white_discrepancy_variance': state['discrepancy'],
                    'mean_new_task_latent_variance': float(total.mean()),
                    'weak_tail_fraction_of_task_function_variance': float(tail.mean()/task.mean()),
                    'weak_tail_fraction_of_new_task_latent_variance': float(tail.mean()/total.mean())})
        print('both-head source scales and covariance budgets reproduced', layout, flush=True)
    summary = {layout: {head: {key: float(np.mean([r[key] for r in rows if r['layout'] == layout and r['head'] == head]))
        for key in ('native_parameter_signal_scale', 'nonnegative_data_signal_scale', 'native_to_data_signal_ratio',
            'weak_tail_fraction_of_task_function_variance', 'weak_tail_fraction_of_new_task_latent_variance')}
        for head in spec['heads']} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'rows': rows, 'native_scales_laws_covariance_budgets_reproduced': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'source_holdouts_indexed': False, 'target_promoted': False, 'independent_confirmation': False}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
