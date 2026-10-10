"""Conditional source information for the four joint spectral parameters."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_joint_compression import source_laws
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_joint_spectral_geometry import JointSpectralGeometryLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_f_joint_spectral_information_execution_v1.json'


def information(model, parameters):
    a, aa, ap = model.covariance(0, parameters[0], parameters[1])
    r, ra, rp = model.covariance(1, parameters[2], parameters[3]); d = model.design
    covariance = d @ a @ d.T + block_diag(*[z @ r @ z.T for z in model.designs]) + np.diag(model.noise)
    directions = (d @ aa @ d.T, d @ ap @ d.T, block_diag(*[z @ ra @ z.T for z in model.designs]), block_diag(*[z @ rp @ z.T for z in model.designs]))
    factor = cho_factor(covariance, lower=True)
    solved = [cho_solve(factor, v) for v in directions]
    fisher = np.asarray([[.5 * np.trace(x @ y) for y in solved] for x in solved]); fisher = .5 * (fisher + fisher.T)
    eigenvalues = np.linalg.eigvalsh(fisher); assert eigenvalues.min() > 0
    inverse = np.linalg.solve(fisher, np.eye(4)); standard_errors = np.sqrt(np.diag(inverse))
    correlation = inverse / np.outer(standard_errors, standard_errors)
    # Whitened derivative Gram independently detects a mistaken Fisher trace.
    root = np.linalg.cholesky(covariance); whitened = [np.linalg.solve(root, np.linalg.solve(root, v).T).T for v in directions]
    gram = np.asarray([[.5 * np.sum(x * y) for y in whitened] for x in whitened])
    np.testing.assert_allclose(fisher, gram, atol=1e-8, rtol=1e-8)
    return {'expected_fisher_information': fisher.tolist(), 'eigenvalues': eigenvalues.tolist(),
        'unconstrained_local_standard_errors': standard_errors.tolist(), 'local_parameter_correlation': correlation.tolist(),
        'source_gradient_per_label': model.evaluate(parameters)[1].tolist(),
        'task_exponent_nuisance_adjusted_information': float(1 / inverse[3, 3]),
        'task_exponent_fixed_other_parameters_information': float(fisher[3, 3]),
        'whitened_gram_error': float(np.max(np.abs(fisher - gram))),
        'task_exponent_at_declared_upper_bound': bool(parameters[3] == 4.)}


def run():
    started = time.process_time(); spec = read(EXECUTION)
    inner, outer = read(spec['inner_source_laws']), read(spec['outer_source_laws'])
    assert inner['all_ten_source_laws_frozen_without_query_values'] and outer['both_source5_laws_frozen_before_query_values']
    templates = read(spec['inner_templates'])['rows']; registry = read(spec['inner_registry'])['rows']
    coverage, profile = source_paths(read(spec['source_context_execution']))
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True); rows = []
    for budget, law in [(4, r) for r in inner['rows']] + [(5, r) for r in outer['rows']]:
        if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
            save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
                'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started}); return
        layout, query = law['layout'], law['query_task']; inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        tasks = [t for t in inputs['source_tasks'] if t['name'] in law['source_task_names']]
        assert len(tasks) == budget and [t['name'] for t in tasks] == law['source_task_names']
        assert f'related_inventory_{query:02d}' not in law['source_task_names']
        public = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(public)
        if budget == 4:
            template = next(r for r in templates if (r['layout'], r['query_task'], r['arm']) == (layout, query, 'joint_native_support'))
            record = next(r for r in registry if r['layout'] == layout and r['inner_query_task'] == f'related_inventory_{query:02d}')
            gp = read(record['source_fit'])['fits']['gp']
        else:
            head = read(Path(profile['output']) / f'layout{layout}' / 'task1' / 'selected_source_fit.json')['heads'][0]; gp = head['joint_gp']
            compressed, _ = source_laws(tasks, public, public, {'fits': {'gp': gp, 'structural': head['joint_structural']}}, 0.)
            template = next(r for r in compressed if r['arm'] == 'joint_native_support')
        assert template['columns'] == law['columns']
        anchor = gp['anchor']; center = anchor['normalization']['mean'] + anchor['normalization']['scale'] * anchor['constant']
        model = JointSpectralGeometryLikelihood(basis, tasks, template['common_prior_coefficient_covariance'],
            template['task_coefficient_covariance'], law['columns'], center, law['white_discrepancy'], spec['positive_tail_fraction'])
        record = information(model, np.asarray(law['parameters']))
        rows.append({'source_task_count': budget, 'layout': layout, 'query_task': query, 'source_task_names': law['source_task_names'],
            'parameters': law['parameters'], **record})
        print('source spectral information', budget, layout, query, 'task exponent SE', record['unconstrained_local_standard_errors'][3], flush=True)
    assert len(rows) == 12
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started,
        'cost': spec['cost'], 'no_model_adopted': True, 'candidate_remains_stopped': True, 'query_outcomes_indexed': False,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False,
        'interpretation': 'Expected Fisher information conditional on fixed empirical source templates,centers,noise and support. Standard errors are unconstrained local information summaries,not posterior credible intervals at p=4. Template/support/GP estimation uncertainty is excluded. No posterior or new predictor fitted.'})


if __name__ == '__main__':
    run()
