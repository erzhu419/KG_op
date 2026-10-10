"""Source-only scalar-equivalence and cost checks for the batched raw filter."""

import copy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save
from representation.joint_function_risk import local_joint_moments
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments
from representation.vectorized_joint_function_risk import joint_variance_field, local_raw_moments, update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_shape_vectorized_filter_execution_v1.json'


def prepare_branch(spec, layout, deleted):
    compact = read(spec['compact_execution']); reference = read(compact['reference_execution'])
    geometry = read(reference['joint_execution']); shape = read(geometry['shape_execution'])
    coverage, profile = source_paths(read(shape['source_context_execution']))
    parent = read(coverage['source_execution']); old = read(spec['mixture_execution'])
    adaptation = read(read(old['reference_execution'])['adaptation_execution'])
    suffix = Path(f'layout{layout}')/f'task{deleted}'
    inputs = read(Path(coverage['output'])/suffix/'inputs.json')
    head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
    names = [t['name'] for t in inputs['source_tasks']]
    assert set(names) == {f'related_inventory_{i:02d}' for i in range(6) if i != deleted}
    assert names == [t['name'] for t in inputs['risk_source_tasks']]
    prefix = next(r['public_prefix'] for r in read(old['public_prefixes'])['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)[:4]
    sites = coverage['layouts'][layout][deleted]; assert not set(prefix).intersection(sites)
    selection = prefix+sites; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    risk_basis = OrthogonalLowOrderBasis(inputs['public_risk_reference'])
    risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
    eta_law = next(h['joint_structural'] for h in risk['heads'] if h['head'] == 3)
    eta = source_prior_moments(restore_joint(risk_basis, eta_law), np.asarray(inputs['public_risk_reference']))
    return {'coverage': coverage, 'basis': basis, 'psi': np.asarray(inputs['public_mean_reference'])[selection],
        'eta': (eta[0][selection], eta[1][np.ix_(selection, selection)]), 'prefix': prefix, 'sites': sites,
        'selection': selection, 'source_task_names': names, 'native_law': head['joint_structural']}


def run():
    started = time.process_time(); spec = read(EXECUTION); package = read(spec['compact_rules'])
    assert package['all12_source_rules_frozen_without_query_data']; rows = []
    for layout in spec['layouts']:
        deleted = spec['deleted_task']; branch = prepare_branch(spec, layout, deleted)
        rule = next(r for r in package['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
        source = rule['source_reference_law']; assert source['source_task_names'] == branch['source_task_names']
        with np.load(ROOT/rule['source_rule_file']) as saved:
            nodes, weights = saved['physical_variance_nodes'].copy(), saved['posterior_weights'].copy()
        native = joint_variance_field(branch['basis'], branch['psi'], source,
            np.asarray([[source['native_level_variance'], source['native_shape_variance']]]), np.ones(1), branch['eta'])
        design = np.column_stack([np.ones(len(branch['psi'])), branch['basis'].transform(branch['psi'])]); law = branch['native_law']
        expected_mean = design@np.asarray(law['coefficient_mean'])
        expected_covariance = design@(np.asarray(law['population_mean_covariance'])+np.diag(law['residual_coefficient_diagonal']))@design.T
        expected_covariance += law['discrepancy_variance']*np.eye(len(design))
        np.testing.assert_allclose(native.means[0, ::2], expected_mean, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(native.covariances[0, ::2, ::2], expected_covariance, atol=1e-8, rtol=1e-8)
        sd = np.sqrt(native.covariances[0, 0, 0]+np.exp(native.means[0, 1]))
        labels = [float(native.means[0, 0]+sd), float(native.means[0, 0]-.5*sd)]
        indices = np.linspace(0, len(nodes)-1, 8, dtype=int)
        scalar = joint_variance_field(branch['basis'], branch['psi'], source, nodes[indices], weights[indices], branch['eta'])
        checks = []
        for phase, label in enumerate(labels):
            mean, covariance = scalar.means[:, :2].copy(), scalar.covariances[:, :2, :2].copy()
            batched = local_raw_moments(mean, covariance, label)
            reference = [local_joint_moments(m, c, label, None, 1) for m, c in zip(mean, covariance)]
            errors = {'local_mean_max_absolute': float(np.max(np.abs(batched['mean']-np.asarray([r['mean'] for r in reference])))),
                'local_covariance_max_absolute': float(np.max(np.abs(batched['covariance']-np.asarray([r['covariance'] for r in reference])))),
                'component_log_evidence_max_absolute': float(np.max(np.abs(batched['log_evidence']-np.asarray([r['log_evidence'] for r in reference]))))}
            vector = copy.deepcopy(scalar); timer = time.process_time(); evidence = update_raw_observation(vector, 0, label)
            vector_cpu = time.process_time()-timer; timer = time.process_time(); scalar_evidence = scalar.update_batch(0, label, None, 1)
            scalar_cpu = time.process_time()-timer
            errors.update(full_conditional_mean_max_absolute=float(np.max(np.abs(vector.means-scalar.means))),
                full_conditional_covariance_max_absolute=float(np.max(np.abs(vector.covariances-scalar.covariances))))
            evidence_error = abs(evidence-scalar_evidence); weight_error = float(np.max(np.abs(vector.log_weights-scalar.log_weights)))
            passed = all(errors[k] <= limit for k, limit in spec['numerical_tolerances'].items()) and evidence_error <= 1e-8 and weight_error <= 1e-8
            checks.append({'phase': phase, 'model_derived_fixture_label': label, 'incoming_max_f_eta_covariance': float(np.max(np.abs(covariance[:, 0, 1]))),
                'errors': errors, 'mixture_log_evidence_error': evidence_error, 'component_log_weight_error': weight_error, 'passed': passed,
                'scalar_eight_component_cpu_seconds': scalar_cpu, 'batched_eight_component_cpu_seconds': vector_cpu,
                'vector_quadrature_relative_error': batched['integration_relative_error'], 'integrand_evaluations': batched['integrand_evaluations']})
        full = joint_variance_field(branch['basis'], branch['psi'], source, nodes, weights, branch['eta'])
        timer = time.process_time(); update_raw_observation(full, 0, labels[0]); elapsed = time.process_time()-timer
        rows.append({'layout': layout, 'deleted_task': deleted, 'component_indices': indices.tolist(), 'checks': checks,
            'native_component_reproduced': True, 'batch_component_count': len(nodes), 'batch_field_site_count': len(branch['selection']),
            'batch_update_cpu_seconds': elapsed, 'batch_field_array_bytes': full.means.nbytes+full.covariances.nbytes+full.log_weights.nbytes,
            'process_peak_rss_bytes': int(next(line for line in Path('/proc/self/status').read_text().splitlines()
                if line.startswith('VmHWM:')).split()[1])*1024,
            'batch_max_quadrature_relative_error': full.max_integration_relative_error})
        print('batched raw filter verified', layout, 'checks', [r['passed'] for r in checks], 'batch CPU', elapsed, flush=True)
        del full, scalar, vector
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'all_scalar_equivalence_checks_passed': all(c['passed'] for r in rows for c in r['checks']),
        'cpu_seconds': time.process_time()-started, 'cost': spec['cost'], 'fixture_labels_are_not_empirical_evidence': True,
        'query_paid_outcomes_read': False, 'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'all_checks_passed': result['all_scalar_equivalence_checks_passed'], 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
