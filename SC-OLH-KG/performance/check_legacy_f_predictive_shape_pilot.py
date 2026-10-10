"""Updated source-predictive shape mixture at two already-frozen designs."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, errors, score, raw_density
from performance.check_legacy_f_source_information_design import comparison
from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_predictive_shape_pilot_execution_v1.json'
ARM = 'source_predictive_shape_mixture'


def run():
    spec = read(EXECUTION); reference = read(spec['reference_results']); package = read(spec['source_rules'])
    assert reference['both_source_integrals_and_rules_verified'] and package['both_source_rules_frozen_without_query_data']
    execute(spec, package)


def execute(spec, package):
    started = time.process_time()
    designs = read(spec['design_results']); control_pilot = read(spec['control_results'])
    assert designs['frozen_before_query_outcomes'] and control_pilot['scientific_gate_interpreted']
    control_rows = list(control_pilot['rows'])
    for filename in spec['additional_control_results']:
        extra = read(filename)
        assert extra['scientific_gate_interpreted']
        control_rows += extra['rows']
    coverage, _ = source_paths(read(spec['source_context_execution'])); parent = read(coverage['source_execution'])
    old = read(spec['retained_source_execution']); adaptation = read(read(old['reference_execution'])['adaptation_execution'])
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); branches, frozen = [], []
    for rule in package['rows']:
        layout, deleted = rule['layout'], rule['deleted_task']; source = rule['source_reference_law']
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
        assert source['source_task_names'] == [t['name'] for t in inputs['source_tasks']]
        assert f'related_inventory_{deleted:02d}' not in source['source_task_names']
        record = next(r for r in designs['rows'] if r['layout'] == layout)
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); public = np.asarray(inputs['public_mean_reference'])
        risk_public = np.asarray(inputs['public_risk_reference']); risk_basis = OrthogonalLowOrderBasis(risk_public)
        risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
        eta_law = next(h['joint_structural'] for h in risk['heads'] if h['head'] == 3)
        eta_all = source_prior_moments(restore_joint(risk_basis, eta_law), risk_public)
        nodes, weights = rule_on_intervals(source, rule['adaptive_u_intervals'], 32)
        for design_name in spec['designs']:
            prefix, sites = record['frozen_prefixes'][design_name], record['scoring_site_indices']
            assert len(prefix) == 4 and not set(prefix).intersection(sites)
            selection = prefix+sites; eta = eta_all[0][selection], eta_all[1][np.ix_(selection, selection)]
            selected = mixture_field(basis, public[selection], source, np.asarray(rule['variance_nodes']), np.asarray(rule['posterior_weights']), eta)
            refined = mixture_field(basis, public[selection], source, nodes, weights, eta)
            branches.append({'layout': layout, 'deleted_task': deleted, 'design': design_name, 'prefix': prefix,
                             'sites': sites, 'selected': selected, 'refined': refined})
            frozen.append({'layout': layout, 'deleted_task': deleted, 'design': design_name, 'paid_prefix': prefix,
                'scoring_sites': sites, 'source_task_names': source['source_task_names'],
                'source_q': source['native_law']['fitted_shape_allocation_q'],
                'source_component_count': len(rule['variance_nodes']), 'reference_component_count': len(nodes)})
    save(out/'frozen_inputs.json', {'execution': spec, 'all_four_prior_designs_frozen_before_paid_values': True, 'rows': frozen})
    raw = read(coverage['source_data']); paid = {(b['layout'], b['design']): [raw['tasks'][b['deleted_task']]['Y_replicates'][j][0][0]
        for j in b['prefix']] for b in branches}; del raw
    rows, numerical = [], []
    for branch in branches:
        forecasts = []; files = []
        for name in ('selected', 'refined'):
            field = branch[name]; evidence = sum(update_raw_observation(field, j, float(y)) for j, y in enumerate(paid[(branch['layout'], branch['design'])]))
            state = compact(field, list(range(4, 16)), evidence); forecasts.append(prediction(state))
            filename = f"layout{branch['layout']}_{branch['design']}_{name}.npz"; np.savez_compressed(out/filename, **state); files.append(filename)
            del branch[name]
        difference = errors(forecasts[0], forecasts[1]); passed = all(difference[k] <= v for k, v in spec['predictive_tolerances'].items())
        numerical.append({'layout': branch['layout'], 'design': branch['design'], 'errors': difference, 'passed': passed})
        rows.append({'layout': branch['layout'], 'deleted_task': branch['deleted_task'], 'episode_count': 4, 'arm': spec['candidate_arm'],
            'design': branch['design'], 'paid_prefix': branch['prefix'], 'scoring_sites': branch['sites'], 'prediction': forecasts[0],
            'refined_prediction': forecasts[1], 'component_marginals_file': files[0], 'refined_component_marginals_file': files[1]})
    save(out/'predictions.json', {'execution': spec, 'all_four_forecasts_frozen_before_score_labels': True, 'rows': rows, 'numerical_checks': numerical})
    base = {'execution': spec, 'numerical_checks': numerical, 'cost': spec['cost'], 'no_model_adopted': True,
            'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    if not all(r['passed'] for r in numerical):
        save(out/'results.json', {**base, 'forecasts_scored': False, 'scientific_gate_interpreted': False, 'cpu_seconds': time.process_time()-started}); return
    density_checks, gates = [], {}
    for row in rows:
        if time.process_time()-started >= spec['maximum_single_thread_cpu_seconds']:
            save(out/'results.json', {**base, 'forecasts_scored': False, 'scientific_gate_interpreted': False,
                'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time()-started}); return
        labels = read(Path(coverage['output'])/f"layout{row['layout']}"/'inputs.json')['source_tasks'][row['deleted_task']]
        with np.load(out/row['component_marginals_file']) as state:
            row['scores'], row['density_integration_error_bound'] = score(row, state, labels)
        with np.load(out/row['refined_component_marginals_file']) as state:
            density, error = raw_density(state, labels)
        difference = abs(row['scores']['f_raw_batch_density']-density)
        density_checks.append({'layout': row['layout'], 'design': row['design'], 'raw_density_refinement_error': difference,
            'reference_density_integration_error_bound': error, 'passed': difference <= spec['density_refinement_tolerance']})
        comparisons = {}
        for arm in spec['controls']:
            control = next(r for r in control_rows if r['layout'] == row['layout'] and r['arm'] == arm and r['design'] == row['design'])
            assert control['paid_prefix'] == row['paid_prefix'] and control['scoring_sites'] == row['scoring_sites']
            comparisons[arm] = comparison(row, control, True)
        gates.setdefault(row['layout'], {})[row['design']] = comparisons
        print('source-predictive shape pilot', row['layout'], row['design'], row['scores'], comparisons, flush=True)
    verified = all(r['passed'] for r in density_checks)
    retained_pass = all(g['source_variance_mixture']['passed'] for group in gates.values() for g in group.values())
    complete_pass = all(v['passed'] for group in gates.values() for design in group.values() for v in design.values())
    result = {**base, 'rows': rows, 'gates': gates, 'density_checks': density_checks, 'numerical_density_verified': verified,
        'forecasts_scored': True, 'scientific_gate_interpreted': verified,
        'retained_mixture_gate_both_designs': retained_pass if verified else None,
        'all_control_gate_both_designs': complete_pass if verified else None,
        'cpu_seconds': time.process_time()-started, 'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz'))}
    save(out/'results.json', result)


if __name__ == '__main__':
    run()
