"""Source-coherent shape budget tangent at the frozen four-query contract."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, errors, score, raw_density
from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.prepare_legacy_f_inner_raw_contract import finite_reference, finite_rule_errors
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_source_shape_scale_direction_execution_v1.json'


def scaled_source_law(inputs, frozen_law, scale):
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); tasks = inputs['source_tasks']
    def design(psi): return np.column_stack([np.ones(len(psi)), basis.transform(np.asarray(psi))])
    local = [design(t['psi']) for t in tasks]; d = np.vstack(local)
    a = np.diag(frozen_law['common_prior_coefficient_diagonal']); m0 = np.asarray(frozen_law['empirical_center_coefficient_mean'])
    native_r = np.asarray(frozen_law['residual_coefficient_diagonal']); tail = frozen_law['geometry']['independent_tail_per_shape_coefficient']
    r = native_r.copy(); r[1:] = tail+scale*(r[1:]-tail)
    assert np.all(r >= 0) and r[0] == native_r[0]
    error = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks])-d@m0
    def condition(residual):
        c = d@a@d.T+block_diag(*[(z*residual)@z.T+np.diag(np.asarray(t['noise'])[:, 0]+frozen_law['discrepancy_variance']) for z, t in zip(local, tasks)])
        factor = cho_factor(c, lower=True); alpha = cho_solve(factor, error)
        beta = m0+a@d.T@alpha; h = a-a@d.T@cho_solve(factor, d@a); h = .5*(h+h.T)
        nll = .5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha)/len(error)
        return beta, h, float(nll)
    baseline = condition(native_r)
    for value, key in zip(baseline, ('coefficient_mean', 'population_mean_covariance', 'source_negative_log_likelihood_per_label')):
        np.testing.assert_allclose(value, frozen_law[key], atol=1e-8, rtol=1e-8)
    beta, h, nll = condition(r)
    return {**frozen_law, 'coefficient_mean': beta.tolist(), 'population_mean_covariance': h.tolist(),
        'source_negative_log_likelihood_per_label': nll, 'residual_coefficient_diagonal': r.tolist(),
        'source_shape_budget_probe_scale': scale, 'fixed_native_point_law_reproduced_before_probe': True,
        'law_contract': 'Only active low-frequency task-shape excess is scaled;native support/spectral ratios,weak tail,common prior,empirical center,independent task intercept and discrepancy unchanged. Common posterior and full-tail variance mixture reconditioned on source4.'}


def run():
    started = time.process_time(); spec = read(EXECUTION); calibration = read(spec['calibration_results'])
    assert calibration['diagnostic_interpreted']
    context = read(spec['source_context_plan']); coverage, _ = source_paths(read(context['source_context_execution']))
    source = read(spec['reference_forecasts']); assert source['diagnostic_interpreted']
    branches, rules = [], []
    for directory in source['execution']['source_packages']:
        branches += read(Path(directory)/'source_branches.json')['rows']
        rules += read(Path(directory)/'source_quadrature.json')['rows']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    prepared, source_records, new_rules, incomplete = [], [], [], []
    for branch in branches:
        layout, query, names = branch['layout'], branch['query_task'], branch['source_task_names']
        inputs = read(Path(coverage['output'])/f'layout{layout}'/'task1'/'inputs.json')
        kept = {**inputs, 'source_tasks': [t for t in inputs['source_tasks'] if t['name'] in names]}
        assert len(kept['source_tasks']) == 4 and all(t['name'] != f'related_inventory_{query:02d}' for t in kept['source_tasks'])
        rule = next(r for r in rules if r['layout'] == layout and r['arm'] == 'shape' and r['source_reference_law']['source_task_names'] == names)
        frozen = rule['source_reference_law']['native_law']
        for scale in spec['fixed_scales']:
            native = scaled_source_law(kept, frozen, scale)
            record, law = finite_reference(kept, native, context); trials = []; selected = None
            for size in context['quadrature']['positive_rule_nodes_per_interval']:
                nodes, weights = rule_on_intervals(law, record['adaptive_u_intervals'], size)
                error = finite_rule_errors(nodes, weights, law); passed = all(error[k] <= limit for k, limit in spec['source_rule_tolerances'].items())
                trials.append({'size': size, 'component_count': len(nodes), 'errors': error, 'passed': passed})
                if passed:
                    selected = {'layout': layout, 'query_task': query, 'scale': scale, 'source_reference_law': law,
                        'variance_nodes': nodes.tolist(), 'posterior_weights': (weights/weights.sum()).tolist(), 'adaptive_u_intervals': record['adaptive_u_intervals']}
                    new_rules.append(selected); break
            source_records.append({'layout': layout, 'query_task': query, 'scale': scale, **record, 'rule_trials': trials, 'rule_complete': selected is not None})
            if selected is None: incomplete.append((layout, query, scale))
            else: prepared.append((branch, selected))
    save(out/'source_quadrature.json', {'execution': spec, 'all_twenty_source_rules_frozen_without_raw_query_values': not incomplete, 'rows': new_rules})
    save(out/'source_results.json', {'execution': spec, 'rows': source_records, 'all_twenty_source_rules_complete': not incomplete,
        'source_preparation_cpu_seconds': time.process_time()-started})
    base = {'execution': spec, 'no_model_adopted': True, 'source6_7_evaluated': False, 'target_promoted': False,
        'independent_confirmation': False, 'cost': spec['cost'], 'training_diagnostic_only': True}
    if incomplete:
        save(out/'results.json', {**base, 'source_rules_incomplete': incomplete, 'forecasts_scored': False, 'direction_interpreted': False, 'cpu_seconds': time.process_time()-started}); return
    # Binding to the unchanged public-coordinate prior is frozen before values.
    public_records = read(Path(ROOT/source['execution']['output'])/'frozen_inputs.json')['rows']
    from performance.diagnose_legacy_risk_transfer import no_outcomes
    from performance.run_legacy_related_inventory_source import problem_for
    from representation.public_control_mean import public_control_mean_coordinates
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    fields, frozen = [], []
    for branch, rule in prepared:
        layout, query = branch['layout'], branch['query_task']; sites = branch['scoring_sites']; prefix = spec['paid_prefix']
        contract = next(r for r in public_records if r['layout'] == layout and r['query_task'] == query and r['arm'] == 'shape_mixture' and r['design'] == 'fixed_public_prefix')
        assert contract['paid_prefix'] == prefix and contract['scoring_sites'] == sites
        inputs = read(Path(coverage['output'])/f'layout{layout}'/'task1'/'inputs.json')
        psi = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][query])), bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
        selection = prefix+sites; basis = OrthogonalLowOrderBasis(branch['public_mean_reference'])
        eta = np.asarray(branch['eta_mean'])[selection], np.asarray(branch['eta_covariance'])[np.ix_(selection, selection)]
        rn, rw = rule_on_intervals(rule['source_reference_law'], rule['adaptive_u_intervals'], 32)
        work = {'row': {'layout': layout, 'query_task': query, 'scale': rule['scale'], 'episode_count': 4, 'paid_prefix': prefix, 'scoring_sites': sites},
            'selected': mixture_field(basis, psi[selection], rule['source_reference_law'], np.asarray(rule['variance_nodes']), np.asarray(rule['posterior_weights']), eta),
            'refined': mixture_field(basis, psi[selection], rule['source_reference_law'], rn, rw, eta)}
        fields.append(work); frozen.append({**work['row'], 'source_task_names': branch['source_task_names']})
    save(out/'frozen_inputs.json', {'execution': spec, 'all_twenty_priors_frozen_before_paid_values': True, 'rows': frozen})
    raw = read(coverage['source_data']); paid = {b['query_task']: [raw['tasks'][b['query_task']]['Y_replicates'][j][0][0] for j in spec['paid_prefix']] for b in branches}; del raw
    rows, numerical = [], []
    for work in fields:
        row = work['row']
        for resolution in ('selected', 'refined'):
            field = work.pop(resolution); evidence = sum(update_raw_observation(field, j, float(y)) for j, y in enumerate(paid[row['query_task']]))
            state = compact(field, list(range(4, 12)), evidence); p = prediction(state)
            filename = f"layout{row['layout']}_task{row['query_task']}_scale{row['scale']}_{resolution}.npz"; np.savez_compressed(out/filename, **state)
            row[f'{resolution}_component_file'], row[f'{resolution}_prediction'] = filename, p
        row['prediction'] = row.pop('selected_prediction'); difference = errors(row['prediction'], row['refined_prediction'])
        numerical.append({'layout': row['layout'], 'query_task': row['query_task'], 'scale': row['scale'], 'errors': difference,
            'passed': all(difference[k] <= limit for k, limit in spec['prediction_tolerances'].items())})
        rows.append(row)
    save(out/'predictions.json', {'execution': spec, 'all_twenty_forecasts_frozen_before_score_labels': True, 'rows': rows, 'numerical_checks': numerical})
    if not all(r['passed'] for r in numerical):
        save(out/'results.json', {**base, 'numerical_checks': numerical, 'forecasts_scored': False, 'direction_interpreted': False, 'cpu_seconds': time.process_time()-started}); return
    density_checks = []
    for row in rows:
        inputs = read(Path(coverage['output'])/f"layout{row['layout']}"/'task1'/'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == f"related_inventory_{row['query_task']:02d}")
        labels = {k: np.asarray(task[k])[4:] for k in ('response', 'noise')}
        with np.load(out/row['selected_component_file']) as state: row['scores'], _ = score(row, state, labels)
        with np.load(out/row['refined_component_file']) as state: density, _ = raw_density(state, labels)
        error = abs(row['scores']['f_raw_batch_density']-density)
        density_checks.append({'layout': row['layout'], 'query_task': row['query_task'], 'scale': row['scale'], 'density_error': error, 'passed': error <= 1e-5})
        baseline = next(r for r in source['rows'] if r['layout'] == row['layout'] and r['query_task'] == row['query_task'] and r['design'] == 'fixed_public_prefix' and r['arm'] == 'shape_mixture')
        row['probe_minus_frozen_baseline'] = {k: row['scores'][k]-baseline['scores'][k] for k in row['scores']}
    summary = {l: {str(s): {k: float(np.mean([r['probe_minus_frozen_baseline'][k] for r in rows if r['layout'] == l and r['scale'] == s]))
        for k in rows[0]['probe_minus_frozen_baseline']} for s in spec['fixed_scales']} for l in ('5', '6')}
    verified = all(r['passed'] for r in density_checks)
    supported = all(g['0.99']['f_raw_batch_density'] > 0 and g['1.01']['f_raw_batch_density'] < 0 for g in summary.values())
    result = {**base, 'rows': rows, 'summary': summary, 'numerical_checks': numerical, 'density_checks': density_checks,
        'direction_interpreted': verified, 'source_coherent_shrink_direction_supported': supported if verified else None,
        'source_derived_laws_frozen_before_paid_values': True, 'forecasts_scored': True,
        'cpu_seconds': time.process_time()-started, 'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz'))}
    save(out/'results.json', result); print('source-coherent shape-scale direction', summary, 'supported', supported if verified else None, 'CPU', result['cpu_seconds'], flush=True)


if __name__ == '__main__':
    run()
