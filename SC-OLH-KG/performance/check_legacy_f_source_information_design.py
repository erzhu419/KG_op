"""Frozen source-derived four-point designs with common-design prior controls."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, errors, score, raw_density
from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments
from representation.source_kernel_population import predict_unequal_site_kernel_population
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_source_information_pilot_execution_v1.json'


def gaussian_field(f, eta):
    mean = np.column_stack([f[0], eta[0]]); n = len(mean); covariance = np.zeros((n, 2, n, 2))
    covariance[:, 0, :, 0] = f[1]; covariance[:, 1, :, 1] = eta[1]
    return JointFunctionRiskPosterior.from_moments(mean, covariance)


def comparison(candidate, reference, include_transcript):
    keys = ['f_mse', 'f_raw_batch_density'] + (['sequential_transcript_density'] if include_transcript else [])
    delta = {key: candidate['scores'][key]-reference['scores'][key] for key in keys}
    passed = delta['f_mse'] <= 0 and delta['f_raw_batch_density'] > 0
    if include_transcript:
        passed = passed and delta['sequential_transcript_density'] >= 0
    return {'candidate_design': candidate['design'], 'reference_arm': reference['arm'], 'reference_design': reference['design'],
        'candidate_minus_reference': delta, 'mse_relative_change': delta['f_mse']/reference['scores']['f_mse'], 'passed': bool(passed)}


def run():
    started = time.process_time(); spec = read(EXECUTION); design_results = read(spec['design_results'])
    assert design_results['frozen_before_query_outcomes'] and not design_results['query_paid_or_score_outcomes_read']
    coverage, profile = source_paths(read(spec['source_context_execution']))
    old = read(spec['retained_source_execution']); rules = read(old['source_rules'])
    assert rules['all14_rules_frozen_without_query_data']
    parent = read(coverage['source_execution']); adaptation = read(read(old['reference_execution'])['adaptation_execution'])
    controls = read(old['controls_profiles'])['rows']; mixtures = read(Path(old['output'])/'calibration_profiles.json')['rows']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); branches, frozen, rows = [], [], []
    for record in design_results['rows']:
        layout, deleted = record['layout'], spec['deleted_task']; suffix = Path(f'layout{layout}')/f'task{deleted}'
        inputs = read(Path(coverage['output'])/suffix/'inputs.json')
        head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
        names = [t['name'] for t in inputs['source_tasks']]
        assert names == record['source_task_names'] and f'related_inventory_{deleted:02d}' not in names
        basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); public = np.asarray(inputs['public_mean_reference'])
        risk_public = np.asarray(inputs['public_risk_reference']); risk_basis = OrthogonalLowOrderBasis(risk_public)
        risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
        eta_law = next(h['joint_structural'] for h in risk['heads'] if h['head'] == 3)
        eta_all = source_prior_moments(restore_joint(risk_basis, eta_law), risk_public)
        rule = next(r for r in rules['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
        assert rule['source_reference_law']['source_task_names'] == names
        source = rule['source_reference_law']; refined_nodes, refined_weights = rule_on_intervals(source, rule['adaptive_u_intervals'], 32)
        sites = record['scoring_site_indices']
        for arm in spec['arms']:
            cached = mixtures if arm == 'source_variance_mixture' else controls
            row = next(r for r in cached if r['layout'] == layout and r['deleted_task'] == deleted and r['episode_count'] == 4 and r['arm'] == arm)
            assert row['paid_prefix'] == record['frozen_prefixes']['fixed_public_prefix'] and row['scoring_sites'] == sites
            rows.append({**row, 'design': 'fixed_public_prefix', 'reused_frozen_forecast_and_scores': True})
        for design_name in spec['new_designs']:
            prefix = record['frozen_prefixes'][design_name]; assert len(prefix) == 4 and not set(prefix).intersection(sites)
            selection = prefix+sites; psi = public[selection]
            eta = eta_all[0][selection], eta_all[1][np.ix_(selection, selection)]
            for arm in spec['arms']:
                if arm == 'source_variance_mixture':
                    field = mixture_field(basis, psi, source, np.asarray(rule['variance_nodes']), np.asarray(rule['posterior_weights']), eta)
                    refined = mixture_field(basis, psi, source, refined_nodes, refined_weights, eta)
                else:
                    f = (predict_unequal_site_kernel_population(psi, head['joint_gp']) if arm == 'source_gp'
                         else source_prior_moments(restore_joint(basis, head['flat' if arm == 'flat' else 'joint_structural']), psi))
                    field = gaussian_field(f, eta); refined = None
                branches.append({'layout': layout, 'arm': arm, 'design': design_name, 'prefix': prefix,
                    'sites': sites, 'field': field, 'refined': refined})
                frozen.append({'layout': layout, 'arm': arm, 'design': design_name, 'paid_prefix': prefix,
                    'source_task_names': names, 'scoring_sites': sites, 'source_component_count': len(field.log_weights),
                    'refined_component_count': len(refined.log_weights) if refined is not None else None})
    save(out/'frozen_inputs.json', {'execution': spec, 'all_priors_and_prefixes_frozen_before_paid_values': True, 'rows': frozen})
    raw = read(coverage['source_data']); paid = {(b['layout'], b['design']): [raw['tasks'][spec['deleted_task']]['Y_replicates'][j][0][0]
            for j in b['prefix']] for b in branches}; del raw
    numerical = []
    for b in branches:
        evidence = sum(update_raw_observation(b['field'], j, float(y)) for j, y in enumerate(paid[(b['layout'], b['design'])]))
        state = compact(b['field'], list(range(4, 16)), evidence); p = prediction(state)
        filename = f"layout{b['layout']}_{b['design']}_{b['arm']}.npz"; np.savez_compressed(out/filename, **state)
        row = {'layout': b['layout'], 'deleted_task': spec['deleted_task'], 'episode_count': 4, 'arm': b['arm'], 'design': b['design'],
            'paid_prefix': b['prefix'], 'scoring_sites': b['sites'], 'prediction': p, 'component_marginals_file': filename}
        if b['refined'] is not None:
            evidence = sum(update_raw_observation(b['refined'], j, float(y)) for j, y in enumerate(paid[(b['layout'], b['design'])]))
            refined_state = compact(b['refined'], list(range(4, 16)), evidence); difference = errors(p, prediction(refined_state))
            passed = all(difference[k] <= tolerance for k, tolerance in spec['predictive_tolerances'].items())
            numerical.append({'layout': b['layout'], 'design': b['design'], 'errors': difference, 'passed': passed})
            refined_filename = filename.replace('.npz', '_refined.npz'); np.savez_compressed(out/refined_filename, **refined_state)
            row['refined_component_marginals_file'] = refined_filename
        rows.append(row); del b['field'], b['refined']
    save(out/'predictions.json', {'execution': spec, 'all_forecasts_frozen_before_score_labels': True, 'numerical_checks': numerical, 'rows': rows})
    base = {'execution': spec, 'numerical_checks': numerical, 'cost': spec['cost'], 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False, 'no_new_prior_adopted': True}
    if not all(r['passed'] for r in numerical):
        save(out/'results.json', {**base, 'forecasts_scored': False, 'scientific_gate_interpreted': False, 'cpu_seconds': time.process_time()-started}); return
    density_checks = []
    for row in rows:
        if row.get('reused_frozen_forecast_and_scores'):
            continue
        if time.process_time()-started >= spec['maximum_single_thread_cpu_seconds']:
            save(out/'results.json', {**base, 'forecasts_scored': False, 'scientific_gate_interpreted': False,
                'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time()-started}); return
        labels = read(Path(coverage['output'])/f"layout{row['layout']}"/'inputs.json')['source_tasks'][spec['deleted_task']]
        with np.load(out/row['component_marginals_file']) as state:
            row['scores'], row['density_integration_error_bound'] = score(row, state, labels)
        if 'refined_component_marginals_file' in row:
            with np.load(out/row['refined_component_marginals_file']) as state:
                density, error = raw_density(state, labels)
            difference = abs(row['scores']['f_raw_batch_density']-density)
            density_checks.append({'layout': row['layout'], 'design': row['design'], 'density_refinement_error': difference,
                'reference_density_integration_error_bound': error, 'passed': difference <= spec['density_refinement_tolerance']})
    gates = {}
    for layout in spec['layouts']:
        def lookup(arm, design):
            return next(r for r in rows if r['layout'] == layout and r['arm'] == arm and r['design'] == design)
        candidate = lookup('source_variance_mixture', 'intercept_mixture_greedy')
        gates[layout] = {'design_effect_fixed_future_roster': comparison(candidate, lookup('source_variance_mixture', 'fixed_public_prefix'), False),
            'own_design_source_gp_comparison': comparison(candidate, lookup('source_gp', 'source_gp_greedy'), False),
            'common_design_prior_controls': {arm: comparison(candidate, lookup(arm, 'intercept_mixture_greedy'), True)
                                            for arm in ('native_f', 'flat', 'source_gp')}}
    verified = all(r['passed'] for r in density_checks)
    design_pass = all(g['design_effect_fixed_future_roster']['passed'] for g in gates.values())
    method_pass = all(g['own_design_source_gp_comparison']['passed'] and all(v['passed'] for v in g['common_design_prior_controls'].values()) for g in gates.values())
    result = {**base, 'rows': rows, 'gates': gates, 'density_checks': density_checks, 'numerical_density_verified': verified,
        'forecasts_scored': True, 'scientific_gate_interpreted': verified, 'source_design_prediction_gate_passed': design_pass if verified else None,
        'source_prior_common_and_own_design_gate_passed': method_pass if verified else None, 'cpu_seconds': time.process_time()-started,
        'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz'))}
    save(out/'results.json', result)
    print({'design_prediction_gate': result['source_design_prediction_gate_passed'], 'source_prior_gate': result['source_prior_common_and_own_design_gate_passed'],
           'gates': gates, 'cpu_seconds': result['cpu_seconds'], 'component_artifact_bytes': result['component_artifact_bytes']}, flush=True)


if __name__ == '__main__':
    run()
