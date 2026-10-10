"""Source f mean/covariance moment probes under the frozen paid prefix."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, errors, raw_density, score
from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments
from representation.source_kernel_population import predict_unequal_site_kernel_population

EXECUTION = 'performance/manifests/legacy_joint_population_f_reference_prior_transplant_execution_v1.json'


def transplant(field, arm, gp, eta):
    if arm == 'gp_source_mean': field.means[:, 0::2] = gp[0]
    else: field.covariances[:, 0::2, 0::2] = gp[1]
    np.testing.assert_allclose(field.means[:, 1::2], np.broadcast_to(eta[0], field.means[:, 1::2].shape), atol=0, rtol=0)
    np.testing.assert_allclose(field.covariances[:, 1::2, 1::2], np.broadcast_to(eta[1], field.covariances[:, 1::2, 1::2].shape), atol=0, rtol=0)
    return field


def run():
    started = time.process_time(); spec = read(EXECUTION); mixture_spec = read(spec['mixture_execution'])
    reference = read(mixture_spec['reference_execution'])
    centering = read(read(reference['source_execution'])['centering_execution'])
    coverage = read(centering['coverage_execution']); parent = read(coverage['source_execution'])
    profile = read(centering['profile_execution']); adaptation = read(reference['adaptation_execution'])
    rules = read(mixture_spec['source_rules']); assert rules['all14_rules_frozen_without_query_data']
    prefixes = read(mixture_spec['public_prefixes'])['rows']
    mixture_controls = [r for r in read(spec['mixture_profiles'])['rows'] if r['episode_count'] == spec['episode_count']]
    gp_controls = [r for r in read(spec['controls_profiles'])['rows'] if r['episode_count'] == spec['episode_count'] and r['arm'] == 'source_gp']
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(x) for x in public['bank']]
    branches, frozen = {}, []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            rule = next(r for r in rules['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
            source = rule['source_reference_law']; names = source['source_task_names']
            assert public['tasks'][deleted]['task_id'] not in names
            assert names == [t['name'] for t in inputs['source_tasks']] == [t['name'] for t in inputs['risk_source_tasks']]
            gp_law = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]['joint_gp']
            assert gp_law['optimizer_converged'] and gp_law['source_task_count'] == 5 and gp_law['source_label_count'] == 60
            basis, risk_basis = (OrthogonalLowOrderBasis(inputs[key]) for key in ('public_mean_reference', 'public_risk_reference'))
            psi = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][deleted])), bank)
                -inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            gp = predict_unequal_site_kernel_population(psi, gp_law)
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            eta_law = next(h['joint_structural'] for h in risk['heads'] if h['head'] == 3)
            eta = source_prior_moments(restore_joint(risk_basis, eta_law), np.asarray(inputs['public_risk_reference']))
            prefix = next(r['public_prefix'] for r in prefixes if r['layout'] == layout and r['deleted_task'] == deleted)
            sites = coverage['layouts'][layout][deleted]; assert not set(prefix).intersection(sites)
            branches[(layout, deleted)] = (rule, basis, psi, eta, gp, prefix, sites)
            frozen.append({'layout': layout, 'deleted_task': deleted, 'source_task_names': names, 'prefix': prefix,
                'gp_prior_mean': gp[0].tolist(), 'gp_prior_covariance': gp[1].tolist()})
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'frozen_priors.json', {'execution': spec, 'all_source_rules_prefixes_gp_moments_frozen_before_paid_values': True, 'rows': frozen})
    raw = read(coverage['source_data']); paid = {key: [raw['tasks'][key[1]]['Y_replicates'][j][0][0]
        for j in b[5]] for key, b in branches.items()}; del raw
    rows, numerical, refined_files = [], [], {}
    for (layout, deleted), (rule, basis, psi, eta, gp, prefix, sites) in branches.items():
        source = rule['source_reference_law']; nodes, weights = rule_on_intervals(source, rule['adaptive_u_intervals'], 32)
        for arm in spec['probes']:
            selected = transplant(mixture_field(basis, psi, source, np.asarray(rule['variance_nodes']), np.asarray(rule['posterior_weights']), eta), arm, gp, eta)
            refined = transplant(mixture_field(basis, psi, source, nodes, weights, eta), arm, gp, eta)
            evidence = refined_evidence = 0.
            for j, value in zip(prefix, paid[(layout, deleted)]):
                evidence += selected.update_batch(j, float(value), None, 1)
                refined_evidence += refined.update_batch(j, float(value), None, 1)
            state, refined_state = compact(selected, sites, evidence), compact(refined, sites, refined_evidence)
            p, rp = prediction(state), prediction(refined_state); e = errors(p, rp)
            passed = e['mean_max_absolute'] <= 1e-7 and e['covariance_relative_frobenius'] <= 1e-5 and e['paid_log_evidence_absolute'] <= 1e-6
            numerical.append({'layout': layout, 'deleted_task': deleted, 'arm': arm, 'errors': e, 'passed': passed})
            filename = f'layout{layout}_task{deleted}_{arm}.npz'; np.savez_compressed(out/filename, **state)
            if deleted == 1:
                refined_filename = f'layout{layout}_task{deleted}_{arm}_refined.npz'
                np.savez_compressed(out/refined_filename, **refined_state); refined_files[(layout, arm)] = refined_filename
            rows.append({'layout': layout, 'deleted_task': deleted, 'arm': arm, 'episode_count': spec['episode_count'],
                'scoring_sites': sites, 'paid_prefix': prefix, 'prediction': p, 'refined_prediction': rp,
                'component_marginals_file': filename})
            print('prior transplant forecast frozen', layout, deleted, arm, passed, flush=True)
            del selected, refined, state, refined_state
    save(out/'predictions.json', {'execution': spec, 'all24_forecasts_frozen_before_scored_labels': True, 'rows': rows, 'numerical_checks': numerical})
    base = {'execution': spec, 'cost': spec['cost'], 'numerical_checks': numerical, 'no_model_adopted': True,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False, 'original_complete_gate_unchanged': True}
    if not all(r['passed'] for r in numerical):
        save(out/'results.json', {**base, 'numerical_forecasts_verified': False, 'forecasts_scored': False,
            'scientific_gate_interpreted': False, 'cpu_seconds': time.process_time()-started}); return
    labels = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks'] for layout in spec['layouts']}
    density_checks = []; contrasts = []
    for row in rows:
        label = labels[row['layout']][row['deleted_task']]
        with np.load(out/row['component_marginals_file']) as state: row['scores'], row['density_integration_error_bound'] = score(row, state, label)
        if row['deleted_task'] == 1:
            with np.load(out/refined_files[(row['layout'], row['arm'])]) as state: density, _ = raw_density(state, label)
            difference = abs(density-row['scores']['f_raw_batch_density'])
            density_checks.append({'layout': row['layout'], 'arm': row['arm'], 'raw_density_refinement_error': difference, 'passed': difference <= 1e-5})
        for name, controls in (('original_mixture', mixture_controls), ('source_gp', gp_controls)):
            control = next(r for r in controls if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'])
            contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'arm': row['arm'],
                'reference': name, 'probe_minus_control': {key: row['scores'][key]-control['scores'][key]
                    for key in control['scores'] if key in row['scores']}})
    groups = {}; summary = {}; gates = {}
    for layout in spec['layouts']:
        groups[layout] = {}; summary[layout] = {}; gates[layout] = {}
        for arm in spec['probes']:
            groups[layout][arm] = {}
            for group, tasks in (('all6', range(6)), ('task1', [1]), ('other5', [0, 2, 3, 4, 5])):
                members = [r for r in rows if r['layout'] == layout and r['arm'] == arm and r['deleted_task'] in tasks]
                groups[layout][arm][group] = {key: float(np.mean([r['scores'][key] for r in members])) for key in members[0]['scores']}
            summary[layout][arm] = groups[layout][arm]['all6']
            old = [r for r in mixture_controls if r['layout'] == layout]
            reference = {key: float(np.mean([r['scores'][key] for r in old])) for key in old[0]['scores']}; actual = summary[layout][arm]
            gates[layout][arm] = {'mse_nonincrease': actual['f_mse'] <= reference['f_mse'],
                'raw_density_strict_improvement': actual['f_raw_batch_density'] > reference['f_raw_batch_density'],
                'transcript_density_nondecrease': actual['sequential_transcript_density'] >= reference['sequential_transcript_density']}
    verified = all(r['passed'] for r in density_checks)
    passed = {arm: all(all(gates[layout][arm].values()) for layout in spec['layouts']) for arm in spec['probes']} if verified else None
    result = {**base, 'summary': summary, 'descriptive_groups': groups, 'matched_contrasts': contrasts,
        'gates_by_layout': gates, 'probe_improvement_both_layouts': passed, 'numerical_forecasts_verified': True,
        'numerical_density_verified': verified, 'density_checks': density_checks, 'forecasts_scored': True,
        'scientific_gate_interpreted': verified, 'cpu_seconds': time.process_time()-started,
        'compact_component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz'))}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'summary': summary, 'probe_improvement': passed, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
