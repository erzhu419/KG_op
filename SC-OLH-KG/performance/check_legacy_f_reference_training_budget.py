"""Frozen source-variance mixtures on matched four/eight-episode prefixes."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_f_training_information_budget import group_summary
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.validate_legacy_paired_taskout import gaussian_score
from performance.vectorized_joint_batch_density import component_batch_log_evidence
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments

EXECUTION = 'performance/manifests/legacy_joint_population_f_reference_training_budget_execution_v1.json'
ARM = 'source_variance_mixture'


def compact(field, sites, evidence):
    points = 2*np.asarray(sites)[:, None]+np.arange(2)
    return {'component_means': field.means[:, points].copy(),
        'component_local_covariances': np.stack([field.covariances[:, p][:, :, p] for p in points], axis=1),
        'f_component_covariances': field.covariances[:, points[:, 0]][:, :, points[:, 0]].copy(),
        'log_weights': field.log_weights.copy(), 'paid_log_evidence': np.asarray(evidence)}


def prediction(state):
    weights = np.exp(state['log_weights']); component_means = state['component_means'][:, :, 0]
    mean = weights@component_means; centered = component_means-mean
    covariance = np.einsum('k,kij->ij', weights, state['f_component_covariances'])+(centered.T*weights)@centered
    return {'mean': mean.tolist(), 'covariance': covariance.tolist(),
        'paid_predictive_log_score': float(state['paid_log_evidence'])}


def errors(actual, reference):
    a, r = np.asarray(actual['covariance']), np.asarray(reference['covariance'])
    return {'mean_max_absolute': float(np.max(np.abs(np.asarray(actual['mean'])-reference['mean']))),
        'covariance_relative_frobenius': float(np.linalg.norm(a-r)/np.linalg.norm(r)),
        'paid_log_evidence_absolute': abs(actual['paid_predictive_log_score']-reference['paid_predictive_log_score'])}


def raw_density(state, labels):
    log_densities = []; max_error = 0.
    for j, value in enumerate(labels['response']):
        densities, error = component_batch_log_evidence(state['component_means'][:, j],
            state['component_local_covariances'][:, j], value[0], value[3], 3)
        log_densities.append(float(logsumexp(state['log_weights']+densities))); max_error = max(max_error, error)
    return float(np.mean(log_densities)), max_error


def score(row, state, labels):
    observed, noise = np.asarray(labels['response'])[:, 0], np.asarray(labels['noise'])[:, 0]
    p = row['prediction']; n = len(observed); raw, density_error = raw_density(state, labels)
    moment = gaussian_score(np.asarray(p['mean']), np.asarray(p['covariance'])+np.diag(noise), observed)/n
    actual = float(logsumexp([weight+gaussian_score(mean, covariance+np.diag(noise), observed)
        for weight, mean, covariance in zip(state['log_weights'], state['component_means'][:, :, 0],
            state['f_component_covariances'])])/n)
    paid = p['paid_predictive_log_score']; denominator = n+row['episode_count']
    return {'f_mse': float(np.mean((np.asarray(p['mean'])-observed)**2)), 'f_gaussian_mean_density': moment,
        'f_raw_batch_density': raw, 'raw_paid_log_evidence': paid,
        'sequential_transcript_density': (paid+n*moment)/denominator,
        'actual_mixture_mean_density': actual, 'actual_mixture_transcript_density': (paid+n*actual)/denominator}, density_error


def run():
    started = time.process_time(); spec = read(EXECUTION); reference_spec = read(spec['reference_execution'])
    centering = read(read(reference_spec['source_execution'])['centering_execution'])
    coverage = read(centering['coverage_execution']); parent = read(coverage['source_execution'])
    adaptation = read(reference_spec['adaptation_execution']); rules = read(spec['source_rules'])
    assert rules['all14_rules_frozen_without_query_data']
    prefixes = read(spec['public_prefixes'])['rows']; old_mixture = read(spec['reference_profiles'])['rows']
    old_native = [r for r in read(reference_spec['native_f_profiles'])['rows'] if r['case'] == 'native_f']
    controls = read(spec['controls_profiles'])['rows']
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(x) for x in public['bank']]
    branches = {}; frozen = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            rule = next(r for r in rules['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
            source = rule['source_reference_law']; names = source['source_task_names']
            assert public['tasks'][deleted]['task_id'] not in names
            assert names == [t['name'] for t in inputs['source_tasks']] == [t['name'] for t in inputs['risk_source_tasks']]
            assert source['native_law']['source_task_count'] == 5 and source['native_law']['source_label_count'] == 60
            prefix = next(r['public_prefix'] for r in prefixes if r['layout'] == layout and r['deleted_task'] == deleted)
            sites = coverage['layouts'][layout][deleted]; assert not set(prefix).intersection(sites)
            basis, risk_basis = (OrthogonalLowOrderBasis(inputs[key]) for key in ('public_mean_reference', 'public_risk_reference'))
            psi = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][deleted])), bank)
                -inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            eta_law = next(h['joint_structural'] for h in risk['heads'] if h['head'] == 3)
            eta = source_prior_moments(restore_joint(risk_basis, eta_law), np.asarray(inputs['public_risk_reference']))
            branches[(layout, deleted)] = (rule, basis, psi, eta, prefix, sites)
            frozen.append({'layout': layout, 'deleted_task': deleted, 'source_task_names': names,
                'source_node_count': len(rule['variance_nodes']), 'prefix': prefix, 'scoring_sites': sites})
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'frozen_inputs.json', {'execution': spec, 'all12_source_rules_and_prefixes_frozen_before_paid_values': True, 'rows': frozen})
    raw = read(coverage['source_data']); paid = {key: [raw['tasks'][key[1]]['Y_replicates'][j][0][0]
        for j in b[4]] for key, b in branches.items()}; del raw
    rows, numerical, refinement_files = [], [], {}
    for (layout, deleted), (rule, basis, psi, eta, prefix, sites) in branches.items():
        source = rule['source_reference_law']
        if (layout, deleted) == ('6', 1):
            native = mixture_field(basis, psi, source, np.asarray([source['native_task_intercept_variance']]), np.ones(1), eta)
            evidence = sum(native.update_batch(j, float(y), None, 1) for j, y in zip(prefix[:2], paid[(layout, deleted)][:2]))
            old = next(r for r in old_native if r['layout'] == layout and r['deleted_task'] == deleted)['predictions']['raw_paid']
            p = prediction(compact(native, sites, evidence))
            for key in p: np.testing.assert_allclose(p[key], old[key], atol=1e-8, rtol=1e-8)
        nodes, weights = rule_on_intervals(source, rule['adaptive_u_intervals'], 32)
        selected = mixture_field(basis, psi, source, np.asarray(rule['variance_nodes']), np.asarray(rule['posterior_weights']), eta)
        refined = mixture_field(basis, psi, source, nodes, weights, eta)
        selected_evidence = refined_evidence = 0.
        for count, (j, value) in enumerate(zip(prefix, paid[(layout, deleted)]), 1):
            selected_evidence += selected.update_batch(j, float(value), None, 1)
            refined_evidence += refined.update_batch(j, float(value), None, 1)
            if count not in spec['raw_episode_budgets']: continue
            state, refined_state = compact(selected, sites, selected_evidence), compact(refined, sites, refined_evidence)
            p, rp = prediction(state), prediction(refined_state)
            if count == 2:
                old = next(r for r in old_mixture if r['layout'] == layout and r['deleted_task'] == deleted)['predictions']['raw_paid']
                for key in p: np.testing.assert_allclose(p[key], old[key], atol=1e-8, rtol=1e-8)
                continue
            e = errors(p, rp); passed = e['mean_max_absolute'] <= 1e-7 and e['covariance_relative_frobenius'] <= 1e-5 and e['paid_log_evidence_absolute'] <= 1e-6
            numerical.append({'layout': layout, 'deleted_task': deleted, 'episode_count': count, 'errors': e, 'passed': passed})
            filename = f'layout{layout}_task{deleted}_episodes{count}.npz'
            np.savez_compressed(out/filename, **state)
            if deleted == 1:
                refined_filename = f'layout{layout}_task{deleted}_episodes{count}_refined.npz'
                np.savez_compressed(out/refined_filename, **refined_state)
                refinement_files[(layout, count)] = refined_filename
            rows.append({'layout': layout, 'deleted_task': deleted, 'episode_count': count, 'arm': ARM,
                'scoring_sites': sites, 'paid_prefix': prefix[:count], 'prediction': p, 'refined_prediction': rp,
                'component_marginals_file': filename, 'component_count': len(rule['variance_nodes']), 'refined_component_count': len(nodes)})
            print('variance mixture forecast frozen', layout, deleted, count, passed, flush=True)
        del selected, refined, state, refined_state
    save(out/'predictions.json', {'execution': spec, 'all24_forecasts_frozen_before_scored_labels': True,
        'cached_two_episode_moments_and_evidence_reproduced': True, 'numerical_checks': numerical, 'rows': rows})
    base = {'execution': spec, 'numerical_checks': numerical, 'cost': spec['cost'], 'no_model_adopted': True,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False, 'original_complete_gate_unchanged': True}
    if not all(r['passed'] for r in numerical):
        save(out/'results.json', {**base, 'numerical_forecasts_verified': False, 'forecasts_scored': False,
            'scientific_gate_interpreted': False, 'cpu_seconds': time.process_time()-started})
        print('Numerical mismatch retained;no scoring', flush=True); return
    labels = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks'] for layout in spec['layouts']}
    density_checks = []
    for row in rows:
        label = labels[row['layout']][row['deleted_task']]
        with np.load(out/row['component_marginals_file']) as state:
            row['scores'], row['density_integration_error_bound'] = score(row, state, label)
        if row['deleted_task'] == 1:
            with np.load(out/refinement_files[(row['layout'], row['episode_count'])]) as state:
                density, error = raw_density(state, label)
            difference = abs(density-row['scores']['f_raw_batch_density'])
            density_checks.append({'layout': row['layout'], 'episode_count': row['episode_count'],
                'raw_density_refinement_error': difference, 'passed': difference <= 1e-5})
        print('variance mixture scored', row['layout'], row['deleted_task'], row['episode_count'], flush=True)
    cached = [{'layout': r['layout'], 'deleted_task': r['deleted_task'], 'episode_count': 2, 'arm': ARM,
        'scores': {**r['scores']['raw_paid'], 'actual_mixture_mean_density': r['mixture_mean_densities']['raw_paid'],
            'actual_mixture_transcript_density': r['mixture_transcript_density']}} for r in old_mixture]
    scored = rows+cached; groups = group_summary(scored, spec)
    summary = {layout: groups[layout]['all6'] for layout in spec['layouts']}; gates = {}; contrasts = []
    for count in spec['raw_episode_budgets']:
        gates[str(count)] = {}
        for layout in spec['layouts']:
            actual = summary[layout][str(count)]; gates[str(count)][layout] = {}
            for arm in ('native_f', 'flat', 'source_gp'):
                members = [r for r in controls if r['layout'] == layout and r['episode_count'] == count and r['arm'] == arm]
                reference = {key: float(np.mean([r['scores'][key] for r in members])) for key in members[0]['scores']}
                gates[str(count)][layout][arm] = {'mse_nonincrease': actual['f_mse'] <= reference['f_mse'],
                    'raw_density_strict_improvement': actual['f_raw_batch_density'] > reference['f_raw_batch_density'],
                    'transcript_density_nondecrease': actual['sequential_transcript_density'] >= reference['sequential_transcript_density']}
                for row in scored:
                    if row['layout'] != layout or row['episode_count'] != count: continue
                    control = next(r for r in members if r['deleted_task'] == row['deleted_task'])
                    contrasts.append({'layout': layout, 'deleted_task': row['deleted_task'], 'episode_count': count,
                        'reference': arm, 'mixture_minus_control': {key: row['scores'][key]-control['scores'][key] for key in reference},
                        'actual_mixture_transcript_minus_control': row['scores']['actual_mixture_transcript_density']-control['scores']['sequential_transcript_density']})
    verified = all(r['passed'] for r in density_checks)
    advantage = {count: all(all(g.values()) for layout in layouts.values() for g in layout.values())
        for count, layouts in gates.items()} if verified else None
    result = {**base, 'summary': summary, 'descriptive_groups': groups, 'matched_contrasts': contrasts,
        'gates_by_episode_count': gates, 'mixture_advantage_by_episode_count': advantage,
        'numerical_forecasts_verified': True, 'numerical_density_verified': verified, 'density_checks': density_checks,
        'forecasts_scored': True, 'scientific_gate_interpreted': verified, 'cpu_seconds': time.process_time()-started,
        'compact_component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz'))}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': scored}); save(out/'results.json', result)
    print({'summary': summary, 'mixture_advantage': advantage, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
