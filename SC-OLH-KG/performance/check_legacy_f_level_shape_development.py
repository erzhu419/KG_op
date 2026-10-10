"""Two frozen task1 four-episode comparisons of coherent joint source mixtures."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_level_shape_vectorized_filter import prepare_branch
from performance.check_legacy_f_reference_training_budget import compact, prediction, errors, score, raw_density
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.vectorized_joint_function_risk import joint_variance_field, update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_shape_development_execution_v1.json'
ARM = 'joint_level_shape_variance_mixture'


def run():
    started = time.process_time(); spec = read(EXECUTION)
    assert read(spec['filter_results'])['all_scalar_equivalence_checks_passed']
    assert read(spec['compact_results'])['all12_compact_source_rules_verified']
    selected_package, reference_package = read(spec['compact_rules']), read(spec['reference_rules'])
    assert selected_package['all12_source_rules_frozen_without_query_data'] and reference_package['all12_source_rules_frozen_without_query_data']
    old = read(spec['mixture_execution']); controls = read(old['controls_profiles'])['rows']
    mixtures = read(Path(old['output'])/'calibration_profiles.json')['rows']
    branches, frozen = [], []
    for layout in spec['layouts']:
        deleted = spec['deleted_task']; branch = prepare_branch(spec, layout, deleted)
        selected = next(r for r in selected_package['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
        reference = next(r for r in reference_package['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
        assert selected['source_reference_law'] == reference['source_reference_law']
        assert selected['source_reference_law']['source_task_names'] == branch['source_task_names']
        control = next(r for r in controls if r['layout'] == layout and r['deleted_task'] == deleted
            and r['episode_count'] == spec['raw_episode_count'] and r['arm'] == 'native_f')
        assert control['paid_prefix'] == branch['prefix'] and control['scoring_sites'] == branch['sites']
        branches.append((layout, deleted, branch, selected, reference, control))
        frozen.append({'layout': layout, 'deleted_task': deleted, 'source_task_names': branch['source_task_names'],
            'paid_prefix': branch['prefix'], 'scoring_sites': branch['sites'], 'exact_marginal_sites': branch['selection'],
            'selected_source_rule_file': selected['source_rule_file'], 'reference_source_rule_file': reference['source_rule_file'],
            'source_component_count': selected['node_count'], 'reference_component_count': reference['node_count'],
            'full_public_basis_reference_preserved': True})
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'frozen_inputs.json', {'execution': spec, 'both_source_laws_and_prefixes_frozen_before_paid_values': True, 'rows': frozen})
    raw = read(branches[0][2]['coverage']['source_data'])
    paid = {layout: [raw['tasks'][deleted]['Y_replicates'][j][0][0] for j in branch['prefix']]
        for layout, deleted, branch, _, _, _ in branches}; del raw
    rows, numerical = [], []
    for layout, deleted, branch, selected, reference, control in branches:
        source = selected['source_reference_law']; score_indices = list(range(4, 16))
        native = joint_variance_field(branch['basis'], branch['psi'], source,
            np.asarray([[source['native_level_variance'], source['native_shape_variance']]]), np.ones(1), branch['eta'])
        evidence = sum(update_raw_observation(native, j, float(value)) for j, value in enumerate(paid[layout]))
        native_prediction = prediction(compact(native, score_indices, evidence))
        for key in native_prediction:
            np.testing.assert_allclose(native_prediction[key], control['prediction'][key], atol=1e-8, rtol=1e-8)
        states, forecasts = [], []
        for tag, rule in [('selected', selected), ('reference', reference)]:
            with np.load(ROOT/rule['source_rule_file']) as saved:
                field = joint_variance_field(branch['basis'], branch['psi'], source,
                    saved['physical_variance_nodes'], saved['posterior_weights'], branch['eta'])
            evidence = sum(update_raw_observation(field, j, float(value)) for j, value in enumerate(paid[layout]))
            state = compact(field, score_indices, evidence); states.append(state); forecasts.append(prediction(state))
            filename = f'layout{layout}_task{deleted}_episodes4_{tag}.npz'
            np.savez_compressed(out/filename, **state); del field
        difference = errors(forecasts[0], forecasts[1]); tolerance = spec['predictive_tolerances']
        passed = all(difference[k] <= tolerance[k] for k in difference)
        numerical.append({'layout': layout, 'deleted_task': deleted, 'errors': difference, 'passed': passed,
            'cached_native_four_episode_prediction_reproduced': True})
        rows.append({'layout': layout, 'deleted_task': deleted, 'episode_count': 4, 'arm': ARM,
            'paid_prefix': branch['prefix'], 'scoring_sites': branch['sites'], 'prediction': forecasts[0], 'refined_prediction': forecasts[1],
            'component_marginals_file': f'layout{layout}_task{deleted}_episodes4_selected.npz',
            'refined_component_marginals_file': f'layout{layout}_task{deleted}_episodes4_reference.npz',
            'component_count': selected['node_count'], 'reference_component_count': reference['node_count']})
        print('joint level/shape four-episode forecast frozen', layout, passed, difference, flush=True)
        del states, native
    save(out/'predictions.json', {'execution': spec, 'both_forecasts_frozen_before_score_labels': True, 'numerical_checks': numerical, 'rows': rows})
    base = {'execution': spec, 'numerical_checks': numerical, 'cost': spec['cost'], 'no_model_adopted': True,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False, 'original_complete_gate_unchanged': True}
    if not all(r['passed'] for r in numerical):
        save(out/'results.json', {**base, 'numerical_forecasts_verified': False, 'forecasts_scored': False,
            'scientific_gate_interpreted': False, 'cpu_seconds': time.process_time()-started}); return
    contrasts, gates, density_checks = [], {}, []
    for row in rows:
        elapsed = time.process_time()-started
        if elapsed >= spec['cost']['maximum_single_thread_cpu_seconds']:
            save(out/'results.json', {**base, 'numerical_forecasts_verified': True, 'forecasts_scored': False,
                'scientific_gate_interpreted': False, 'stopped_at_cpu_budget': True, 'cpu_seconds': elapsed}); return
        branch = next(b[2] for b in branches if b[0] == row['layout'])
        labels = read(Path(branch['coverage']['output'])/f"layout{row['layout']}"/'inputs.json')['source_tasks'][row['deleted_task']]
        with np.load(out/row['component_marginals_file']) as state:
            row['scores'], row['density_integration_error_bound'] = score(row, state, labels)
        with np.load(out/row['refined_component_marginals_file']) as state:
            density, error_bound = raw_density(state, labels)
        difference = abs(row['scores']['f_raw_batch_density']-density)
        density_checks.append({'layout': row['layout'], 'raw_density_refinement_error': difference,
            'reference_density_integration_error_bound': error_bound, 'passed': difference <= spec['density_refinement_tolerance']})
        gates[row['layout']] = {}
        for arm in ['native_f', 'flat', 'source_gp', 'source_variance_mixture']:
            members = mixtures if arm == 'source_variance_mixture' else controls
            control = next(r for r in members if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task']
                and r['episode_count'] == row['episode_count'] and r['arm'] == arm)
            assert control['paid_prefix'] == row['paid_prefix'] and control['scoring_sites'] == row['scoring_sites']
            delta = {key: row['scores'][key]-control['scores'][key] for key in ['f_mse', 'f_raw_batch_density', 'sequential_transcript_density']}
            gates[row['layout']][arm] = {'mse_nonincrease': delta['f_mse'] <= 0., 'raw_density_strict_improvement': delta['f_raw_batch_density'] > 0.,
                'transcript_density_nondecrease': delta['sequential_transcript_density'] >= 0.}
            contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'reference': arm, 'candidate_minus_control': delta,
                'mse_relative_change': delta['f_mse']/control['scores']['f_mse']})
        print('joint level/shape task1 scores', row['layout'], row['scores'], flush=True)
    verified = all(r['passed'] for r in density_checks)
    result = {**base, 'rows': rows, 'gates': gates, 'matched_contrasts': contrasts, 'density_checks': density_checks,
        'numerical_forecasts_verified': True, 'numerical_density_verified': verified, 'forecasts_scored': True, 'scientific_gate_interpreted': verified,
        'pilot_advantage_against_all_four_controls': all(all(g.values()) for layout in gates.values() for g in layout.values()) if verified else None,
        'cpu_seconds': time.process_time()-started, 'compact_component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz')),
        'process_peak_rss_bytes': int(next(line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:')).split()[1])*1024}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'pilot_advantage': result['pilot_advantage_against_all_four_controls'], 'gates': gates, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
