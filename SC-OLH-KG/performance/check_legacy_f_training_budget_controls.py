"""Matched cached flat/GP f priors on the fixed public information prefix."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_training_information_budget import score_new, group_summary
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_source_mean_covariance_swap import g_prediction
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates, source_prior_moments
from representation.source_kernel_population import predict_unequal_site_kernel_population

EXECUTION = 'performance/manifests/legacy_joint_population_f_training_budget_controls_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); information = read(spec['information_execution'])
    coverage = read(information['coverage_execution']); parent = read(coverage['source_execution'])
    centering = read(information['centering_execution']); profile = read(centering['profile_execution'])
    adaptation = read(information['adaptation_execution']); prefixes = read(spec['public_prefixes'])['rows']
    native_rows = read(spec['information_profiles'])['rows']
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    branches, frozen = {}, []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            names = [t['name'] for t in inputs['source_tasks']]
            assert public['tasks'][deleted]['task_id'] not in names
            assert names == [t['name'] for t in inputs['risk_source_tasks']]
            head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
            prefix = next(r['public_prefix'] for r in prefixes if r['layout'] == layout and r['deleted_task'] == deleted)
            sites = coverage['layouts'][layout][deleted]; assert not set(sites).intersection(prefix)
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            basis, risk_basis = (OrthogonalLowOrderBasis(inputs[key]) for key in ('public_mean_reference', 'public_risk_reference'))
            eta_law = next(h['joint_structural'] for h in risk['heads'] if h['head'] == 3)
            eta = source_prior_moments(restore_joint(risk_basis, eta_law), np.asarray(inputs['public_risk_reference']))
            psi = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][deleted])), bank)
                -inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            for arm, key in (('flat', 'flat'), ('source_gp', 'joint_gp')):
                law = head[key]
                assert law['optimizer_converged'] and law['source_task_count'] == 5 and law['source_label_count'] == 60
                f = source_prior_moments(restore_joint(basis, law), psi) if arm == 'flat' else predict_unequal_site_kernel_population(psi, law)
                n = len(bank); mean = np.column_stack([f[0], eta[0]]); covariance = np.zeros((n, 2, n, 2))
                covariance[:, 0, :, 0], covariance[:, 1, :, 1] = f[1], eta[1]
                field = JointFunctionRiskPosterior.from_moments(mean, covariance)
                m, c = field.predict(full_cov=True)
                np.testing.assert_array_equal(m[:, 1], eta[0]); np.testing.assert_array_equal(c[:, 1, :, 1], eta[1])
                branches[(layout, deleted, arm)] = {'field': field, 'prefix': prefix, 'sites': sites}
                frozen.append({'layout': layout, 'deleted_task': deleted, 'arm': arm, 'prefix': prefix,
                    'source_fit': str(Path(profile['output'])/suffix/'selected_source_fit.json'), 'source_fit_key': key,
                    'source_task_names': names, 'same_eta_f_prior': True})
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'frozen_priors.json', {'execution': spec, 'all24_priors_frozen_before_paid_values': True, 'rows': frozen})
    raw = read(coverage['source_data']); paid = {key: [raw['tasks'][key[1]]['Y_replicates'][j][0][0]
        for j in b['prefix']] for key, b in branches.items()}; del raw
    rows, fields = [], {}
    for key, b in branches.items():
        layout, deleted, arm = key; field = b['field']; evidence = 0.
        for count, (j, value) in enumerate(zip(b['prefix'], paid[key]), 1):
            evidence += field.update_batch(j, float(value), None, 1)
            if count not in spec['raw_episode_budgets']: continue
            mean, covariance = field.predict(full_cov=True)
            prediction = g_prediction(mean[:, 0], covariance[:, 0, :, 0], b['sites'])
            prediction['paid_predictive_log_score'] = float(evidence)
            rows.append({'layout': layout, 'deleted_task': deleted, 'arm': arm, 'episode_count': count,
                'scoring_sites': b['sites'], 'paid_prefix': b['prefix'][:count], 'prediction': prediction})
            fields[(*key, count)] = deepcopy(field)
        print('matched budget forecasts frozen', layout, deleted, arm, flush=True)
    save(out/'predictions.json', {'execution': spec, 'all72_forecasts_frozen_before_score_labels': True, 'rows': rows})
    labels = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks'] for layout in spec['layouts']}
    for row in rows:
        key = (row['layout'], row['deleted_task'], row['arm'], row['episode_count'])
        row['scores'] = score_new(row, fields[key], labels[key[0]][key[1]])
    scored = rows+[{**r, 'arm': 'native_f'} for r in native_rows]
    groups = {arm: group_summary([r for r in scored if r['arm'] == arm], spec) for arm in spec['mean_arms']}
    summary = {layout: {arm: groups[arm][layout]['all6'] for arm in spec['mean_arms']} for layout in spec['layouts']}
    gates = {}; contrasts = []
    for count in spec['raw_episode_budgets']:
        gates[str(count)] = {}
        for layout in spec['layouts']:
            actual = summary[layout]['native_f'][str(count)]
            gates[str(count)][layout] = {}
            for arm in ('flat', 'source_gp'):
                reference = summary[layout][arm][str(count)]
                gates[str(count)][layout][arm] = {'mse_nonincrease': actual['f_mse'] <= reference['f_mse'],
                    'raw_density_strict_improvement': actual['f_raw_batch_density'] > reference['f_raw_batch_density'],
                    'transcript_density_nondecrease': actual['sequential_transcript_density'] >= reference['sequential_transcript_density']}
        for row in scored:
            if row['arm'] != 'native_f' or row['episode_count'] != count: continue
            for arm in ('flat', 'source_gp'):
                reference = next(r for r in rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task']
                    and r['arm'] == arm and r['episode_count'] == count)
                contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'episode_count': count,
                    'reference': arm, 'native_minus_control': {k: row['scores'][k]-reference['scores'][k] for k in row['scores']}})
    passed = {count: all(all(g.values()) for layout in layouts.values() for g in layout.values()) for count, layouts in gates.items()}
    result = {'execution': spec, 'summary': summary, 'descriptive_groups': groups, 'matched_contrasts': contrasts,
        'gates_by_episode_count': gates, 'native_advantage_by_episode_count': passed, 'cost': spec['cost'],
        'cpu_seconds': time.process_time()-started, 'no_model_adopted': True, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False, 'original_complete_gate_unchanged': True}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': scored}); save(out/'results.json', result)
    print({'summary': summary, 'native_advantage': passed, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
