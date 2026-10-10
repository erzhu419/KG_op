"""Cheap opened-target diagnosis of source-policy-layout instability."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import encode_diverse_source
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast
from performance.check_legacy_source_layout_robustness import models_at_query, ARMS, CASES
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_independent_pilot import target_problem
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.public_control_mean import public_control_mean_coordinates

EXECUTION = 'performance/manifests/legacy_joint_population_source_layout_target_diagnosis_v1.json'


def run(execution=EXECUTION):
    started = time.process_time(); spec = read(execution); source = read(spec['source_execution'])
    data = read(spec['source_data']); public = read(Path(spec['source_data']).with_name('frozen_inputs.json'))
    inputs = read(spec['target_inputs']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    bank = [tuple(p) for p in inputs['bank']]
    source_desc = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], t)),
        [tuple(p) for p in public['bank']], 50) for i, t in enumerate(public['tasks'])}
    providers = {t['task_id']: no_outcomes(target_problem(inputs['target_plan'], t['config'])) for t in inputs['targets']}
    target_desc = {k: legacy_exposure_descriptors(p, bank, 50) for k, p in providers.items()}
    target_mean = {k: public_control_mean_coordinates(p, bank) for k, p in providers.items()}
    rows, snapshots = [], []
    for name, sites in source['layouts'].items():
        folder = Path(source['output'])/f'layout{name}'
        members = [(read(folder/'inputs.json'), read(folder/'source_fit.json'), read(folder/'risk_source_fit.json'))]
        members += [(read(folder/f'task{i}'/'inputs.json'), read(folder/f'task{i}'/'gp_source_fit.json'),
                     read(folder/f'task{i}'/'risk_source_fit.json')) for i in source['training_tasks']]
        maps = []
        for deleted, (inp, _, _) in zip([None]+source['training_tasks'], members):
            keep = [(i, s) for i, s in zip(source['training_tasks'], sites) if i != deleted]
            coordinate, _, center, scale = encode_diverse_source(data, [i for i, _ in keep], [s for _, s in keep],
                source_desc, public['plan']['task_distribution']['L'])
            np.testing.assert_array_equal(center, inp['risk_coordinate_center'])
            np.testing.assert_array_equal(scale, inp['risk_coordinate_scale'])
            np.testing.assert_array_equal(coordinate.transform(source_desc[source['training_tasks'][0]]), inp['public_risk_reference'])
            maps.append(coordinate)
        for target in inputs['targets']:
            task = target['task_id']; query = {'task_id': task, 'task_index': len(snapshots)%len(inputs['targets']),
                'kind': 'target', 'tau': target['tau'], 'alpha': providers[task].alpha, **target['initialization']}
            mean_queries = [((target_mean[task]-np.asarray(m[0]['mean_coordinate_center']))/
                np.asarray(m[0]['mean_coordinate_scale'])).tolist() for m in members]
            risk_queries = [c.transform(target_desc[task]).tolist() for c in maps]
            snapshots.append({'layout': name, 'query': query, 'mean_queries': mean_queries, 'risk_queries': risk_queries})
            for a, model in models_at_query(members, mean_queries, risk_queries):
                prior = forecast(query, a, CASES[a], model)
                for j, value in zip(query['candidate_indices'], query['paired_observations']):
                    model.update_raw(j, [value[0], value[1]-query['tau']])
                row = forecast(query, a, CASES[a], model); row['layout'] = name; row['prior'] = prior; rows.append(row)
        print('source layout target forecasts frozen', name, flush=True)
    save(out/'inputs.json', {'execution': spec, 'snapshots': snapshots})
    save(out/'target_predictions.json', {'execution': spec, 'rows': rows})
    # Both layouts and all36 forecasts precede cached-target truth scoring.
    scoring = read(spec['target_scoring']); points = []
    for row in rows:
        fold = next(f for f in scoring['folds'] if f['task_id'] == row['task_id'])
        truth = np.asarray(fold['scoring_truth'])[:, [0, 2, 3, 1]]
        mean, variance = np.asarray(row['mean']), np.asarray(row['variance']); error = mean-truth
        row['scores'] = {'truth_mse': (error[2:]**2).mean(axis=0).tolist(),
            'moment_interval_95_truth_coverage': (abs(error[2:]) <= 1.96*np.sqrt(variance[2:])).mean(axis=0).tolist()}
        alpha = providers[row['task_id']].alpha
        for j in spec['diagnostic_candidates']:
            points.append({'layout': row['layout'], 'task_id': row['task_id'], 'mean_arm': row['mean_arm'], 'candidate': j,
                'prior_bias': (np.asarray(row['prior']['mean'][j])-truth[j]).tolist(),
                'posterior_bias': error[j].tolist(), 'posterior_sd': np.sqrt(variance[j]).tolist(),
                'mean_eta_plugin_margin': float(mean[j, 1]+norm.ppf(1-alpha)*np.exp(mean[j, 2]/2)),
                'true_chance_margin': float(truth[j, 1]+norm.ppf(1-alpha)*np.exp(truth[j, 2]/2))})
    reference_risk = read(spec['reference_risk_profiles'])['rows']; reference_gp = read(spec['reference_gp_profiles'])['rows']
    references = [r for r in reference_risk if r['kind']=='target' and r['mean_arm'] in ARMS[:2] and r['risk_case']==CASES[r['mean_arm']]]
    references += reference_gp
    all_rows = [{**r, 'layout':'0'} for r in references]+rows
    summary = {name: {a: {k: np.mean([r['scores'][k] for r in all_rows if r['layout']==name and r['mean_arm']==a],axis=0).tolist()
        for k in rows[0]['scores']} for a in ARMS} for name in ('0',*source['layouts'])}
    save(out/'calibration_profiles.json', {'rows': rows})
    result = {'execution': spec, 'summary': summary, 'diagnostic_rows': points, 'source_map_reproduction_passed': True,
        'cost': {'new_source_fits': 0, 'source_map_reconstructions': 14, 'new_simulator_calls': 0,
            'new_verification_calls': 0, 'new_terminal_integrations': 0}, 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result); print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else EXECUTION)
