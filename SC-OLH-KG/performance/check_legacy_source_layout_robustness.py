"""Frozen source-layout contrasts at144 calls with source-selected treatments."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_complete_source_uncertainty import mean_views
from performance.check_legacy_diverse_source_design import encode_diverse_source
from performance.check_legacy_public_mean_coordinates import fit as fit_mean
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.check_legacy_public_mean_source_half import source_views, fit_risk
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_response_mixture import JointResponseMixture
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments
from representation.source_kernel_population import (fit_unequal_site_kernel_population,
    fit_joint_common_unequal_site_kernel_population, predict_unequal_site_kernel_population)
from representation.source_known_noise_gp import fit_known_noise_gp

EXECUTION = 'performance/manifests/legacy_joint_population_source_layout_robustness_execution_v1.json'
ARMS = ('joint_structural', 'flat', 'joint_gp')
CASES = {'joint_structural': 'full_plus_six_relearned_map_deletion_risks',
         'flat': 'full_plus_six_relearned_map_deletion_risks',
         'joint_gp': 'full_plus_six_complete_source_deletions'}


def branch_views(data, public, source, deleted):
    bank = [tuple(p) for p in public['bank']]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], t)), bank, 50)
                   for i, t in enumerate(public['tasks'])}
    keep = [(i, s) for i, s in zip(source['training_tasks'], source['source_site_indices_by_task']) if i != deleted]
    coordinate, tasks, center, scale = encode_diverse_source(data, [i for i, _ in keep],
        [s for _, s in keep], descriptors, public['plan']['task_distribution']['L'])
    return {**mean_views(data, public, source, deleted),
        'risk_source_tasks': [{**t, **{k: t[k].tolist() for k in ('psi', 'features', 'response', 'noise')}} for t in tasks],
        'public_risk_reference': coordinate.transform(descriptors[source['training_tasks'][0]]).tolist(),
        'risk_coordinate_center': center.tolist(), 'risk_coordinate_scale': scale.tolist(),
        'coordinate': coordinate.diagnostics(),
        'heldout_risk_psi': {str(i): coordinate.transform(descriptors[i]).tolist() for i in source['heldout_tasks']}}


def fit_gp(spec, inputs):
    path = ROOT/spec['output']/'gp_source_fit.json'
    fitted = read(path) if path.exists() else {'heads': [], 'cpu_seconds': 0.}
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
    for h in range(len(fitted['heads']), 2):
        started = time.process_time()
        anchor = fit_known_noise_gp(tasks, h, spec['gp_anchor_max_iterations'])
        fixed = fit_unequal_site_kernel_population(tasks, h, anchor, optimizer=spec['optimizer'])
        joint = fit_joint_common_unequal_site_kernel_population(tasks, h, fixed, optimizer=spec['optimizer'])
        record = {'anchor': anchor, 'fixed_gp': fixed, 'joint_gp': joint}
        fitted['heads'].append(record); fitted['cpu_seconds'] += time.process_time()-started; save(path, fitted)
    if not all(d['optimizer_converged'] for h in fitted['heads'] for d in h.values()):
        raise RuntimeError('complete failed native GP source objectives before forecasts')
    return fitted


def models_at_query(members, mean_queries, risk_queries):
    full_inputs, full_means, _ = members[0]
    full_basis = OrthogonalLowOrderBasis(full_inputs['public_mean_reference'])
    full_psi = np.asarray(mean_queries[0])
    static = {a: [source_prior_moments(restore_joint(full_basis, h[a]), full_psi)
                  for h in full_means['heads']] for a in ARMS[:2]}
    fields = {a: [] for a in ARMS}
    for (inputs, means, risks), mean_psi, risk_psi in zip(members, mean_queries, risk_queries):
        risk_basis = OrthogonalLowOrderBasis(inputs['public_risk_reference'])
        eta = [source_prior_moments(restore_joint(risk_basis, h['joint_structural']),
            np.asarray(risk_psi)) for h in risks['heads']]
        gp = [predict_unequal_site_kernel_population(np.asarray(mean_psi), h['joint_gp'])
              for h in means['heads']]
        for a in ARMS:
            fields[a].append(paired_fields((gp if a == 'joint_gp' else static[a])+eta))
    names = ['full144']+[f'drop{i}' for i in range(6)]
    return [(a, JointResponseMixture(names, fields[a])) for a in ARMS]


def prepare_layout(spec, base, data, public, name, sites):
    out = ROOT/spec['output']/f'layout{name}'; out.mkdir(parents=True, exist_ok=True)
    source = dict(base, source_site_indices_by_task=sites, output=str(out.relative_to(ROOT)))
    inputs = {'execution': source, **source_views(data, public, source)}; save(out/'inputs.json', inputs)
    means, risks = fit_mean(source, inputs), fit_risk(source, inputs)
    members = [(inputs, means, risks)]; fitting_cpu = means['cpu_seconds']+risks['cpu_seconds']
    fits = 20+sum('continued_from_optimizer_iterations' in h['joint_structural'] for h in means['heads']+risks['heads'])
    branches = []
    for deleted in source['training_tasks']:
        branch = out/f'task{deleted}'; branch.mkdir(parents=True, exist_ok=True)
        view = branch_views(data, public, source, deleted); save(branch/'inputs.json', view)
        branch_spec = dict(source, output=str(branch.relative_to(ROOT)))
        eta, gp = fit_risk(branch_spec, view), fit_gp(branch_spec, view)
        members.append((view, gp, eta)); fitting_cpu += eta['cpu_seconds']+gp['cpu_seconds']
        fits += 12+sum('continued_from_optimizer_iterations' in h['joint_structural'] for h in eta['heads'])
        branches.append({'deleted_task': deleted, 'mean_design_rank': view['mean_source_design_rank'],
                         'input_records': sum(len(t['response']) for t in view['source_tasks'])})
    print('source layout fits frozen', name, 'native fits', fits, flush=True)
    return source, members, {'layout': name, 'historical_calls': 144, 'records': 48,
        'distinct_source_policies': len(set(j for s in sites for j in s)),
        'mean_design_rank': inputs['mean_source_design_rank'], 'risk_design_rank': inputs['risk_source_design_rank'],
        'candidate_source_coverage': {str(j): sum(j in s for s in sites) for j in spec['diagnostic_candidates']},
        'branches': branches, 'native_source_fits': fits, 'source_fit_cpu_seconds': fitting_cpu}


def run(execution=EXECUTION):
    started = time.process_time(); spec = read(execution); base = read(spec['reference_source_execution'])
    data = read(spec['source_data']); public = read(Path(spec['source_data']).with_name('frozen_inputs.json'))
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows, jobs, fits = [], {}, []
    for name, sites in spec['layouts'].items():
        source, members, fitting = prepare_layout(spec, base, data, public, name, sites); fits.append(fitting)
        for i in spec['heldout_tasks']:
            task = data['tasks'][i]; provider = no_outcomes(problem_for(public['plan'], public['tasks'][i]))
            query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source', 'tau': task['tau'],
                'alpha': provider.alpha, 'candidate_indices': [0, 1],
                'paired_observations': [task['Y_replicates'][j][0] for j in (0, 1)]}
            for a, model in models_at_query(members,
                [m[0]['heldout_mean_psi'][str(i)] for m in members],
                [m[0]['heldout_risk_psi'][str(i)] for m in members]):
                prior = forecast(query, a, CASES[a], model)
                for j, value in zip(query['candidate_indices'], query['paired_observations']):
                    model.update_raw(j, [value[0], value[1]-query['tau']])
                row = forecast(query, a, CASES[a], model); row['layout'] = name; row['prior'] = prior
                rows.append(row); jobs[(name, i, a)] = model
        save(out/'source_predictions.json', {'execution': spec, 'all_forecasts_frozen': False, 'rows': rows})
    save(out/'source_predictions.json', {'execution': spec, 'all_forecasts_frozen': True, 'rows': rows})
    # Both layouts, every native fit and all12 forecasts precede holdout scoring.
    reference_risk = read(spec['reference_risk_profiles'])['rows']
    reference_gp = read(spec['reference_gp_profiles'])['rows']
    references = [r for r in reference_risk if r['kind'] == 'source' and r['mean_arm'] in ARMS[:2]
                  and r['risk_case'] == CASES[r['mean_arm']]]
    references += [r for r in reference_gp if r['mean_arm'] == 'joint_gp']
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    contrasts, points = [], []
    for row in rows:
        i, a, name = row['task_index'], row['mean_arm'], row['layout']
        score_source(row, labels[i], jobs[(name, i, a)], 3)
        old = next(r for r in references if r['task_id'] == row['task_id'] and r['mean_arm'] == a)
        contrasts.append({'layout': name, 'task_id': row['task_id'], 'mean_arm': a,
            'g_mse_difference_from_layout0': row['scores']['observed_head_mse'][1]-old['scores']['observed_head_mse'][1],
            'joint_batch_score_difference_from_layout0': row['scores']['joint_batch_log_score']-old['scores']['joint_batch_log_score'],
            'g_batch_score_difference_from_layout0': row['scores']['response_batch_log_scores'][1]-old['scores']['response_batch_log_scores'][1]})
        for j in spec['diagnostic_candidates']:
            points.append({'layout': name, 'task_id': row['task_id'], 'mean_arm': a, 'candidate': j,
                'prior_mean': row['prior']['mean'][j], 'prior_sd': np.sqrt(row['prior']['variance'][j]).tolist(),
                'posterior_mean': row['mean'][j], 'posterior_sd': np.sqrt(row['variance'][j]).tolist(),
                'observed_labels': labels[i]['response'][j].tolist()})
    all_rows = [{**r, 'layout': '0'} for r in references]+rows
    summary = {name: {a: {k: np.mean([r['scores'][k] for r in all_rows if r['layout'] == name
        and r['mean_arm'] == a], axis=0).tolist() for k in rows[0]['scores']} for a in ARMS}
        for name in ('0', *spec['layouts'])}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows})
    result = {'execution': spec, 'summary': summary, 'layout_fits': fits, 'contrasts': contrasts,
        'diagnostic_rows': points, 'cost': {'historical_calls_per_model': 144,
            'new_native_source_fits': sum(f['native_source_fits'] for f in fits), 'new_simulator_calls': 0,
            'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else EXECUTION)
