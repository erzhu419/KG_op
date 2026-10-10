"""Source common mean versus covariance transplant, with the raw risk filter."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_paid_covariance_gain import fixed_noise_condition
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.joint_function_risk import JointFunctionRiskPosterior, joint_batch_log_score
from representation.public_control_mean import public_control_mean_coordinates

EXECUTION = 'performance/manifests/legacy_joint_population_source_mean_covariance_swap_execution_v1.json'


def g_prediction(mean, covariance, sites):
    return {'mean': np.asarray(mean)[sites].tolist(), 'covariance': np.asarray(covariance)[np.ix_(sites, sites)].tolist()}


def run():
    started = time.process_time(); spec = read(EXECUTION); gain = read(spec['gain_execution'])
    adaptation = read(gain['adaptation_execution']); coverage = read(adaptation['coverage_execution'])
    deletion, parent = read(adaptation['deleted_mean_execution']), read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    references = read(Path(read(gain['adaptation_results'])['execution']['output'])/'predictions.json')['rows']
    raw = read(coverage['source_data']); paid_sites = spec['paid_candidates']
    values = {i: [raw['tasks'][i]['Y_replicates'][j][0][1]-raw['tasks'][i]['tau'] for j in paid_sites] for i in spec['training_tasks']}
    del raw
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); rows, models = [], {}
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            branch = Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'
            inputs, gp = read(branch/'inputs.json'), read(branch/'gp_source_fit.json')
            means = read(Path(deletion['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
            means = dict(means, heads=[dict(h, joint_gp=g['joint_gp']) for h, g in zip(means['heads'], gp['heads'])])
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            fields = components(inputs, means, risk, psi, np.asarray(inputs['public_risk_reference']))
            native = {arm: pair[1].predict(full_cov=True) for arm, pair in zip(adaptation['mean_arms'], fields) if arm != 'flat'}
            sm, sc = native['joint_structural']; gm, gc = native['joint_gp']
            np.testing.assert_array_equal(sm[:, 1], gm[:, 1]); np.testing.assert_array_equal(sc[:, 1, :, 1], gc[:, 1, :, 1])
            episode_noise = np.exp(sm[:, 1]+.5*np.diag(sc[:, 1, :, 1])); n = len(sm)
            sites = coverage['layouts'][layout][deleted]
            for case, parts in spec['cases'].items():
                mean = native[parts['mean']][0][:, 0]
                covariance = native[parts['covariance']][1][:, 0, :, 0]
                m = np.column_stack([mean, sm[:, 1]]); c = np.zeros((n, 2, n, 2))
                c[:, 0, :, 0], c[:, 1, :, 1] = covariance, sc[:, 1, :, 1]
                prior = JointFunctionRiskPosterior.from_moments(m, c); paid = deepcopy(prior)
                for j, y in zip(paid_sites, values[deleted]):
                    paid.update_batch(j, float(y), None, 1)
                pm, pc = paid.predict(full_cov=True)
                if parts['mean'] == parts['covariance']:
                    reference = next(r for r in references if r['layout'] == layout and r['deleted_task'] == deleted and r['mean_arm'] == parts['mean'])
                    for actual_m, actual_c, old in ((m, c, reference['prior']), (pm, pc, reference)):
                        np.testing.assert_allclose(actual_m, np.asarray(old['mean'])[:, [1, 2]], atol=1e-8)
                        np.testing.assert_allclose(np.diag(actual_c.reshape(2*n, 2*n)).reshape(n, 2), np.asarray(old['variance'])[:, [1, 2]], atol=1e-8)
                fm, fc, _ = fixed_noise_condition(mean, covariance, paid_sites, values[deleted], episode_noise[paid_sites])
                rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'parts': parts, 'scoring_sites': sites,
                    'predictions': {'prior': g_prediction(mean, covariance, sites), 'raw_paid': g_prediction(pm[:, 0], pc[:, 0, :, 0], sites),
                        'fixed_noise_paid': g_prediction(fm, fc, sites)}})
                models[(layout, deleted, case)] = prior, paid
        print('source mean/covariance swap forecasts frozen', layout, flush=True)
    save(out/'predictions.json', {'execution': spec, 'all48_case_forecasts_frozen': True, 'SS_GG_all73_moments_reproduced': True, 'rows': rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        layout, deleted, case = row['layout'], row['deleted_task'], row['case']
        labels = source[layout]['source_tasks'][deleted]; observed = np.asarray(labels['response']); noise = np.asarray(labels['noise'])[:, 1]
        row['scores'] = {}
        for stage, prediction in row['predictions'].items():
            mean, covariance = np.asarray(prediction['mean']), np.asarray(prediction['covariance'])
            row['scores'][stage] = {'g_mse': float(np.mean((mean-observed[:, 1])**2)),
                'g_gaussian_mean_density': gaussian_score(mean, covariance+np.diag(noise), observed[:, 1])/len(mean)}
        for stage, model in zip(('prior', 'raw_paid'), models[(layout, deleted, case)]):
            row['scores'][stage]['g_raw_batch_density'] = float(np.mean([joint_batch_log_score(model.local_components(j), value[1], value[2], 3)
                for j, value in zip(row['scoring_sites'], observed)]))
    summary = {layout: {case: {stage: {k: np.mean([r['scores'][stage][k] for r in rows if r['layout'] == layout and r['case'] == case]).item()
        for k in rows[0]['scores'][stage]} for stage in ('prior', 'raw_paid', 'fixed_noise_paid')} for case in spec['cases']} for layout in spec['layouts']}
    substitutions = []
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            group = {r['case']: r for r in rows if r['layout'] == layout and r['deleted_task'] == deleted}
            for stage in ('prior', 'raw_paid', 'fixed_noise_paid'):
                for changed, baseline, component in [('SG', 'SS', 'covariance_at_structural_mean'), ('GG', 'GS', 'covariance_at_gp_mean'),
                    ('GS', 'SS', 'mean_at_structural_covariance'), ('GG', 'SG', 'mean_at_gp_covariance')]:
                    a, b = group[changed]['scores'][stage], group[baseline]['scores'][stage]
                    substitutions.append({'layout': layout, 'deleted_task': deleted, 'stage': stage, 'component': component,
                        'changed_case': changed, 'baseline_case': baseline, 'differences': {k: a[k]-b[k] for k in a}})
    result = {'execution': spec, 'summary': summary, 'substitutions': substitutions, 'SS_GG_reproduction_passed': True,
        'no_model_selected': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
