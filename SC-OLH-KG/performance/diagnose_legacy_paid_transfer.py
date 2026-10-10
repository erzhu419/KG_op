"""Exact prefix, identity-weight and Gaussian gain attribution of fixed priors."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_common_task_attribution import assemble_common_task
from performance.check_legacy_counterbalanced_complete_mean_uncertainty import load_group
from performance.check_legacy_crossfit_variance_noise import risk_models
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save, training_view
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis

EXECUTION = 'performance/manifests/legacy_joint_population_paid_transfer_attribution_execution_v1.json'
ARM = 'joint_structural'


def means_by_identity(model):
    return np.asarray([g.predict()[0][:, 0] for _, g in model.components])


def mse(prediction, observed):
    return float(np.mean((prediction[2:]-observed[2:])**2))


def rms(vector):
    return float(np.sqrt(np.mean(vector[2:]**2)))


def update_with_gain(model, candidate, values):
    before = []
    for _, field in model.components:
        assert len(field.means) == 1
        mean, covariance = field.predict(full_cov=True)
        gain = np.linalg.solve(covariance[candidate, :, candidate, :], covariance[:, 0, candidate, :].T).T
        before.append((mean, gain))
    model.update_raw(candidate, values); results = []
    for (_, field), (previous, gain) in zip(model.components, before):
        current, _ = field.predict()
        innovation = current[candidate]-previous[candidate]
        np.testing.assert_allclose(current[:, 0]-previous[:, 0], gain@innovation, atol=1e-10, rtol=1e-8)
        results.append({'gain': gain, 'innovation': innovation, 'update': current[:, 0]-previous[:, 0]})
    return results


def run():
    started = time.process_time(); spec = read(EXECUTION); component = read(spec['component_execution'])
    native, restricted = read(component['native_execution']), read(component['restricted_execution'])
    coverage = read(native['coverage_execution']); parent = read(coverage['source_execution'])
    saved = read(Path(component['output'])/'calibration_profiles.json')['rows']
    prefixes = read(Path(component['output'])/'paid_prefix_means.json')['rows']
    raw = read(coverage['source_data']); public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    labels = {i: np.asarray(training_view(raw, [i], public['plan']['task_distribution']['L'])[0]['response'])[:, 1]
        for i in spec['heldout_tasks']}
    rows, detailed, gain_rows = [], [], []
    for layout in spec['layouts']:
        group = load_group(native, coverage, parent, layout); full = group[0][0]
        nf = read(Path(native['output'])/f'layout{layout}'/'selected_source_fit.json')
        rf = read(Path(restricted['output'])/f'layout{layout}'/'source_fit.json')
        nl, rl = nf['heads'][1][ARM], rf['heads'][1][ARM]
        basis = OrthogonalLowOrderBasis(full['public_mean_reference'])
        for task_index in spec['heldout_tasks']:
            mq = np.asarray(full['heldout_mean_psi'][str(task_index)])
            rq = [np.asarray(g[0]['heldout_risk_psi'][str(task_index)]) for g in group]
            design = np.column_stack([np.ones(len(mq)), basis.transform(mq)])
            template = risk_models(group, nf, mq, rq, [ARM])[0][1]
            models = [assemble_common_task(template, design, nl, task) for task in (nl, rl)]
            observed = labels[task_index]; task = raw['tasks'][task_index]; previous_difference = 0.
            for prefix in range(len(spec['paid_candidates'])+1):
                gains = None
                if prefix:
                    candidate = spec['paid_candidates'][prefix-1]; value = task['Y_replicates'][candidate][0]
                    gains = [update_with_gain(model, candidate, [value[0], value[1]-task['tau']]) for model in models]
                matrices = [means_by_identity(model) for model in models]
                weights = [model.weights.copy() for model in models]
                mixed = [w@m for w, m in zip(weights, matrices)]
                for case, prediction in zip(spec['cases'], mixed):
                    if not prefix:
                        old = next(r for r in saved if r['layout'] == layout and r['task_index'] == task_index and r['case'] == case)
                        expected = np.asarray(old['prior']['mean'])[:, 1]
                    else:
                        old = next(r for r in prefixes if r['layout'] == layout and r['task_index'] == task_index and r['case'] == case and r['paid_count'] == prefix)
                        expected = np.asarray(old['g_mean'])
                    np.testing.assert_allclose(prediction, expected, atol=1e-8, rtol=1e-8)
                within = weights[0]@(matrices[1]-matrices[0])
                reweight = (weights[1]-weights[0])@matrices[1]
                np.testing.assert_allclose(mixed[1]-mixed[0], within+reweight, atol=1e-12)
                old_mse, new_mse = mse(mixed[0], observed), mse(mixed[1], observed)
                intermediate_mse = mse(weights[0]@matrices[1], observed)
                difference = new_mse-old_mse
                if not prefix:
                    np.testing.assert_allclose(mixed[0], mixed[1], atol=1e-12)
                if prefix == len(spec['paid_candidates']):
                    endpoint = {case: next(r for r in saved if r['layout'] == layout and r['task_index'] == task_index and r['case'] == case)
                        for case in spec['cases']}
                    np.testing.assert_allclose(difference, endpoint[spec['cases'][1]]['stage_metrics']['raw_paid']['g_mse']-
                        endpoint[spec['cases'][0]]['stage_metrics']['raw_paid']['g_mse'], atol=1e-12)
                row = {'layout': layout, 'task_index': task_index, 'paid_count': prefix,
                    'native_g_mse': old_mse, 'restricted_task_g_mse': new_mse,
                    'restricted_minus_native_g_mse': difference, 'incremental_mse_difference': difference-previous_difference,
                    'within_identity_mse_effect': intermediate_mse-old_mse, 'identity_reweighting_mse_effect': new_mse-intermediate_mse,
                    'within_identity_mean_shift_rms': rms(within), 'identity_reweighting_mean_shift_rms': rms(reweight)}
                rows.append(row); previous_difference = difference
                detailed.append({'layout': layout, 'task_index': task_index, 'paid_count': prefix,
                    'identity_names': models[0].names, 'native_weights': weights[0].tolist(), 'restricted_weights': weights[1].tolist(),
                    'native_identity_g_means': matrices[0].tolist(), 'restricted_identity_g_means': matrices[1].tolist()})
                if gains:
                    propagation, local, eta_paths = [], [], [[], []]
                    for n, r in zip(*gains):
                        propagation.append((r['gain']-n['gain'])@n['innovation'])
                        local.append(r['gain']@(r['innovation']-n['innovation']))
                        np.testing.assert_allclose(r['update']-n['update'], propagation[-1]+local[-1], atol=1e-10)
                        for index, g in enumerate((n, r)):
                            eta_paths[index].append(g['gain'][:, 1]*g['innovation'][1])
                    p, l = weights[0]@np.asarray(propagation), weights[0]@np.asarray(local)
                    gain_rows.append({'layout': layout, 'task_index': task_index, 'paid_count': prefix,
                        'native_weighted_gain_change_update': p.tolist(), 'native_weighted_local_innovation_update': l.tolist(),
                        'gain_change_rms': rms(p), 'local_innovation_change_rms': rms(l),
                        'native_eta_gain_path_rms': rms(weights[0]@np.asarray(eta_paths[0])),
                        'restricted_eta_gain_path_rms': rms(weights[0]@np.asarray(eta_paths[1]))})
            print('paid transfer attribution reproduced', layout, task_index, flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'identity_means.json', {'execution': spec, 'rows': detailed})
    result = {'execution': spec, 'prefix_rows': rows, 'gain_rows': gain_rows,
        'all_saved_prefixes_and_final_mse_reproduced': True, 'mean_and_update_decompositions_passed': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'independent_confirmation': False, 'target_promoted': False}
    save(out/'results.json', result)
    print({'prefix_rows': rows, 'gain_rms': [{k: r[k] for k in ('layout', 'task_index', 'paid_count', 'gain_change_rms',
        'local_innovation_change_rms')} for r in gain_rows], 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
