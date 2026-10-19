"""Compare actual neighborhood geometry in frozen public and learned inputs."""
import argparse

import numpy as np
from scipy.spatial.distance import cdist

from performance.diagnose_mainline_legacy_source_geometry import source_support
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save
from representation.legacy_function_prior import legacy_public_coordinates
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.observable_exposure import observable_state_descriptor_names
from representation.source_kernel_population import rbf_cross


MANIFEST = ROOT / 'performance/manifests/mainline_initialization_audit_v1.json'


def correlations(covariance, head):
    block = covariance[:, head, :, head]
    return block / np.sqrt(np.outer(np.diag(block), np.diag(block)))


def quantiles(values):
    return np.quantile(values, [0., .5, .95, 1.])


def library_extrapolation(coordinate, descriptors):
    names = observable_state_descriptor_names()
    names = ['signed_log1p_' + name for name in names] + ['log1p_squared_' + name for name in names]
    library = np.asarray([coordinate._input_library(row) for row in descriptors])
    z = (library-coordinate.feature_mean)/coordinate.feature_scale
    return [{'maximum_absolute_source_z': float(np.max(np.abs(row))),
             'largest_axes': [{'name': names[index], 'source_z': float(row[index])}
                              for index in np.argsort(-np.abs(row))[:3]]} for row in z]


def run(plan):
    output, parent = ROOT / plan['output_dir'], ROOT / plan['parent_output']
    destination = output / 'geometry.json'
    if destination.exists():
        return read(destination)
    with attempt(output, 'initialization-geometry', plan['domain']) as cost:
        cost.update(new_truth_function_calls=0, new_source_fits=0, download_bytes=0)
        bundle = load_pickle(ROOT / plan['frozen_assets']['source_models_file'])
        source_mean = np.asarray(bundle['gp']['heads'][1]['joint_common']['source_features'])
        source_risk = np.asarray(bundle['gp']['heads'][2]['joint_common']['source_features'])
        source_joint = np.column_stack([source_mean, source_risk])
        cases = []
        for seed in plan['target_seeds']:
            directory = output / f'seed_{seed}'
            records = {arm: read(directory / arm / 'initialization.json') for arm in plan['arms']}
            gp = records['same_source_gp']
            bank = np.load(directory / 'same_source_gp/bank.npz')['points']
            for arm in plan['arms']:
                np.testing.assert_array_equal(bank, np.load(directory / arm / 'bank.npz')['points'])
            problem = load_pickle(parent / f'seed_{seed}/initialization_only/search_engine.pkl').problem
            normalized = np.asarray([problem.normalize(point) for point in bank])
            public = (legacy_public_coordinates(normalized)-bundle['mean_center'])/bundle['mean_scale']
            descriptors = legacy_exposure_descriptors(problem, bank, 50)
            learned = (bundle['encoder'].transform(descriptors)-bundle['risk_center'])/bundle['risk_scale']
            initial = np.asarray(gp['initial_bank_indices'])
            neighbor_ids = np.asarray(gp['neighbor_bank_indices'])
            new_ids = neighbor_ids[gp['new_neighbor_indices']]
            distance = cdist(normalized, normalized[initial])/np.sqrt(normalized.shape[1])
            nearest = initial[np.argmin(distance, axis=1)]
            models = {}
            for arm in plan['arms']:
                covariance = np.load(directory / arm / 'source0_covariance.npz')['covariance']
                models[arm] = {'g': correlations(covariance, 2), 'eta_g': correlations(covariance, 3)}
            anchor = bundle['gp']['heads'][1]['joint_common']['anchor']
            rbf = rbf_cross(public, public[initial], anchor['lengthscale'])
            extrapolation = {
                'g_mean_encoder': library_extrapolation(bundle['encoder'].g_mean, descriptors),
                'g_variance_encoder': library_extrapolation(bundle['encoder'].g_variance.coordinate, descriptors),
            }
            pairs = []
            for neighbor_index, index in enumerate(neighbor_ids):
                near = nearest[index]
                row = {'neighbor_index': neighbor_index, 'bank_index': int(index),
                    'new_neighbor': neighbor_index in gp['new_neighbor_indices'],
                    'nearest_initial_row': int(np.where(initial == near)[0][0]),
                    'physical_rms': float(distance[index].min()),
                    'head_change': float(normalized[index, 0]-normalized[near, 0]),
                    'tail_mean_change': float(normalized[index, 1:].mean()-normalized[near, 1:].mean()),
                    'public10_distance': float(np.linalg.norm(public[index]-public[near])),
                    'learned10_distance': float(np.linalg.norm(learned[index]-learned[near])),
                    'gp_g_rbf_to_nearest_initial': float(rbf[index, np.where(initial == near)[0][0]]),
                    'gp_g_maximum_rbf_to_initial': float(rbf[index].max()),
                    'source_correlations': {arm: {
                        'g_to_nearest_initial': float(value['g'][index, near]),
                        'eta_g_to_nearest_initial': float(value['eta_g'][index, near]),
                        'g_to_initial0': float(value['g'][index, initial[0]]),
                        'eta_g_to_initial0': float(value['eta_g'][index, initial[0]]),
                    } for arm, value in models.items()},
                    'encoder_library_extrapolation': {name: values[index] for name, values in extrapolation.items()}}
                pairs.append(row)
            supports = {name: source_support(source, target) for name, source, target in (
                ('public10', source_mean, public), ('learned10', source_risk, learned),
                ('joint20', source_joint, np.column_stack([public, learned])))}
            summaries = {name: {
                'source_nearest_distinct_distance_quantiles': value['source_nearest_distinct_distance_quantiles'],
                'new_neighbor_nearest_source_distance_quantiles': quantiles(np.asarray(value['nearest_source_distance'])[new_ids]),
                'new_neighbor_outside_box_axis_count_quantiles': quantiles(np.asarray(value['outside_source_box_axis_count'])[new_ids]),
            } for name, value in supports.items()}
            new_pairs = [row for row in pairs if row['new_neighbor']]
            summaries['neighbor_pairs'] = {name: quantiles([row[name] for row in new_pairs]) for name in (
                'physical_rms', 'public10_distance', 'learned10_distance',
                'gp_g_maximum_rbf_to_initial')}
            summaries['source_correlations'] = {arm: {name: quantiles([
                row['source_correlations'][arm][name] for row in new_pairs])
                for name in ('g_to_nearest_initial', 'eta_g_to_nearest_initial', 'g_to_initial0', 'eta_g_to_initial0')}
                for arm in plan['arms']}
            cases.append({'seed': seed, 'bank_count': len(bank), 'new_neighbor_count': len(new_ids),
                'pairs': pairs, 'support': supports, 'summary': summaries})
        result = {'status': 'complete', 'seeds': cases,
            'quantile_order': [0., .5, .95, 1.],
            'head_input_mapping': {
                'same_source_gp': {'f_g': 'public10', 'eta_f_eta_g': 'learned10'},
                'structural_head_specific': {'f_g': 'public10', 'eta_f_eta_g': 'learned10'},
                'structural_joint': {'all_heads': 'public10 concatenated with learned10'}},
            'source_identity_weights': dict(zip(bundle['identities'], np.exp(bundle['log_weights']))),
            'source_tasks': bundle['diagnostics']['source_task_names'],
            'gp_g_lengthscales': anchor['lengthscale'],
            'limits': 'Nearest initial point is a distance diagnostic, not recovered generator parent. Source correlations measure frozen prior coupling; no new response or fitted model is used.'}
    result['cost'] = cost
    save(destination, result)
    print({'status': 'complete', 'seeds': len(cases), 'cpu_seconds': cost['cpu_seconds']}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    run(read(args.manifest))


if __name__ == '__main__':
    main()
