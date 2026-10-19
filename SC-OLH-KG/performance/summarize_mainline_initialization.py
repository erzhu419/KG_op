"""Summarize saved initialization predictions with existing truth only."""
import argparse

import numpy as np

from performance.diagnose_mainline_shared_neighbors import load_truth_cache
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save


MANIFEST = ROOT / 'performance/manifests/mainline_initialization_audit_v1.json'


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {'minimum': float(np.min(values)), 'median': float(np.median(values)),
            'maximum': float(np.max(values))}


def complete_truth(value):
    return value is not None and all(key in value for key in
        ('objective', 'constraint_mean_centered', 'constraint_sigma'))


def snapshot_metrics(snapshot, indices, truths):
    known = [(index, value) for index, value in zip(indices, truths) if complete_truth(value)]
    result = {'policy_count': len(indices), 'known_truth_count': len(known)}
    if not known:
        return result
    idx = [index for index, _ in known]
    truth = np.asarray([[value['objective'], value['constraint_mean_centered'],
                         value['constraint_sigma']**2] for _, value in known])
    mean, variance, noise = (snapshot[key][idx] for key in ('mean', 'variance', 'noise'))
    error = np.column_stack([mean[:, 0], mean[:, 2], noise[:, 1]])-truth
    eta = mean[:, 3]
    result.update(
        mean_bias_f_g_noise_variance=np.mean(error, axis=0),
        mse_f_g_noise_variance=np.mean(error**2, axis=0),
        sigma_prediction_to_truth=distribution(np.sqrt(noise[:, 1]/truth[:, 2])),
        g_mean=distribution(mean[:, 2]),
        g_epistemic_variance=distribution(variance[:, 2]),
        eta_g_mean=distribution(eta), eta_g_variance=distribution(variance[:, 3]),
        exp_mean_eta_g=distribution(np.exp(eta)),
        exact_expected_exp_eta_g=distribution(noise[:, 1]),
        noise_to_exp_mean_eta_g=distribution(noise[:, 1]/np.exp(eta)))
    return result


def point_prediction(snapshot, index, truth):
    mean, variance, noise = (snapshot[key][index] for key in ('mean', 'variance', 'noise'))
    result = {'paid_prefix': snapshot['paid_prefix'], 'f_mean': mean[0],
        'g_mean': mean[2], 'g_epistemic_variance': variance[2],
        'eta_g_mean': mean[3], 'eta_g_variance': variance[3],
        'exp_mean_eta_g': np.exp(mean[3]), 'exact_expected_exp_eta_g': noise[1],
        'constraint_sigma_prediction': np.sqrt(noise[1]),
        'noise_to_exp_mean_eta_g': noise[1]/np.exp(mean[3]),
        'identity_weights': snapshot['identity_weights']}
    if complete_truth(truth):
        result.update(g_error=mean[2]-truth['constraint_mean_centered'],
            sigma_prediction_to_truth=np.sqrt(noise[1])/truth['constraint_sigma'])
    return result


def observation_changes(record, snapshots, points, new_indices, neighbor_indices, cache, L):
    neighbor_lookup = {bank: index for index, bank in enumerate(neighbor_indices)}
    new_points = points[new_indices].astype(float)/L
    rows = []
    for step, observed in enumerate(record['pre_observation']):
        index = observed['bank_index']
        before, after = snapshots[step:step+2]
        raw_g = observed['centered_raw'][1]
        residual = raw_g-before['mean'][index, 2]
        mean_change = after['mean'][index, 2]-before['mean'][index, 2]
        distances = np.sqrt(np.mean((new_points-points[index]/L)**2, axis=1))
        neighbor = new_indices[int(np.argmin(distances))]
        truth = cache.get(tuple(map(int, points[neighbor])))
        rows.append({
            'initialization_row': step, 'bank_index': index,
            'raw_f_g_centered': observed['centered_raw'],
            'pre_standardized_residual_f_g': observed['standardized_residual'],
            'pre_log_predictive': observed['log_predictive'],
            'pre_g_mean': before['mean'][index, 2],
            'post_g_mean': after['mean'][index, 2],
            'pre_g_residual': residual, 'post_g_residual': raw_g-after['mean'][index, 2],
            'g_mean_change': mean_change,
            'fraction_raw_residual_followed_by_g_mean': None if abs(residual) <= 1e-12 else mean_change/residual,
            'pre_g_epistemic_variance': before['variance'][index, 2],
            'post_g_epistemic_variance': after['variance'][index, 2],
            'pre_g_noise_variance': before['noise'][index, 1],
            'post_g_noise_variance': after['noise'][index, 1],
            'g_noise_variance_multiplier': after['noise'][index, 1]/before['noise'][index, 1],
            'g_eta_mean_change': after['mean'][index, 3]-before['mean'][index, 3],
            'g_eta_variance_change': after['variance'][index, 3]-before['variance'][index, 3],
            'nearest_new_unsampled_neighbor': {
                'neighbor_index': neighbor_lookup[neighbor], 'bank_index': neighbor,
                'physical_normalized_rms': float(np.min(distances)), 'truth': truth,
                'pre_g_mean': before['mean'][neighbor, 2],
                'post_g_mean': after['mean'][neighbor, 2],
                'g_mean_change': after['mean'][neighbor, 2]-before['mean'][neighbor, 2],
                'pre_g_noise_variance': before['noise'][neighbor, 1],
                'post_g_noise_variance': after['noise'][neighbor, 1],
                'g_noise_variance_multiplier': after['noise'][neighbor, 1]/before['noise'][neighbor, 1]},
        })
    return rows


def observation_summary(rows):
    return {
        'count': len(rows),
        'absolute_pre_g_standardized_residual': distribution(
            [abs(row['pre_standardized_residual_f_g'][1]) for row in rows]),
        'absolute_pre_g_raw_residual': distribution([abs(row['pre_g_residual']) for row in rows]),
        'absolute_post_g_raw_residual': distribution([abs(row['post_g_residual']) for row in rows]),
        'fraction_raw_residual_followed_by_g_mean': distribution([
            row['fraction_raw_residual_followed_by_g_mean'] for row in rows
            if row['fraction_raw_residual_followed_by_g_mean'] is not None]),
        'g_noise_variance_multiplier': distribution([row['g_noise_variance_multiplier'] for row in rows]),
        'g_mean_moved_toward_raw_count': sum(row['g_mean_change']*row['pre_g_residual'] > 0 for row in rows),
        'g_noise_increased_count': sum(row['g_noise_variance_multiplier'] > 1 for row in rows),
    }


def case_summary(plan, seed, arm, cache, seed_truth):
    directory = ROOT/plan['output_dir']/f'seed_{seed}'/arm
    record = read(directory/'initialization.json')
    if record['status'] != 'complete':
        raise RuntimeError(f'Initialization trajectory is incomplete: {directory}')
    points = np.load(directory/'bank.npz')['points']
    snapshots = [{**snapshot, **{key: np.asarray(snapshot[key]) for key in ('mean', 'variance', 'noise')}}
                 for snapshot in record['snapshots']]
    neighbors = record['neighbor_bank_indices']
    new_indices = [neighbors[i] for i in record['new_neighbor_indices']]
    truths = [cache.get(tuple(map(int, points[index]))) for index in new_indices]
    baseline = seed_truth['initial_truth']
    improving = [i for i in record['new_neighbor_indices']
                 if seed_truth['neighbor_truth'][i]['true_feasible']
                 and seed_truth['neighbor_truth'][i]['objective'] < baseline['objective']-1e-12]
    focus = [('initial0', record['initial_bank_indices'][0], None)]
    focus += [(f'neighbor_{i}', neighbors[i], i) for i in improving]
    focused = []
    for name, index, neighbor in focus:
        truth = cache.get(tuple(map(int, points[index])))
        focused.append({'name': name, 'bank_index': index, 'neighbor_index': neighbor,
                        'truth': truth,
                        'snapshots': [point_prediction(s, index, truth) for s in snapshots]})
    changes = observation_changes(record, snapshots, points, new_indices, neighbors, cache, plan['task']['L'])
    final = snapshots[-1]
    initial_indices = record['initial_bank_indices']
    residuals = np.asarray([row['centered_raw'][1] for row in record['pre_observation']])-final['mean'][initial_indices, 2]
    result = {'seed': seed, 'arm': arm, 'new_neighbor_indices': record['new_neighbor_indices'],
        'improving_neighbor_indices': improving, 'identity_names': record['identity_names'],
        'new_neighbor_trajectory': [{'paid_prefix': snapshot['paid_prefix'],
            'identity_weights': snapshot['identity_weights'],
            **snapshot_metrics(snapshot, new_indices, truths)} for snapshot in snapshots],
        'focus_policies': focused, 'observation_changes': changes,
        'observation_summary': observation_summary(changes),
        'initial10_reused_raw_residual': {
            'g_residual': distribution(residuals),
            'squared_g_residual': distribution(residuals**2),
            'predicted_g_noise_variance': distribution(final['noise'][initial_indices, 1]),
            'squared_g_residual_to_predicted_noise': distribution(residuals**2/final['noise'][initial_indices, 1]),
            'scope': 'Retrospective residual of one reused raw per point, not replicated noise validation.'},
        'existing_parity': record['parity'], 'trajectory_file': str(directory/'initialization.json')}
    return result, snapshots, new_indices, truths


def summarize(plan):
    output, parent = ROOT/plan['output_dir'], ROOT/plan['parent_output']
    # Missing or incomplete files must be resolved by their existing producer;
    # this summarizer never invokes a model or simulator to fill a gap.
    for seed in plan['target_seeds']:
        for arm in plan['arms']:
            path = output/f'seed_{seed}'/arm/'initialization.json'
            if not path.exists() or read(path)['status'] != 'complete':
                raise RuntimeError(f'Initialization trajectory required: {path}')
    with attempt(output, 'summarize-initialization', plan['domain']) as cost:
        cost.update(new_truth_function_calls=0, new_source_fits=0, download_bytes=0)
        cache = load_truth_cache(plan, parent)
        prior = read(parent/'diagnosis.json')
        rows, retained = [], {}
        for seed in plan['target_seeds']:
            seed_truth = next(row for row in prior['seeds'] if row['seed'] == seed)
            for arm in plan['arms']:
                row, snapshots, indices, truths = case_summary(plan, seed, arm, cache, seed_truth)
                rows.append(row); retained[seed, arm] = snapshots, indices, truths
        aggregate = []
        for arm in plan['arms']:
            trajectory = []
            for prefix in plan['initialization_path']['prefixes']:
                parts = [retained[seed, arm] for seed in plan['target_seeds']]
                snapshot = {key: np.concatenate([snapshots[prefix][key][indices]
                    for snapshots, indices, _ in parts]) for key in ('mean', 'variance', 'noise')}
                truths = [truth for _, _, values in parts for truth in values]
                trajectory.append({'paid_prefix': prefix,
                    **snapshot_metrics(snapshot, list(range(len(truths))), truths)})
            observations = [change for row in rows if row['arm'] == arm for change in row['observation_changes']]
            aggregate.append({'arm': arm, 'new_neighbor_trajectory': trajectory,
                              'observation_summary': observation_summary(observations)})
        result = {'status': 'complete', 'rows': rows, 'aggregate': aggregate,
            'scope': 'Three observation seeds on one already-opened task geometry. All56 new neighbors per seed remain unsampled throughout initial10; aggregated168 entries are not168 independent tasks.',
            'noise_interpretation': 'Exact mixture E[exp eta_g] is episode variance. Its ratio to exp(E eta_g) describes uncertainty amplification; total eta mean/variance are not substituted into a single Gaussian.',
            'missing_truth_policy': 'Use cached compatible truth only; no analytic function is called.',
            'supplemental_files': {name: {'file': str(output/name), 'available': (output/name).exists()}
                                  for name in ('geometry.json', 'native_self_cross.json')}}
        if (output/'geometry.json').exists():
            geometry = read(output/'geometry.json')
            improving = {row['seed']: row['improving_neighbor_indices'] for row in rows}
            result['geometry'] = {
                key: geometry[key] for key in ('quantile_order', 'head_input_mapping',
                    'source_identity_weights', 'source_tasks', 'gp_g_lengthscales')}
            result['geometry']['seeds'] = [
                {'seed': row['seed'], 'summary': row['summary'],
                 'improving_neighbor_pairs': [pair for pair in row['pairs']
                     if pair['neighbor_index'] in improving[row['seed']]]}
                for row in geometry['seeds']]
        if (output/'native_self_cross.json').exists():
            native = read(output/'native_self_cross.json')
            result['native_partition'] = {key: native[key] for key in (
                'paid11_12_summary', 'all_full_and_frozen_scores_match',
                'all_full_and_frozen_experts_match', 'definition', 'limitations')}
    result['cost'] = cost
    save(output/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    result = summarize(read(args.manifest))
    print({'status': result['status'], 'cases': len(result['rows']),
           'cpu_seconds': result['cost']['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    main()
