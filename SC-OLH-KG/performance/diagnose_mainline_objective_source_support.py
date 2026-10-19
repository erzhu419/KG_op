"""Observed source-pair support of the frozen objective law; no model fitting."""
import argparse
import numpy as np

from baselines.transfer_archive import FrozenTransferArchive
from representation.legacy_function_prior import legacy_public_coordinates, legacy_source_labels
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save


PARENT = ROOT/'performance/manifests/mainline_role_replication_v1.json'
ARCHIVE = ROOT/'paper_artifacts/legacy_joint_bridge_v1/inputs/FactorShockStatePolicyRZDT1/archive.json'
OUTPUT = ROOT/'paper_artifacts/mainline_objective_attribution_v1'
FOCUS = {2026101672: [24, 36], 2026101675: [33]}


def nearest_pairs(coordinates):
    distances = np.linalg.norm(coordinates[:, None]-coordinates[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.argmin(axis=1)
    pairs = sorted({tuple(sorted((i, int(j)))) for i, j in enumerate(nearest)})
    return np.asarray(pairs), distances[np.arange(len(nearest)), nearest]


def summary(observed, observed_variance, prediction, variance, noise_signal):
    error = prediction-observed
    informative = noise_signal >= 2.
    standardized = error/np.sqrt(variance+observed_variance)
    return dict(pair_count=len(observed), observed_difference_rms=float(np.sqrt(np.mean(observed**2))),
        predicted_difference_rms=float(np.sqrt(np.mean(prediction**2))),
        difference_rmse=float(np.sqrt(np.mean(error**2))),
        direction_agreement=int(np.sum(observed*prediction > 0.)),
        observed_signal_at_least_2_se_count=int(informative.sum()),
        direction_agreement_at_least_2_se=int(np.sum((observed*prediction > 0.) & informative)),
        rms_standardized_residual=float(np.sqrt(np.mean(standardized**2))),
        absolute_standardized_residual_above_2=int(np.sum(abs(standardized) > 2.)))


def source_rows(archive, tasks, bundle):
    masses = np.exp(bundle['log_weights'])
    all_coordinates = (np.vstack([t['psi'] for t in tasks])-bundle['mean_center'])/bundle['mean_scale']
    result, start = [], 0
    for archived, task in zip(archive.tasks, tasks, strict=True):
        n = len(task['response'])
        coordinates = all_coordinates[start:start+n]
        pairs, nearest_distances = nearest_pairs(coordinates)
        left, right = pairs.T
        observed = task['response'][right, 0]-task['response'][left, 0]
        observation_variance = task['noise'][right, 0]+task['noise'][left, 0]
        signal = abs(observed)/np.sqrt(observation_variance)
        identities, prior_means, posterior_means, prior_variances, posterior_variances = [], [], [], [], []
        for identity, pair in zip(bundle['identities'], bundle['components'], strict=True):
            record = pair[0]
            law = record['law']
            design = law['source_design'][start:start+n]
            delta = design[right]-design[left]
            common = np.einsum('np,pq,nq->n', delta, law['common'], delta)
            task_variance = np.einsum('np,pq,nq->n', delta, law['task'], delta)
            white = np.full(len(pairs), 2.*law['white'])
            prior_mean = delta@law['mean']
            node_masses = np.exp(record['log_weights'])
            node_means = record['mean']@delta.T
            posterior_mean = node_masses@node_means
            posterior_common = np.einsum('np,pq,nq->n', delta,
                np.einsum('k,kpq->pq', node_masses, record['covariance']), delta)
            node_between = node_masses@((node_means-posterior_mean)**2)
            prior_variance = common+task_variance+white
            posterior_variance = posterior_common+node_between+task_variance+white
            identities.append(dict(identity=identity, source_identity_mass=float(masses[len(identities)]),
                design_columns=design.shape[1], prior_difference_mean=prior_mean,
                posterior_common_difference_mean=posterior_mean,
                prior_common_difference_variance=common, task_difference_variance=task_variance,
                white_difference_variance=white, posterior_common_difference_variance=posterior_common,
                posterior_node_between_difference_variance=node_between,
                prior_summary=summary(observed, observation_variance, prior_mean, prior_variance, signal),
                posterior_common_summary=summary(observed, observation_variance,
                    posterior_mean, posterior_variance, signal)))
            prior_means.append(prior_mean)
            posterior_means.append(posterior_mean)
            prior_variances.append(prior_variance)
            posterior_variances.append(posterior_variance)
        ensemble = {}
        for name, means, variances in [('prior', prior_means, prior_variances),
                                      ('posterior_common', posterior_means, posterior_variances)]:
            mean = masses@np.asarray(means)
            within = masses@np.asarray(variances)
            between = masses@((np.asarray(means)-mean)**2)
            ensemble[name] = dict(difference_mean=mean, within_identity_difference_variance=within,
                between_identity_difference_variance=between,
                summary=summary(observed, observation_variance, mean, within+between, signal))
        result.append(dict(family=archived.name, raw_dimension=archived.X.shape[1], source_sites=n,
            source_raw_calls=archived.simulator_calls, pairs=pairs,
            pair_public_distance=np.linalg.norm(coordinates[right]-coordinates[left], axis=1),
            source_nearest_distances=nearest_distances, observed_objective_difference=observed,
            observed_difference_variance=observation_variance, observed_signal_to_se=signal,
            identities=identities, ensemble=ensemble))
        start += n
    return result, all_coordinates


def target_support(plan, tasks, bundle, source_coordinates):
    parent = ROOT/plan['output_dir']
    boundaries = np.cumsum([0]+[len(t['psi']) for t in tasks])
    result = []
    for seed, focuses in FOCUS.items():
        directory = parent/f'seed_{seed}'
        with np.load(directory/'neighbors.npz') as values:
            points = values['points'][[0, *focuses]]
        initial = read(directory/'initialization.json')['history'][0]['point']
        if not np.array_equal(points[0], initial):
            raise ValueError('Neighbor 0 must be the recorded initial incumbent for this contrast')
        coordinates = (legacy_public_coordinates(points/plan['task']['L'])
                       -bundle['mean_center'])/bundle['mean_scale']
        rows = []
        for index, coordinate in zip([0, *focuses], coordinates, strict=True):
            families = []
            for family, begin, end in zip(tasks, boundaries[:-1], boundaries[1:], strict=True):
                source = source_coordinates[begin:end]
                distance = np.linalg.norm(source-coordinate, axis=1)
                closest = int(distance.argmin())
                _, source_nearest = nearest_pairs(source)
                families.append(dict(family=family['name'], closest_source_site=closest,
                    distance=float(distance[closest]), source_nearest_distance_median=float(np.median(source_nearest)),
                    source_nearest_distance_95th=float(np.quantile(source_nearest, .95)),
                    source_nearest_distance_max=float(source_nearest.max()),
                    source_nearest_fraction_at_or_below_target=float(np.mean(source_nearest <= distance[closest])),
                    coordinates_outside_source_marginal_range=int(np.sum(
                        (coordinate < source.min(axis=0)) | (coordinate > source.max(axis=0))))))
            rows.append(dict(neighbor_index=index, distance_to_initial=float(np.linalg.norm(coordinate-coordinates[0])),
                source_families=families))
        result.append(dict(seed=seed, raw_dimension=plan['task']['d'], policies=rows))
    return result


def function_support():
    """Locate target policies in the actual frozen f design, without filtering."""
    from performance.benchmark_lodo_meta_prior import build_scalarized_problem
    from performance.diagnose_legacy_risk_transfer import no_outcomes
    from performance.mainline_legacy_source_models import _query_design
    from performance.mainline_role_prior import role_inputs
    from representation.legacy_observable_prior import legacy_exposure_descriptors

    def distances(source, targets):
        _, baseline = nearest_pairs(source)
        return [dict(distance_to_initial=float(np.linalg.norm(target-targets[0])),
            nearest_source_site=int(np.argmin(np.linalg.norm(source-target, axis=1))),
            nearest_source_distance=float(np.min(np.linalg.norm(source-target, axis=1))),
            source_nearest_distance_median=float(np.median(baseline)),
            source_nearest_distance_95th=float(np.quantile(baseline, .95)),
            source_nearest_distance_max=float(baseline.max()),
            columns_outside_source_range=np.flatnonzero(
                (target < source.min(axis=0)-1e-12) | (target > source.max(axis=0)+1e-12)))
            for target in targets]

    with attempt(OUTPUT, 'objective-target-function-support', 'FactorShockStatePolicyRZDT1') as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0,
            source_model_restorations=1, target_engine_loads=2)
        plan = read(PARENT)
        bundle = load_pickle(ROOT/plan['frozen_assets']['source_models_file'])
        archive = FrozenTransferArchive.load(ARCHIVE)
        source_mean = (np.vstack([legacy_public_coordinates(t.X) for t in archive.tasks])
                       -bundle['mean_center'])/bundle['mean_scale']
        source_descriptors = []
        for task in archive.tasks:
            problem = no_outcomes(build_scalarized_problem(task.name, task.X.shape[1],
                100, .04, .05, [.5, .5]))
            source_descriptors.append(legacy_exposure_descriptors(problem, task.X_integer, 50))
        source_risk = (bundle['encoder'].transform(np.vstack(source_descriptors))
                       -bundle['risk_center'])/bundle['risk_scale']
        source_coordinates = (source_mean, source_risk)
        # A wrong provider, scale, or archive order would invalidate this support comparison.
        parity = []
        for identity, pair in zip(bundle['identities'], bundle['components'], strict=True):
            projected = _query_design(bundle, identity, 0, source_coordinates, pair[0]['law'])
            error = float(np.max(abs(projected-pair[0]['law']['source_design'])))
            if error > 1e-9:
                raise ValueError(f'Source f projection does not reproduce frozen design: {identity}: {error}')
            parity.append(dict(identity=identity, maximum_source_design_error=error))
        boundaries = np.cumsum([0]+[len(task.X) for task in archive.tasks])
        rows = []
        for seed, focuses in FOCUS.items():
            directory = ROOT/plan['output_dir']/f'seed_{seed}'
            engine = load_pickle(directory/'legacy_role_full/search_engine.pkl')
            no_outcomes(engine.problem)
            with np.load(directory/'neighbors.npz') as values:
                points = values['points'][[0, *focuses]]
            _, designs = role_inputs(bundle, engine, points)
            target_mean = (legacy_public_coordinates(points/engine.problem.L)
                           -bundle['mean_center'])/bundle['mean_scale']
            target_risk = (bundle['encoder'].transform(legacy_exposure_descriptors(engine.problem, points, 50))
                           -bundle['risk_center'])/bundle['risk_scale']
            joint = np.column_stack([target_mean, target_risk])
            families = []
            for task, begin, end in zip(archive.tasks, boundaries[:-1], boundaries[1:], strict=True):
                functions = []
                for identity, pair, design in zip(bundle['identities'], bundle['components'], designs, strict=True):
                    functions.append(dict(identity=identity, design_columns=design[0][0].shape[1],
                        policies=distances(pair[0]['law']['source_design'][begin:end], design[0][0])))
                families.append(dict(family=task.name,
                    public_mean10=distances(source_mean[begin:end], target_mean),
                    learned_risk10=distances(source_risk[begin:end], target_risk),
                    joint20=distances(np.column_stack(source_coordinates)[begin:end], joint),
                    function_designs=functions))
            rows.append(dict(seed=seed, neighbor_order=[0, *focuses], source_families=families))
        save(OUTPUT/'target_function_support.json', dict(status='complete', rows=rows,
            source_projection_parity=parity,
            interpretation='Distances use the actual frozen standardized coordinates and unrescaled learned design columns. Per-space distances are compared only against source distances in that same space; numerical scales differ across spaces. Column-range inclusion and nearest distance are descriptive support measures, not convex-hull membership or proof of transferability.',
            source_archive=str(ARCHIVE.relative_to(ROOT)), source_bundle=plan['frozen_assets']['source_models_file'],
            new_source_calls=0, new_target_search_calls=0, new_verification_calls=0,
            new_source_fits=0, new_truth_function_calls=0, download_bytes=0))


def main():
    with attempt(OUTPUT, 'objective-source-support', 'FactorShockStatePolicyRZDT1') as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        plan = read(PARENT)
        bundle_path = ROOT/plan['frozen_assets']['source_models_file']
        bundle = load_pickle(bundle_path)
        cost['source_model_restorations'] = 1
        archive = FrozenTransferArchive.load(ARCHIVE)
        tasks = legacy_source_labels(archive)
        # A mismatched source order would silently attach frozen designs to wrong labels.
        if [t.name for t in archive.tasks] != bundle['diagnostics']['source_task_names']:
            raise ValueError('Archive and stored source-design family order differ')
        rows, coordinates = source_rows(archive, tasks, bundle)
        result = dict(status='complete', source_archive=str(ARCHIVE.relative_to(ROOT)),
            source_bundle=str(bundle_path.relative_to(ROOT)), source_raw_calls_reused=archive.simulator_calls,
            source_identity_weights=dict(zip(bundle['identities'], np.exp(bundle['log_weights']))),
            pair_selection='Within each source family, k=1 in source-standardized public mean10; symmetric dedup; no response selection.',
            difference_direction='right_source_site minus left_source_site; source site indices are zero based.',
            prediction_semantics='Stored law mean and source-conditioned common coefficient law, with existing source joint identity weights. Source task coefficients are not individually reconditioned. Prior label uses pre-source law coefficients but post-source frozen identity weights.',
            uncertainty_semantics='Pair variance separates common, task, two independent white terms, integration-node mixture and outer identity mixture. Observed difference variance sums raw f sample-mean variances from independent source draws.',
            limits='All source data trained the frozen law: this is descriptive source support, not held-out calibration. Two-SE counts are descriptive with only three replicates per site. Mean10 proximity does not establish joint20 or learned-function support. No per-task coefficient fit, model change, target truth, or simulated observation.',
            source_families=rows, target_public_support=target_support(plan, tasks, bundle, coordinates),
            new_source_calls=0, new_target_search_calls=0, new_verification_calls=0,
            new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        save(OUTPUT/'source_support.json', result)
        for row in rows:
            print(row['family'], row['ensemble']['posterior_common']['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-function-support-only', action='store_true')
    args = parser.parse_args()
    function_support() if args.target_function_support_only else main()
