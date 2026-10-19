"""Compare equal-budget geometry allocations using saved Stage55 paired effects."""
import json

import numpy as np

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

SOURCE = ROOT / 'paper_artifacts/mainline_observation_seed_attribution_v1/summary.json'
OUTPUT = ROOT / 'paper_artifacts/mainline_geometry_allocation_v1'
ZERO_COST_FIELDS = ('new_target_initial_calls', 'new_source_fits', 'new_truth_function_calls',
                    'posterior_restorations', 'raw_replays', 'new_rng_draws', 'download_bytes')


def analyze_metric(primary, metric):
    geometry = primary['geometry_mean_effects']
    seeds = [row['seed'] for row in geometry]
    replicas = primary['by_replicate']
    assert len(seeds) == len(set(seeds)) == 12
    assert [rep['replicate'] for rep in replicas] == [1, 2]
    indexed = [{row['seed']: row for row in rep['rows']} for rep in replicas]
    assert all(len(rep['rows']) == len(index) == 12 and set(index) == set(seeds)
               for rep, index in zip(replicas, indexed))
    effects = np.array([[index[seed][metric] for index in indexed] for seed in seeds])
    assert np.all(np.isfinite(effects))
    assert np.allclose(effects.mean(axis=1), [row[metric] for row in geometry],
                       rtol=0, atol=1e-12)
    families = []
    for domain in dict.fromkeys(row['domain'] for row in geometry):
        family = effects[[row['domain'] == domain for row in geometry]]
        assert family.shape == (4, 2)
        within = float(np.mean((family[:, 0] - family[:, 1])**2 / 2))
        mean_variance = float(np.var(family.mean(axis=1), ddof=1))
        between = mean_variance - within / 2
        families.append(dict(domain=domain, geometry_count=4,
            mean=float(family.mean()), replicate_means=family.mean(axis=0).tolist(),
            within_noise_variance=within, geometry_mean_variance=mean_variance,
            between_geometry_variance_raw=between,
            between_geometry_variance_nonnegative=max(0., between)))
    assert len(families) == 3
    allocations = []
    for count, streams in ((12, 2), (24, 1)):
        n_family = count // 3
        between_part = sum(f['between_geometry_variance_nonnegative'] / n_family
                           for f in families) / 9
        noise_part = sum(f['within_noise_variance'] / (n_family * streams)
                         for f in families) / 9
        allocations.append(dict(new_geometries=count, geometries_per_family=n_family,
            streams_per_geometry=streams, task_noise_units=count*streams,
            terminal_cells=count*streams*9,
            between_variance_contribution=between_part,
            within_variance_contribution=noise_part,
            projected_standard_error=float(np.sqrt(between_part + noise_part))))
    first, second = allocations
    means = effects.mean(axis=1)
    largest = np.argsort(-means)[:2]
    return dict(metric=metric, mean=float(means.mean()), families=families,
        allocations=allocations,
        relative_standard_error_24x1_to_12x2=(second['projected_standard_error'] /
                                             first['projected_standard_error']),
        between_share_current_projected_variance=(first['between_variance_contribution'] /
            (first['between_variance_contribution'] + first['within_variance_contribution'])),
        concentration=dict(top_two_geometry_seeds=[seeds[i] for i in largest],
            top_two_share_of_signed_sum=float(means[largest].sum()/means.sum()),
            remaining_ten_mean=float(np.delete(means, largest).mean()),
            leave_one_family_out=[dict(excluded=f['domain'], remaining_mean=float(np.mean([
                other['mean'] for other in families if other['domain'] != f['domain']])))
                for f in families]))


def main():
    with attempt(OUTPUT, 'saved-pilot-geometry-allocation') as cost:
        cost.update({field: 0 for field in ZERO_COST_FIELDS})
        source = read(SOURCE)
        assert source['status'] == 'complete' and source['terminal_cells'] == 216
        primary = source['primary']
        result = dict(status='complete', source=str(SOURCE.relative_to(ROOT)),
            comparison=dict(left=primary['left'], right=primary['right']),
            metrics=[analyze_metric(primary, metric)
                     for metric in ('signed_gain', 'new_policy_gain')],
            model='d_fgr = mu_f + b_fg + e_fgr; three fixed equally weighted families.',
            estimators=dict(within='mean_g((d_fg1-d_fg2)^2/2)',
                between_raw='sample_variance_g(mean_r(d_fgr)) - within/2',
                between_for_projection='max(0, between_raw)',
                mean_variance='(1/9)*sum_f(between/n_f + within/(n_f*r))'),
            recommendation='24 new geometries x 1 stream, eight per family; retain all nine arms.',
            limits=[
                'Descriptive pilot allocation comparison; no new outcomes or inference cohort.',
                'Assumes new geometries follow the same within-family distribution and noise streams '
                'are independent conditional on geometry; two streams and four geometries per family '
                'give weak variance estimates.',
                'Between-geometry variation includes task-specific optimizer/proposal RNG held fixed '
                'within geometry; it is not solely variation in physical geometry.',
                'A zero observed within-noise variance does not establish zero population variance. '
                'Negative raw between estimates are retained and clipped only for projection.',
                'Concentration exclusions are sensitivity summaries, not replacement primary cohorts. '
                'Original frozen confidence intervals remain unchanged; no power guarantee.',
                'One stream per new geometry cannot independently estimate its within-noise variance.',
            ])
        save(OUTPUT / 'results.json', result)
    attempts = [json.loads(line) for line in (OUTPUT/'attempts.jsonl').read_text().splitlines()]
    fields = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_search_calls',
              'new_verification_calls', 'source_model_restorations', *ZERO_COST_FIELDS)
    completion = dict(status='complete', attempts=len(attempts),
        failed_attempts=[row for row in attempts if row['status'] != 'completed'],
        cost={field: sum(row.get(field, 0) for row in attempts) for field in fields})
    save(OUTPUT/'completion.json', completion)
    print(json.dumps(dict(status='complete', cost=completion['cost'],
        projected_standard_errors={row['metric']: [a['projected_standard_error']
            for a in row['allocations']] for row in result['metrics']})))


if __name__ == '__main__':
    main()
