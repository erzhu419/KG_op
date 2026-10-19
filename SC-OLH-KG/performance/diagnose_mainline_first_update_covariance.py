"""Split the first saved objective update using its original covariance law."""
import json
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from scipy.special import softmax

from performance.mainline_role_prior import role_inputs
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save

MANIFEST = ROOT/'performance/manifests/mainline_first_update_covariance_v1.json'
PARTS = ('source_common', 'new_task', 'point_discrepancy')


def check(actual, expected, errors, name):
    """A failed reconstruction invalidates attribution, not the frozen filter."""
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10, err_msg=name)
    errors[name] = float(np.max(np.abs(np.asarray(actual)-expected)))


def diagnose(plan, seed, bundle):
    parent = ROOT/plan['input_dir']/f'seed_{seed}'
    cached = ROOT/plan['projection_dir']/f'seed_{seed}'
    projection = read(cached/'projection.json')
    focus = plan['focus_neighbors'][str(seed)]
    assert projection['identity_names'] == bundle['identities']
    assert projection['focus_neighbors'] == focus
    rows = read(parent/'legacy_role_full/search.json')['history']
    neighbors = np.load(parent/'neighbors.npz')['points']
    bank = [tuple(rows[0]['point'])]+[tuple(neighbors[i]) for i in focus]
    assert len(set(bank)) == len(bank)
    engine = load_pickle(parent/'legacy_role_full/search_engine.pkl')
    with ExitStack() as stack:
        for name in ('simulate', 'true_objective', 'true_constraint_mean', 'true_sigma'):
            stack.enter_context(patch.object(engine.problem, name, side_effect=AssertionError(name)))
        replacement, designs = role_inputs(bundle, engine, bank)
    with np.load(cached/'node_projections.npz') as z:
        mu = z['mean'][:, :2]
        q = softmax(z['log_mass'][:, :2], axis=2)
        prior = z['source_log_weights']
        w = softmax(prior[:, None]+z['head_mass'][:, :2].sum(axis=2), axis=0)
        saved_variance = z['delta_variance'][:, 0]
        saved_cross = z['delta_observation_covariance'][:, 0]
    errors = {}
    check(prior, bundle['log_weights'], errors, 'source_identity_log_weights')
    check(w[:, 0], projection['summary']['identity_weights'][0], errors, 'source_identity_weights')
    changes, identity_rows = [], []
    for i, (identity, records, pair_design) in enumerate(zip(
            bundle['identities'], replacement['components'], designs)):
        record, D = records[0], pair_design[0][0]
        covariance, task, white = record['covariance'], record['law']['task'], record['law']['white']
        d0, contrast = D[0], D[1:]-D[:1]
        common_cross = np.einsum('ap,kpq,q->ka', D, covariance, d0)
        task_cross = np.broadcast_to(D@task@d0, common_cross.shape)
        white_cross = np.zeros_like(common_cross)
        white_cross[:, 0] = white
        cross = np.stack((common_cross, task_cross, white_cross))
        total_cross = cross.sum(axis=0)
        variance0 = total_cross[:, 0]
        delta_variance = (np.einsum('ap,kpq,aq->ka', contrast, covariance, contrast)
                          + np.einsum('ap,pq,aq->a', contrast, task, contrast)+2*white)
        check(record['mean']@D.T, mu[i, 0], errors, identity+'/source_mean')
        check(delta_variance, saved_variance[i], errors, identity+'/difference_variance')
        check(total_cross[:, 1:]-variance0[:, None], saved_cross[i], errors,
              identity+'/difference_observation_covariance')
        node_change = mu[i, 1]-mu[i, 0]
        contributions = cross/variance0[None, :, None]*node_change[None, :, :1]
        check(contributions.sum(axis=0), node_change, errors, identity+'/node_update')
        point_change = np.einsum('k,jka->ja', q[i, 0], contributions)
        changes.append(point_change)
        identity_rows.append(dict(identity=identity, source_weight=w[i, 0],
            node_weighted_variance0=q[i, 0]@variance0,
            variance0_shares=dict(zip(PARTS, np.einsum('k,jk->j', q[i, 0], cross[:, :, 0]/variance0))),
            transfer_coefficients=dict(zip(PARTS, np.einsum('k,jka->ja', q[i, 0], cross/variance0[None, :, None]))),
            point_contributions=dict(zip(PARTS, point_change)),
            difference_contributions=dict(zip(PARTS, point_change[:, 1:]-point_change[:, :1]))))
    mixed = np.einsum('i,ija->ja', w[:, 0], np.asarray(changes))
    point_terms = dict(zip(PARTS, mixed))
    first = projection['summary']['steps'][0]
    check(mixed.sum(axis=0), first['point_contributions']['within_node'], errors, 'mixed_within_node')
    for key in ('node_reweight', 'identity_f', 'identity_g'):
        point_terms[key] = np.asarray(first['point_contributions'][key])
    difference_terms = {key:value[1:]-value[0] for key,value in point_terms.items()}
    means = np.einsum('it,itk,itka->ta', w, q, mu)
    differences = means[:, 1:]-means[:, :1]
    check(sum(point_terms.values()), means[1]-means[0], errors, 'full_point_update')
    check(sum(difference_terms.values()), differences[1]-differences[0], errors, 'full_difference_update')
    return dict(seed=seed, focus_neighbors=focus, observed_raw_f=rows[0]['raw'][0],
        mean_before=means[0], mean_after=means[1],
        difference_before=differences[0], difference_after=differences[1],
        point_contributions=point_terms, difference_contributions=difference_terms,
        identities=identity_rows, parity=errors)


def summarize_source_pairs(plan):
    """Read existing fixed source pairs; their residuals still include task effects."""
    old = read(ROOT/plan['projection_dir']/'source_support.json')
    families = []
    for family in old['source_families']:
        residual = (np.asarray(family['observed_objective_difference'])
                    - family['ensemble']['posterior_common']['difference_mean'])
        noise = np.asarray(family['observed_difference_variance'])
        parts = {}
        for label, key in [('source_common', 'posterior_common_difference_variance'),
                           ('new_task', 'task_difference_variance'),
                           ('point_discrepancy', 'white_difference_variance')]:
            parts[label] = sum(row['source_identity_mass']*np.mean(row[key])
                               for row in family['identities'])
        nearest = int(np.argmin(family['pair_public_distance']))
        families.append(dict(family=family['family'], pair_count=len(residual),
            mean_squared_residual=float(np.mean(residual**2)),
            mean_observation_variance=float(np.mean(noise)),
            noise_subtracted_mean_squared_residual=float(np.mean(residual**2-noise)),
            mean_model_difference_variance_parts=parts,
            pairs=family['pairs'], residual_difference=residual,
            closest_public_pair=dict(pair=family['pairs'][nearest],
                distance=family['pair_public_distance'][nearest], residual=residual[nearest],
                observation_standard_error=float(np.sqrt(noise[nearest])))))
    save(ROOT/plan['output_dir']/'source_pair_support.json', dict(source_families=families,
        interpretation='Training-data pair residuals include task effects and common-law estimation. '
        'Noise subtraction is descriptive, not an unbiased held-out error or a fitted residual kernel. '
        'Neighbor pairs share sites; the 51 pairs are not independent source tasks.'))


def complete(plan, cases):
    output = ROOT/plan['output_dir']
    attempts = [json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines()]
    keys = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_initial_calls',
            'new_target_search_calls', 'new_verification_calls', 'new_source_fits',
            'new_truth_function_calls', 'download_bytes', 'source_model_restorations',
            'saved_raw_rows_replayed', 'cached_task_projections_read')
    completion = dict(status='complete', attempts=len(attempts),
        failed_attempts=[row for row in attempts if row['status'] != 'completed'],
        cost={key:sum(row.get(key, 0) for row in attempts) for key in keys},
        reconstruction_max_absolute_error=max(v for case in cases for v in case['parity'].values()))
    save(output/'completion.json', completion)
    plan.update(status='complete', entrypoint='performance/diagnose_mainline_first_update_covariance.py')
    save(MANIFEST, plan)
    print(json.dumps(completion, indent=2))


def main():
    plan = read(MANIFEST)
    output = ROOT/plan['output_dir']
    with attempt(output, 'first-update-covariance-accounting', 'FactorShockStatePolicyRZDT1') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0, new_truth_function_calls=0,
                    download_bytes=0, source_model_restorations=1, saved_raw_rows_replayed=0,
                    cached_task_projections_read=len(plan['focus_neighbors']))
        parent = read(ROOT/plan['parent_manifest'])
        bundle = load_pickle(ROOT/parent['frozen_assets']['source_models_file'])
        cases = [diagnose(plan, int(seed), bundle) for seed in plan['focus_neighbors']]
        # Save attribution before reading already cached truth for interpretation.
        save(output/'attribution.json', dict(cases=cases))
        for case in cases:
            old = read(ROOT/plan['projection_dir']/f"seed_{case['seed']}"/'diagnosis.json')
            case['true_objective_differences'] = old['true_objective_differences']
        save(output/'summary.json', dict(cases=cases))
        summarize_source_pairs(plan)
    complete(plan, cases)
    for case in cases:
        print(case['seed'], 'before', case['difference_before'], 'after', case['difference_after'])
        for key,value in case['difference_contributions'].items():
            print(' ', key, value)


if __name__ == '__main__':
    main()
