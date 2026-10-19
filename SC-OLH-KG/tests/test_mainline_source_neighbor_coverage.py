"""Tiny deterministic fixtures for reserved labels, noise and fixed-pair masks."""
import copy

import numpy as np
import pytest

from performance import mainline_source_neighbor_coverage as coverage


def fixed_pair(i, j):
    return dict(i=i, j=j, policy_ids=[i, j], distance_squared=.1, selected_from=[i],
        observed={str(prefix): dict(i=i < prefix, j=j < prefix,
            pair_group=('observed_observed' if j < prefix else
                'observed_unobserved' if i < prefix else 'unobserved_unobserved')) for prefix in (10, 13)})


def test_unused_pair_evidence_excludes_adaptation_raw_and_preserves_never_observed_three():
    raw = np.asarray([[[i, -1.], [i+.2, -1.], [i+.4, -1.]] for i in range(16)])
    task = dict(Y_replicates=raw, tau=.3)
    held = list(range(16))
    labels = coverage.scoring_labels(task, held)
    before = coverage.score_fixed_pair(fixed_pair(0, 13), labels,
        coverage.unused_repeats(task, held), .05)
    raw[:13, 0] = 1e9
    after = coverage.score_fixed_pair(fixed_pair(0, 13), coverage.scoring_labels(task, held),
        coverage.unused_repeats(task, held), .05)
    assert before == after
    assert after['unused_replicates'] == [2, 3]
    assert after['unused_mean_difference'] == pytest.approx(.3 - 13.2)
    assert after['empirical_constraint_margins'] == pytest.approx([-1.3, -1.3])
    assert after['crn_paired_difference_variance'] is None


def test_non_crn_row_permutation_cannot_change_noise_or_hide_label_sign_instability():
    left, right = [0., 4.], [1., 2.]
    original = coverage.difference_evidence(left, right)
    permuted = coverage.difference_evidence(left[::-1], right[::-1])
    for name in ('unused_mean_difference', 'endpoint_sample_variances',
                 'independent_episode_difference_variance_estimate',
                 'independent_mean_difference_standard_error'):
        assert original[name] == permuted[name]
    assert original['independent_episode_difference_variance_estimate'] == pytest.approx(8.5)
    assert original['independent_mean_difference_standard_error'] == pytest.approx(np.sqrt(4.25))
    assert original['crn_paired_difference_variance'] is None
    evidence = original['leave_one_endpoint_repeat']
    assert original['unused_mean_difference'] == .5
    assert evidence['minimum'] == -1.5 and evidence['maximum'] == 2.5
    assert evidence['all_same_sign_as_full'] is False
    assert sorted(evidence['differences']) == sorted(permuted['leave_one_endpoint_repeat']['differences'])


def test_fixed_pair_population_survives_label_changes_and_common_qualification_is_symmetric():
    pairs = [dict(fixed_pair(i, j), unused_mean_difference=gap,
        empirical_both_feasible=feasible, leave_one_endpoint_repeat=dict(all_same_sign_as_full=True))
        for i, j, gap, feasible in [(0, 13, -1., True), (13, 14, -2., False)]]
    snapshot = dict(status='complete', mean=np.column_stack([np.arange(16), np.zeros((16, 3))]).tolist(),
        probability_violation=[.1] * 16)
    left = coverage.predict_fixed_pairs(snapshot, pairs, 10)
    other = copy.deepcopy(snapshot)
    other['probability_violation'][13] = .9
    right = coverage.predict_fixed_pairs(other, pairs, 10)
    paired = coverage.compare_snapshots(left, right, pairs, ('left', 'right'))
    reverse = coverage.compare_snapshots(right, left, pairs, ('right', 'left'))
    full = next(row for row in paired['groups'] if
        (row['pair_group'], row['scope']) == ('all', 'all_fixed_pairs'))
    common = next(row for row in paired['groups'] if
        (row['pair_group'], row['scope']) == ('all', 'common_predicted_eligible'))
    assert full['left']['pairs'] == 2 and full['delta']['wrong_order_fraction'] == 0
    assert common['left']['pairs'] == 0 and common['delta']['wrong_order_fraction'] is None
    assert reverse['groups'] == paired['groups']
    altered = copy.deepcopy(pairs)
    altered[0]['unused_mean_difference'] = 100.
    changed = coverage.predict_fixed_pairs(snapshot, altered, 10)
    assert [row['pair_index'] for row in changed['pairs']] == [0, 1]
    assert changed['pairs'][0]['wrong_order'] is True
    failed = coverage.predict_fixed_pairs(dict(status='failed', error='saved failure'), pairs, 10)
    unavailable = coverage.compare_snapshots(left, failed, pairs, ('left', 'failed'))
    assert unavailable['status'] == 'unavailable' and unavailable['groups'] == []
