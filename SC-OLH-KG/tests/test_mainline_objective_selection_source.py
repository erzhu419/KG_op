"""Small source-ranking fixtures; no real archives, models or random draws."""
import numpy as np
import pytest

from performance import mainline_objective_selection_source as source


def fixture(mean, probability=None, labels=None, prefix=2):
    count = len(mean)
    labels = np.arange(count, dtype=float) if labels is None else np.asarray(labels, dtype=float)
    snapshot = dict(status='complete',
        mean=np.column_stack([mean, np.zeros(count), -np.ones(count), np.zeros(count)]).tolist(),
        variance=np.ones((count, 4)).tolist(), noise=np.ones((count, 2)).tolist(),
        probability_violation=[.25] * count if probability is None else probability)
    scoring = dict(mean=np.column_stack([labels, -np.ones(count)]),
        variance=np.full((count, 2), .01), replicate_counts=np.full(count, 2))
    return source.analyze_snapshot(snapshot, scoring, .05, list(range(count)), prefix)


def test_common_eligibility_removes_risk_only_selection_change_and_absolute_shift():
    # Different risk masks change each arm's own nomination. A common mask must
    # not call that an objective-ranking improvement, even with a large offset.
    left = fixture([100, 101, 102, 103], [.1, .1, .9, .1])
    right = fixture([0, 1, 2, 3], [.9, .1, .1, .1])
    assert left['selections'][0]['selected'] == 0
    assert right['selections'][0]['selected'] == 1
    paired = source.paired_snapshot(left, right, source.AFTER, source.POINT)
    assert paired['common_eligible_indices'] == [1, 3]
    assert paired['eligibility_disagreements'] == 2
    assert paired['selections'][0]['changed'] is False
    assert paired['selections'][0]['unused_label_objective_delta'] == 0
    assert all(row['delta']['order_error'] in (None, 0) for row in paired['groups'])


def test_small_label_gap_stratum_is_fixed_before_eligibility_filter():
    snap = fixture([0, 6, 4, 8], [.1, .9, .1, .1], labels=[0, 6, 4, 8])
    all_near = next(row for row in snap['groups'] if
        (row['pair_group'], row['scope'], row['subset']) ==
        ('observed_unobserved', 'all', 'small_label_gap_quarter'))
    eligible_near = next(row for row in snap['groups'] if
        (row['pair_group'], row['scope'], row['subset']) ==
        ('observed_unobserved', 'predicted_eligible', 'small_label_gap_quarter'))
    assert all_near['metrics']['pairs'] == 1
    assert eligible_near['metrics']['pairs'] == 0
    assert eligible_near['metrics']['order_error'] is None


def test_pair_labels_ties_and_signed_selection_effects_stay_distinct():
    rows = source.rank_pairs([0, 0, 1, 3], [0, 1, 1, 2], 2)
    metrics = source.pair_metrics(rows)
    assert metrics['untied_label_pairs'] == 5  # Equal scoring labels aren't errors.
    assert next(row for row in rows if (row['i'], row['j']) == (0, 1))['wrong_order'] is True
    left, right = fixture([0, 1, 2, 3], labels=[2, 0, 3, 4]), fixture([1, 0, 2, 3], labels=[2, 0, 3, 4])
    positive = source.paired_snapshot(left, right, source.AFTER, source.POINT)['selections'][0]
    negative = source.paired_snapshot(right, left, source.AFTER, source.POINT)['selections'][0]
    assert positive['unused_label_objective_delta'] == 2
    assert negative['unused_label_objective_delta'] == -2
    assert positive['left']['unused_label_lead_over_runner_up'] == -2
    assert positive['left']['predicted_lead'] == 1


def test_empty_eligibility_is_complete_but_saved_failure_and_denominator_remain_unknown():
    empty = fixture([0, 1, 2, 3], [.9] * 4)
    failed = source.analyze_snapshot(dict(status='failed', error='saved failure'), {}, .05, [0, 1, 2, 3], 2)
    assert empty['status'] == 'complete' and empty['selections'][0]['status'] == 'no_eligible'
    assert source.paired_snapshot(empty, empty, source.AFTER, source.POINT)['status'] == 'complete'
    assert source.paired_snapshot(empty, failed, source.AFTER, source.POINT)['status'] == 'unavailable'
    metric = source.describe([.2, None])
    assert metric['registered_groups'] == 2 and metric['available_groups'] == 1
    assert metric['mean'] is None and metric['conditional_mean'] == pytest.approx(.2)


def test_compact_keeps_all_registered_arms_prefixes_and_failed_source_case():
    arms = [source.NATIVE, source.FULL, source.AFTER, source.POINT]
    good = fixture([0, 1, 2, 3], prefix=2)
    # Production prefix labels are 10/13; the tiny arrays do not represent a real task.
    good10, good13 = dict(good, prefix=10), dict(good, prefix=13)
    failed = source.analyze_snapshot(dict(status='failed', error='saved native failure'), {}, .05, [], 13)
    case = dict(archive_heldout_family='heldout', fold=0, family='fixture', d=2,
        held_policy_ids=[11, 12, 13, 14], arms=[dict(arm=arm,
            status='partial_failure' if arm == source.NATIVE else 'complete',
            failures=['saved native failure'] if arm == source.NATIVE else [],
            snapshots={'10': good10, '13': failed if arm == source.NATIVE else good13}) for arm in arms],
        comparisons=[])
    index = {arm['arm']: arm for arm in case['arms']}
    for left, right in source.COMPARISONS:
        for prefix in ('10', '13'):
            case['comparisons'].append(source.paired_snapshot(index[left]['snapshots'][prefix],
                index[right]['snapshots'][prefix], left, right))
    raw = dict(status='complete', registered_source_groups=1, registered_arm_cases=4,
        registered_snapshots=8, completed_snapshots=7, cases=[case],
        aggregates=source.aggregate_cases([case], arms))
    compact = source.compact_source(raw)
    assert compact['coverage']['arm_cases'] == 4 and compact['coverage']['snapshots'] == 8
    assert compact['coverage']['completed_snapshots'] == 7
    assert compact['coverage']['failed_arm_cases'][0]['arm'] == source.NATIVE
    assert len(compact['cases'][0]['comparisons']) == 6
    assert compact['cases'][0]['arms'][0]['snapshot_status']['13'] == 'failed'
    assert compact['cases'][0]['comparisons'][-1]['status'] == 'unavailable'
