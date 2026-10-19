"""Algebraic propagation decomposition of Stage70's already-scored fixed pairs."""
from collections import Counter
from pathlib import Path

from performance.run_mainline_legacy_backend_bridge import read
from performance.mainline_objective_selection_source import AFTER, POINT, FULL, NATIVE, describe


ARMS = {'AFTER': AFTER, 'POINT': POINT, 'FULL': FULL, 'NATIVE': NATIVE}
CONTRASTS = {'AP10': ('POINT', 'AFTER', '10'), 'AP13': ('POINT', 'AFTER', '13'),
             'FA10': ('AFTER', 'FULL', '10'), 'FA13': ('AFTER', 'FULL', '13'),
             'AN10': ('NATIVE', 'AFTER', '10'), 'AN13': ('NATIVE', 'AFTER', '13')}
TRANSITIONS = ('wrong_to_correct', 'correct_to_wrong', 'same_correct', 'same_wrong', 'label_tie', 'unknown')
COHORTS = ('all_fixed_pairs', 'initial_observed_observed', 'initial_observed_unobserved',
           'initial_unobserved_unobserved', 'final_unobserved_unobserved')
METRICS = ('wrong_fraction_before', 'wrong_fraction_after', 'wrong_fraction_change',
           'absolute_gap_error_change_mean')


def sign(value):
    return (value > 0) - (value < 0)


def subtract(left, right):
    return None if left is None or right is None else left - right


def compare_gap(base, updated, label):
    result = dict(base_gap=base, updated_gap=updated, label_gap=label, increment=subtract(updated, base),
        transition='unknown', base_correct=None, updated_correct=None, absolute_gap_error_change=None,
        direction=None, crossed_label=None, new_wrong_mechanism=None)
    if base is None or updated is None:
        return result
    change = updated - base
    error_change = abs(updated - label) - abs(base - label)
    direction = ('unchanged' if change == 0 else 'toward_label' if change * (label-base) > 0 else 'away_from_label')
    crossed = (base-label) * (updated-label) < 0
    result.update(direction=direction, crossed_label=crossed, absolute_gap_error_change=error_change)
    if label == 0:
        result['transition'] = 'label_tie'
        return result
    before, after = sign(base) == sign(label), sign(updated) == sign(label)
    transition = ('same_correct' if before and after else 'correct_to_wrong' if before else
                  'wrong_to_correct' if after else 'same_wrong')
    result.update(base_correct=before, updated_correct=after, transition=transition)
    if transition == 'correct_to_wrong':
        if direction == 'toward_label' and crossed:
            effect = 'improved' if error_change < 0 else 'worsened' if error_change > 0 else 'unchanged'
            result['new_wrong_mechanism'] = 'toward_label_overshoot_absolute_error_' + effect
        else:
            result['new_wrong_mechanism'] = 'away_from_label'
    return result


def decompose_pair(pair, states):
    label = pair['unused_mean_difference']
    contrasts = {}
    for name, (base, updated, prefix) in CONTRASTS.items():
        a, b = states[base][prefix], states[updated][prefix]
        contrasts[name] = dict(compare_gap(a['gap'], b['gap'], label),
            base_both_eligible=a['both_predicted_eligible'], updated_both_eligible=b['both_predicted_eligible'],
            common_both_eligible=(None if a['both_predicted_eligible'] is None or b['both_predicted_eligible'] is None
                else a['both_predicted_eligible'] and b['both_predicted_eligible']))
    p10, p13 = [states['POINT'][prefix]['gap'] for prefix in ('10', '13')]
    a10, a13 = [states['AFTER'][prefix]['gap'] for prefix in ('10', '13')]
    f13 = states['FULL']['13']['gap']
    initialization = subtract(a10, p10)
    own = subtract(p13, p10)
    interaction = subtract(subtract(a13, p13), initialization)
    continued = subtract(f13, a13)
    terms = [p10, own, initialization, interaction, continued]
    reconstructed = sum(terms) if all(value is not None for value in terms) else None
    return dict(i=pair['i'], j=pair['j'], policy_ids=pair['policy_ids'],
        distance_squared=pair['distance_squared'], selected_from=pair['selected_from'],
        initial_pair_group=pair['observed']['10']['pair_group'], final_pair_group=pair['observed']['13']['pair_group'],
        label_gap=label, label_standard_error=pair['independent_mean_difference_standard_error'],
        label_loo=pair['leave_one_endpoint_repeat'], empirical_feasible=pair['empirical_feasible'],
        empirical_both_feasible=pair['empirical_both_feasible'], states=states, contrasts=contrasts,
        timeline=dict(point10=p10, point_self_change_13_minus_10=own,
            initialization_effect_after10_minus_point10=initialization,
            initialization_self_update_interaction=interaction,
            after_self_change_13_minus_10=subtract(a13, a10),
            after13_minus_point13=subtract(a13, p13),
            continued_full13_minus_after13=continued, after13=a13, full13=f13,
            reconstructed_full13=reconstructed, reconstruction_residual=subtract(f13, reconstructed)))


def analyze_case(case):
    source_arms = {arm['arm']: arm for arm in case['arms']}
    indexed = {name: {prefix: {row['pair_index']: row for row in source_arms[arm]['snapshots'][prefix]['pairs']}
        for prefix in ('10', '13')} for name, arm in ARMS.items()}
    result = {key: case[key] for key in ('archive_heldout_family', 'fold', 'family', 'd', 'held_policy_ids')}
    result['arm_status'] = {name: dict(status=source_arms[arm]['status'], failures=source_arms[arm]['failures'],
        snapshots={prefix: source_arms[arm]['snapshots'][prefix]['status'] for prefix in ('10', '13')})
        for name, arm in ARMS.items()}
    result['pairs'] = []
    for index, pair in enumerate(case['pairs']):
        states = {}
        for name, arm in ARMS.items():
            states[name] = {}
            for prefix in ('10', '13'):
                snapshot = source_arms[arm]['snapshots'][prefix]
                prediction = indexed[name][prefix][index] if snapshot['status'] == 'complete' else None
                states[name][prefix] = dict(status=snapshot['status'], failure=snapshot.get('failure'),
                    gap=None if prediction is None else prediction['predicted_difference'],
                    both_predicted_eligible=None if prediction is None else prediction['both_predicted_eligible'],
                    endpoint_eligible=None if prediction is None else prediction['endpoint_eligible'],
                    endpoint_probability_violation=None if prediction is None else prediction['endpoint_probability_violation'],
                    point_reference_semantics=('prior_only_unobserved_pair' if name == 'POINT' and
                        pair['observed'][prefix]['pair_group'] == 'unobserved_unobserved' else
                        'point_only_reference' if name == 'POINT' else None))
        result['pairs'].append(decompose_pair(pair, states))
    return result


def in_cohort(pair, cohort):
    if cohort == 'all_fixed_pairs':
        return True
    if cohort == 'final_unobserved_unobserved':
        return pair['final_pair_group'] == 'unobserved_unobserved'
    return pair['initial_pair_group'] == cohort.removeprefix('initial_')


def summarize_pairs(pairs, contrast):
    rows = [pair['contrasts'][contrast] for pair in pairs]
    known = [row for row in rows if row['base_correct'] is not None]
    numeric = [row for row in rows if row['absolute_gap_error_change'] is not None]
    counts = dict.fromkeys(TRANSITIONS, 0)
    for row in rows:
        counts[row['transition']] += 1
    before = sum(not row['base_correct'] for row in known) / len(known) if known else None
    after = sum(not row['updated_correct'] for row in known) / len(known) if known else None
    return dict(pairs=len(pairs), available_untied_pairs=len(known), transitions=counts,
        stable_label_transitions=dict(Counter(pair['contrasts'][contrast]['transition'] for pair in pairs
            if pair['label_loo']['all_same_sign_as_full'])),
        unstable_label_transitions=dict(Counter(pair['contrasts'][contrast]['transition'] for pair in pairs
            if not pair['label_loo']['all_same_sign_as_full'])),
        both_empirically_feasible_transitions=dict(Counter(pair['contrasts'][contrast]['transition'] for pair in pairs
            if pair['empirical_both_feasible'])),
        common_eligible_transitions=dict(Counter(row['transition'] for row in rows if row['common_both_eligible'] is True)),
        new_wrong_mechanisms=dict(Counter(row['new_wrong_mechanism'] for row in rows if row['new_wrong_mechanism'])),
        metrics=dict(wrong_fraction_before=before, wrong_fraction_after=after,
            wrong_fraction_change=None if before is None else after-before,
            absolute_gap_error_change_mean=(sum(row['absolute_gap_error_change'] for row in numeric) / len(numeric)
                if numeric else None)))


def aggregate_cases(cases):
    result = []
    for cohort in COHORTS:
        for contrast in CONTRASTS:
            groups = [summarize_pairs([pair for pair in case['pairs'] if in_cohort(pair, cohort)], contrast)
                      for case in cases]
            fields = ('transitions', 'stable_label_transitions', 'unstable_label_transitions',
                      'both_empirically_feasible_transitions', 'common_eligible_transitions', 'new_wrong_mechanisms')
            totals = {field: dict(sum((Counter(group[field]) for group in groups), Counter())) for field in fields}
            totals['transitions'] = {name: totals['transitions'].get(name, 0) for name in TRANSITIONS}
            totals['transitions'] = {key: totals['transitions'].get(key, 0) for key in TRANSITIONS}
            result.append(dict(cohort=cohort, contrast=contrast, registered_source_groups=len(cases),
                groups_with_pairs=sum(group['pairs'] > 0 for group in groups),
                pairs=sum(group['pairs'] for group in groups),
                available_untied_pairs=sum(group['available_untied_pairs'] for group in groups), **totals,
                metrics={key: describe([group['metrics'][key] for group in groups]) for key in METRICS}))
    return result


def analyze_propagation(root, plan):
    source = read(Path(root) / plan['inputs']['neighbor_analysis'])
    cases = [analyze_case(case) for case in source['cases']]
    uu = [pair for case in cases for pair in case['pairs'] if pair['final_pair_group'] == 'unobserved_unobserved']
    changes = {}
    for name in ARMS:
        values = [subtract(pair['states'][name]['13']['gap'], pair['states'][name]['10']['gap']) for pair in uu]
        known = [value for value in values if value is not None]
        changes[name] = dict(registered_pairs=len(uu), available_pairs=len(known),
            maximum_absolute_gap_change=max(map(abs, known)) if known else None)
    return dict(status='complete', coverage=dict(source['coverage'],
        original_uu_pairs=sum(pair['initial_pair_group'] == 'unobserved_unobserved' for case in cases for pair in case['pairs']),
        final_uu_pairs=len(uu)), source_analysis=plan['inputs']['neighbor_analysis'],
        cases=cases, aggregates=aggregate_cases(cases),
        family_aggregates=[dict(family=family, aggregates=aggregate_cases([case for case in cases if case['family'] == family]))
            for family in sorted({case['family'] for case in cases})],
        final_uu_saved_gap_changes=changes,
        prior_paid_accounting=source['prior_paid_accounting'],
        source_fit_flags_location=plan['inputs']['neighbor_analysis'] + ':source_fit_flags',
        definitions=dict(fixed_population='All Stage70 geometry-sealed pairs are retained. Initial UU30 is followed unchanged to prefix13; final UU6 is separate.',
            point_reference='POINT is an unconditioned prior gap only while both endpoints are unobserved; other POINT gaps are point-only references.',
            shared_mean_shift='The same additive shift at both unobserved endpoints cancels in their objective gap; UU gap changes describe differential propagation, not a common intercept correction.',
            transition='Exact objective-gap signs versus the frozen unused label gap; a predicted tie counts as wrong for a nonzero label. Zero label gaps remain label_tie.',
            overshoot='An increment starts toward the label, crosses the label and ends with the wrong objective-gap sign; its absolute error may improve or worsen, reported separately.',
            timeline='FULL13 = POINT10 + (POINT13-POINT10) + (AFTER10-POINT10) + [(AFTER13-POINT13)-(AFTER10-POINT10)] + (FULL13-AFTER13).',
            interaction='The difference in subsequent point-local updates under the two initialized states. This is aggregate saved-state algebra, not attribution to individual observation anchors.',
            statistical_unit='Within-source-group summaries precede means across24 overlapping groups; pair counts are descriptive, not independent experiments.'),
        limits=['Frozen source labels use only2-3 repeats; leave-one sign stability and empirical feasibility are not truth or certificates.',
            'No posterior covariance or target acquisition sampling margin can be reconstructed from these gap decompositions.',
            'No interpolation scale, sampling temperature, weights or correction parameters are fitted or searched.'],
        new_work=dict(new_source_calls=0, new_target_calls=0, new_verification_calls=0, new_truth_function_calls=0,
            new_source_fits=0, posterior_restorations=0, raw_replays=0, new_rng_draws=0))


def compact_pair(case, pair):
    return dict(archive_heldout_family=case['archive_heldout_family'], fold=case['fold'], family=case['family'],
        i=pair['i'], j=pair['j'], policy_ids=pair['policy_ids'],
        initial_group=pair['initial_pair_group'], final_group=pair['final_pair_group'], label_gap=pair['label_gap'],
        label_standard_error=pair['label_standard_error'], label_loo_stable=pair['label_loo']['all_same_sign_as_full'],
        label_loo_range=[pair['label_loo']['minimum'], pair['label_loo']['maximum']],
        empirical_feasible=pair['empirical_feasible'], timeline=pair['timeline'],
        saved_gaps={name: [states[prefix]['gap'] for prefix in ('10', '13')] for name, states in pair['states'].items()},
        endpoint_eligible={name: [states[prefix]['endpoint_eligible'] for prefix in ('10', '13')]
            for name, states in pair['states'].items()},
        transitions={name: row['transition'] for name, row in pair['contrasts'].items()},
        new_wrong_mechanisms={name: row['new_wrong_mechanism'] for name, row in pair['contrasts'].items()
            if row['new_wrong_mechanism'] is not None})


def compact_propagation(result):
    compact = {key: value for key, value in result.items() if key not in ('cases', 'family_aggregates')}
    compact['family_aggregates'] = [dict(family=family['family'], aggregates=[row for row in family['aggregates']
        if row['contrast'] in ('AP10', 'AP13', 'FA13')]) for family in result['family_aggregates']]
    compact['original_uu_pairs'] = [compact_pair(case, pair) for case in result['cases'] for pair in case['pairs']
        if pair['initial_pair_group'] == 'unobserved_unobserved']
    compact['changed_pair_columns'] = ['i', 'j', 'contrast', 'transition', 'base_gap', 'increment', 'updated_gap',
        'label_gap', 'absolute_gap_error_change', 'new_wrong_mechanism', 'label_loo_stable', 'both_empirically_feasible',
        'base_both_eligible', 'updated_both_eligible']
    compact['cases'] = []
    for case in result['cases']:
        saved = {key: case[key] for key in ('archive_heldout_family', 'fold', 'family', 'held_policy_ids', 'arm_status')}
        saved['counts'] = {name: summarize_pairs(case['pairs'], name) for name in CONTRASTS}
        saved['changed_pairs'] = [[pair['i'], pair['j'], name, row['transition'], row['base_gap'], row['increment'],
            row['updated_gap'], pair['label_gap'], row['absolute_gap_error_change'], row['new_wrong_mechanism'],
            pair['label_loo']['all_same_sign_as_full'], pair['empirical_both_feasible'],
            row['base_both_eligible'], row['updated_both_eligible']]
            for pair in case['pairs'] for name, row in pair['contrasts'].items()
            if row['transition'] in ('correct_to_wrong', 'wrong_to_correct', 'unknown')]
        compact['cases'].append(saved)
    return compact
