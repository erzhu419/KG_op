"""Small fixed arrays exercise actual eligibility, physical identity and failed raws."""
from copy import deepcopy

import numpy as np
import pytest

from performance import mainline_safe_unsampled_competition as audit


INITIAL = [dict(point=[i], raw=[float(i), -1.]) for i in range(10)]
SHARED = [(10,), (11,), (12,)]
ACTIONS = [(0,), (10,), (11,), (12,), (13,)]


@pytest.fixture(autouse=True)
def no_random_draws(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved-array fixture cannot draw randomness')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    monkeypatch.setattr(np.random, 'SeedSequence', forbidden)


def truths():
    result = {(i,):dict(objective=5., quality=.5, true_feasible=True, chance_margin=-1.) for i in range(10)}
    result.update({(10,):dict(objective=1., quality=.9, true_feasible=True, chance_margin=-1.),
        (11,):dict(objective=2., quality=.8, true_feasible=True, chance_margin=-1.),
        (12,):dict(objective=0., quality=1., true_feasible=False, chance_margin=1.),
        (13,):dict(objective=3., quality=.7, true_feasible=True, chance_margin=-1.)})
    return result


def bundle(specifications, *, reverse_bank=False, returned=None, status=None):
    points = [(i,) for i in range(14)]
    if reverse_bank:
        points.reverse()
    ids = {point:index for index, point in enumerate(points)}
    history, stages = deepcopy(INITIAL), []
    returned = len(specifications) if returned is None else returned
    for step, (mean, sampled, margin) in enumerate(specifications):
        chosen, total, mode, count = audit.score_choice(sampled, margin)
        stages.append(dict(paid_prefix=10+step, action_ids=[ids[p] for p in ACTIONS],
            observed_ids=[ids[tuple(row['point'])] for row in history],
            selected_index=chosen, selected_id=ids[ACTIONS[chosen]],
            action_sources=['fixture']*len(ACTIONS), score=dict(objective_mean=mean,
                sampled_objective=sampled, sampled_margin=margin, total=total.tolist(),
                sampled_feasibility_mode=mode, sampled_feasible_count=count,
                objective_epistemic=[.1]*5, constraint_mean=[0.]*5,
                constraint_epistemic=[.1]*5, constraint_aleatoric=[.1]*5,
                sampled_constraint=margin, probability_violation=[0.01]*5)))
        if step < returned:
            history.append(dict(point=list(ACTIONS[chosen]), raw=[float(step), -1.]))
    return dict(points=points, audit=dict(shared_ids=[ids[p] for p in SHARED], stages=stages),
        history=history, status=status or ('complete' if len(specifications) == 3 else 'partial'))


def run(bundles, truth=None, cases=(), outcomes=()):
    return audit.analyze_task(1, 'fixture', bundles, truths() if truth is None else truth,
                             INITIAL, SHARED, list(cases), list(outcomes))


def state(result, arm=audit.AFTER, prefix=10):
    return next(row for row in result['states'] if row['arm'] == arm and row['paid_prefix'] == prefix)


def test_actual_mask_safe_pairs_and_regret_do_not_use_terminal_probability():
    spec = ([9., 1., 2., -99., 3.], [9., 2., 1., -99., 3.], [1., -1., -1., 1., 1.])
    result = run({audit.AFTER:bundle([spec])})
    row = state(result)
    assert row['replay']['valid'] and row['selected']['truth']['objective'] == 2.
    assert row['counts']['unobserved_risk_table'] == dict(true_feasible_eligible=2,
        true_feasible_excluded=1, true_infeasible_eligible=0, true_infeasible_excluded=1,
        truth_unknown_eligible=0, truth_unknown_excluded=0)
    assert row['pairs']['all']['pairs'] == 3  # Includes the safe risk-excluded point.
    assert row['pairs']['all']['mean_order'] == {'correct':3}
    assert row['pairs']['all']['sampled_order'] == {'wrong':1, 'correct':2}
    assert row['safe_competition'] == dict(status='selected_in_safe_competition', objective_regret=1.)
    assert row['counts']['improving_safe_unobserved'] == 3
    # The marginal terminal probability says all qualify, but the actual TS mask does not.
    assert next(r for r in row['candidates'] if r['truth']['objective'] == 0.)['eligible'] is False


def test_zero_one_and_tied_safe_competition_remain_explicit():
    mean, sampled = [9., 1., 1., -99., 3.], [9., 1., 1., -99., 3.]
    fallback = state(run({audit.AFTER:bundle([(mean, sampled, [1., .1, .2, .3, .4])])}))
    assert fallback['replay']['mode'] == 'minimum_violation'
    assert fallback['counts']['safe_competition_size'] == 'zero'
    assert fallback['safe_competition']['objective_regret'] is None
    single = state(run({audit.AFTER:bundle([(mean, sampled, [1., -1., .2, .3, .4])])}))
    assert single['counts']['safe_competition_size'] == 'one'
    assert single['safe_competition']['objective_regret'] is None
    pair = single['pairs']['all']
    assert pair['mean_order']['predicted_tie'] == 1
    assert pair['mean_misorder_rate'] == 0. and pair['mean_tie_rate'] == 1/3
    unknown = truths()
    del unknown[(13,)]
    row = state(run({audit.AFTER:bundle([(mean, sampled, [1., -1., .2, .3, .4])])}, unknown))
    assert row['counts']['unobserved_risk_table']['truth_unknown_excluded'] == 1
    assert row['pairs']['all']['pairs'] == 1


def test_common_pool_uses_physical_identity_and_prefix_observation_status():
    mean = [9., 1., 2., 0., 3.]
    a = bundle([(mean, mean, [1., -1., -1., 1., 1.]),
                (mean, [9., 2., 1., 0., 3.], [1., -1., -1., 1., 1.])])
    p = bundle([(mean, mean, [1., 1., -1., 1., -1.]),
                (mean, [9., 2., 1., 0., .5], [1., 1., -1., 1., -1.])], reverse_bank=True)
    result = run({audit.AFTER:a, audit.POINT:p})
    comparisons = [row for row in result['comparisons'] if row['contrast'] == 'AP']
    first, second = comparisons[:2]
    assert first['same_physical_pool'] and first['same_paid_history']
    assert first['first_divergence'] and len(first['substitutions']) == 4
    assert {row['group']:row['count'] for row in first['eligibility_groups']} == dict(
        both_eligible=1, left_only_eligible=1, right_only_eligible=1, both_excluded=1)
    assert first['common_unobserved_count'] == 4
    assert second['common_unobserved_count'] == 2 and not second['same_paid_history']
    assert second['initial_common_unobserved_cohort']['original_count'] == 4
    assert state(result, prefix=11)['initial_unobserved_cohort']['observed'] == 1
    assert state(result, prefix=11)['pairs']['all']['pairs'] == 1
    assert second['substitutions'] == []


def test_partial_selection_is_not_returned_sampling_and_nullable_false_certificates_stay():
    spec = ([9., 1., 2., 0., 3.], [9., 1., 2., 0., 3.], [1., -1., -1., 1., -1.])
    bundles = {audit.AFTER:bundle([spec]*3), audit.POINT:bundle([spec]*3),
               audit.FULL:bundle([spec]*2, returned=1, status='partial')}
    item = dict(frozen_shortlist=[dict(point=list(p)) for p in SHARED])
    def descriptor(certified, point):
        return dict(status='complete', item=item, record=dict(deployed=list(point),
            verification=dict(certified=certified, attempts=[dict(candidate_index=2,
                certified=certified, upper_margin=-1. if certified else 1.)], objective_comparisons=[])))
    cases, outcomes = [], []
    for arm in (audit.AFTER, audit.POINT):
        false_certificate = arm == audit.AFTER
        cases.append(dict(arm=arm, history_arm=arm, old=descriptor(false_certificate, (12,)),
                          new=descriptor(False, (12,))))
        old = dict(A=int(false_certificate), C=0, false_certificate=int(false_certificate), R=0., G=None, P=None, L=None)
        outcomes.append(dict(arm=arm, old=old, new=dict(old, A=0, false_certificate=0)))
    truth = truths()
    for i in range(10):
        truth[(i,)]['true_feasible'] = False
    result = run(bundles, truth, cases, outcomes)
    full = state(result, audit.FULL, 11)
    assert full['replay']['valid'] and not full['replay']['observation_returned']
    assert full['replay']['backend_conditioned'] is None
    chain = next(row for row in result['query_chains'] if row['arm'] == audit.FULL and row['paid_prefix'] == 11)
    assert not chain['observation_returned']
    assert result['status'] == 'partial' and result['coverage']['captured_states'] == 8
    effect = next(row for row in result['terminal_effects'] if row['contrast'] == 'AP' and row['protocol'] == 'old')
    assert effect['delta']['false_certificate'] == 1 and effect['delta']['C'] == 0
    assert effect['delta']['G'] is None
    assert all(value == 0 for value in effect['signed_origin_contributions']['C'].values())
    assert all(value is None for value in effect['signed_origin_contributions']['G'].values())
    compact = audit.compact_task(result)
    summary = audit.aggregate_tasks([compact])
    assert summary['coverage']['registered_states'] == 9
    assert summary['coverage']['captured_states'] == 8
    assert summary['per_task'][0]['status'] == 'partial'
    assert 'candidates' not in state(compact) and 'safe_unobserved_pairs' not in state(compact)
