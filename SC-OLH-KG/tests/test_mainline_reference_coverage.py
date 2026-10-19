"""Truth-free reference wiring and real verifier branches with substitutes."""
from copy import deepcopy
from itertools import product
import json
import math

import numpy as np
import pytest

from core import terminal_two_comparison as tournament
from core import terminal_verification as original
from performance import mainline_reference_coverage as wiring


def stop_input():
    return dict(seed=79, domain='fixture', task=dict(d=1000), arm='initialization_only',
        search_evaluation_count=10, terminal_recording=dict(status='frozen', paid_prefix=10,
            observed=[dict(point=[i], observation_indices=[i]) for i in range(10)], stages=[
                dict(phase='primary_pool', pool=[[i] for i in range(10)],
                    components=dict(probability_violation=[.01 * i for i in range(10)],
                                    objective=[-i for i in range(10)]))]))


def old_input():
    return dict(seed=79, domain='fixture', task=dict(d=1000), arm='legacy_role_after_init_point_only',
        protocol=wiring.OLD_PROTOCOL, search_evaluation_count=13,
        candidate_budgets=[80, 128, 128], objective_incumbent_position=2,
        objective_comparison_budget=8, objective_comparison_delta=.05 / 3,
        familywise_delta=.05, verification_seed=19, method='normal_quantile_tolerance',
        mean_delta_fraction=.5, shortlist_mode='posterior_objective_challenger_then_safe',
        frozen_shortlist=[dict(point=[point], shortlist_position=i + 1,
            shortlist_role=('primary', 'incumbent', 'third')[i], posterior_rank=21 + i,
            policy_fingerprint='preexisting-value') for i, point in enumerate((11, 1, 2))],
        terminal_recording=dict(status='frozen', stages=[dict(phase='same_policy_transform')]))


def frozen(item=None):
    return wiring.freeze_initial_order(stop_input() if item is None else item,
        stream='observation_2', input_file='saved/stop/input.json')


@pytest.fixture(autouse=True)
def forbid_real_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Reference preparation must not construct real random streams')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    monkeypatch.setattr(np.random, 'SeedSequence', forbidden)


def test_order_uses_all_initials_saved_scores_and_first_occurrence_tie_only():
    item = stop_input()
    pool = item['terminal_recording']['stages'][0]
    pool['components']['probability_violation'] = [.1] * 10
    pool['components']['objective'] = [3.] * 10
    pool['components']['objective'][4] = -1.
    pool['pool'].reverse()
    for values in pool['components'].values():
        values.reverse()
    before = deepcopy(item)
    result = frozen(item)
    assert [row['point'] for row in result['order']] == [[4], [0], [1], [2], [3], [5], [6], [7], [8], [9]]
    assert result['initial_observation_count'] == result['initial_physical_policy_count'] == 10
    assert result['selection_uses_truth'] is result['selection_uses_verification'] is False
    assert result['selection_uses_final_search_scores'] is False and result['new_candidate_scores'] == 0
    assert item == before and json.loads(json.dumps(result)) == result
    # A radically different truth, terminal result or final-score annotation is irrelevant.
    item.update(truth=dict(best_policy=[9]), verification=dict(deployed=[8]),
                final_search_scores=dict(preferred=[7]))
    assert frozen(item) == result


def test_repeated_initial_observations_are_one_physical_reference():
    item = stop_input()
    record = item['terminal_recording']
    record['observed'][9]['point'] = [0]
    record['stages'][0]['pool'].pop()
    for values in record['stages'][0]['components'].values():
        values.pop()
    result = frozen(item)
    assert result['initial_observation_count'] == 10 and result['initial_physical_policy_count'] == 9
    assert result['order'][0]['observation_indices'] == [0, 9]
    record['observed'][0]['observation_indices'].append(9)
    record['observed'].pop()
    assert frozen(item) == result


def test_saved_mixture_probability_roundoff_keeps_raw_order_without_clipping():
    item = stop_input()
    components = item['terminal_recording']['stages'][0]['components']
    value = math.nextafter(1., math.inf)
    components['probability_violation'][-2:] = [1., value]
    components['objective'][-2:] = [100., -100.]
    result = frozen(item)
    # Clipping to one would incorrectly promote policy9 through its objective.
    assert [row['point'] for row in result['order'][-2:]] == [[8], [9]]
    assert result['order'][-1]['probability_violation'] == value
    assert components['probability_violation'][-1] == value


@pytest.mark.parametrize('damage', ['missing_score', 'missing_policy', 'missing_observation',
                                  'not_stop', 'not_frozen', 'nan_score'])
def test_incomplete_initial_evidence_is_a_failure_not_a_reference_fallback(damage):
    item = stop_input()
    record, pool = item['terminal_recording'], item['terminal_recording']['stages'][0]
    if damage == 'missing_score':
        del pool['components']['objective']
    elif damage == 'missing_policy':
        pool['pool'].pop()
    elif damage == 'missing_observation':
        record['observed'].pop()
    elif damage == 'not_stop':
        item['arm'] = 'legacy_role_after_init_point_only'
    elif damage == 'not_frozen':
        record['status'] = 'incomplete'
    else:
        pool['components']['objective'][0] = float('nan')
    with pytest.raises(ValueError):
        frozen(item)


def test_four_variants_keep_slots_search_inputs_and_displaced_metadata():
    item, order = old_input(), frozen()
    before_item, before_order = deepcopy(item), deepcopy(order)
    variants = wiring.make_variants(item, order)
    assert tuple(variants) == wiring.VARIANTS
    assert item == before_item and order == before_order
    for name, variant in variants.items():
        assert variant['variant_name'] == name
        assert variant['frozen_shortlist'][:2] == item['frozen_shortlist'][:2]
        assert variant['search_evaluation_count'] == 13 and variant['verification_seed'] == 19
        assert variant['objective_incumbent_position'] == 2
        meta = variant['reference_coverage']
        assert meta['displaced_third'] == item['frozen_shortlist'][2]
        assert meta['first_two_slots_unchanged'] and meta['reference_shortlist_changed']
        assert not meta['reference_selection_uses_truth']
        if name.startswith('original_'):
            assert variant['frozen_shortlist'] == item['frozen_shortlist']
            assert variant['terminal_recording'] == item['terminal_recording']
        else:
            third = variant['frozen_shortlist'][2]
            assert third['point'] == [0] and third['shortlist_position'] == 3
            assert third['source_initial_order']['rank'] == 1
            assert third['source_initial_order']['probability_violation'] == 0.
            assert 'posterior_rank' not in third and 'policy_fingerprint' not in third
            assert variant['terminal_recording']['stages'][-1]['after'] == variant['frozen_shortlist']
        if name.endswith('two_comparison'):
            assert variant['protocol'] == wiring.two.PROTOCOL
            assert variant['candidate_budgets'] == [80, 120, 120]
            assert variant['objective_comparison_delta'] == .05 / 6
        else:
            assert variant['protocol'] == wiring.OLD_PROTOCOL
            assert variant['candidate_budgets'] == [80, 128, 128]
            assert variant['objective_comparison_delta'] == .05 / 3
    assert json.loads(json.dumps(variants)) == variants


def test_primary_incumbent_collisions_are_skipped_and_same_third_is_identity():
    item = old_input()
    item['frozen_shortlist'][0]['point'] = [0]
    variants = wiring.make_variants(item, frozen())
    meta = variants['reference_old']['reference_coverage']
    assert [row['point'] for row in meta['skipped_primary_incumbent_collisions']] == [[0], [1]]
    assert variants['reference_old']['frozen_shortlist'][2]['point'] == [2]
    assert meta['physical_shortlist_identity'] and not meta['reference_shortlist_changed']


@pytest.mark.parametrize('damage', ['incumbent_slot', 'duplicate', 'budget', 'wrong_task', 'no_distinct'])
def test_invalid_fixed_terminal_or_reference_order_is_not_repaired(damage):
    item, order = old_input(), frozen()
    if damage == 'incumbent_slot':
        item['objective_incumbent_position'] = 1
    elif damage == 'duplicate':
        item['frozen_shortlist'][1]['point'] = [11]
    elif damage == 'budget':
        item['candidate_budgets'][1] = 120
    elif damage == 'wrong_task':
        order['seed'] += 1
    else:
        order['order'] = [row for row in order['order'] if row['point'] == [1]]
    with pytest.raises(ValueError):
        wiring.make_variants(item, order)


class SubstituteProblem:
    def __init__(self):
        self.calls = []

    def simulate(self, point, rng=None):
        self.calls.append(tuple(point))


@pytest.mark.parametrize('safe', list(product((False, True), repeat=3)))
@pytest.mark.parametrize('comparison_win', [False, True])
def test_four_wired_variants_reach_all_safety_branches_and_account_resource_trade(
        monkeypatch, safe, comparison_win):
    variants = wiring.make_variants(old_input(), frozen())
    results = {}
    for name, item in variants.items():
        problem, events = SubstituteProblem(), []
        points = [tuple(row['point']) for row in item['frozen_shortlist']]

        def safety(task, point, **kw):
            index = points.index(tuple(point))
            assert kw['candidate_index'] == index and kw['seed'] == 19
            assert kw['delta'] == .05 / 3
            assert kw['verification_budget'] == item['candidate_budgets'][index]
            events.append(('safety', index))
            for _ in range(kw['verification_budget']):
                task.simulate(point)
            return dict(candidate_index=index, certified=safe[index],
                        verification_budget=kw['verification_budget'])

        def paired(task, challenger, incumbent, **kw):
            challenger_index, incumbent_index = points.index(tuple(challenger)), points.index(tuple(incumbent))
            assert safe[challenger_index] and safe[incumbent_index]
            assert kw['comparison_budget'] == 8 and kw['delta'] == item['objective_comparison_delta']
            block = kw.get('comparison_block_index', 0)
            assert block == (1 if challenger_index == 2 else 0)
            events.append(('comparison', challenger_index, incumbent_index, block))
            for _ in range(8):
                task.simulate(challenger)
                task.simulate(incumbent)
            return dict(challenger_dominates=comparison_win, simulation_calls=16)

        for module in (original, tournament):
            monkeypatch.setattr(module, 'verify_frozen_policy', safety)
            monkeypatch.setattr(module, 'verify_paired_objective_dominance', paired)
        deployed, report = wiring.run_variant(problem, item)
        results[name] = report
        selected = report['selected_shortlist_rank']
        assert report['certified'] == any(safe)
        if selected is not None:
            assert safe[selected - 1] and deployed == points[selected - 1]
        else:
            assert report['status'] == 'abstained_uncertified'
        assert report['verification_budget'] == len(problem.calls) <= 352
        assert report['total_evaluation_count'] == 13 + len(problem.calls)
        assert report['max_verification_budget'] == 352
        if name.endswith('two_comparison'):
            assert [row['candidate_index'] for row in report['attempts']] == [0, 1, 2]
            assert report['safety_verification_budget'] == 320
            third_compared = any(event[0] == 'comparison' and event[1] == 2 for event in events)
            assert third_compared == (safe[2] and any(safe[:2]))
            if safe[2] and (comparison_win or not any(safe[:2])):
                assert selected == 3
        else:
            expected_attempts = [0, 1] if any(safe[:2]) else [0, 1, 2]
            assert [row['candidate_index'] for row in report['attempts']] == expected_attempts
            assert report['verification_budget'] in (208, 224, 336)
            assert not any(event[0] == 'comparison' and event[1] == 2 for event in events)
    for prefix in ('original', 'reference'):
        old, new = results[prefix + '_old'], results[prefix + '_two_comparison']
        difference = new['verification_budget'] - old['verification_budget']
        assert difference == ((128 if safe[2] else 112) if any(safe[:2]) else -16)
    if all(safe):
        assert results['reference_old']['verification_budget'] == 224
        assert results['reference_two_comparison']['verification_budget'] == 352
    if not any(safe[:2]):
        assert results['reference_old']['verification_budget'] == 336
        assert results['reference_two_comparison']['verification_budget'] == 320
