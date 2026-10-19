"""Prepare the fixed initial-reference/coverage factorial without inference.

The common reference order uses only already saved STOP prefix-ten scores.
Verification adapters are exposed for deterministic substitutes; registration
does not call them or authorize a new scientific evaluation.
"""
from copy import deepcopy
import math

from performance import prepare_mainline_two_comparison_terminal as two
from performance import run_mainline_terminal_equal as old


OLD_PROTOCOL = 'same_policy_distinct_challenger'
VARIANTS = ('original_old', 'original_two_comparison',
            'reference_old', 'reference_two_comparison')


def _point(value):
    return tuple(int(x) for x in value)


def freeze_initial_order(stop_item, *, stream, input_file):
    """Freeze all physical initial policies, never reading truth or outcomes."""
    try:
        record = stop_item['terminal_recording']
        if (stop_item['arm'] != 'initialization_only' or
                stop_item['search_evaluation_count'] != 10 or
                stop_item['task']['d'] != 1000 or record['paid_prefix'] != 10 or
                record['status'] != 'frozen'):
            raise ValueError('Reference order requires a frozen d=1000 STOP prefix ten')
        observed = {}
        all_indices = []
        for row in record['observed']:
            point, indices = _point(row['point']), list(row['observation_indices'])
            if not indices:
                raise ValueError('An initial physical policy has no observation indices')
            observed.setdefault(point, []).extend(indices)
            all_indices.extend(indices)
        if sorted(all_indices) != list(range(10)):
            raise ValueError('Initial policies must cover exactly observations zero through nine')
        pools = [row for row in record['stages'] if row['phase'] == 'primary_pool']
        if len(pools) != 1:
            raise ValueError('Reference order requires one saved complete primary_pool')
        pool = pools[0]
        points = [_point(point) for point in pool['pool']]
        components = pool['components']
        if len(points) != len(set(points)) or set(points) != set(observed):
            raise ValueError('Saved primary_pool must cover every distinct initial physical policy once')
        for field in ('probability_violation', 'objective'):
            if field not in components or len(components[field]) != len(points):
                raise ValueError('Saved primary_pool has missing or unaligned ' + field)
        rows = []
        for i, point in enumerate(points):
            probability = float(components['probability_violation'][i])
            objective = float(components['objective'][i])
            # Frozen mixture scores include 1 + one ULP. Preserve the same raw
            # ordering as the producer; clipping would introduce new ties.
            if not math.isfinite(probability) or not math.isfinite(objective):
                raise ValueError('Reference order requires finite saved objective and probability')
            indices = sorted(observed[point])
            rows.append(dict(point=list(point), observation_indices=indices,
                first_initial_index=indices[0], probability_violation=probability,
                objective=objective, source_pool_index=i))
        rows.sort(key=lambda row: (row['probability_violation'], row['objective'],
                                   row['first_initial_index']))
        for rank, row in enumerate(rows, 1):
            row['source_initial_rank'] = rank
        return dict(status='frozen', seed=stop_item['seed'], domain=stop_item['domain'],
            stream=stream, input_file=str(input_file), paid_prefix=10,
            order=rows, initial_observation_count=10,
            initial_physical_policy_count=len(rows),
            ordering=['probability_violation', 'objective', 'first_initial_index'],
            score_source='STOP prefix-ten terminal_recording.primary_pool',
            selection_uses_truth=False, selection_uses_verification=False,
            selection_uses_final_search_scores=False, new_candidate_scores=0)
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing or malformed frozen STOP reference input: ' + str(exc)) from exc


def _check_old(item):
    try:
        expected = dict(protocol=OLD_PROTOCOL, candidate_budgets=[80, 128, 128],
            objective_incumbent_position=2, objective_comparison_budget=8,
            objective_comparison_delta=.05 / 3, familywise_delta=.05)
        if any(item[key] != value for key, value in expected.items()) or item['task']['d'] != 1000:
            raise ValueError('Expected the frozen same_policy_distinct_challenger terminal contract')
        rows = item['frozen_shortlist']
        if len(rows) != 3 or len({_point(row['point']) for row in rows}) != 3:
            raise ValueError('Reference replacement requires three distinct policies and incumbent in slot two')
        if item['search_evaluation_count'] != (10 if item['arm'] == 'initialization_only' else 13):
            raise ValueError('Frozen search count must remain ten for STOP and thirteen otherwise')
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing or malformed original terminal input: ' + str(exc)) from exc


def make_variants(old_item, reference_order):
    """Keep primary/incumbent fixed; cross one reference rule with two verifiers."""
    _check_old(old_item)
    try:
        if (reference_order['status'] != 'frozen' or reference_order['paid_prefix'] != 10 or
                any(reference_order[key] != old_item[key] for key in ('seed', 'domain'))):
            raise ValueError('Frozen common reference order does not identify this task')
        fixed = {_point(row['point']) for row in old_item['frozen_shortlist'][:2]}
        selected = next((row for row in reference_order['order'] if _point(row['point']) not in fixed), None)
        if selected is None:
            raise ValueError('No distinct initial reference remains after primary/incumbent collisions')
        selected = deepcopy(selected)
        displaced = deepcopy(old_item['frozen_shortlist'][2])
        changed = _point(selected['point']) != _point(displaced['point'])
        source = dict(input_file=reference_order['input_file'], stream=reference_order['stream'],
            rank=selected['source_initial_rank'], pool_index=selected['source_pool_index'],
            first_initial_index=selected['first_initial_index'],
            observation_indices=selected['observation_indices'],
            probability_violation=selected['probability_violation'], objective=selected['objective'])
        reference = deepcopy(old_item)
        reference['frozen_shortlist'][2] = dict(point=selected['point'], shortlist_position=3,
            shortlist_role='common_stop_initial_reference', source_initial_order=source,
            probability_violation=selected['probability_violation'], objective=selected['objective'])
        metadata = dict(reference_input_file=reference_order['input_file'],
            reference_stream=reference_order['stream'], selected_initial_rank=selected['source_initial_rank'],
            skipped_primary_incumbent_collisions=[deepcopy(row) for row in reference_order['order']
                if row['source_initial_rank'] < selected['source_initial_rank']],
            reference_shortlist_changed=changed, physical_shortlist_identity=not changed,
            first_two_slots_unchanged=True, displaced_third=displaced,
            reference_selection_uses_truth=False, reference_selection_uses_verification=False,
            reference_order_shared_across_all_arms=True)
        if reference.get('terminal_recording') is not None:
            reference['terminal_recording']['stages'].append(dict(
                phase='reference_coverage_replace_third', changed=changed,
                before=deepcopy(old_item['frozen_shortlist']),
                after=deepcopy(reference['frozen_shortlist']), source_initial_order=deepcopy(source)))
        result = dict(original_old=deepcopy(old_item), original_two_comparison=two.candidate_input(old_item),
            reference_old=reference, reference_two_comparison=two.candidate_input(reference))
        for name, item in result.items():
            item['variant_name'] = name
            item['reference_coverage'] = deepcopy(metadata)
        return result
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing or malformed frozen reference order: ' + str(exc)) from exc


def run_variant(problem, item):
    """Reuse the existing verifier; callers control substitute/science admission."""
    name = item.get('variant_name')
    if name not in VARIANTS:
        raise ValueError('Unknown reference-coverage variant')
    if name.endswith('two_comparison'):
        if item.get('protocol') != two.PROTOCOL:
            raise ValueError('Two-comparison variant protocol mismatch')
        return two.run_verifier(problem, item)
    if item.get('protocol') != OLD_PROTOCOL:
        raise ValueError('Original-verifier variant protocol mismatch')
    return old.run_verifier(problem, item)
