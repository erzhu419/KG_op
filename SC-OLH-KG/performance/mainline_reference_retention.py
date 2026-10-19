"""Trace frozen initial references through saved terminals without new inference.

Reference identities are frozen from the initialization-only input before any
truth or verification outcome is joined. Role groups overlap; physical policies
occur once in the initial reference universe.
"""
from copy import deepcopy
import json
from pathlib import Path


STOP = 'initialization_only'
PROTOCOLS = ('old', 'new')
ROLES = ('empirical_incumbent', 'stop_primary', 'stop_safe_fallback',
         'stop_final_slot1', 'stop_final_slot2', 'stop_final_slot3')


def read(path):
    return json.loads(Path(path).read_text())


def point_key(point):
    return tuple(map(int, point))


def references_from_input(item, *, stream, input_file):
    """Freeze only the saved prefix-ten input; never consult a certificate."""
    recording = item['terminal_recording']
    if item['arm'] != STOP or item['search_evaluation_count'] != 10 or recording['paid_prefix'] != 10:
        raise ValueError('Initial references require the saved initialization-only prefix ten')
    if recording['status'] != 'frozen':
        raise ValueError('Initial references require a complete frozen selection record')
    observed = recording['observed']
    positions = [i for row in observed for i in row['observation_indices']]
    if sorted(positions) != list(range(10)):
        raise ValueError('Initial reference universe must cover exactly the initial ten observations')
    references, by_point = [], {}
    for row in observed:
        point = point_key(row['point'])
        if point in by_point:
            raise ValueError('Saved initial physical reference occurs twice')
        reference = dict(reference_id=f"initial_{min(row['observation_indices']):02d}",
            point=list(point), observation_indices=list(row['observation_indices']), roles=[])
        references.append(reference)
        by_point[point] = reference
    stages = {row['phase']: row for row in recording['stages']}
    shortlist = item['frozen_shortlist']
    roles = dict(empirical_incumbent=shortlist[item['objective_incumbent_position']-1]['point'],
        stop_primary=stages['primary_order']['selected'],
        stop_safe_fallback=stages['safe_fallback']['selected'].get('point'))
    roles.update({f'stop_final_slot{i+1}': row['point'] for i, row in enumerate(shortlist)})
    if len(shortlist) != 3:
        raise ValueError('This frozen reference protocol requires the existing three slots')
    for role, point in roles.items():
        if point is None:
            continue
        reference = by_point.get(point_key(point))
        if reference is None:
            raise ValueError('Initialization role points outside the initial physical universe')
        reference['roles'].append(role)
    return dict(status='frozen', seed=item['seed'], domain=item['domain'], stream=stream,
        input_file=str(input_file), paid_prefix=10, references=references,
        role_status={role: 'unavailable' if point is None else 'frozen' for role, point in roles.items()},
        initial_observation_count=10, initial_physical_policy_count=len(references),
        selection_uses_truth=False, selection_uses_verification=False,
        selection_uses_final_search_scores=False,
        scope='All initial physical policies; saved STOP role labels are overlapping descriptive groups.')


def freeze_references(root, child_plan, seed, *, stream):
    """Read only the STOP frozen input, with no saved truth/outcome analysis."""
    root = Path(root)
    seal = read(root / child_plan['output_dir'] / 'execution_status.json')
    if seal['status'] != 'sealed':
        raise ValueError('Reference registration requires the completed original terminal seal')
    case = next(row for row in seal['cases'] if row['seed'] == seed and row['arm'] == STOP)
    input_file = case['old']['input_file']
    return references_from_input(read(root / input_file), stream=stream, input_file=input_file)


def _terminal_structure(protocol, item, execution, registry):
    if item is None:
        return dict(status='input_unavailable', shortlist=[], safety=[], comparisons=[],
            costs=deepcopy(execution.get('costs', {})))
    shortlist = [dict(policy_id=registry[point_key(row['point'])], slot=i+1,
        role=row.get('shortlist_role'), incumbent=i+1 == item['objective_incumbent_position'])
        for i, row in enumerate(item['frozen_shortlist'])]
    safety = [dict(deepcopy(row), policy_id=shortlist[row['candidate_index']]['policy_id'])
        for row in execution.get('safety', [])]
    comparisons = deepcopy(execution.get('comparisons', []))
    for row in comparisons:
        for role in ('challenger', 'incumbent', 'winner'):
            index = row.get(role + '_index')
            row[role + '_policy_id'] = None if index is None else shortlist[index]['policy_id']
    fields = ('candidate_budgets', 'familywise_delta', 'objective_comparison_budget',
        'objective_comparison_delta', 'objective_incumbent_position', 'verification_seed',
        'search_evaluation_count', 'maximum_objective_comparisons', 'method',
        'mean_delta_fraction', 'shortlist_mode', 'tau')
    return dict(status='available', protocol=protocol, shortlist=shortlist,
        **{key: deepcopy(item.get(key)) for key in fields},
        safety=safety, comparisons=comparisons, costs=deepcopy(execution.get('costs', {})),
        random_blocks=dict(safety='Saved verification_seed, shortlist slot and replication',
            comparison='Saved verification_seed, replication and comparison block; same slot is not same policy'))


def _reference_row(reference, observed, protocol, descriptor, structure):
    row = next((r for r in observed if r['policy_id'] == reference['policy_id']), None)
    base = {key: deepcopy(reference[key]) for key in
        ('reference_id', 'policy_id', 'roles', 'observation_indices')}
    if row is None:
        return dict(**base, status='initial_reference_missing_from_saved_history',
            shortlisted=None, attempted=None, certified=None, deployed=None,
            comparison_attempted=None, truth=None, quality_minus_actual_deployed=None,
            path=None, selection_record=None)
    path = deepcopy(row.get('protocols', {}).get(protocol, {}))
    input_available = structure['status'] == 'available'
    complete = descriptor['status'] == 'complete'
    shortlisted = (None if not input_available else
        reference['policy_id'] in {r['policy_id'] for r in structure['shortlist']})
    attempted = (None if not complete else bool(path.get('safety_attempted', False)))
    certified = (None if not complete else bool(path.get('safety_passed', False)))
    deployed = (None if not complete else bool(path.get('deployed', False)))
    compared = (None if not complete else any(r.get('executed') and
        reference['policy_id'] in (r.get('challenger_policy_id'), r.get('incumbent_policy_id'))
        for r in structure['comparisons']))
    truth = deepcopy(row.get('truth'))
    actual_quality = descriptor.get('sets', {}).get('deployed', {}).get('max_quality')
    quality_gap = (None if truth is None or truth.get('true_feasible') is not True or
        truth.get('quality') is None or actual_quality is None else truth['quality']-actual_quality)
    selection = deepcopy(row.get('selection_record'))
    return dict(**base, status='available', shortlisted=shortlisted, attempted=attempted,
        certified=certified, deployed=deployed, comparison_attempted=compared,
        true_certified=(None if certified is None or truth is None or truth.get('true_feasible') is None
            else bool(certified and truth['true_feasible'])),
        true_deployed=(None if deployed is None or truth is None or truth.get('true_feasible') is None
            else bool(deployed and truth['true_feasible'])),
        truth=truth, quality_minus_actual_deployed=quality_gap,
        pre_incumbent_slot=None if selection is None else selection.get('pre_incumbent_slot'),
        post_incumbent_slot=None if selection is None else selection.get('post_incumbent_slot'),
        final_slot=None if selection is None else selection.get('final_slot'),
        path=path, selection_record=selection)


def analyze_task(root, stage77_task, reference_snapshot):
    """Join all saved controls to previously frozen physical reference identities."""
    root = Path(root)
    task = stage77_task
    if task['status'] != 'complete' or reference_snapshot['status'] != 'frozen':
        raise ValueError('Reference attribution requires the saved completed path analysis and frozen references')
    if any(task[key] != reference_snapshot[key] for key in ('seed', 'stream', 'domain')):
        raise ValueError('Frozen initial references do not identify this task stream')
    registry = {point_key(row['point']): row['policy_id'] for row in task['policies']}
    references = [dict(reference_id=row['reference_id'], policy_id=registry[point_key(row['point'])],
        roles=list(row['roles']), observation_indices=list(row['observation_indices']), status='frozen')
        for row in reference_snapshot['references']]
    initial_ids = {row['policy_id'] for row in references}
    arms, inputs = [], {}
    for original in task['arms']:
        arm = {key: deepcopy(original.get(key)) for key in
            ('arm', 'history_arm', 'terminal_backend', 'search_status', 'recording')}
        arm['protocols'] = {}
        for protocol in PROTOCOLS:
            descriptor = original['protocols'][protocol]
            execution = descriptor.get('execution', {})
            filename = execution.get('files', {}).get('input_file')
            if filename is not None and filename not in inputs:
                inputs[filename] = read(root / filename)
            item = inputs.get(filename)
            structure = _terminal_structure(protocol, item, execution, registry)
            result = {key: deepcopy(descriptor.get(key)) for key in ('status', 'metrics',
                'deployed_policy_id', 'reported_policy_id', 'no_deployment_reason')}
            result['terminal_structure'] = structure
            result['reference_rows'] = [_reference_row(reference, original['observed'],
                protocol, descriptor, structure) for reference in references]
            result['reference_counts'] = {key: sum(row[key] is True for row in result['reference_rows'])
                for key in ('shortlisted', 'attempted', 'certified', 'deployed', 'comparison_attempted')}
            result['unknown_reference_counts'] = {key: sum(row[key] is None for row in result['reference_rows'])
                for key in ('shortlisted', 'attempted', 'certified', 'deployed', 'comparison_attempted')}
            result['other_observed_policies'] = [dict(
                **{key: deepcopy(row.get(key)) for key in
                    ('policy_id', 'origin', 'truth', 'final_scores', 'selection_record')},
                path=deepcopy(row.get('protocols', {}).get(protocol)),
                quality_minus_deployed=deepcopy(row.get('quality_minus_deployed', {}).get(protocol)))
                for row in original['observed'] if row['policy_id'] not in initial_ids]
            result['execution'] = deepcopy(execution)
            arm['protocols'][protocol] = result
        arms.append(arm)
    return dict(status='complete', seed=task['seed'], stream=task['stream'], domain=task['domain'],
        reference_source=reference_snapshot['input_file'], references=references,
        reference_role_status=deepcopy(reference_snapshot['role_status']), arms=arms,
        baseline=deepcopy(task.get('baseline')), coverage=dict(initial_observations=10,
            physical_references=len(references), arms=len(arms), terminal_cells=2*len(arms)),
        limits=['Reference identities were frozen from initial STOP inputs before truth or verification was joined.',
            'Role groups overlap; all-initial counts deduplicate physical policies.',
            'Actual role/native differences combine shortlist, verification exposure and random-slot effects.',
            'A missed feasible reference is an opportunity, not an unexecuted counterfactual certificate.'])
