"""Read sealed Stage65 terminal records; never evaluate or restore anything.

One task per scheduler item.  Policy labels are local to a task; exact integer
points join the saved shortlist, paid search history and existing truth cache.
"""
import json
from collections import Counter
from pathlib import Path


ZERO = dict(new_simulation_calls=0, new_truth_function_calls=0, new_source_fits=0,
            posterior_restorations=0, raw_replays=0, new_rng_draws=0)


def read(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _safe(attempt, terminal_status):
    if attempt is None:
        return dict(status='not_attempted' if terminal_status == 'complete' else
                    'unknown_after_' + terminal_status, passed=None)
    return dict(status='passed' if attempt['certified'] else 'failed',
                passed=bool(attempt['certified']),
                **{key: attempt[key] for key in ('upper_margin', 'sample_mean',
                   'sample_std', 'tolerance_factor', 'verification_budget') if key in attempt})


def _comparison(record, labels, truths, seed, block, pair):
    if not record or not record.get('simulation_calls', 0):
        return None
    challenger, incumbent = pair
    a, b = truths[challenger], truths[incumbent]
    truth_difference = None if a is None or b is None else a['objective'] - b['objective']
    keys = ('status', 'challenger_dominates', 'sample_mean_difference',
            'sample_std_difference', 'standard_error', 'one_sided_upper_confidence_bound',
            'critical_value', 'delta', 'simulation_calls', 'comparison_budget_per_policy',
            'comparison_stream_tag', 'comparison_stream_independent_from_search_and_safety')
    result = {key: record[key] for key in keys if key in record}
    result.update(block_index=block, challenger=labels[challenger],
        incumbent=labels[incumbent],
        winner=labels[challenger if record['challenger_dominates'] else incumbent],
        true_objective_difference=truth_difference,
        observed_difference_error=None if truth_difference is None else
            record['sample_mean_difference'] - truth_difference,
        seed_namespace=[seed, 'replication', record['comparison_stream_tag']] + ([block] if block else []))
    return result


def _protocol(descriptor, saved, item, labels, truths):
    status = descriptor['status']
    result = dict(status=status, input_file=descriptor.get('input_file'),
                  verification_file=descriptor.get('verification_file'))
    if status != 'complete':
        result.update(failure_file=descriptor.get('failure_file'),
            reason=descriptor.get('reason', descriptor.get('failure')),
            certified=None, safety=[_safe(None, status) for _ in labels],
            comparisons=[], deployed=None, reported_deployed=None,
            logical_terminal_calls=None, new_physical_terminal_calls=None)
        return result
    verification = saved['verification']
    attempts = {row['candidate_index']: row for row in verification['attempts']}
    comparisons = []
    if 'objective_comparisons' in verification:
        for row in verification['objective_comparisons']:
            comparisons.append(_comparison(row, labels, truths, item['verification_seed'],
                row['block_index'], (row['challenger_index'], row['incumbent_index'])))
    else:
        row = _comparison(verification.get('objective_comparison'), labels, truths,
            item['verification_seed'], 0, (0, item['objective_incumbent_position'] - 1))
        if row is not None:
            comparisons.append(row)
    points = [row['point'] for row in item['frozen_shortlist']]
    deployed = labels[points.index(saved['deployed'])]
    result.update(certified=bool(verification['certified']),
        reported_deployed=deployed, deployed=deployed if verification['certified'] else None,
        safety=[_safe(attempts.get(index), status) for index in range(len(labels))],
        comparisons=comparisons, initial_stage_winner_index=verification.get('initial_stage_winner_index'),
        logical_terminal_calls=verification['verification_budget'],
        new_physical_terminal_calls=saved.get('cumulative_new_simulation_invocations',
                                             saved.get('new_verification_calls')),
        reused_old_raw_calls=saved.get('reused_old_raw_calls', 0))
    return result


def _exposure(protocol, policy, truth):
    safety = protocol['safety']
    index, label = policy['index'], policy['policy']
    safe = safety[index]
    blocks = [row['block_index'] for row in protocol['comparisons']
              if label in (row['challenger'], row['incumbent'])]
    deployed = protocol['deployed'] == label
    disposition = ('unknown_terminal_' + protocol['status'] if protocol['status'] != 'complete' else
        'deployed' if deployed else 'not_safety_tested' if safe['passed'] is None else
        'safety_failed' if not safe['passed'] else 'objective_comparison_not_selected' if blocks else
        'safety_passed_without_objective_comparison')
    return dict(safety=safe, objective_blocks=blocks, deployed=deployed,
        disposition=disposition,
        safety_false_certificate=None if safe['passed'] is None or truth is None else
            bool(safe['passed'] and not truth['true_feasible']))


def analyze_pair(case, reference, items, saved, history, truth):
    """Pure record join, also usable with tiny deterministic fixtures."""
    anchor = items.get('new') or items.get('old')
    labels = reference['shortlist']
    base = {key: reference[key] for key in ('arm', 'history_arm', 'quality_reference_status',
        'quality_baseline_defined', 'best_initial_quality', 'old', 'new', 'B',
        'new_minus_old', 'third_challenge', 'prechange')}
    if anchor is None:
        return dict(base, status='inputs_unavailable', candidates=[], protocols={},
                    coverage=None, attribution_status='unknown_without_frozen_inputs')
    if items.get('old') and items.get('new'):
        # A join error would attribute different policies or search paths to the terminal rule.
        assert [r['point'] for r in items['old']['frozen_shortlist']] == [
            r['point'] for r in items['new']['frozen_shortlist']]
        assert items['old']['source_case'] == items['new']['source_case']
        assert items['old']['verification_seed'] == items['new']['verification_seed']
    points = [tuple(row['point']) for row in anchor['frozen_shortlist']]
    truths = [truth.get(point) for point in points]
    initial = {tuple(row['point']) for row in history[:10]}
    protocols = {key: _protocol(case[key], saved.get(key), items.get(key) or anchor, labels, truths)
                 for key in ('old', 'new')}
    candidates = []
    for index, (point, label, actual) in enumerate(zip(points, labels, truths)):
        frozen = anchor['frozen_shortlist'][index]
        good = None if actual is None or not reference['quality_baseline_defined'] else bool(
            point not in initial and actual['true_feasible'] and
            actual['quality'] > reference['best_initial_quality'] + 1e-12)
        policy = dict(index=index, policy=label, role=frozen.get('shortlist_role'),
            source_role=frozen.get('source_shortlist_role'), posterior_rank=frozen.get('posterior_rank'),
            observed_at=[i for i, row in enumerate(history) if tuple(row['point']) == point],
            initial_policy=point in initial, cached_truth=actual, good_new=good)
        policy.update({key: _exposure(protocol, policy, actual) for key, protocol in protocols.items()})
        candidates.append(policy)
    new = protocols['new']
    third = next((row for row in new['comparisons'] if row['block_index'] == 1), None)
    if new['status'] != 'complete':
        third_state = 'unknown_terminal_' + new['status']
    elif new['safety'][2]['passed'] is not True:
        third_state = 'third_safety_failed'
    elif new['initial_stage_winner_index'] is None:
        third_state = 'third_only_fallback_without_objective_comparison'
    else:
        third_state = 'third_won_comparison' if third['challenger_dominates'] else 'first_stage_winner_retained'
    if third is not None:
        # The actual opponent follows the first-stage winner, not always initial slot one.
        assert third['incumbent'] == labels[new['initial_stage_winner_index']]
    diagnostics = reference.get('matched_diagnostics', {})
    safety_flips = [row for row in diagnostics.get('safety', []) if row['pass_flip'] not in (None, 0)]
    coverage = {}
    good = [row for row in candidates if row['good_new'] is True]
    for key, protocol in protocols.items():
        coverage[key] = dict(shortlisted_good_new=len(good),
            safety_attempted_good_new=sum(row[key]['safety']['passed'] is not None for row in good),
            safety_passed_good_new=sum(row[key]['safety']['passed'] is True for row in good),
            objective_compared_good_new=sum(bool(row[key]['objective_blocks']) for row in good),
            deployed_good_new=sum(row[key]['deployed'] for row in good),
            untested_good_new=[row['policy'] for row in good if row[key]['safety']['passed'] is None],
            unknown_truth=sum(row['cached_truth'] is None for row in candidates),
            unknown_good_classification=sum(row['good_new'] is None for row in candidates))
    old_calls, new_calls = (protocols[key]['logical_terminal_calls'] for key in ('old', 'new'))
    return dict(base, status=reference['status'], source_case=anchor['source_case'],
        protocols=protocols, candidates=candidates, coverage=coverage,
        third_comparison=dict(state=third_state, actual=third,
            independent_namespace_from_block0=None if third is None else third['block_index'] == 1,
            independence_evidence='saved block index and existing verifier seed mapping; no raw replay',
            metrics=reference['third_challenge']),
        prechange_evidence=dict(safety_flips=safety_flips,
            matched_block0=diagnostics.get('block0'), metrics=reference['prechange']),
        false_certificates={key: [row['policy'] for row in candidates
            if row[key]['deployed'] and row[key]['safety_false_certificate'] is True]
            for key in ('old', 'new')},
        logical_terminal_calls_delta=None if old_calls is None or new_calls is None else new_calls - old_calls)


def analyze_task(root, plan, seed):
    """Return all nine terminal-pair attributions for one registered geometry.

    ``plan`` supplies ``source_output_dir`` or the Stage65 ``source_manifest``.
    Only saved JSON records are read; unavailable evidence remains unknown.
    """
    root = Path(root)
    output = plan.get('source_output_dir')
    if output is None:
        output = read(root / plan['source_manifest'])['output_dir']
    output = root / output
    summary, seal = read(output / 'summary.json'), read(output / 'execution_status.json')
    if summary['status'] != 'complete' or seal['status'] != 'sealed':
        raise ValueError('Stage65 must be complete and sealed before attribution')
    references = {row['arm']: row for row in summary['rows'] if row['seed'] == seed}
    cases = [row for row in seal['cases'] if row['seed'] == seed]
    if {row['arm'] for row in cases} != set(references) or not cases:
        raise ValueError('Saved terminal outcomes and summary disagree on registered arms')
    histories, truths, rows = {}, {}, []
    for case in cases:
        items, saved = {}, {}
        for protocol in ('old', 'new'):
            descriptor = case[protocol]
            path = descriptor.get('input_file')
            if path and (root / path).is_file():
                items[protocol] = read(root / path)
            if descriptor['status'] == 'complete':
                saved[protocol] = read(root / descriptor['verification_file'])
        anchor = items.get('new') or items.get('old')
        history, truth = [], {}
        if anchor:
            path = anchor['source_case']
            if path not in histories:
                histories[path] = read(root / path / 'search.json')['history']
            history = histories[path]
            cache_path = str(Path(path).parent / 'truth.json')
            if cache_path not in truths:
                truths[cache_path] = {tuple(row['point']): row['truth'] for row in read(root / cache_path)}
            truth = truths[cache_path]
        rows.append(analyze_pair(case, references[case['arm']], items, saved, history, truth))
    return dict(seed=seed, domain=cases[0]['domain'], status='complete', arms=rows,
        source_output_dir=str(output.relative_to(root)),
        quality_reference_status=next(iter(references.values()))['quality_reference_status'],
        new_work=dict(ZERO), limits=[
            'Truth is limited to the existing cache; untested policies have no counterfactual certificate.',
            'Terminal-rule effects are paired within a fixed search history; they are not new learning.'])


def _compact_comparison(row):
    return dict(block=row['block_index'], pair=[row['challenger'], row['incumbent']],
        winner=row['winner'], mean_difference=row['sample_mean_difference'],
        true_difference=row['true_objective_difference'],
        upper_bound=row['one_sided_upper_confidence_bound'], delta=row.get('delta'))


def compact_task(result):
    """Small collector payload; arm-local labels never join physical policies."""
    compact = {key: result[key] for key in ('seed', 'domain', 'status', 'quality_reference_status')}
    compact.update(arms=[])
    for row in result['arms']:
        arm = {key: row[key] for key in ('arm', 'history_arm', 'status',
            'quality_baseline_defined', 'best_initial_quality', 'old', 'new', 'B',
            'new_minus_old', 'third_challenge', 'prechange')}
        arm.update(candidates=[], protocols={}, coverage=row['coverage'])
        for candidate in row['candidates']:
            truth = candidate['cached_truth']
            current = {key: candidate[key] for key in ('policy', 'role', 'observed_at', 'good_new')}
            current['truth'] = None if truth is None else {
                key: truth.get(key) for key in ('quality', 'true_feasible', 'chance_margin')}
            for key in ('old', 'new'):
                exposure = candidate[key]
                safe = exposure['safety']
                current[key] = dict(safety=safe['status'], passed=safe['passed'],
                    margin=safe.get('upper_margin'), budget=safe.get('verification_budget'),
                    objective_blocks=exposure['objective_blocks'], deployed=exposure['deployed'])
            arm['candidates'].append(current)
        for key, protocol in row['protocols'].items():
            arm['protocols'][key] = {field: protocol[field] for field in (
                'status', 'certified', 'deployed', 'reported_deployed',
                'logical_terminal_calls', 'new_physical_terminal_calls')}
            arm['protocols'][key]['comparisons'] = [_compact_comparison(c) for c in protocol['comparisons']]
            if protocol['status'] != 'complete':
                arm['protocols'][key]['failure_file'] = protocol.get('failure_file')
                arm['protocols'][key]['reason'] = protocol.get('reason')
        third = row.get('third_comparison')
        if third:
            actual = third['actual']
            arm['third_comparison'] = dict(state=third['state'],
                actual_opponent=None if actual is None else actual['incumbent'],
                block=None if actual is None else actual['block_index'],
                independent_namespace_from_block0=third['independent_namespace_from_block0'],
                seed_namespace=None if actual is None else actual['seed_namespace'])
        evidence = row.get('prechange_evidence', {})
        block0 = evidence.get('matched_block0') or {}
        arm['prechange_evidence'] = dict(safety_flips=evidence.get('safety_flips', []),
            same_block0=block0.get('matched_actual_pair_stream'),
            stricter_delta_changed_decision=block0.get('stricter_delta_changed_decision'))
        arm['false_certificates'] = row.get('false_certificates', {})
        arm['logical_terminal_calls_delta'] = row.get('logical_terminal_calls_delta')
        compact['arms'].append(arm)
    return compact


def aggregate_terminals(compact_tasks):
    """Retain all signs and unknowns; existing Stage65 inference is not refit."""
    grouped = {}
    for task in compact_tasks:
        for row in task['arms']:
            grouped.setdefault(row['arm'], []).append((task, row))
    arms = []
    for name, rows in grouped.items():
        result = dict(arm=name, registered_pairs=len(rows),
            completed_pairs=sum(row['status'] == 'complete' for _, row in rows),
            third_state_counts=dict(Counter(row.get('third_comparison', {}).get(
                'state', 'unknown_without_frozen_inputs') for _, row in rows)),
            coverage={}, costs={}, changes={}, false_certificate_cases=[])
        for protocol in ('old', 'new'):
            coverage = [row['coverage'][protocol] for _, row in rows if row['coverage'] is not None]
            result['coverage'][protocol] = dict(available_pairs=len(coverage),
                **{key: sum(c[key] for c in coverage) for key in (
                    'shortlisted_good_new', 'safety_attempted_good_new', 'safety_passed_good_new',
                    'objective_compared_good_new', 'deployed_good_new', 'unknown_truth',
                    'unknown_good_classification')},
                untested_good_new=sum(len(c['untested_good_new']) for c in coverage))
            result['costs'][protocol] = {}
            for key in ('logical_terminal_calls', 'new_physical_terminal_calls'):
                values = [row['protocols'].get(protocol, {}).get(key) for _, row in rows]
                known = [value for value in values if value is not None]
                result['costs'][protocol][key] = dict(known_sum=sum(known),
                    available_pairs=len(known), unknown_pairs=len(values) - len(known))
            for task, row in rows:
                for policy in row.get('false_certificates', {}).get(protocol, []):
                    result['false_certificate_cases'].append(dict(seed=task['seed'],
                        domain=task['domain'], protocol=protocol, policy=policy))
        for field in ('new_minus_old', 'third_challenge', 'prechange'):
            result['changes'][field] = {}
            for metric in ('C', 'G', 'false_certificate'):
                counts = dict(positive=0, zero=0, negative=0, unknown=0)
                events, known = [], []
                for task, row in rows:
                    value = row[field][metric]
                    sign = ('unknown' if value is None else 'positive' if value > 1e-12 else
                            'negative' if value < -1e-12 else 'zero')
                    counts[sign] += 1
                    if value is not None:
                        known.append(value)
                    if sign != 'zero':
                        events.append(dict(seed=task['seed'], domain=task['domain'], value=value,
                            sign=sign, old=row['protocols'].get('old', {}).get('deployed'),
                            new=row['protocols'].get('new', {}).get('deployed')))
                result['changes'][field][metric] = dict(counts=counts,
                    known_sum=sum(known), conditional_mean=sum(known) / len(known) if known else None,
                    events=events)
        arms.append(result)
    return dict(task_count=len(compact_tasks), arms=arms, new_work=dict(ZERO),
                interpretation='Descriptive fixed-history terminal decomposition; no new inference or tuning.')
