"""Plan precision and old-terminal costs, then register metadata without science."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.mainline_confirmation_precision import analyze_precision


MANIFEST = 'performance/manifests/mainline_precision_confirmation_preparation_v1.json'
ZERO = dict(new_source_calls=0, new_target_initial_calls=0, new_target_search_calls=0,
    new_verification_calls=0, new_truth_function_calls=0, new_candidate_scores=0,
    new_source_fits=0, source_model_restorations=0, engine_restorations=0,
    posterior_restorations=0, raw_replays=0, simulation_random_draws=0)
SHARED_STAGES = {'search', 'prepare-neighbors', 'freeze-native-terminal',
                 'freeze-modified-only-terminal-inputs', 'freeze-paired-terminal-inputs'}
OLD_VERIFY = 'verify'


def stage_costs(attempts):
    groups = defaultdict(lambda: dict(cpu_seconds=0., attempts=0, failed_attempts=0))
    for row in attempts:
        group = groups[row['stage']]
        group['cpu_seconds'] += row['cpu_seconds']
        group['attempts'] += 1
        group['failed_attempts'] += row['status'] != 'completed'
    return dict(groups)


def paired_source_cost(completion, name):
    if completion['status'] != 'complete':
        raise ValueError('Saved source cost account must be complete')
    groups = stage_costs(completion['attempts'])
    initial = completion['new_work']['new_target_initial_calls']
    online = completion['new_work']['new_target_search_calls']
    terminal = completion['terminal_costs']['old_physical_calls']
    if terminal and OLD_VERIFY not in groups:
        raise ValueError('Old-terminal CPU cannot be replaced by the two-terminal total')
    shared_cpu = sum(row['cpu_seconds'] for stage, row in groups.items() if stage in SHARED_STAGES)
    old_cpu = groups.get(OLD_VERIFY, {}).get('cpu_seconds', 0.)
    return dict(name=name, task_streams=completion['registered_geometries'],
        initial_physical_calls=initial, online_physical_calls=online, old_terminal_physical_calls=terminal,
        old_only_physical_calls=initial + online + terminal, shared_search_cpu_seconds=shared_cpu,
        old_terminal_cpu_seconds=old_cpu, old_only_search_terminal_cpu_seconds=shared_cpu + old_cpu,
        source_all_protocols_target_calls=completion['new_physical_target_calls'],
        source_all_stages_cpu_seconds=completion['recorded_cpu_seconds'],
        source_new_truth_field_calls=completion['new_work']['new_truth_function_calls'],
        source_failed_attempts=completion['failed_attempts'], source_execution_failures=completion['execution_failures'],
        source_by_stage=groups, source_terminal_costs=completion['terminal_costs'],
        cpu_scope='Shared search/neighbors/native/input preparation plus old verifier only; other protocol, truth and audit CPU remain in the source account.')


def four_cell_source_cost(completion, seal, root):
    if completion['status'] != 'complete' or seal['status'] != 'sealed':
        raise ValueError('The new-geometry source must have a completed terminal seal')
    costs = completion['costs']
    old_cells = [row for row in seal['cells'] if row['variant'] == 'original_old']
    if len(old_cells) != completion['registered_geometries'] * 9:
        raise ValueError('Keep every old-terminal control in the cost projection')
    ledger_paths = {Path(row['result_file']).parent / 'attempts.jsonl'
                    for row in old_cells if row.get('result_file')}
    attempts = [dict(json.loads(line), ledger=str(path)) for path in sorted(ledger_paths)
                for line in (root / path).read_text().splitlines() if line.strip()]
    old_cpu = sum(row['cpu_seconds'] for row in attempts)
    shared_cpu = sum(row['cpu_seconds'] for stage, row in costs['by_stage'].items() if stage in SHARED_STAGES)
    terminal = costs['terminal_logical_and_raw_ownership_by_variant']['original_old']['cumulative_new_verification_calls']
    if terminal and not attempts:
        raise ValueError('Original-old CPU must use its own saved attempt ledgers')
    initial, online = costs['shared_initial_physical_calls'], costs['shared_online_physical_calls']
    return dict(name='new_geometries_one_stream', task_streams=completion['registered_geometries'],
        initial_physical_calls=initial, online_physical_calls=online, old_terminal_physical_calls=terminal,
        old_only_physical_calls=initial + online + terminal, shared_search_cpu_seconds=shared_cpu,
        old_terminal_cpu_seconds=old_cpu, old_only_search_terminal_cpu_seconds=shared_cpu + old_cpu,
        source_all_protocols_target_calls=costs['successful_target_physical_calls'],
        source_all_stages_cpu_seconds=costs['attempt_cpu_seconds'],
        source_new_truth_field_calls=costs['analytic_truth_field_calls'],
        source_failed_attempts=costs['failed_attempts'],
        source_execution_failures=[row for row in seal['cells'] if row['status'] != 'complete'],
        source_by_stage=costs['by_stage'], old_terminal_attempt_ledger_files=[str(path) for path in sorted(ledger_paths)],
        source_call_accounting_complete=costs['call_accounting_complete'],
        source_search_failed_invocation_counter=costs['search_failed_invocation_counter'],
        cpu_scope='Shared search/neighbors/native preparation plus actual original_old attempt CPU; four-cell verifier CPU is not divided by four.')


def cost_projection(sources, inventories, plan, selected_n):
    """Scale historical accounts separately; never charge warm-cache misses as cold truth."""
    sources = list(sources)
    fresh = [row for row in sources if row['name'] in ('observation_1', 'observation_2')]
    independent = dict(name='old_geometries_two_independent_streams',
        task_streams=sum(row['task_streams'] for row in fresh),
        old_only_physical_calls=sum(row['old_only_physical_calls'] for row in fresh),
        old_only_search_terminal_cpu_seconds=sum(row['old_only_search_terminal_cpu_seconds'] for row in fresh))
    cost_cohorts = [independent, next(row for row in sources if row['name'] == 'new_geometries_one_stream')]
    if len(inventories) != 2 or any(row['status'] != 'sealed' for row in inventories):
        raise ValueError('Both existing complete-pool inventories must be sealed')
    counts = [{row['seed']: row['distinct_policies'] for row in inventory['tasks']} for inventory in inventories]
    if set(counts[0]) != set(counts[1]):
        raise ValueError('Cold-cache inventory estimates must pair the same geometries')
    fields = [dict(seed=seed, separate_stream_policy_counts=[counts[0][seed], counts[1][seed]],
        maximum_fields_before_cross_stream_deduplication=3 * (counts[0][seed] + counts[1][seed]))
        for seed in counts[0]]
    average_fields = sum(row['maximum_fields_before_cross_stream_deduplication'] for row in fields) / len(fields)
    maximum_fields = max(row['maximum_fields_before_cross_stream_deduplication'] for row in fields)
    table = []
    for n in plan['candidate_geometries']:
        scenarios = [dict(cohort=row['name'],
            projected_old_only_physical_calls=row['old_only_physical_calls'] / row['task_streams'] * (2 * n),
            projected_search_terminal_cpu_seconds=row['old_only_search_terminal_cpu_seconds'] / row['task_streams'] * (2 * n))
            for row in cost_cohorts]
        table.append(dict(geometries=n, observation_streams=2, task_streams=2*n,
            search_histories=14*n, terminal_cells=18*n, initial_physical_calls=20*n,
            adaptive_physical_calls=36*n, reachable_successful_physical_call_cap=6104*n,
            conservative_physical_call_cap_using352=6392*n, expanded_logical_call_cap=6564*n,
            historical_cost_scenarios=scenarios,
            historical_no_cross_stream_dedup_cold_truth_fields=average_fields*n,
            historical_maximum_pair_size_cold_truth_fields=maximum_fields*n))
    return dict(source_accounts=sources, projected_cohorts=cost_cohorts, projections=table,
        selected=next((row for row in table if row['geometries'] == selected_n), None),
        cold_truth=dict(existing_two_stream_geometry_inventory=fields,
            total_separate_stream_field_bound=sum(row['maximum_fields_before_cross_stream_deduplication'] for row in fields),
            fields_per_two_stream_geometry_historical_mean=average_fields,
            fields_per_two_stream_geometry_historical_maximum=maximum_fields,
            cross_stream_physical_union_computed=False, policy_arrays_read=False,
            interpretation='Existing inventories include all saved role pools and both historical terminals. '
                'Their sum is a conservative no-dedup field count for those saved pools, not a hard bound on future pools. '
                'New geometries start without old truth caches;570 historical missing-field calls are not a cold-cache estimate. '
                'Future exact cost is at most3 times the globally sealed distinct physical-policy inventory.'),
        limits=['Success caps do not erase failure or explicit recovery spending; all historical failures stay in source accounts.',
            'Search/old-terminal CPU is an empirical projection, not a wall-time or complete pipeline guarantee. '
            'Full-pool truth/audit CPU, scheduling overhead and uninstrumented failed search invocations are not estimated as zero.',
            'Both historical cost cohorts are retained separately; the two-terminal or four-terminal total is not used as the old-only cost.',
            'No source fit, model load, posterior restore, truth function or candidate scorer runs during preparation.'])


def analyze_costs(plan, precision, root):
    sources = [paired_source_cost(read(root / row['completion']), row['name']) for row in plan['paired_cost_sources']]
    descriptor = plan['new_geometry_cost_source']
    sources.append(four_cell_source_cost(read(root / descriptor['completion']), read(root / descriptor['seal']), root))
    return cost_projection(sources, [read(root / path) for path in plan['independent_inventories']],
                           plan, precision['selected_N'])


def prepare(plan, manifest, checks, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    if (output / 'completion.json').exists():
        return read(output / 'completion.json')
    if not checks or read(root / checks)['status'] != 'passed':
        raise ValueError('Targeted substitute checks must pass before preparation')
    with attempt(output / 'planning', 'plan-frozen-confirmation-precision-cost') as cost:
        cost.update(ZERO, geometry_uniform_draws=0)
        old, newer = read(root / plan['source83_readout']), read(root / plan['source82_readout'])
        precision = analyze_precision(old, newer, plan)
        fees = analyze_costs(plan, precision, root)
        result = dict(status='planning_complete', precision=precision, costs=fees,
            all_nine_controls=dict(old_reference=old['streams'][0]['controls'],
                old_independent=old['independent']['controls'], new_geometry=newer['controls']),
            original_analysis_failures=dict(old=old['analysis_failures'], newer=newer['analysis_failures']),
            historical_accounts=plan['historical_accounts'], new_work=dict(ZERO))
        save(output / 'planning.json', result)
    from performance.prepare_mainline_precision_confirmation import register
    registration = register(plan, precision, checks, root=root)
    attempts = [dict(json.loads(line), ledger=str(path.relative_to(root)))
        for scope in ('planning', 'registration') for path in (output / scope).rglob('attempts.jsonl')
        for line in path.read_text().splitlines() if line.strip()]
    validations = [dict(file=str(path.relative_to(root)), **read(path))
        for path in sorted((output / 'cluster_control/validation').glob('*/checks.json'))]
    completion = dict(status='complete' if precision['selection_pass'] else 'complete_not_registered',
        completed_utc=datetime.now(timezone.utc).isoformat(), selected_N=precision['selected_N'],
        registration=registration, scientific_execution_started=False,
        attempts=attempts, failed_attempts=[row for row in attempts if row['status'] != 'completed'],
        preparation_cpu_seconds=sum(row['cpu_seconds'] for row in attempts),
        geometry_uniform_draws=sum(row.get('geometry_uniform_draws', 0) for row in attempts),
        validations=validations, historical_accounts=plan['historical_accounts'], new_work=dict(ZERO),
        accounting='Preparation attempts counted once; validation/preflight/wrapper and historical costs remain separate. '
            'Registration draws geometry metadata only, never simulator observations or optimizer samples.')
    save(output / 'completion.json', completion)
    readout = dict(result, status=completion['status'], registration=registration,
                   preparation_cpu_seconds=completion['preparation_cpu_seconds'])
    save(output / 'readout.json', readout)
    plan.update(status=completion['status'], completed_utc=completion['completed_utc'],
                selected_N=precision['selected_N'], scientific_execution_started=False)
    save(root / manifest, plan)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare'])
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--checks', required=True)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Run preparation through scheduler on node001-node006')
    result = prepare(read(ROOT / args.manifest), args.manifest, args.checks)
    print(json.dumps({key: result[key] for key in ('status', 'selected_N', 'scientific_execution_started')}))


if __name__ == '__main__':
    main()
