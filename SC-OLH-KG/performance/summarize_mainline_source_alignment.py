"""Post-seal five-role pool scoring and the registered 11-arm factorial summary.

Reuse the existing analytic truth cache, saved-choice replay and terminal metrics.
Real work runs through scheduleurm; there is no model/prediction restoration here.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import socket
import traceback
from unittest.mock import patch

import numpy as np

from performance import summarize_mainline_precision_confirmation as previous
from performance import mainline_source_alignment_summary as factorial
from performance import mainline_pathway_replication as pathway
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save


MANIFEST = 'performance/manifests/mainline_source_alignment_targets_v1.json'
ROLES = (factorial.AFTER, factorial.POINT, factorial.FULL,
         factorial.PERMUTED_AFTER, factorial.PERMUTED_POINT)
QUERY_CONTRASTS = (*previous.competition.CONTRASTS,
    ('source_AFTER', factorial.AFTER, factorial.PERMUTED_AFTER),
    ('source_POINT', factorial.POINT, factorial.PERMUTED_POINT),
    ('permuted_AP', factorial.PERMUTED_AFTER, factorial.PERMUTED_POINT))
ZERO = dict(previous.ZERO)
_ANALYZE_TASK = previous.competition.analyze_task


def _analyze_five_roles(*args, **kwargs):
    raw = _ANALYZE_TASK(*args, **kwargs)
    captured = raw['coverage']['captured_states']
    registered = len(ROLES) * len(previous.competition.PREFIXES)
    raw['coverage'].update(registered_states=registered, unavailable_states=registered-captured,
                           registered_query_arms=len(ROLES))
    raw['status'] = ('invalid_replay' if raw['coverage']['invalid_replays'] else
        'partial' if raw['coverage']['failures'] or captured < registered else 'complete')
    return raw


@contextmanager
def five_role_context(root):
    """One geometry per process; restore unchanged three-role defaults on exit."""
    with patch.object(previous.audit, 'ROOT', Path(root)), \
            patch.object(previous.audit, 'ROLES', ROLES), \
            patch.object(previous.competition, 'ARMS', ROLES), \
            patch.object(previous.competition, 'CONTRASTS', QUERY_CONTRASTS), \
            patch.object(previous.competition, 'analyze_task', _analyze_five_roles), \
            patch.object(pathway, 'ROLES', ROLES):
        yield


def global_seal(plan, root):
    seal = previous.global_seal(plan, root)
    factorial.require_seal(plan, seal)
    return seal


def require_inventory(plan, root):
    global_seal(plan, root)  # A saved inventory never bypasses the full terminal barrier.
    inventory = previous.require_inventory(plan, root)
    if (inventory.get('query_arms') != list(ROLES)
            or inventory.get('registered_query_states') != len(plan['tasks']) * 2 * 15
            or len(inventory['tasks']) != len(plan['tasks'])
            or {row['seed'] for row in inventory['tasks']} != {row['geometry_task_seed'] for row in plan['tasks']}):
        raise ValueError('The frozen inventory must cover both streams and all five saved role histories')
    return inventory


def freeze(plan, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    seal = global_seal(plan, root)
    target = output / 'mechanism/inventory.json'
    if target.exists():
        return require_inventory(plan, root)
    children = previous.child_plans(plan, root)
    indices = []
    with attempt(output / 'mechanism/inventory', 'freeze-source-alignment-five-role-pools') as cost:
        cost.update(ZERO)
        for task in plan['tasks']:
            seed = task['geometry_task_seed']
            folder = output / 'mechanism' / f'seed_{seed}'
            streams, union = [], {}
            for descriptor, child in children:
                cases = [row for row in seal['cases'] if row['seed'] == seed and row['stream'] == descriptor['name']]
                item = dict(name=descriptor['name'], manifest=descriptor['manifest'], cases=cases)
                try:
                    loaded = previous.load_cases(cases, root)
                    with five_role_context(root):
                        directory, _, initial, _, bundles, _ = previous.audit.task_material(child, task, loaded)
                    points = previous.inventory_points(initial, bundles, loaded)
                    union.update(dict.fromkeys(points))
                    path = folder / descriptor['name'] / 'policies.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, points=np.asarray(points, dtype=np.int16).reshape((-1, child['frozen']['d'])))
                    captured = {arm: len(bundles[arm]['audit']['stages']) for arm in ROLES}
                    complete = all(bundles[arm]['status'] == 'complete' and captured[arm] == 3 for arm in ROLES)
                    item.update(status='frozen', policies_file=str(path.relative_to(root)), distinct_policies=len(points),
                        source_directory=str(directory.relative_to(root)), initial_rows=len(initial),
                        role_status={arm: bundles[arm]['status'] for arm in ROLES}, captured_stages=captured,
                        saved_query_coverage_complete=complete)
                except Exception:
                    item.update(status='failed', error=traceback.format_exc(), distinct_policies=None,
                                saved_query_coverage_complete=False)
                streams.append(item)
            union_path = folder / 'policies.npz'
            union_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(union_path, points=np.asarray(list(union), dtype=np.int16).reshape((-1, plan['frozen']['d'])))
            complete = all(row['status'] == 'frozen' and row['saved_query_coverage_complete'] for row in streams)
            index = dict(seed=seed, domain=task['domain'], streams=streams, query_arms=list(ROLES),
                union_policies_file=str(union_path.relative_to(root)), distinct_policy_task_units=len(union),
                maximum_analytic_function_calls=3 * len(union), inventory_complete=complete)
            path = folder / 'index.json'
            save(path, index)
            indices.append(dict(seed=seed, index_file=str(path.relative_to(root)),
                distinct_policy_task_units=len(union), inventory_complete=complete))
        complete = all(row['inventory_complete'] for row in indices)
        total = sum(row['distinct_policy_task_units'] for row in indices)
        result = dict(status='sealed', registered_geometries=len(plan['tasks']), registered_task_streams=2 * len(plan['tasks']),
            registered_terminal_cells=len(seal['cases']), query_arms=list(ROLES),
            registered_query_states=2 * len(plan['tasks']) * 15,
            truth_access_allowed=True, truth_used=False, tasks=indices, inventory_complete=complete,
            known_distinct_policy_task_units=total, maximum_analytic_function_calls=3 * total if complete else None,
            inventory_failures=[row for row in indices if not row['inventory_complete']],
            inventory_scope='All saved five-role action pools and returned histories, initial10, and all11 old-terminal lists/deployments. '
                'Missing saved states remain missing; their unsaved candidate union is unknown.', **ZERO)
        save(target, result)
    return result


def score_task(plan, seed, *, root=ROOT):
    require_inventory(plan, root)
    with five_role_context(root):
        return previous.score_task(plan, seed, root=root)


def score_worker(args):
    manifest, seed, root = args
    root = Path(root)
    plan = read(root / manifest)
    require_inventory(plan, root)  # Barrier rejection is not a fabricated scientific failure.
    try:
        result = score_task(plan, seed, root=root)
    except Exception:
        task = next(row for row in plan['tasks'] if row['geometry_task_seed'] == seed)
        error = traceback.format_exc()
        result = dict(seed=seed, domain=task['domain'], status='scoring_failed', error=error,
            streams=[previous.failed_stream(plan, task, descriptor, error) for descriptor in plan['streams']],
            unclosed_scoring_cpu_seconds=None)
        save(root / plan['output_dir'] / 'cases' / f'seed_{seed}' / 'result.json', result)
    return dict(seed=seed, status=result['status'])


def score_batch(plan, manifest, start, end, workers, *, root=ROOT):
    require_inventory(plan, root)
    with patch.object(previous, 'score_worker', score_worker):
        return previous.score_batch(plan, manifest, start, end, workers, root=root)


def accounting(plan, tasks, *, root=ROOT, seal=None):
    accounts = previous.accounting(plan, tasks, root=root, seal=seal)
    # The target cohort shares a root with already paid Stage89–91 work.
    # Preserve every ledger, but separate that work from new target execution.
    groups = dict(target=[], retained_source=[], retained_checks=[])
    output = Path(plan['output_dir'])
    for row in accounts['attempts']:
        relative = Path(row['ledger']).relative_to(output)
        scope = ('retained_source' if relative.parts[0] in ('source_models', 'source_preparation')
                 else 'retained_checks' if relative.parts[0] == 'cluster_control' else 'target')
        groups[scope].append(row)
    fields = tuple(accounts['known_recorded'])
    totals = {name: {key: sum(row.get(key) or 0 for row in rows) for key in fields} for name, rows in groups.items()}
    accounts.update(known_recorded=totals['target'], retained_source_known_recorded=totals['retained_source'],
        retained_check_known_recorded=totals['retained_checks'], attempt_scope_counts={name: len(rows) for name, rows in groups.items()},
        accounting='Every physical ledger is retained once. known_recorded covers target execution and postseal analysis only; '
            'source preparation/fits and checks are separate. Historical account links are provenance, not additional charges. '
            'Completed-cell logical budgets are not added to physical calls; failed costs and unknown fees remain explicit.')
    return accounts


def _compact_unit(unit):
    controls = [{key: value for key, value in row.items() if key != 'rows'} for row in unit['controls']]
    comparisons = []
    for row in unit['comparisons']:
        value = {key: entry for key, entry in row.items() if key != 'rows'}
        value['rows'] = [{key: saved[key] for key in ('seed', 'domain', 'delta', 'positive_gain_component', 'avoided_loss_component')}
                         for saved in row['rows']]
        comparisons.append(value)
    return dict(name=unit['name'], registered_geometries=unit['registered_geometries'], controls=controls, comparisons=comparisons)


def collect(plan, manifest, *, root=ROOT):
    root, output = Path(root), Path(root) / plan['output_dir']
    inventory = require_inventory(plan, root)
    seal = global_seal(plan, root)
    target = output / 'completion.json'
    if target.exists():
        return read(target)
    tasks = []
    for task in plan['tasks']:
        path = output / 'cases' / f"seed_{task['geometry_task_seed']}" / 'result.json'
        if not path.is_file():
            raise ValueError('Every registered geometry needs saved scoring or a retained failure')
        tasks.append(read(path))
    source_seal = read(root / plan['source_asset_seal_file'])
    accounts = accounting(plan, tasks, root=root, seal=seal)
    with attempt(output / 'aggregation', 'aggregate-source-alignment-factorial') as cost:
        cost.update(ZERO)
        result = factorial.summarize_loaded(plan, tasks, seal=seal, accounts=accounts,
            accounts_source=plan['output_dir'] + '/completion.json#costs')
        result.update(source_asset_seal=source_seal, source_warning_document=plan['source_warning_document'],
            source_numerical_limits=source_seal['numerical_limits'],
            completed_scoring_geometries=sum(task['status'] == 'complete' for task in tasks),
            completed_scoring_streams=sum(stream['status'] == 'complete' for task in tasks for stream in task['streams']),
            registered_query_states=inventory['registered_query_states'])
        result['limits'].append(source_seal['numerical_limits'])
    accounts = accounting(plan, tasks, root=root, seal=seal)
    accounts.update(registered_pool_inventory_complete=inventory['inventory_complete'],
        known_distinct_policy_task_units=inventory['known_distinct_policy_task_units'],
        maximum_analytic_function_calls=inventory['maximum_analytic_function_calls'])
    result['costs'] = accounts
    save(output / 'summary.json', result)
    compact_accounts = {key: value for key, value in accounts.items() if key not in ('attempts', 'historical_accounts')}
    headline = {key: result[key] for key in ('status', 'registered_geometries', 'registered_task_streams',
        'registered_terminal_cells', 'registered_query_states', 'completed_scoring_geometries', 'completed_scoring_streams',
        'terminal_failures', 'scoring_failures', 'scoring_failed_outcomes', 'source_numerical_limits', 'source_warning_document', 'limits')}
    headline.update(independent=_compact_unit(result['independent']), streams=[_compact_unit(unit) for unit in result['streams']],
        source_asset_seal_file=plan['source_asset_seal_file'], historical_accounts=plan['historical_accounts'], costs=compact_accounts)
    save(output / 'headline.json', headline)
    completion = dict(status=result['status'], completed_utc=datetime.now(timezone.utc).isoformat(),
        registered_geometries=len(tasks), registered_task_streams=2 * len(tasks),
        registered_terminal_cells=result['registered_terminal_cells'], registered_query_states=inventory['registered_query_states'],
        completed_scoring_geometries=result['completed_scoring_geometries'], completed_scoring_streams=result['completed_scoring_streams'],
        scoring_failures=result['scoring_failures'], terminal_failures=result['terminal_failures'], costs=accounts,
        source_asset_seal_file=plan['source_asset_seal_file'], source_warning_document=plan['source_warning_document'],
        source_numerical_limits=source_seal['numerical_limits'], historical_accounts=plan['historical_accounts'],
        summary_file=plan['output_dir'] + '/summary.json', headline_file=plan['output_dir'] + '/headline.json')
    save(target, completion)
    plan.update(status=result['status'], execution_started=True, completed_utc=completion['completed_utc'],
                summary_file=completion['summary_file'], completion_file=plan['output_dir'] + '/completion.json')
    save(root / manifest, plan)
    for descriptor, child in previous.child_plans(plan, root):
        failed = any(row['stream'] == descriptor['name'] for row in result['scoring_failures'] + result['terminal_failures'])
        child.update(status='complete_with_retained_failures' if failed else 'complete', execution_started=True,
            completed_utc=completion['completed_utc'], global_completion_file=plan['output_dir'] + '/completion.json')
        save(root / descriptor['manifest'], child)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'score-batch', 'collect'))
    parser.add_argument('--manifest', default=MANIFEST)
    for key in ('start', 'end', 'workers'):
        parser.add_argument('--' + key, type=int)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Run source-alignment scoring through scheduler on node001-node006')
    plan = read(ROOT / args.manifest)
    if args.stage == 'freeze':
        result = freeze(plan)
    elif args.stage == 'score-batch':
        result = score_batch(plan, args.manifest, args.start, args.end, args.workers)
    else:
        result = collect(plan, args.manifest)
    print(json.dumps({key: result[key] for key in ('status', 'registered_geometries', 'registered_task_streams',
        'registered_terminal_cells', 'registered_query_states') if key in result}), flush=True)


if __name__ == '__main__':
    main()
