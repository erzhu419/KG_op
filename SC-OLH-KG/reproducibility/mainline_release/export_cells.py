"""Export only saved cell metrics and attempt ledgers after both cohort seals.

No raw trajectory, source model, posterior, checkpoint or truth cache is opened.
Invoke once on the existing server through its scheduler, not by local science.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .statistics import METRICS, validate, read_json, write_json

COHORTS = (
    ('stage85', 'performance/manifests/mainline_precision_confirmation_v1.json'),
    ('stage93', 'performance/manifests/mainline_source_alignment_targets_v1.json'),
)
COST_FIELDS = ('known_recorded', 'physical_target_calls', 'known_recorded_physical_target_calls',
    'physical_accounting_scope', 'retained_source_known_recorded', 'retained_check_known_recorded',
    'unknown_target_fee_records', 'historical_accounts', 'source_model_restorations', 'engine_restorations',
    'posterior_restorations', 'failed_simulator_invocations', 'simulation_rng_draws', 'accounting', 'instrumentation')
ATTEMPT_FIELDS = ('stage', 'family', 'arm', 'seed', 'stream', 'status', 'cpu_seconds', 'wall_seconds',
    'new_source_calls', 'new_source_fits', 'new_target_initial_calls', 'new_target_search_calls',
    'new_verification_calls', 'new_truth_function_calls', 'error', 'reused', 'started_utc', 'finished_utc')


def _sealed(plan, seal, completion):
    expected = {(stream['name'], task['geometry_task_seed'], arm['name'])
        for stream in plan['streams'] for task in plan['tasks'] for arm in plan['terminal_arms']}
    actual = [(row['stream'], row['seed'], row['arm']) for row in seal['cases']]
    if (seal.get('status') != 'sealed' or not seal.get('truth_access_allowed')
            or completion.get('status') not in ('complete', 'complete_with_retained_failures')
            or len(actual) != len(expected) or set(actual) != expected):
        raise ValueError('Both complete registered cohort grids must already be sealed and scored')
    return {(row['stream'], row['seed'], row['arm']): row for row in seal['cases']}


def export(root, descriptors=COHORTS):
    root = Path(root)
    inputs = []
    for name, manifest in descriptors:
        plan = read_json(root / manifest)
        output = Path(plan['output_dir'])
        seal, completion = (read_json(root / output / filename) for filename in ('execution_status.json', 'completion.json'))
        indexed = _sealed(plan, seal, completion)
        inputs.append((name, manifest, plan, output, seal, completion, indexed))
    # Open scored metrics only after every whole-cohort barrier has passed.
    cells, cohorts, attempts = [], [], []
    for name, manifest, plan, output, seal, completion, indexed in inputs:
        science = read_json(root / output / 'summary.json')
        scored = {}
        for unit in science['streams']:
            for control in unit['controls']:
                for row in control['rows']:
                    key = unit['name'], row['seed'], control['arm']
                    if key in scored:
                        raise ValueError('Duplicate saved metric cell')
                    scored[key] = row['values']
        if set(scored) != set(indexed):
            raise ValueError('Saved metric cells do not match the sealed grid')
        for task in plan['tasks']:
            seed = task['geometry_task_seed']
            result_file = output / 'cases' / f'seed_{seed}' / 'result.json'
            saved_task = read_json(root / result_file)
            outcomes = {(stream['name'], row['arm']): row for stream in saved_task['streams'] for row in stream['outcomes']}
            for stream in plan['streams']:
                for arm in plan['terminal_arms']:
                    key = stream['name'], seed, arm['name']
                    sealed, outcome, values = indexed[key], outcomes[stream['name'], arm['name']], scored[key]
                    if outcome['metrics'] != values:
                        raise ValueError('Per-geometry metrics differ from the saved summary')
                    old = sealed['old']
                    quality = outcome.get('best_initial_quality')
                    baseline = ('defined' if quality is not None else 'undefined_initial_baseline'
                        if outcome['status'] == 'complete' else 'unavailable_execution_or_scoring')
                    verification_calls = outcome.get('verification_budget')
                    if old['status'] == 'missing':
                        verification_calls = 0  # No frozen terminal input: verifier never invoked.
                    elif verification_calls is None and old['status'] == 'failed':
                        verification_calls = old.get('failure', {}).get('new_verification_calls')
                    cells.append(dict(cohort=name, seed=seed, domain=task['domain'], stream=stream['name'],
                        arm=arm['name'], history_arm=arm['history_arm'], protocol='old', status=outcome['status'],
                        terminal_status=old['status'], metrics=deepcopy(values), baseline_status=baseline,
                        best_initial_quality=quality, selected_truth=outcome.get('selected_truth'),
                        reported_policy_id=outcome.get('reported_policy_id'), deployed_policy_id=outcome.get('deployed_policy_id'),
                        verification_calls=verification_calls,
                        references=dict(scored_result=str(result_file), summary=str(output / 'summary.json'),
                            terminal_seal=str(output / 'execution_status.json'), source_case=sealed.get('source_case'),
                            **{key: old.get(key) for key in ('input_file', 'verification_file', 'failure_file')})))
        accounts = completion['costs']
        counters = Counter()
        for row in accounts.get('attempts', []):
            ledger = row['ledger']
            counters[ledger] += 1
            relative = Path(ledger).relative_to(output)
            scope = ('retained_source' if relative.parts[0] in ('source_models', 'source_preparation')
                else 'retained_checks' if relative.parts[0] == 'cluster_control' else 'target')
            attempts.append(dict(cohort=name, ledger=ledger, ledger_row=counters[ledger], accounting_scope=scope,
                **{field: deepcopy(row[field]) for field in ATTEMPT_FIELDS if field in row}))
        cohorts.append(dict(name=name, streams=[row['name'] for row in plan['streams']],
            arms=[row['name'] for row in plan['terminal_arms']],
            geometries=[dict(seed=row['geometry_task_seed'], domain=row['domain']) for row in plan['tasks']],
            sources=dict(manifest=manifest, completion=str(output / 'completion.json'), seal=str(output / 'execution_status.json'),
                summary=str(output / 'summary.json')),
            saved_cost_accounts={field: deepcopy(accounts[field]) for field in COST_FIELDS if field in accounts}))
    result = dict(format='mainline_release_cells_v1', cohorts=cohorts, cells=cells, attempts=attempts,
        provenance='Saved post-seal metric values and uniquely owned historical ledger rows; no rescoring or imputation.',
        new_scientific_work=dict(simulation_calls=0, source_fits=0, posterior_restorations=0, rng_draws=0))
    validate(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = export(args.root)
    write_json(args.output, result)
    print(dict(status='complete', cells=len(result['cells']), attempts=len(result['attempts']),
        terminal_counts=dict(Counter(row['terminal_status'] for row in result['cells']))))


if __name__ == '__main__':
    main()
