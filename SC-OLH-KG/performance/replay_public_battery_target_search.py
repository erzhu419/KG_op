#!/usr/bin/env python3
"""Fixed-budget retrospective search and finite-window verification replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.terminal_verification import exact_binomial_lower
from performance.inspect_public_battery_announced_population import key, lines, read
from performance.run_public_battery_population_parallel import write_json

PROTOCOL = ROOT / 'performance/manifests/public_battery_target_search_replay_v1_20261002.json'
OUTPUT = ROOT / 'paper_artifacts/public_battery_target_search_replay_v1_20261002'


def roster(protocol):
    return [dict(period=day, algorithm_seed=seed, arm=arm)
            for day in protocol['periods'] for seed in protocol['algorithm_seeds'] for arm in protocol['arms']]


def sample_starts(starts, seed, stream, index, count):
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream, index]))
    return [int(starts[i]) for i in rng.integers(0, len(starts), size=count)]


def shortlist(records, size):
    """Failed costs stay null; no arbitrary penalty enters the ordering."""
    if any((r['window_success'] and r['cost_GBP'] is None) or
           (not r['window_success'] and r['cost_GBP'] is not None) for r in records):
        raise ValueError('observation success/cost support mismatch')
    return sorted(records, key=lambda r: (not r['window_success'],
                  r['cost_GBP'] if r['window_success'] else 0, r['evaluation_index']))[:size]


def evaluate(task, protocol, source, cells):
    ids = source['full_library_profiles']
    initial = source['designs'][task['arm']]
    seed = protocol['seed_base'] + 10000 * protocol['periods'].index(task['period']) + task['algorithm_seed']
    # The same full-library permutation drives both arms, with observed IDs skipped.
    rng = np.random.default_rng(np.random.SeedSequence([seed, protocol['continuation_stream']]))
    remaining = [ids[i] for i in rng.permutation(len(ids)) if ids[i] not in initial]
    chosen = initial + remaining[:protocol['search_budget'] - len(initial)]
    starts, rule = protocol['window_start_minutes'], protocol['dispatch_rule']
    observations = []
    for i, profile in enumerate(chosen):
        minute = sample_starts(starts, seed, protocol['search_stream'], i, 1)[0]
        row = cells[(task['period'], minute, profile, rule)]
        observations.append(dict(profile_id=profile, evaluation_index=i, window_start_minute=minute,
                                 window_success=row['window_success'], cost_GBP=row['cost_GBP']))
    frozen = shortlist(observations, protocol['shortlist_size'])
    attempts, deployed = [], None
    for i, candidate in enumerate(frozen):
        minutes = sample_starts(starts, seed, protocol['verification_stream'], i, protocol['verification_trials'])
        rows = [cells[(task['period'], minute, candidate['profile_id'], rule)] for minute in minutes]
        successes = sum(r['window_success'] for r in rows)
        lower = exact_binomial_lower(successes, len(rows), protocol['familywise_delta'] / protocol['shortlist_size'])
        certified = successes == len(rows) and lower >= 1 - protocol['failure_probability']
        attempts.append(dict(profile_id=candidate['profile_id'], candidate_index=i, sampled_window_starts=minutes,
                             successes=successes, trials=len(rows), exact_lower_bound=lower, certified=certified))
        if certified:
            deployed = candidate['profile_id']
            break
    # Finite-population truth is accessed only after the replay decision is fixed.
    all_rows = [cells[(task['period'], minute, deployed, rule)] for minute in starts] if deployed else []
    probability = sum(r['window_success'] for r in all_rows) / len(all_rows) if all_rows else None
    safe = probability is not None and probability >= 1 - protocol['failure_probability']
    objective = float(np.mean([r['cost_GBP'] for r in all_rows])) if all_rows and all(r['window_success'] for r in all_rows) else None
    return dict(**task, design_seed=seed, initial_profile_ids=initial, observations=observations,
                frozen_shortlist_profile_ids=[r['profile_id'] for r in frozen], verification_attempts=attempts,
                deployed_profile_id=deployed, certified=deployed is not None, certified_finite_population_feasible=safe,
                false_certificate=deployed is not None and not safe, deployment_success_probability=probability,
                deployed_full_population_mean_cost_GBP=objective, represented_search_queries=len(observations),
                represented_verification_queries=sum(r['trials'] for r in attempts),
                new_controller_window_evaluations=0, target_truth_used_for_selection=False,
                independent_confirmation=False, confirmation_year_access=False)


def prepare(protocol, out):
    if (out / 'plan.json').exists():
        raise ValueError('replay already registered; use saved plan')
    source = read(ROOT / protocol['source_designs'])
    if source['status'] != 'source_first_center_designs_frozen' or source['source_selection_uses_target_outcomes']:
        raise ValueError('source-only designs must already be frozen')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out, 'source_designs_snapshot.json', source)
    write_json(out, 'plan.json', dict(protocol_id=protocol['protocol_id'], tasks=roster(protocol),
                                    fixed_at_utc=datetime.now(timezone.utc).isoformat()))


def run(protocol, out, index):
    task = read(out / 'plan.json')['tasks'][index]
    folder = out / 'tasks' / f'{index:03d}'
    if (folder / 'result.json').exists():
        print(json.dumps(dict(index=index, status='saved_replay')), flush=True)
        return
    source = read(out / 'source_designs_snapshot.json')
    saved = lines(ROOT / protocol['population_outputs'] / 'cells.jsonl')
    cells = {key(r): r for r in saved}
    expected = {(day, minute, profile, protocol['dispatch_rule']) for day in protocol['periods']
                for minute in protocol['window_start_minutes'] for profile in source['full_library_profiles']}
    if len(cells) != len(saved) or not expected.issubset(cells):
        raise ValueError('complete saved server population required; local migration snapshot cannot run replay')
    result = evaluate(task, protocol, source, cells)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder, 'result.json', result)
    print(json.dumps(dict(index=index, certified=result['certified'], deployed=result['deployed_profile_id'])), flush=True)


def collect(protocol, out):
    tasks = read(out / 'plan.json')['tasks']
    rows = [read(out / 'tasks' / f'{i:03d}' / 'result.json') for i in range(len(tasks))]
    if [(r['period'], r['algorithm_seed'], r['arm']) for r in rows] != [(r['period'], r['algorithm_seed'], r['arm']) for r in tasks]:
        raise ValueError('complete registered paired replay required')
    groups = []
    for arm in protocol['arms']:
        selected = [r for r in rows if r['arm'] == arm]
        source_once = protocol['source_archive_queries_paid_once'] if arm == protocol['arms'][0] else 0
        search = sum(r['represented_search_queries'] for r in selected)
        verification = sum(r['represented_verification_queries'] for r in selected)
        # Five repeats are alternatives for each of three actual seasonal tasks.
        avg_queries = sum(np.mean([r['represented_search_queries'] + r['represented_verification_queries']
                                   for r in selected if r['period'] == day]) for day in protocol['periods'])
        avg_certificates = sum(np.mean([r['certified_finite_population_feasible'] for r in selected if r['period'] == day])
                               for day in protocol['periods'])
        groups.append(dict(arm=arm, certified_runs=sum(r['certified'] for r in selected), total_runs=len(selected),
                           finite_population_feasible_certificates=sum(r['certified_finite_population_feasible'] for r in selected),
                           false_certificates=sum(r['false_certificate'] for r in selected),
                           represented_search_queries=search, represented_verification_queries=verification,
                           source_archive_queries_paid_once=source_once,
                           expected_queries_for_three_seasonal_tasks=avg_queries,
                           expected_certificates_for_three_seasonal_tasks=avg_certificates,
                           all_in_represented_queries_per_expected_certificate=(source_once+avg_queries)/avg_certificates
                           if avg_certificates else None))
    contrasts = [dict(period=day, source_certified_runs=sum(r['certified'] for r in rows if r['period']==day and r['arm']==protocol['arms'][0]),
                       structural_certified_runs=sum(r['certified'] for r in rows if r['period']==day and r['arm']==protocol['arms'][1]))
                 for day in protocol['periods']]
    result = dict(protocol_id=protocol['protocol_id'], status='paired_target_replay_complete', completed_runs=len(rows),
                  groups=groups, seasonal_contrasts=contrasts, new_controller_window_evaluations=0,
                  represented_target_queries=sum(r['represented_search_queries']+r['represented_verification_queries'] for r in rows),
                  retrospective=True, independent_confirmation=False, confirmation_year_access=False,
                  source_comparison_gate='HOLD', limitations=protocol['limitations'],
                  next_action='retain_all_in_source_cost_and_simple_constant_baseline_before_confirmation_registration')
    write_json(out, 'rows.json', rows)
    write_json(out, 'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('mode', choices=['prepare','run','collect'])
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    protocol = read(args.protocol)
    if args.mode == 'prepare': prepare(protocol,args.output)
    elif args.mode == 'run': run(protocol,args.output,args.index)
    else: collect(protocol,args.output)
