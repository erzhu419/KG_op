"""Consolidate Stage61 saved diagnostics; no model, simulator or verifier imports."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'paper_artifacts/mainline_objective_contrast_exposure_v1'
ZERO = ('new_source_calls', 'new_target_initial_calls', 'new_target_search_calls',
        'new_verification_calls', 'new_truth_function_calls', 'new_source_fits',
        'source_model_restorations', 'engine_restorations', 'posterior_restorations',
        'raw_conditioning', 'raw_replays', 'new_rng_draws', 'download_bytes')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def consolidate():
    source, target, terminal = [read(OUTPUT / f'{name}.json') for name in
                               ('source_contrasts', 'target_contrasts', 'comparison_exposure')]
    # Incomplete inputs must keep this stage open; a missing registered failure is an error.
    assert all(x['status'] == 'complete' for x in (source, target, terminal))
    assert source['completed_arm_cases'] == source['registered_arm_cases'] == 96
    assert source['completed_snapshots'] == source['registered_snapshots'] == 192
    assert target['outcome_counts'] == {'complete': 71, 'registered_prior_search_failure': 1}
    assert terminal['registered_cells'] == 216 and terminal['completed_cells'] == 215
    better = [dict(seed=c['seed'], domain=c['domain'], arm=c['arm'],
                   deployed=c['certified_deployment'],
                   terminal_calls=c['logical_verification_calls'],
                   unused_terminal_allowance=c['unused_terminal_allowance'], **r)
              for c in terminal['cases'] for r in c.get('observed', [])
              if r['better_than_certified_deployment']]
    source_table = [dict(arm=r['arm'], pair_group=r['pair_group'],
                        difference_mse=r['metrics']['prediction_difference_mse']['mean'],
                        order_error=r['metrics']['prediction_order_error']['mean'])
                    for r in source['aggregates'] if r['prefix'] == 13
                    and r['partition'] == 'current_prefix' and r['family'] is None
                    and r['subset'] == 'all']
    summary = dict(status='complete',
        coverage=dict(source_cases=24, source_arm_cases=96, source_snapshots=192,
            target_geometries=24, registered_role_units=72, completed_role_units=71,
            observed_policy_arm_records=sum(c.get('observed_policy_count', 0) for c in target['cases']),
            target_pairs=sum(len(c.get('pairs', [])) for c in target['cases']),
            registered_terminal_cells=216, completed_terminal_cells=215),
        source_final13=source_table,
        source_initialization_propagation_scope='AFTER retains joint initial10 information then updates only the observed policy; POINT has no cross-policy updates from source0. AFTER versus POINT does not identify continued online propagation.',
        source_paired_contrasts=[r for r in source['contrasts'] if r['prefix'] == 13
            and r['pair_group'] == 'observed_unobserved' and r['subset'] == 'all'],
        target_by_family=[dict(arm=r['arm'], domain=r['domain'],
            registered_tasks=r['registered_tasks'], completed_tasks=r['completed_tasks'],
            strata={k:r['strata'][k] for k in ('all', 'both_true_feasible', 'both_predicted_eligible')})
            for r in target['by_family']],
        focal_3016=target['focal_3016'],
        focal_3016_existing_comparisons=[dict(seed=c['seed'], arm=c['arm'],
            pair=c['actual_objective_pair'], deployed=c['certified_deployment'],
            stop_reason=c['stop_reason']) for c in terminal['cases'] if c['seed'] == 2026103016],
        better_observed_policy_cells=better,
        better_dispositions=dict(Counter(r['disposition'] for r in better)),
        all9_terminal_arms=terminal['arms'],
        prior_failures=terminal['prior_failures'],
        decision='Keep the deployed model, search, shortlist, risk and precision frozen. Source data support retaining initialization coupling, not globally replacing objective means with raw observations. Register only a zero-call preparation of one same-budget three-candidate comparison candidate; no adoption or new certificate.',
        limitations=source['limits'] + terminal['limits'],
        artifacts={name:str((OUTPUT / f'{name}.json').relative_to(ROOT)) for name in
                   ('source_contrasts', 'target_contrasts', 'comparison_exposure')})
    save(OUTPUT / 'summary.json', summary)


def main():
    start, wall = time.process_time(), time.perf_counter()
    entry = dict(stage='consolidate-saved-diagnostics', started_utc=datetime.now(timezone.utc).isoformat(),
                 status='failed', **dict.fromkeys(ZERO, 0))
    try:
        consolidate()
        entry['status'] = 'completed'
    except Exception as exc:
        entry['error'] = repr(exc)
        raise
    finally:
        entry.update(cpu_seconds=time.process_time()-start, wall_seconds=time.perf_counter()-wall,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        directory = OUTPUT / 'consolidation'
        directory.mkdir(exist_ok=True)
        with (directory / 'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
    attempts = [dict(json.loads(line), ledger=str(p.relative_to(ROOT)))
                for p in sorted(OUTPUT.rglob('attempts.jsonl'))
                for line in p.read_text().splitlines() if line.strip()]
    costs = {k:sum(a.get(k, 0) or 0 for a in attempts) for k in ZERO}
    # Any paid operation invalidates the zero-call completion claim.
    assert not any(costs.values())
    completion = dict(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
        recorded_cpu_seconds=sum(a.get('cpu_seconds', 0) or 0 for a in attempts),
        recorded_wall_seconds=sum(a.get('wall_seconds', 0) or 0 for a in attempts),
        attempt_statuses=dict(Counter(a['status'] for a in attempts)),
        unmetered_attempts=[a for a in attempts if a.get('cpu_seconds') is None],
        new_work=costs, attempts=attempts,
        inherited_accounts=['paper_artifacts/mainline_source_observed_ranking_v1/completion.json',
            'paper_artifacts/mainline_target_terminal_risk_reconstruction_v1/completion.json',
            'paper_artifacts/mainline_unseen_geometry_confirmation_v1/completion.json'],
        accounting_note='Recorded analysis CPU includes both target analysis attempts. Startup failure CPU, shell inspection, documentation and final account serialization are unmetered; historical costs are referenced, not charged again.')
    save(OUTPUT / 'completion.json', completion)
    print(json.dumps({k:completion[k] for k in ('status', 'recorded_cpu_seconds', 'attempt_statuses', 'new_work')}))


if __name__ == '__main__':
    main()
