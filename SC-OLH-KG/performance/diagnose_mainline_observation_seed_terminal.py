"""Reuse saved terminal funnels across all registered observation streams."""
import argparse
from collections import Counter

from performance.diagnose_mainline_three_family_terminal import analyze, FIELDS
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.summarize_mainline_observation_seed_replication import AFTER, FULL, PROTOCOL

MANIFEST = ROOT / 'performance/manifests/mainline_observation_seed_attribution_v1.json'
COUNTS = ('tasks', 'registered_tasks', 'missing_terminal_tasks', 'known_baseline_tasks',
          'baseline_undefined_tasks')


def known_sum(values):
    return None if any(value is None for value in values) else sum(values)


def summarize_arms(replicates):
    combined = []
    for arm in replicates[0]['arms']:
        rows = [next(row for row in rep['arms'] if row['arm'] == arm['arm']) for rep in replicates]
        record = dict(arm=arm['arm'], **{key: sum(row[key] for row in rows) for key in COUNTS},
            funnel={key: known_sum([row['funnel'][key] for row in rows]) for key in FIELDS},
            known_baseline_funnel={key: sum(row['known_baseline_funnel'][key] for row in rows)
                                  for key in FIELDS})
        for key in ('origins', 'dispositions', 'better_than_actual_deployment_by_disposition'):
            counts = Counter()
            for row in rows:
                counts.update(row[key])
            record[key] = dict(counts)
        for key in ('unselected_not_better_than_actual_deployment', 'no_certified_deployment'):
            record[key] = sum(row[key] for row in rows)
        record['by_family'] = []
        for family in arm['by_family']:
            selected = [next(f for f in row['by_family'] if f['domain'] == family['domain']) for row in rows]
            record['by_family'].append(dict(domain=family['domain'],
                **{key: sum(row[key] for row in selected) for key in COUNTS},
                **{key: known_sum([row[key] for row in selected]) for key in FIELDS},
                known_baseline_funnel={key: sum(row['known_baseline_funnel'][key] for row in selected)
                                      for key in FIELDS}))
        combined.append(record)
    return combined


def diagnose(plan):
    prepared = read(ROOT / plan['prepared_file'])
    source = read(ROOT / plan['source_output_dir'] / 'results.json')
    scored = {(row['replicate'], row['seed'], row['arm']): row for row in source['rows']}
    cases, replicas, missing_scores = [], [], []
    for replicate in prepared['replicates']:
        identifier = replicate['replicate']
        family_rows = {}
        failures = []
        for family in replicate['families']:
            terminal = read(ROOT / family['terminal_manifest'])
            results = read(ROOT / terminal['output_dir'] / 'results.json')
            family_rows.update({(family['domain'], row['seed'], row['arm']): row
                                for row in results['rows']})
            failures.extend(terminal.get('execution_failures', []))
        parent = dict(replicate, target_seeds=plan['target_seeds'], execution_failures=failures)
        result = analyze(parent, terminal_protocol=PROTOCOL, result_rows=family_rows)
        for case in result['cases']:
            row = scored[identifier, case['seed'], case['arm']]
            assert row['status'] == 'complete'
            assert case['funnel']['deployed'] == row['good_new_deployment']
            assert case['best_feasible_initial_truth'] == row['best_feasible_initial_truth']
            case.update(replicate=identifier, status='complete', terminal_variant=PROTOCOL,
                case_id=f"replicate_{identifier}/seed_{case['seed']}/{case['arm']}",
                best_initial_quality_increment=row['quality_increment'])
            namespace = f"replicate_{identifier}/seed_{case['seed']}/"
            for policy in case['good_new_policies'] or []:
                policy['policy_id'] = namespace + policy['policy']
            for entry in case['shortlist']:
                entry['policy_id'] = namespace + entry['policy']
            assert len({entry['policy'] for entry in case['shortlist']}) == len(case['shortlist'])
            if case['baseline_defined']:
                assert all(case['funnel'][left] >= case['funnel'][right]
                           for left, right in zip(FIELDS, FIELDS[1:]))
        cases.extend(result['cases'])
        missing_scores.extend(dict(row, replicate=identifier) for row in result['missing_final_scores'])
        replicas.append(dict(replicate=identifier, terminal_cells=result['terminal_cells'],
                             arms=result['arms'], execution_failures=failures))
    index = {(row['replicate'], row['seed'], row['arm']): row for row in cases}
    crosses = []
    for replica in plan['replicates']:
        for seed in plan['target_seeds']:
            for history in (FULL, AFTER):
                role = index.get((replica, seed, history))
                native = index.get((replica, seed, history + '_native_terminal'))
                if role is None or native is None:
                    continue
                assert role['history_arm'] == native['history_arm'] and role['queries'] == native['queries']
                crosses.append(dict(replicate=replica, seed=seed, domain=role['domain'], history_arm=history,
                    queries=role['queries'], role_deployed=role['deployed'], native_deployed=native['deployed'],
                    role_certified=role['certified'], native_certified=native['certified'],
                    role_minus_native_reward=role['actual_safe_reward']-native['actual_safe_reward'],
                    role_minus_native_quality_increment=(None if not role['baseline_defined'] else
                        role['best_initial_quality_increment']-native['best_initial_quality_increment'])))
    assert len(cases) == source['completed_cells']
    return dict(status=source['status'], registered_terminal_cells=source['registered_cells'],
        terminal_cells=len(cases), failed_terminal_cells=source['registered_cells']-len(cases),
        cases=cases, arms=summarize_arms(replicas), by_replicate=replicas, native_crosses=crosses,
        execution_failures=plan['source_execution_failures'],
        missing_cells=[row for row in source['rows'] if row['status'] != 'complete'],
        undefined_baseline_cells=[{key: row[key] for key in ('replicate', 'seed', 'domain', 'arm',
            'baseline_unavailable_reason')} for row in cases if not row['baseline_defined']],
        missing_final_scores=missing_scores, definition=result['definition'],
        scope=plan.get('scope', 'Saved JSON/NPZ and cached truth only. Replicate-qualified policy IDs; counts are unique within each task-noise-arm history.'),
        limitations=result['limitations'] + [
            'Native crosses share query histories; slot, order, exposure, budget and verification noise can differ together.',
            'Policy absence from a shortlist is not itself deployment loss; compare with actual certified deployed quality.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    plan = read(ROOT / args.manifest)
    output = ROOT / plan['output_dir']
    with attempt(output, 'saved-observation-terminal-funnel') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0, new_truth_function_calls=0,
                    posterior_restorations=0, download_bytes=0)
        result = diagnose(plan)
        save(output / 'terminal.json', result)
    print({row['arm']: row['funnel'] for row in result['arms']}, flush=True)


if __name__ == '__main__':
    main()
