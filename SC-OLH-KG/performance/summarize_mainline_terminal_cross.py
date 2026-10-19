"""Score the frozen-history terminal crossover using existing truth caches only."""
from collections import Counter
from datetime import datetime, timezone
import json

import numpy as np

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_terminal_equal import case_dir, cases, require_all

FULL = 'legacy_role_full'
LOCAL = 'legacy_role_after_init_point_only'
NATIVE_SUFFIX = '_native_terminal'
METRICS = {
    'reward': 'actual_safe_reward',
    'quality': 'certified_feasible_quality_gain',
    'certification_reference': 'certified_initial_quality_component',
}


def mean_known(values):
    return None if any(value is None for value in values) else float(np.mean(values))


def contrast(rows, seeds, terms, name):
    differences = []
    for seed in seeds:
        values = {}
        for metric, field in METRICS.items():
            inputs = [rows.get((seed, arm), {}).get(field) for arm in terms]
            values[metric] = (None if any(value is None for value in inputs) else
                             sum(weight * value for weight, value in
                                 zip(terms.values(), inputs, strict=True)))
        differences.append(dict(seed=seed, **values))
    counts = {}
    for metric in METRICS:
        known = [row[metric] for row in differences if row[metric] is not None]
        counts[metric] = dict(wins=sum(value > 1e-12 for value in known),
            ties=sum(abs(value) <= 1e-12 for value in known),
            losses=sum(value < -1e-12 for value in known),
            unknown_tasks=len(seeds) - len(known))
    return dict(name=name, coefficients=terms, task_count=len(seeds),
        mean_difference={metric: mean_known([row[metric] for row in differences])
                         for metric in METRICS},
        win_tie_loss=counts, task_differences=differences)


def summarize(output, plan):
    # A missing verifier must stop scoring before target truth is opened.
    require_all(output, plan, 'verification.json')
    seeds = plan['target_seeds']
    definitions = {arm['name']: arm for arm in plan['arms']}
    rows, source_histories = [], {}
    with attempt(output, 'summarize-terminal-crossover') as cost:
        cost.update(new_source_fits=0, download_bytes=0, new_truth_function_calls=0)
        caches = {seed: {tuple(row['point']): row['truth'] for row in
                  read(ROOT / plan['source_output_dir'] / f'seed_{seed}' / 'truth.json')}
                  for seed in seeds}
        for seed, arm, protocol in cases(plan):
            directory = case_dir(output, seed, arm, protocol)
            item = read(directory / 'input.json')
            saved = read(directory / 'verification.json')
            terminal = saved['verification']
            initial = caches[seed][tuple(item['initial'])]
            truth = caches[seed].get(tuple(saved['deployed']))
            certified = bool(terminal['certified'])
            safe = None if truth is None else bool(truth['true_feasible'])
            if not certified or safe is False:
                reward = quality = reference = 0.
            elif safe is True:
                reward, reference = truth['quality'], initial['quality']
                quality = reward - reference
            else:
                reward = quality = reference = None
            source_histories[item['source_case']] = item['search_evaluation_count']
            rows.append(dict(seed=seed, arm=arm, protocol=protocol,
                history_arm=definitions[arm]['history_arm'],
                terminal_backend=definitions[arm]['terminal_backend'],
                verification_seed=saved['verification_seed'], deployed=saved['deployed'],
                initial_truth=initial, deployed_truth=truth, certified=certified,
                certified_true_feasible=(False if not certified else safe),
                false_certificate=(False if not certified else
                                   None if safe is None else not safe),
                actual_safe_reward=reward, certified_feasible_quality_gain=quality,
                certified_initial_quality_component=reference,
                certified_feasible_improving=(False if not certified or safe is False else
                    None if safe is None else truth['objective'] < initial['objective'] - 1e-12),
                selected_shortlist_rank=terminal['selected_shortlist_rank'],
                branch=terminal['objective_comparison']['status'],
                returned_policy_changed=saved['deployed'] != item['initial'],
                new_verification_calls=saved['new_verification_calls'],
                logical_search_calls=item['search_evaluation_count'],
                logical_total_calls=item['search_evaluation_count'] + saved['new_verification_calls']))

        indexed = {(row['seed'], row['arm']): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError('Crossover summary requires one terminal protocol per task-cell')
        errors = [abs(row['actual_safe_reward'] - row['certified_feasible_quality_gain']
                      - row['certified_initial_quality_component'])
                  for row in rows if row['actual_safe_reward'] is not None]
        if errors and max(errors) > 1e-12:
            raise ValueError('Certification reference and quality gain do not sum to reward')
        groups = []
        for arm, definition in definitions.items():
            group = [indexed[seed, arm] for seed in seeds if (seed, arm) in indexed]
            groups.append(dict(arm=arm, history_arm=definition['history_arm'],
                terminal_backend=definition['terminal_backend'], task_count=len(group),
                expected_task_count=len(seeds), execution_failure_count=len(seeds)-len(group),
                certified=sum(row['certified'] for row in group),
                certified_true_feasible=sum(row['certified_true_feasible'] is True for row in group),
                certified_feasible_improving=sum(row['certified_feasible_improving'] is True for row in group),
                false_certificates=sum(row['false_certificate'] is True for row in group),
                certified_truth_unknown=sum(row['certified'] and row['deployed_truth'] is None for row in group),
                truth_cache_missing=sum(row['deployed_truth'] is None for row in group),
                **{'mean_' + metric: mean_known([
                    indexed.get((seed, arm), {}).get(field) for seed in seeds])
                   for metric, field in METRICS.items()},
                verification_calls=sum(row['new_verification_calls'] for row in group),
                logical_target_calls=sum(row['logical_total_calls'] for row in group),
                branches=dict(Counter(row['branch'] for row in group))))
        paired = [contrast(indexed, seeds, {history + NATIVE_SUFFIX: 1, history: -1},
                           'native_minus_role_terminal_on_' + history)
                  for history in (FULL, LOCAL)]
        paired.extend(contrast(indexed, seeds, {FULL + suffix: 1, LOCAL + suffix: -1},
                               'full_minus_after_init_on_' + terminal)
                      for suffix, terminal in (('', 'role_terminal'), (NATIVE_SUFFIX, 'native_terminal')))
        interaction = contrast(indexed, seeds,
            {FULL + NATIVE_SUFFIX: 1, LOCAL + NATIVE_SUFFIX: -1, FULL: -1, LOCAL: 1},
            '(full-after_init)_native_minus_(full-after_init)_role')
        main = {FULL, LOCAL, FULL + NATIVE_SUFFIX, LOCAL + NATIVE_SUFFIX}
        references = [contrast(indexed, seeds, {history + NATIVE_SUFFIX: 1, reference: -1},
                               history + NATIVE_SUFFIX + '_minus_' + reference)
                      for history in (FULL, LOCAL) for reference in definitions if reference not in main]
        save(output / 'results.json', dict(status='complete', task_count=len(seeds),
            protocol=plan, rows=rows, arms=groups, paired=paired,
            execution_failures=plan.get('execution_failures', []),
            interaction=interaction, reference_comparisons=references,
            improving_deployments=[dict(seed=row['seed'], arm=row['arm'],
                initial_objective=row['initial_truth']['objective'],
                deployed_objective=row['deployed_truth']['objective'],
                quality_gain=row['certified_feasible_quality_gain'])
                for row in rows if row['certified_feasible_improving'] is True],
            decomposition_maximum_error=max(errors, default=0.),
            truth_cache_complete=all(row['deployed_truth'] is not None for row in rows),
            metric_definitions=dict(
                reward='I(certified and true feasible) * deployed quality',
                quality='I(certified and true feasible) * (deployed quality - initial quality)',
                certification_reference='I(certified and true feasible) * initial quality',
                unknown='Uncertified reward and both components are zero; failed execution or certified missing truth remains unknown. Registered-task means remain unknown if any task is missing.'),
            task_weighting='equal; win/tie/loss absolute tolerance 1e-12',
            scope=('Fixed 2x2 terminal intervention on saved full and after-init-point histories, '
                'with five unchanged references. Fresh terminal streams on the same 12 task geometries; '
                'no new search, model fitting or truth evaluation.')))

    attempts = [json.loads(line) for path in output.rglob('attempts.jsonl')
                for line in path.read_text().splitlines()]
    fields = ('cpu_seconds', 'wall_seconds', 'new_source_calls', 'new_target_initial_calls',
              'new_target_search_calls', 'new_verification_calls', 'new_source_fits',
              'download_bytes', 'new_truth_function_calls')
    totals = {key: sum(row.get(key, 0) for row in attempts) for key in fields}
    source_cost = read(ROOT / plan['source_output_dir'] / 'completion.json')['cost']
    save(output / 'completion.json', dict(status='complete', recommendations=len(rows),
        cost=totals, attempt_count=len(attempts),
        failed_attempts=[row for row in attempts if row['status'] == 'failed'],
        new_physical_target_calls=sum(totals[key] for key in
            ('new_target_initial_calls', 'new_target_search_calls', 'new_verification_calls')),
        new_physical_verification_calls=totals['new_verification_calls'],
        successful_verification_calls=sum(row['new_verification_calls'] for row in rows),
        logical_target_calls=sum(row['logical_total_calls'] for row in rows),
        reused_source_history_count=len(source_histories),
        reused_logical_search_calls_unique_histories=sum(source_histories.values()),
        reused_logical_search_calls_expanded_cells=sum(row['logical_search_calls'] for row in rows),
        source_stage_physical_initial_and_search_calls=(source_cost['new_target_initial_calls'] +
                                                       source_cost['new_target_search_calls']),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    return read(output / 'results.json')
