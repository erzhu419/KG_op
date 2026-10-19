"""Score the sealed terminal development cohort using existing truth caches only."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.prepare_mainline_two_comparison_terminal import ZERO

MANIFEST = ROOT / 'performance/manifests/mainline_terminal_two_comparison_development_v1.json'
OLD_RESULTS = 'paper_artifacts/mainline_unseen_geometry_confirmation_v1/results.json'
EXPOSURE = 'paper_artifacts/mainline_objective_contrast_exposure_v1/comparison_exposure.json'
METRICS = ('A', 'C', 'false_certificate', 'R', 'G', 'P', 'L')
FIELDS = ('old', 'new', 'B', 'new_minus_old', 'third_challenge', 'prechange')
AFTER, POINT, FULL = ('legacy_role_after_init_point_only', 'legacy_role_point_only', 'legacy_role_full')


def metrics(certified=None, truth=None, best=None, *, failed=False):
    """Keep a missing baseline unknown even when there is no deployment."""
    if failed:
        return dict.fromkeys(METRICS)
    a = int(certified)
    c = 0 if not a else None if truth is None else int(truth['true_feasible'])
    reward = None if c is None else float(truth['quality']) if c else 0.
    gain = None if c is None or best is None else reward-c*float(best['quality'])
    return dict(A=a, C=c, false_certificate=None if c is None else a-c, R=reward,
                G=gain, P=None if gain is None else max(gain, 0.),
                L=None if gain is None else max(-gain, 0.))


def difference(left, right):
    return {k: None if left[k] is None or right[k] is None else left[k]-right[k]
            for k in METRICS}


def decompose(old, new, baseline):
    result = dict(old=old, new=new, B=baseline, new_minus_old=difference(new, old),
                  third_challenge=difference(new, baseline), prechange=difference(baseline, old))
    for key in METRICS:
        values = [result[field][key] for field in ('new_minus_old', 'third_challenge', 'prechange')]
        if all(v is not None for v in values):
            assert abs(values[0]-values[1]-values[2]) < 1e-12
    return result


def estimate(values):
    known = np.asarray([v for v in values if v is not None], dtype=float)
    n, registered = len(known), len(values)
    center = float(known.mean()) if n else None
    interval = None
    if n > 1:
        width = float(t.ppf(.975, n-1)*known.std(ddof=1)/np.sqrt(n))
        interval = [center-width, center+width]
    positive = known[known > 1e-12]
    negative = known[known < -1e-12]
    return dict(registered_geometries=registered, available_geometries=n,
        all_registered_mean=center if n == registered else None,
        all_registered_descriptive_95pct_t_interval=interval if n == registered else None,
        conditional_available=dict(n=n, mean=center, descriptive_95pct_t_interval=interval,
            degrees_of_freedom=max(n-1, 0), positive=int(len(positive)), negative=int(len(negative)),
            zero=int(n-len(positive)-len(negative)), positive_sum=float(positive.sum()),
            negative_sum=float(negative.sum()),
            largest_positive_share=None if not len(positive) else float(positive.max()/positive.sum())))


def certification_bounds(rows):
    """Binary-support identification bounds, not a CI or zero imputation."""
    low = high = 0.
    for row in rows:
        old, new = row['old']['C'], row['new']['C']
        low += (0 if new is None else new)-(1 if old is None else old)
        high += (1 if new is None else new)-(0 if old is None else old)
    return [low/len(rows), high/len(rows)]


def aggregates(rows, fields=FIELDS):
    result = {field: {metric: estimate([r[field][metric] for r in rows])
                      for metric in METRICS} for field in fields}
    if 'new_minus_old' in fields:
        result['new_minus_old']['C']['all_registered_binary_support_bounds'] = certification_bounds(rows)
    return result


def intermediate_index(verification):
    index = verification['initial_stage_winner_index']
    if index is not None:
        return index, True, 'first_stage_winner'
    rank = verification['selected_shortlist_rank']
    return (rank-1 if rank is not None else None), verification['certified'], 'original_third_only_fallback'


def comparison_record(record, labels):
    if record is None:
        return None
    result = {k: v for k, v in record.items() if 'fingerprint' not in k}
    if 'challenger_index' in record:
        result['pair'] = [labels[record[k]] for k in ('challenger_index', 'incumbent_index')]
        result['winner'] = labels[record['winner_index']]
    return result


def matched_diagnostics(old_input, new_input, old, new, labels):
    assert old_input['verification_seed'] == new_input['verification_seed']
    assert [r['point'] for r in old_input['frozen_shortlist']] == [r['point'] for r in new_input['frozen_shortlist']]
    old_safety = {r['candidate_index']: r for r in old['attempts']}
    safety = []
    for current in new['attempts']:
        index = current['candidate_index']
        previous = old_safety.get(index)
        safety.append(dict(index=index, policy=labels[index],
            old_budget=None if previous is None else previous['verification_budget'],
            new_budget=current['verification_budget'], old_pass=None if previous is None else previous['certified'],
            new_pass=current['certified'], old_margin=None if previous is None else previous['upper_margin'],
            new_margin=current['upper_margin'], matched_stream=previous is not None,
            pass_flip=None if previous is None else int(current['certified'])-int(previous['certified'])))
    previous = old['objective_comparison']
    previous = previous if previous.get('simulation_calls', 0) else None
    current = next((r for r in new['objective_comparisons'] if r['block_index'] == 0), None)
    matched = previous is not None and current is not None
    if matched:
        assert (current['challenger_index'], current['incumbent_index']) == (0, 1)
        for key in ('sample_mean_difference', 'sample_std_difference'):
            assert abs(previous[key]-current[key]) < 1e-12
        assert not (current['challenger_dominates'] and not previous['challenger_dominates'])
    return dict(safety=safety, block0=dict(matched_actual_pair_stream=matched,
        old=comparison_record(previous, labels), new=comparison_record(current, labels),
        stricter_delta_changed_decision=None if not matched else
            previous['challenger_dominates'] != current['challenger_dominates']),
        block1=comparison_record(next((r for r in new['objective_comparisons'] if r['block_index'] == 1), None), labels))


def score_case(item, old_input, old_saved, new_saved, old_reference, exposure, truth):
    labels = exposure['shortlist']
    points = [tuple(r['point']) for r in item['frozen_shortlist']]
    best = old_reference['best_feasible_initial_truth']
    old_v = old_saved['verification']
    old_point = tuple(old_saved['deployed'])
    recomputed_old = metrics(old_v['certified'], truth.get(old_point), best)
    # A join/scaling mismatch must not silently change the established old control.
    aliases = dict(A='actual_certificate', C='certification', false_certificate='false_certificate',
                   R='reward', G='quality_increment', P='new_policy_gain', L='below_best_regression')
    old_metrics = {key: old_reference[alias] for key, alias in aliases.items()}
    for key, alias in aliases.items():
        expected, actual = old_reference[alias], recomputed_old[key]
        if expected is not None and actual is not None:
            assert abs(expected-actual) < 1e-12
    base = dict(seed=item['seed'], arm=item['arm'], domain=item['domain'],
        quality_baseline_defined=best is not None, best_initial_quality=None if best is None else best['quality'],
        shortlist=labels, old_deployed=exposure['certified_deployment'],
        old_reported_deployed=exposure['reported_deployed'])
    if new_saved is None:
        return dict(base, status='execution_failure', new_deployed=None, B_deployed=None,
                    **decompose(old_metrics, metrics(failed=True), metrics(failed=True)))
    new_v = new_saved['verification']
    new_point = tuple(new_saved['deployed'])
    assert new_point in points
    new_metrics = metrics(new_v['certified'], truth.get(new_point), best)
    index, certified, kind = intermediate_index(new_v)
    b_truth = truth.get(points[index]) if index is not None else None
    b_metrics = metrics(certified, b_truth, best)
    return dict(base, status='complete', new_deployed=labels[points.index(new_point)] if new_v['certified'] else None,
        new_reported_deployed=labels[points.index(new_point)], B_deployed=labels[index] if index is not None else None,
        B_definition=kind, new_selected_truth=truth.get(new_point), B_selected_truth=b_truth,
        cached_label_missing=bool(new_v['certified'] and truth.get(new_point) is None),
        **decompose(old_metrics, new_metrics, b_metrics),
        matched_diagnostics=matched_diagnostics(old_input, item, old_v, new_v, labels))


def summarize_rows(rows, arms):
    groups = []
    for arm in arms:
        group = [r for r in rows if r['arm'] == arm]
        groups.append(dict(arm=arm, registered=len(group), complete=sum(r['status']=='complete' for r in group),
            **aggregates(group), by_family=[dict(domain=domain,
                **aggregates([r for r in group if r['domain']==domain]))
                for domain in dict.fromkeys(r['domain'] for r in group)]))
    indexed = {(r['seed'], r['arm']): r for r in rows}
    secondary = []
    for left, right in ((AFTER, POINT), (FULL, AFTER)):
        if left not in arms or right not in arms:
            continue
        pairs = []
        for seed in sorted({r['seed'] for r in rows}):
            a, b = indexed[seed, left], indexed[seed, right]
            pairs.append(dict(seed=seed, domain=a['domain'], old_contrast=difference(a['old'], b['old']),
                new_contrast=difference(a['new'], b['new']),
                change=difference(a['new_minus_old'], b['new_minus_old'])))
        fields = ('old_contrast', 'new_contrast', 'change')
        secondary.append(dict(left=left, right=right, rows=pairs, **aggregates(pairs, fields),
            by_family=[dict(domain=domain, **aggregates([r for r in pairs if r['domain']==domain], fields))
                       for domain in dict.fromkeys(r['domain'] for r in pairs)]))
    return groups, secondary


def sealed_execution(plan, inventory):
    """This barrier runs before opening old scores or any cached truth labels."""
    output = ROOT / plan['output_dir']
    path = output / 'execution_status.json'
    if not path.is_file():
        raise RuntimeError('All candidate verifications/failures must be sealed before scoring')
    seal = read(path)
    expected = {(r['seed'], r['arm']) for r in inventory['rows']}
    actual = {(r['seed'], r['arm']) for r in seal['cases']}
    assert seal['status'] == 'sealed'
    assert len(seal['cases']) == len(actual) and actual == expected
    assert seal['available_cases'] == len(expected) == inventory['available_cells']
    assert seal['registered_cases'] == inventory['registered_cells']
    result = {}
    for row in seal['cases']:
        assert row['status'] in ('complete', 'failed')
        name = 'verification_file' if row['status'] == 'complete' else 'failure_file'
        file = ROOT / row[name]
        if not file.is_file():
            raise RuntimeError(f'Sealed terminal record is absent: {file}')
        record = read(file)
        assert record['status'] == row['status']
        result[row['seed'], row['arm']] = (record, file)
    return seal, result


def load_cached_truth(item):
    path = (ROOT / item['source_case']).parent / 'truth.json'
    if not path.is_file():
        return {}
    return {tuple(row['point']): row['truth'] for row in read(path)}


def case_costs(inventory_row, record, file):
    ledger = file.parent / 'attempts.jsonl'
    attempts = [json.loads(line) for line in ledger.read_text().splitlines() if line] if ledger.is_file() else []
    old = inventory_row['old_raw_rows']
    physical = record['cumulative_new_verification_calls']
    logical = record['logical_verification_calls'] if record['status'] == 'complete' else None
    return dict(old_physical_verification_calls=old, new_unique_physical_verification_calls=physical,
        historical_old_plus_new_physical_calls=old+physical,
        failed_simulation_invocations=record['failed_simulation_calls'],
        new_simulation_invocations=record['cumulative_new_simulation_invocations'],
        historical_simulation_invocations=record['historical_simulation_invocations'],
        randomized_new_observations=physical,
        exact_new_rng_draws=None if physical or record['failed_simulation_calls'] else 0,
        old_logical_verification_calls=old, new_logical_verification_calls=logical,
        new_logical_total_calls=None if logical is None else logical+inventory_row['search_evaluation_count'],
        unchanged_logical_search_calls=inventory_row['search_evaluation_count'],
        old_raw_reused_last_attempt=record['reused_old_raw_calls'],
        new_stage_raw_replayed_last_attempt=record['reused_new_raw_calls'],
        logical_raw_consumption_last_attempt=record['logical_verification_calls'],
        attempts=attempts,
        failed_attempts=[r for r in attempts if r['status'] != 'completed'])


def recorded_attempt_costs(output):
    """Read each ledger once; row-level references do not multiply spending."""
    stages, files = {}, []
    for path in sorted(output.rglob('attempts.jsonl')):
        files.append(str(path.relative_to(ROOT)))
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            group = stages.setdefault(row['stage'], dict(attempts=0, failed_attempts=0,
                cpu_seconds=0., wall_seconds=0., new_verification_calls=0))
            group['attempts'] += 1
            group['failed_attempts'] += row['status'] != 'completed'
            for key in ('cpu_seconds', 'wall_seconds', 'new_verification_calls'):
                group[key] += row.get(key, 0)
    return dict(ledger_files=files, by_stage=stages,
        cpu_seconds=sum(r['cpu_seconds'] for r in stages.values()),
        wall_seconds_sum_of_attempts=sum(r['wall_seconds'] for r in stages.values()),
        new_verification_calls=sum(r['new_verification_calls'] for r in stages.values()))


def score(plan):
    output = ROOT / plan['output_dir']
    with attempt(output / 'summary', 'summarize-sealed-two-comparison-development') as cost:
        cost.update(ZERO)
        inventory = read(ROOT / plan['block_inventory'])
        seal, execution = sealed_execution(plan, inventory)
        # No labels above this line. The source/task/model is never reconstructed.
        old_results = read(ROOT / OLD_RESULTS)
        old_rows = {(r['seed'], r['arm']): r for r in old_results['rows']}
        exposure = {(r['seed'], r['arm']): r for r in read(ROOT / EXPOSURE)['cases']}
        rows, truth_cache = [], {}
        for registered in inventory['rows']:
            key = registered['seed'], registered['arm']
            item, old_input = read(ROOT/registered['candidate_input']), read(ROOT/registered['original_input'])
            record, file = execution[key]
            task = item['domain'], item['seed']
            if task not in truth_cache:
                truth_cache[task] = load_cached_truth(item)
            row = score_case(item, old_input, read(ROOT/registered['old_verification']),
                record if record['status']=='complete' else None, old_rows[key], exposure[key], truth_cache[task])
            row['costs'] = case_costs(registered, record, file)
            row['files'] = dict(candidate_input=registered['candidate_input'],
                original_verification=registered['old_verification'], new_terminal_record=str(file.relative_to(ROOT)))
            if record['status'] == 'failed':
                row['execution_failure'] = record
            rows.append(row)
        for missing in inventory['missing_cells']:
            empty = metrics(failed=True)
            rows.append(dict(seed=missing['seed'], domain=missing['domain'], arm=missing['arm'],
                status='inherited_search_failure', quality_baseline_defined=old_rows[missing['seed'],missing['arm']]['best_feasible_initial_truth'] is not None,
                inherited_failure=next(r for r in plan['inherited_failures'] if
                    (r['seed'],r['arm'])==(missing['seed'],missing['arm'])),
                **decompose(empty, empty, empty)))
        assert len(rows) == inventory['registered_cells']
        assert len({(r['seed'],r['arm']) for r in rows}) == len(rows)
        arms, secondary = summarize_rows(rows, plan['terminal_arms'])
        cost_keys = ('old_physical_verification_calls', 'new_unique_physical_verification_calls',
            'historical_old_plus_new_physical_calls', 'old_raw_reused_last_attempt',
            'new_stage_raw_replayed_last_attempt', 'logical_raw_consumption_last_attempt',
            'failed_simulation_invocations', 'new_simulation_invocations',
            'historical_simulation_invocations', 'randomized_new_observations')
        totals = {k:sum(r['costs'][k] for r in rows if 'costs' in r) for k in cost_keys}
        for group in arms:
            group['costs'] = {k:sum(r['costs'][k] for r in rows if r['arm']==group['arm'] and 'costs' in r)
                              for k in cost_keys}
        result = dict(status='complete' if all(r['status']=='complete' for r in rows) else 'complete_with_missing_cells',
            scope=plan['scope'], registered_cells=len(rows), available_candidate_cells=len(execution),
            completed_candidate_cells=sum(r['status']=='complete' for r in rows),
            new_failed_cells=sum(r['status']=='execution_failure' for r in rows),
            inherited_missing_cells=sum(r['status']=='inherited_search_failure' for r in rows),
            statistical_unit='24 opened geometries; nine related arms are not independent replicates.',
            rows=rows, arms=arms, secondary=secondary,
            primary_candidate=plan['primary_analysis']['candidate_arm'],
            focal_cases=[r for r in rows if r['seed'] in (2026103007,2026103016,2026103021)],
            costs=dict(totals=totals, execution_seal=seal.get('costs'), inherited_accounts=plan['inherited_accounts'],
                exact_new_rng_draws=None if totals['new_simulation_invocations'] else 0,
                interpretation='Physical new calls count unique stage observations. Last-attempt reuse is not historical spending; all failed attempts remain attached. Shared historical searches are charged once in the inherited accounts.'),
            metric_definitions=plan['primary_analysis']['metrics'],
            decomposition='new-old=(new-B)+(B-old); B preserves the original third-only fallback.',
            limits=['Opened-cohort development, not an independent confirmation or automatic adoption.',
                'Registered means and intervals stay null whenever any registered metric is missing; conditional summaries name their available denominator.',
                'The safety-count and stricter-delta residual is joint; matching diagnostics are not additive causal interventions.',
                'The two protocols have the same maximum budget, not necessarily the same actual logical spending.',
                'Missing cached truth labels stay unknown. No truth function, fit, model restore or posterior replay is used.',
                'Terminal improvements are not new search learning; compare both propagation controls and all references.'])
    result['costs']['recorded_stage_attempts'] = recorded_attempt_costs(output)
    result['costs']['inherited_account_records'] = [dict(file=path, completion=read(ROOT/path))
        for path in plan['inherited_accounts']]
    save(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    print(score(read(parser.parse_args().manifest))['status'])
