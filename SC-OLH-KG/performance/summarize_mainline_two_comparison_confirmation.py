"""Score fresh paired terminal confirmation only after both protocols are sealed."""
import argparse
from unittest.mock import patch

import numpy as np

from performance.diagnose_mainline_shared_neighbors import truth_value
from performance.prepare_mainline_two_comparison_terminal import ZERO
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_two_comparison_development import build_problem
from performance.summarize_mainline_two_comparison_development import (
    AFTER, METRICS, decompose, intermediate_index, matched_diagnostics, metrics,
    recorded_attempt_costs, summarize_rows,
)

MANIFEST = ROOT / 'performance/manifests/mainline_terminal_two_comparison_confirmation_v1.json'


def arm_names(plan):
    return [row['name'] if isinstance(row, dict) else row for row in plan['terminal_arms']]


def load_sealed(plan):
    """Every old/new outcome, including registered failures, precedes all truth."""
    path = ROOT / plan['output_dir'] / 'execution_status.json'
    if not path.is_file():
        raise RuntimeError('Both registered terminal protocols must be globally sealed before truth scoring')
    seal = read(path)
    if seal['status'] != 'sealed':
        raise RuntimeError('Both registered terminal protocols must be globally sealed before truth scoring')
    expected = {(task['geometry_task_seed'], arm) for task in plan['tasks'] for arm in arm_names(plan)}
    actual = {(row['seed'], row['arm']) for row in seal['cases']}
    assert len(actual) == len(seal['cases']) and actual == expected
    # Validate the entire two-protocol barrier before reading even one result.
    for row in seal['cases']:
        for protocol in ('old', 'new'):
            descriptor = row[protocol]
            assert descriptor['status'] in ('complete', 'failed', 'missing')
            if descriptor['status'] == 'complete':
                for key in ('input_file', 'verification_file'):
                    if not (ROOT / descriptor[key]).is_file():
                        raise RuntimeError(f'Sealed {protocol} terminal file is absent: {descriptor[key]}')
            elif descriptor.get('failure_file') and not (ROOT / descriptor['failure_file']).is_file():
                raise RuntimeError(f'Sealed terminal failure file is absent: {descriptor["failure_file"]}')
    cases = []
    for row in seal['cases']:
        case = dict(row)
        for protocol in ('old', 'new'):
            descriptor = dict(row[protocol])
            descriptor['item'] = (read(ROOT / descriptor['input_file'])
                if descriptor.get('input_file') and (ROOT / descriptor['input_file']).is_file() else None)
            key = 'verification_file' if descriptor['status'] == 'complete' else 'failure_file'
            descriptor['record'] = read(ROOT / descriptor[key]) if descriptor.get(key) else None
            if descriptor['status'] == 'complete':
                assert 'verification' in descriptor['record'] and 'deployed' in descriptor['record']
            case[protocol] = descriptor
        cases.append(case)
    return seal, cases


def policy_labels(initial, neighbors, history, history_arm):
    labels = {}
    for index, row in enumerate(initial):
        labels.setdefault(tuple(row['point']), f'initial:{index}')
    for index, point in enumerate(neighbors):
        labels.setdefault(tuple(map(int, point)), f'neighbor:{index}')
    for index, row in enumerate(history):
        labels.setdefault(tuple(row['point']), f'{history_arm}:history:{index}')
    return labels


def initial_reference(initial, cache):
    values = [cache.get(tuple(row['point'])) for row in initial]
    if len(initial) != 10 or any(value is None for value in values):
        return None, 'initial_reference_unavailable'
    feasible = [value for value in values if value['true_feasible']]
    return (min(feasible, key=lambda value:value['objective']), 'defined') if feasible else (
        None, 'no_true_feasible_initial')


def collect_task_truth(cases, cost):
    """Fresh task-local cache; no archived target labels or model restoration."""
    contexts = {}
    for case in cases:
        key = case['domain'], case['seed']
        context = contexts.setdefault(key, dict(items=[], cases=[]))
        context['cases'].append(case)
        context['items'].extend(case[p]['item'] for p in ('old','new') if case[p]['item'] is not None)
    counts = []
    for (domain, seed), context in contexts.items():
        if not context['items']:
            context.update(cache={}, initial=[], neighbors=[], histories={}, best=None,
                reference_status='initial_reference_unavailable')
            counts.append(dict(domain=domain, seed=seed, status='no_available_frozen_input',
                               new_truth_function_calls=0, cached_policies=0))
            continue
        anchor = context['items'][0]
        directory = (ROOT / anchor['source_case']).parent
        initial = read(directory / 'initialization.json')['history']
        neighbors = np.load(directory / 'neighbors.npz')['points'].tolist()
        cache_path = directory / 'truth.json'
        cache = ({tuple(row['point']):row['truth'] for row in read(cache_path)} if cache_path.is_file() else {})
        points = dict.fromkeys(tuple(row['point']) for row in initial)
        for item in context['items']:
            points.update(dict.fromkeys(tuple(row['point']) for row in item['frozen_shortlist']))
        for case in context['cases']:
            for protocol in ('old','new'):
                if case[protocol]['status'] == 'complete':
                    points[tuple(case[protocol]['record']['deployed'])] = None
        before = cost['new_truth_function_calls']
        problem, extra = None, {}
        try:
            for point in points:
                if point in cache and all(k in cache[point] for k in ('objective','quality','true_feasible')):
                    continue
                if problem is None:
                    problem = build_problem(anchor)
                    cost['truth_problem_constructions'] += 1
                    cost['new_rng_draws'] = None  # Factory internals are not instrumented.
                with patch.object(problem, 'simulate', side_effect=RuntimeError(
                        'Sealed truth scoring cannot generate simulation observations')):
                    truth_value(problem, point, cache, extra, cost)
        finally:
            save(cache_path, [dict(point=list(point), truth=value) for point,value in cache.items()])
        best, status = initial_reference(initial, cache)
        histories = {}
        for item in context['items']:
            path = item['source_case']
            if path not in histories:
                history = read(ROOT / path / 'search.json')['history']
                histories[path] = policy_labels(initial, neighbors, history, item.get('history_arm',item['arm']))
        context.update(cache=cache, initial=initial, neighbors=neighbors, histories=histories,
            best=best, reference_status=status)
        counts.append(dict(domain=domain, seed=seed, status=status, cached_policies=len(cache),
            required_policies=len(points), new_truth_function_calls=cost['new_truth_function_calls']-before,
            truth_cache=str(cache_path.relative_to(ROOT))))
    return contexts, counts


def outcome_metrics(descriptor, context):
    if descriptor['status'] != 'complete':
        return metrics(failed=True)
    saved = descriptor['record']
    return metrics(saved['verification']['certified'], context['cache'].get(tuple(saved['deployed'])), context['best'])


def score_pair(case, context):
    old, new = (outcome_metrics(case[p], context) for p in ('old','new'))
    baseline = metrics(failed=True)
    anchor = case['new']['item'] or case['old']['item']
    labels, points = [], []
    if anchor is not None:
        points = [tuple(row['point']) for row in anchor['frozen_shortlist']]
        mapping = context['histories'][anchor['source_case']]
        labels = [mapping[point] for point in points]
    details = {}
    for protocol in ('old', 'new'):
        descriptor = case[protocol]
        if descriptor['status'] != 'complete':
            details[protocol+'_deployed'] = None
            continue
        saved = descriptor['record']
        point = tuple(saved['deployed'])
        details[protocol+'_reported_deployed'] = labels[points.index(point)]
        details[protocol+'_deployed'] = details[protocol+'_reported_deployed'] if saved['verification']['certified'] else None
        details[protocol+'_selected_truth'] = context['cache'].get(point)
    if case['new']['status'] == 'complete':
        verification = case['new']['record']['verification']
        index, certified, kind = intermediate_index(verification)
        truth = context['cache'].get(points[index]) if index is not None else None
        baseline = metrics(certified, truth, context['best'])
        details.update(B_definition=kind, B_deployed=labels[index] if index is not None else None,
                       B_selected_truth=truth)
    if case['old']['status'] == case['new']['status'] == 'complete':
        details['matched_diagnostics'] = matched_diagnostics(case['old']['item'],case['new']['item'],
            case['old']['record']['verification'],case['new']['record']['verification'],labels)
    return dict(seed=case['seed'], domain=case['domain'], arm=case['arm'],
        history_arm=case.get('history_arm'), old_status=case['old']['status'], new_status=case['new']['status'],
        status='complete' if case['old']['status']==case['new']['status']=='complete' else 'incomplete_pair',
        quality_baseline_defined=context['best'] is not None, quality_reference_status=context['reference_status'],
        best_initial_quality=None if context['best'] is None else context['best']['quality'], shortlist=labels,
        **decompose(old,new,baseline), **details)


def leave_largest_positive(rows, field, metric):
    """Sensitivity only: the primary row set and primary mean are unchanged."""
    known = [(r['seed'],r[field][metric]) for r in rows if r[field][metric] is not None]
    positive = [(seed,value) for seed,value in known if value > 1e-12]
    if not positive:
        return dict(status='no_positive_contribution', excluded_seed=None, available_geometries=len(known))
    seed, value = max(positive, key=lambda pair:(pair[1],-pair[0]))
    remaining = [v for s,v in known if s != seed]
    mean = float(np.mean(remaining)) if remaining else None
    return dict(status='descriptive_sensitivity_only', excluded_seed=seed, excluded_contribution=value,
        registered_geometries=len(rows), available_geometries=len(known), remaining_available=len(remaining),
        all_registered_leave_one_out_mean=mean if len(known)==len(rows) else None,
        conditional_available_leave_one_out_mean=mean,
        conditional_direction=None if mean is None else 'positive' if mean>1e-12 else 'negative' if mean < -1e-12 else 'zero',
        primary_sample_changed=False)


def add_sensitivity(groups, secondary, rows):
    for group in groups:
        selected = [r for r in rows if r['arm']==group['arm']]
        for field in ('new_minus_old','third_challenge','prechange'):
            for metric in METRICS:
                group[field][metric]['leave_largest_positive_geometry_out'] = leave_largest_positive(selected,field,metric)
    for contrast in secondary:
        for field in ('old_contrast','new_contrast','change'):
            for metric in METRICS:
                contrast[field][metric]['leave_largest_positive_geometry_out'] = leave_largest_positive(contrast['rows'],field,metric)


def outcome_record(case, protocol, values):
    descriptor = case[protocol]
    result = dict(seed=case['seed'], domain=case['domain'], arm=case['arm'], protocol=protocol,
        status=descriptor['status'], metrics=values,
        files={key:value for key,value in descriptor.items() if key.endswith('_file')})
    saved = descriptor['record']
    if saved is not None:
        if descriptor['status'] != 'complete':
            result['failure'] = saved
        result['costs'] = ({key:saved[key] for key in ('new_verification_calls','cumulative_new_verification_calls',
            'logical_verification_calls','reused_old_raw_calls','reused_new_raw_calls','failed_simulation_calls',
            'cumulative_new_simulation_invocations','historical_simulation_invocations') if key in saved}
            if descriptor['status'] != 'missing' else dict(protocol_execution_started=False,
                newly_charged_physical_calls=0, inherited_failure_cost_reference=descriptor.get('failure_file')))
        if descriptor['status'] == 'complete':
            result['costs']['logical_terminal_calls'] = saved['verification']['verification_budget']
            result['costs']['logical_total_calls'] = saved['verification']['total_evaluation_count']
    for key in ('reason','failure','search_failure'):
        if key in descriptor:
            result[key] = descriptor[key]
    return result


def score(plan, *, saved_contexts=None, saved_truth_counts=None):
    output = ROOT / plan['output_dir']
    with attempt(output/'summary','summarize-sealed-fresh-two-comparison-confirmation') as cost:
        cost.update(ZERO, truth_problem_constructions=0)
        seal, cases = load_sealed(plan)
        if saved_contexts is None:
            contexts, truth_counts = collect_task_truth(cases,cost)
        else:
            # The full-pool audit has already evaluated its sealed inventory.
            # A read-only summary must preserve any missing truth, not retry it.
            contexts, truth_counts = saved_contexts, saved_truth_counts
        rows, outcomes = [], []
        for case in cases:
            row = score_pair(case,contexts[case['domain'],case['seed']])
            rows.append(row)
            outcomes.extend(outcome_record(case,p,row[p]) for p in ('old','new'))
        groups, secondary = summarize_rows(rows,arm_names(plan))
        add_sensitivity(groups,secondary,rows)
        result = dict(status='complete' if all(r['status']=='complete' for r in outcomes) else 'complete_with_missing_cells',
            scope=plan['scope'], registered_geometry_count=len(plan['tasks']),
            registered_paired_cells=len(rows), registered_outcome_cells=len(outcomes),
            completed_outcome_cells=sum(r['status']=='complete' for r in outcomes),
            completed_old_cells=sum(r['status']=='complete' and r['protocol']=='old' for r in outcomes),
            completed_new_cells=sum(r['status']=='complete' and r['protocol']=='new' for r in outcomes),
            rows=rows, outcomes=outcomes, arms=groups, secondary=secondary,
            primary=next(group for group in groups if group['arm']==AFTER), truth_by_task=truth_counts,
            primary_definition='AFTER new-minus-old true-feasible certification C and signed G beyond best true-feasible initial10.',
            statistical_unit='Fresh geometry, with paired old/new protocols and correlated arms; one frozen observation/verification stream per geometry.',
            metric_definitions=dict(A='Actual certificate',C='I(certified and true feasible)',R='C*Qdeployed',
                G='C*(Qdeployed-Qbest_true_feasible_initial10), signed',P='max(G,0)',L='max(-G,0)'),
            costs=dict(execution_seal=seal.get('costs'), scoring_new_truth_function_calls=cost['new_truth_function_calls'],
                truth_problem_constructions=cost['truth_problem_constructions'], new_model_fits=0,
                scoring_source_model_restorations=0, scoring_engine_restorations=0, scoring_posterior_restorations=0,
                fresh_search_model_loading_count=None,
                model_loading_scope='The reused fresh-search runner does not instrument model loading; scoring restores none.',
                interpretation='Old and new terminal cells share each paid search history. Reused old raw is charged once as physical spending; each protocol retains its own logical consumption.'),
            limits=['Only this fresh registered cohort enters estimates; Stage64 opened geometries are not pooled.',
                'Missing executions and undefined best-feasible-initial baselines remain null; full-cohort estimates are null whenever incomplete, with separately named available-subset summaries.',
                'Largest-positive-geometry removal is a labeled sensitivity; all registered geometries remain in the primary estimate.',
                'The three known benchmark families and one stream per geometry do not establish universal superiority or within-geometry noise stability.',
                'Fixed histories identify terminal conversion, not new search learning; generic terminal gains are not automatically prior-propagation gains.'])
    result['costs']['recorded_stage_attempts'] = recorded_attempt_costs(output)
    save(output/'summary.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default=str(MANIFEST))
    print(score(read(parser.parse_args().manifest))['status'])
