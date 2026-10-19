"""Score registered geometries only after every noise replicate finishes verification."""
from shutil import copyfile

import numpy as np
from scipy.stats import t

from performance.run_mainline_legacy_backend_bridge import ROOT, read, save
from performance.run_mainline_terminal_equal import require_all
from performance.summarize_mainline_terminal_confirmation import summarize as summarize_family
from performance.summarize_mainline_terminal_cross import mean_known


FULL = 'legacy_role_full'
AFTER = 'legacy_role_after_init_point_only'
POINT = 'legacy_role_point_only'
PROTOCOL = 'same_policy_distinct_challenger'
METRICS = ('certification', 'reward', 'best_initial_reference', 'quality_increment',
           'new_policy_gain', 'below_best_regression', 'dynamic_new_gain', 'shared_new_gain',
           'dynamic_good_new_deployment',
           'good_new_deployment', 'actual_certificate', 'false_certificate',
           'physical_verification_calls', 'logical_total_calls')
SCOPE = ('Fixed Stage50 task geometries with new observation and verification noise. '
         'Sources, model, initial physical policies, shared neighbors and terminal rule stay frozen.')


def estimates(rows, tasks):
    domains = list(dict.fromkeys(task['domain'] for task in tasks))
    means = {key: mean_known([row[key] for row in rows]) for key in METRICS}
    by_family = [dict(domain=domain, means={key: mean_known([
        row[key] for row in rows if row['domain'] == domain]) for key in METRICS})
        for domain in domains]
    return dict(expected_geometry_count=len(tasks),
        completed_geometry_count=sum(row['reward'] is not None for row in rows),
        available_geometry_count_by_metric={key: sum(row[key] is not None for row in rows)
                                            for key in METRICS},
        task_equal_means=means, by_family=by_family,
        family_equal_means={key: mean_known([row['means'][key] for row in by_family])
                            for key in METRICS})


def comparisons(rows, tasks, replicates, arms, interval_scope=None):
    indexed = {(row['replicate'], row['seed'], row['arm']): row for row in rows}
    pairs = [(AFTER, POINT), (FULL, AFTER)]
    pairs.extend((AFTER, arm) for arm in arms if arm not in (FULL, AFTER, POINT))
    results = []
    for left, right in pairs:
        per_replicate = []
        for replicate in replicates:
            effects = []
            for task in tasks:
                seed = task['geometry_task_seed']
                a, b = (indexed[replicate, seed, arm] for arm in (left, right))
                effects.append(dict(seed=seed, domain=task['domain'], **{
                    key: None if a[key] is None or b[key] is None else a[key]-b[key]
                    for key in METRICS}))
            per_replicate.append(dict(replicate=replicate, rows=effects,
                                     **estimates(effects, tasks)))
        geometry = [dict(seed=task['geometry_task_seed'], domain=task['domain'], **{
            key: mean_known([rep['rows'][index][key] for rep in per_replicate])
            for key in METRICS}) for index, task in enumerate(tasks)]
        intervals = {}
        for key in METRICS:
            values = [row[key] for row in geometry]
            if any(value is None for value in values):
                intervals[key] = None
            else:
                half_width = float(t.ppf(.975, len(tasks)-1) *
                                   np.std(values, ddof=1) / np.sqrt(len(tasks)))
                center = float(np.mean(values))
                intervals[key] = [center-half_width, center+half_width]
        jointly_certified = [dict(replicate=replicate,
            completed_pairs=sum(indexed[replicate, task['geometry_task_seed'], left]['certification'] == 1
                and indexed[replicate, task['geometry_task_seed'], right]['certification'] == 1
                for task in tasks),
            quality_differences=[dict(seed=task['geometry_task_seed'], domain=task['domain'],
                difference=indexed[replicate, task['geometry_task_seed'], left]['quality_increment']-
                           indexed[replicate, task['geometry_task_seed'], right]['quality_increment'])
                for task in tasks if
                indexed[replicate, task['geometry_task_seed'], left]['certification'] == 1 and
                indexed[replicate, task['geometry_task_seed'], right]['certification'] == 1 and
                indexed[replicate, task['geometry_task_seed'], left]['quality_increment'] is not None and
                indexed[replicate, task['geometry_task_seed'], right]['quality_increment'] is not None],
            quality_reference_missing_seeds=[task['geometry_task_seed'] for task in tasks if
                indexed[replicate, task['geometry_task_seed'], left]['certification'] == 1 and
                indexed[replicate, task['geometry_task_seed'], right]['certification'] == 1 and
                (indexed[replicate, task['geometry_task_seed'], left]['quality_increment'] is None or
                 indexed[replicate, task['geometry_task_seed'], right]['quality_increment'] is None)])
            for replicate in replicates]
        results.append(dict(left=left, right=right, by_replicate=per_replicate,
            geometry_mean_effects=geometry, **estimates(geometry, tasks),
            descriptive_95pct_t_interval=intervals, degrees_of_freedom=len(tasks)-1,
            interval_scope=interval_scope or (
                f'Paired geometry-level intervals over {len(tasks)} registered geometries and '
                f'{len(replicates)} observation stream(s) per geometry.'),
            jointly_certified_pairs_conditional=jointly_certified))
    return results


def scored_rows(replicate, family, source, terminal, result, tasks):
    indexed = {(row['seed'], row['arm']): row for row in result['rows']}
    failures = {(row['seed'], row['arm']) for row in terminal.get('execution_failures', [])}
    rows = []
    for task in tasks:
        seed = task['geometry_task_seed']
        directory = ROOT / source['output_dir'] / f'seed_{seed}'
        initial = {tuple(row['point']) for row in read(directory / 'initialization.json')['history']}
        neighbors = {tuple(point) for point in np.load(directory / 'neighbors.npz')['points']}
        cache = {tuple(row['point']): row['truth'] for row in read(directory / 'truth.json')}
        best = min((cache[point] for point in initial if cache[point]['true_feasible']),
                   key=lambda truth: truth['objective'], default=None)
        for definition in terminal['arms']:
            arm = definition['name']
            row = indexed.get((seed, arm))
            record = dict(replicate=replicate, seed=seed, domain=family['domain'], arm=arm,
                history_arm=definition['history_arm'], best_feasible_initial_truth=best,
                quality_reference_status='defined' if best is not None else 'no_true_feasible_initial')
            if row is None:
                assert (seed, arm) in failures, 'Missing score without a registered execution failure'
                rows.append(dict(record, status='execution_failure', **dict.fromkeys(METRICS),
                                 history_physical_calls=None))
                continue
            c = row['certified_true_feasible']
            truth = row['deployed_truth']
            gain = (None if c is None or best is None else truth['quality']-best['quality'] if c else 0.)
            reference = None if c is None or best is None else float(c)*best['quality']
            positive = None if gain is None else max(gain, 0.)
            regression = None if gain is None else max(-gain, 0.)
            deployed = tuple(row['deployed'])
            dynamic = deployed not in initial and deployed not in neighbors
            if gain is not None:
                assert abs(row['actual_safe_reward']-reference-gain) < 1e-12
                assert positive <= 1e-12 or deployed not in initial
            search = read(directory / definition['history_arm'] / 'search.json')
            rows.append(dict(record, status='complete', deployed=row['deployed'],
                certification=None if c is None else int(c),
                reward=row['actual_safe_reward'], best_initial_reference=reference,
                quality_increment=gain,
                new_policy_gain=positive, below_best_regression=regression,
                dynamic_new_gain=None if positive is None else positive if dynamic else 0.,
                shared_new_gain=None if positive is None else 0. if dynamic else positive,
                dynamic_good_new_deployment=(None if gain is None else int(c and dynamic and gain > 1e-12)),
                good_new_deployment=(None if gain is None else int(c and
                    deployed not in initial and gain > 1e-12)),
                actual_certificate=int(row['certified']),
                false_certificate=None if row['false_certificate'] is None else int(row['false_certificate']),
                physical_verification_calls=row['new_verification_calls'],
                logical_total_calls=row['logical_total_calls'],
                history_physical_calls=search['new_initial_calls']+search['new_search_calls'],
                legacy_first_initial_quality_increment=row['certified_feasible_quality_gain']))
    return rows


def score(plan, prepared):
    """Enforce the global verifier barrier before reusing or collecting task-owned truth."""
    registered_replicates = [row['replicate'] for row in plan['replicates']]
    assert [row['replicate'] for row in prepared['replicates']] == registered_replicates
    arms = [row['name'] for row in plan['terminal_arms']]
    tasks = plan['tasks']
    scope = plan.get('scope', SCOPE if all('initial_policy_file' in task for task in tasks) else
        'Registered fresh geometries with frozen sources, model, frontend and terminal rules.')
    families = []
    for replicate in prepared['replicates']:
        for family in replicate['families']:
            source = read(ROOT / family['search_manifest'])
            terminal = read(ROOT / family['terminal_manifest'])
            assert terminal['protocols'] == {PROTOCOL: [80, 128, 128]}
            assert [row['name'] for row in terminal['arms']] == arms
            expected = [task['geometry_task_seed'] for task in tasks if task['domain'] == family['domain']]
            assert source['target_seeds'] == terminal['target_seeds'] == expected
            families.append((replicate, family, source, terminal))
    assert len(families) == len(registered_replicates) * len({task['domain'] for task in tasks})
    # No cached truth, truth collection or family scoring may precede this global barrier.
    for _, _, _, terminal in families:
        require_all(ROOT / terminal['output_dir'], terminal, 'verification.json')
    for _, family, source, _ in families:
        for task in tasks:
            if task['domain'] != family['domain']:
                continue
            destination = ROOT / source['output_dir'] / f"seed_{task['geometry_task_seed']}" / 'truth.json'
            if 'initial_policy_file' in task and not destination.exists():
                copyfile((ROOT / task['initial_policy_file']).with_name('truth.json'), destination)
    rows, costs = [], []
    for replicate, family, source, terminal in families:
        output = ROOT / terminal['output_dir']
        result = summarize_family(output, terminal)
        result.update(scope=scope, statistical_unit=(
            f"{len(source['target_seeds'])} registered geometries in this observation replicate; "
            'paired arms are not independent statistical units.'))
        save(output / 'results.json', result)
        selected = [task for task in tasks if task['domain'] == family['domain']]
        rows.extend(scored_rows(replicate['replicate'], family, source, terminal, result, selected))
        costs.append(dict(replicate=replicate['replicate'], domain=family['domain'],
            search=read(ROOT / source['output_dir'] / 'completion.json')['cost'],
            terminal=read(output / 'completion.json')['cost']))
    per_replicate = []
    for replicate in prepared['replicates']:
        selected = [row for row in rows if row['replicate'] == replicate['replicate']]
        per_replicate.append(dict(replicate=replicate['replicate'], rows=selected,
            arms=[dict(arm=arm, **estimates([row for row in selected if row['arm'] == arm], tasks))
                  for arm in arms], costs=[row for row in costs if row['replicate'] == replicate['replicate']]))
        save(ROOT / replicate['output_dir'] / 'results.json', per_replicate[-1])
    contrasts = comparisons(rows, tasks, registered_replicates, arms, plan.get('interval_scope'))
    result = dict(status='complete_with_execution_failures' if any(row['status'] != 'complete'
                    for row in rows) else 'complete', scope=scope, rows=rows,
        statistical_unit=plan.get('statistical_unit',
            f'{len(tasks)} registered geometries; {len(registered_replicates)} observation stream(s) per geometry.'),
        registered_cells=len(tasks)*len(arms)*len(registered_replicates),
        completed_cells=sum(row['status'] == 'complete' for row in rows),
        by_replicate=per_replicate, primary=contrasts[0], comparisons=contrasts,
        costs=costs, cost_scope=('Search ledgers count each shared history once; terminal ledgers '
            'count each terminal cell. Row history_physical_calls is shared by its crossed cells '
            'and must not be summed over all terminal arms. Failed-attempt costs stay in ledgers.'),
        metric_definitions=dict(certification='C=I(certified and true feasible)',
            quality_increment='Signed C*(Qdeployed-Qbest_true_feasible_initial10), without clipping',
            new_policy_gain='P=max(G,0); positive gain must deploy a policy outside initial10',
            below_best_regression='L=max(-G,0); signed G=P-L remains the primary quality metric',
            dynamic_new_gain='P for deployed policies outside initial10 and the physical shared-neighbor bank',
            shared_new_gain='P for deployed policies in the physical shared-neighbor bank',
            dynamic_good_new_deployment='Certified true-feasible dynamic deployment with G>1e-12',
            reward='C*Qdeployed = C*Qbest_true_feasible_initial10 + quality_increment',
            missing='Failed cells stay null. If initial10 has no true-feasible policy, its best-initial '
                'reference and all dependent quality metrics stay null even when C=0; certification, '
                'reward and costs remain observed. Registered-cohort means and intervals stay null '
                'for each incomplete metric.'),
        weighting=plan.get('weighting', 'Average all registered observation streams within each geometry '
            'before geometry-equal and family-equal means.'))
    save(ROOT / plan['output_dir'] / 'results.json', result)
    return result
