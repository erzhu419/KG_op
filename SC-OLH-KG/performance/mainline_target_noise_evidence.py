"""Join saved Stage41 query noise with Stage42 attribution; never replay raw."""
import numpy as np
from scipy.stats import norm

from performance.run_mainline_legacy_backend_bridge import ROOT, read

FULL = 'legacy_role_full'
LOCAL = 'legacy_role_after_init_point_only'
ARMS = (FULL, LOCAL)


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), maximum=float(values.max()))


def point_evidence(stage, index, bank_ids, neighbor_ids, truth, history, z):
    score = stage['score']
    point = bank_ids[stage['scored_ids'][index]]
    actual = truth.get(point)
    sigma = float(np.sqrt(max(score['constraint_aleatoric'][index], 1e-12)))
    result = dict(full_bank_id=neighbor_ids['full'][point],
        neighbor_index=neighbor_ids['neighbors'].get(point),
        previously_observed=point in history,
        selected=stage['selected_id'] == stage['scored_ids'][index],
        risk_mean=float(score['constraint_mean'][index]),
        risk_epistemic_variance=float(score['constraint_epistemic'][index]),
        risk_noise_variance=float(score['constraint_aleatoric'][index]),
        risk_noise_sigma=sigma, risk_noise_margin=z*sigma,
        risk_draw=float(score['sampled_constraint'][index]-score['constraint_mean'][index]),
        objective_mean=float(score['objective_mean'][index]),
        sampled_margin=float(score['sampled_margin'][index]), truth_available=actual is not None)
    if actual is not None:
        result.update(true_risk_sigma=actual['constraint_sigma'],
            sigma_ratio=sigma/actual['constraint_sigma'],
            noise_margin_bias=z*(sigma-actual['constraint_sigma']),
            risk_mean_bias=result['risk_mean']-actual['constraint_mean_centered'],
            objective_mean_bias=result['objective_mean']-actual['objective'],
            true_feasible=actual['true_feasible'])
    return result


def summarize_rows(rows):
    known = [r for r in rows if r['truth_available']]
    return dict(candidate_count=len(rows), cached_truth_count=len(known),
        missing_truth_count=len(rows)-len(known),
        risk_noise_variance=distribution([r['risk_noise_variance'] for r in rows]),
        risk_noise_sigma=distribution([r['risk_noise_sigma'] for r in rows]),
        risk_epistemic_variance=distribution([r['risk_epistemic_variance'] for r in rows]),
        sigma_ratio=distribution([r['sigma_ratio'] for r in known]),
        absolute_noise_margin_bias=distribution([abs(r['noise_margin_bias']) for r in known]),
        absolute_risk_mean_bias=distribution([abs(r['risk_mean_bias']) for r in known]),
        risk_mean_mse=float(np.mean([r['risk_mean_bias']**2 for r in known])) if known else None,
        objective_mean_mse=float(np.mean([r['objective_mean_bias']**2 for r in known])) if known else None,
        noise_bias_larger_than_mean_bias=sum(abs(r['noise_margin_bias']) > abs(r['risk_mean_bias']) for r in known))


def evidence(plan):
    """Return compact saved-score records and full-bank IDs for source-geometry joins.

    Required plan keys: source_output_dir, target_seeds, attribution_dir.
    """
    source = ROOT/plan['source_output_dir']
    attribution = ROOT/plan['attribution_dir']
    queries, terminal = read(attribution/'queries.json'), read(attribution/'terminal.json')
    original = read(ROOT/'performance/manifests/mainline_terminal_confirmation_search_v1.json')
    z = float(norm.ppf(1-original['task']['alpha']))
    first = {r['seed']: r for r in queries['cases'] if r['first_divergence']}
    focus = {r['seed']: r for r in terminal['full_only_good_policies']}
    states, divergences, focal, groups, terminals = [], [], [], {}, []
    for seed in plan['target_seeds']:
        task = source/f'seed_{seed}'
        truth = {tuple(r['point']): r['truth'] for r in read(task/'truth.json')}
        with np.load(task/FULL/'query_policies.npz') as data:
            full_points = [tuple(p) for p in data['points']]
        full_ids = {p: i for i, p in enumerate(full_points)}
        with np.load(task/'neighbors.npz') as data:
            neighbor_ids = {tuple(p): i for i, p in enumerate(data['points'])}
        labels = dict(full=full_ids, neighbors=neighbor_ids)
        pair_ids = {r['full_bank_id'] for r in first.get(seed, {}).get('selected_pair', [])}
        focal_id = None
        if seed in focus:
            neighbor = int(focus[seed]['policy'].split(':')[1])
            focal_id = full_ids[next(p for p, i in neighbor_ids.items() if i == neighbor)]
        task_groups = dict(all_scored=[], initial=[], shared_neighbors=sorted(
            full_ids[p] for p in neighbor_ids), first_divergence_selected_pair=sorted(pair_ids),
            focal_missed_good=[] if focal_id is None else [focal_id], observed_queries=[])
        group_sets = {k: set(v) for k, v in task_groups.items()}
        task_rows = {}
        for arm in ARMS:
            case = task/arm
            audit, search = read(case/'query_audit.json'), read(case/'search.json')
            with np.load(case/'query_policies.npz') as data:
                points = [tuple(p) for p in data['points']]
            group_sets['initial'].update(full_ids[tuple(r['point'])] for r in search['history'][:10])
            group_sets['observed_queries'].update(full_ids[tuple(r['point'])] for r in search['history'][10:])
            for stage in audit['stages']:
                prefix = stage['paid_prefix']
                if len(stage['scored_ids']) != len(stage['score']['constraint_aleatoric']):
                    raise ValueError('Saved score length differs from scored physical-policy IDs')
                history = {tuple(r['point']) for r in search['history'][:prefix]}
                rows = [point_evidence(stage, i, points, labels, truth, history, z)
                        for i in range(len(stage['scored_ids']))]
                task_rows[arm, prefix] = {r['full_bank_id']: r for r in rows}
                group_sets['all_scored'].update(r['full_bank_id'] for r in rows)
                states.append(dict(seed=seed, arm=arm, paid_prefix=prefix,
                    all_candidates=summarize_rows(rows), unobserved=summarize_rows(
                        [r for r in rows if not r['previously_observed']]),
                    shared_neighbors=summarize_rows([r for r in rows if r['neighbor_index'] is not None])))
                if focal_id is not None:
                    row = task_rows[arm, prefix].get(focal_id)
                    focal.append(dict(seed=seed, arm=arm, paid_prefix=prefix,
                        full_bank_id=focal_id, present=row is not None, evidence=row,
                        scope='Saved prequery state; prefix12 is not the final prefix13 terminal.'))
        if seed in first:
            old = first[seed]
            prefix = old['paid_prefix']
            pair = [dict(full_bank_id=identifier, arms={arm: task_rows[arm, prefix][identifier]
                for arm in ARMS}) for identifier in sorted(pair_ids)]
            divergences.append(dict(seed=seed, paid_prefix=prefix,
                same_paid_history=old['same_paid_history'], same_ordered_pool=old['same_ordered_pool'],
                selected_pair=pair, selected_pair_gap_full_minus_local=old['selected_pair_gap_full_minus_local'],
                preceding_shared_observation=old['preceding_shared_observation']))
        if seed in focus:
            old = focus[seed]
            terminals.append(dict(seed=seed, full_bank_id=focal_id, policy=old['policy'],
                role_final=old['role_final'], native_final=old['native_final'],
                native_primary_failure=old['native_primary_failure'],
                native_objective_difference_to_initial=old['native_objective_difference_to_initial'],
                raw_objective_difference_to_initial=old['raw_objective_difference_to_initial'],
                terminal_exposure=old['terminal_exposure']))
        groups[str(seed)] = {k: sorted(v) for k, v in group_sets.items()}
    direct_swaps = [r for r in queries['first_divergence_substitutions']
        if r['substitution'] in ('risk_risk_mean', 'risk_noise_margin', 'risk_risk_draw',
                                'risk_risk_mean+noise_margin+risk_draw', 'all_objective')]
    return dict(status='complete', task_count=len(plan['target_seeds']),
        query_state_count=len(states), first_divergence_count=len(divergences),
        query_states=states, first_divergences=divergences, focal_query_states=focal,
        terminal_focal_saved=terminals, direct_component_swaps=direct_swaps,
        geometry_groups_by_seed=groups,
        scope='Saved Stage41 JSON/NPZ plus existing Stage42 attribution; no model restore, fitting, raw replay or new truth evaluation.',
        limitations=['Query scores save risk noise expectation and risk-mean epistemic variance, not eta mean/variance or objective noise.',
            'Full/local query states at paid12 can have different adaptive histories.',
            'Terminal role omitted-policy numeric posterior is unavailable; only saved eligibility evidence is used.',
            'Noise calibration uses cached truth where available; uncached candidates remain unknown.'])
