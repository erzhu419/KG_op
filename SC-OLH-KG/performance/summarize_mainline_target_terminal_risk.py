"""Join cached target labels only after all registered terminal reconstructions seal."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import product

import numpy as np
from scipy.stats import norm

from acquisition.decision_backends import feasible_first_terminal_order
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT/'performance/manifests/mainline_target_terminal_risk_reconstruction_v1.json'
FULL, AFTER, POINT = ('legacy_role_full', 'legacy_role_after_init_point_only',
                      'legacy_role_point_only')
ERRORS = ('objective_error', 'centered_constraint_error', 'constraint_noise_error',
          'stochastic_margin_error')


def unit_key(row):
    return row['replicate'], row['seed'], row['arm']


def rank(objective, probability, labels):
    order, mode = feasible_first_terminal_order(objective, probability,
                                               maximum_violation_probability=.5)
    return dict(order=[labels[i] for i in order], primary=labels[order[0]],
                eligible=[labels[i] for i, p in enumerate(probability) if p <= .5], mode=mode)


def precdf_rank(objective, probability, standardized, labels):
    """Describe saturation ties without changing the frozen CDF decision."""
    eligible = np.asarray(probability) <= .5
    if np.any(eligible):
        order = sorted(range(len(labels)), key=lambda i: (not bool(eligible[i]),
            objective[i] if eligible[i] else standardized[i], standardized[i], i))
    else:
        order = sorted(range(len(labels)), key=lambda i: (standardized[i], objective[i], i))
    return [labels[i] for i in order]


def policy_labels(history, reference):
    labels = {tuple(row['point']): f'initial:{i}' for i, row in reversed(list(enumerate(history[:10])))}
    assert len(reference['queries']) == len(history[10:])
    for row, label in zip(history[10:], reference['queries']):
        point = tuple(row['point'])
        if point in labels:
            assert labels[point] == label
        labels[point] = label
    return labels


def reference_summary(cases, arms, registered):
    result = []
    for arm in arms:
        rows = [row for row in cases if row['arm'] == arm]
        known = [row for row in rows if row['baseline_defined']]
        certified = sum(row['certified'] and row['deployed_truth']['true_feasible'] for row in rows)
        result.append(dict(arm=arm, registered_tasks=registered, completed_tasks=len(rows),
            missing_tasks=registered-len(rows), known_quality_baselines=len(known),
            undefined_quality_baselines=len(rows)-len(known),
            certified_true_feasible=certified,
            false_certificates=sum(row['certified'] and not row['deployed_truth']['true_feasible'] for row in rows),
            actual_safe_reward_mean=(sum(row['actual_safe_reward'] for row in rows)/registered
                                     if len(rows) == registered else None),
            completed_case_reward_mean=sum(row['actual_safe_reward'] for row in rows)/len(rows),
            registered_quality_increment_mean=None if len(known) != registered else
                sum(row['best_initial_quality_increment'] for row in known)/registered,
            known_baseline_quality_increment_mean=(sum(row['best_initial_quality_increment'] for row in known)/len(known)
                                                  if known else None),
            physical_verification_calls=sum(row['verification_calls'] for row in rows),
            logical_verification_calls=sum(row['logical_verification_calls'] for row in rows)))
    return result


def prediction_summary(rows):
    known = [row for row in rows if row['cached_truth_available']]
    return dict(observed_policies=len(rows), known_truth_policies=len(known),
        eligible_count=sum(row['eligible'] for row in rows),
        eligible_true_feasible=sum(row['eligible'] and row['truth']['true_feasible'] for row in known),
        eligible_true_infeasible=sum(row['eligible'] and not row['truth']['true_feasible'] for row in known),
        ineligible_true_feasible=sum(not row['eligible'] and row['truth']['true_feasible'] for row in known),
        cdf_zero_count=sum(row['probability_violation'] == 0 for row in rows),
        cdf_one_count=sum(row['probability_violation'] == 1 for row in rows),
        **{name: dict(bias=float(np.mean([row[name] for row in known])),
                     mse=float(np.mean([row[name]**2 for row in known]))) if known else None
           for name in ERRORS})


def paired_channels(left, right, left_history, right_history):
    """Static moment substitutions only for identical ordered raw histories."""
    same_history = left_history == right_history
    same_pool = {tuple(row['point']) for row in left_history} == {tuple(row['point']) for row in right_history}
    result = dict(replicate=left['replicate'], seed=left['seed'], domain=left['domain'],
        left=left['arm'], right=right['arm'], same_ordered_raw_history=same_history,
        same_physical_pool=same_pool, terminal_mechanism_attribution=same_history and same_pool,
        primary_changed=left['original_primary'] != right['original_primary'],
        transformed_primary_changed=left['transformed_primary'] != right['transformed_primary'],
        reported_deployed_policy_changed=left['reference']['deployed'] != right['reference']['deployed'],
        certified_deployment_changed=(left['reference']['deployed'] if left['reference']['certified'] else None) !=
            (right['reference']['deployed'] if right['reference']['certified'] else None),
        risk_channel_substitutions=[])
    if not (same_history and same_pool):
        return result
    assert (left['tau'], left['alpha'], left['guard']) == (right['tau'], right['alpha'], right['guard'])
    labels = [row['policy'] for row in left['observed']]
    by_side = {side: {row['policy']: row for row in value['observed']}
               for side, value in (('L', left), ('R', right))}
    rows = {side: [mapping[label] for label in labels] for side, mapping in by_side.items()}
    for objective_side, channels in product(('L', 'R'), product('LR', repeat=3)):
        mean = np.array([row['constraint_mean']-left['tau']+left['guard'] for row in rows[channels[0]]])
        noise = np.array([row['constraint_aleatoric'] for row in rows[channels[1]]])
        variance = np.array([row['constraint_epistemic'] for row in rows[channels[2]]])
        standardized = (mean+norm.ppf(1-left['alpha'])*np.sqrt(noise))/np.sqrt(np.maximum(variance, 1e-12))
        probability = norm.cdf(standardized)
        objective = [row['objective'] for row in rows[objective_side]]
        choice = rank(objective, probability, labels)
        actual = by_side['L'][choice['primary']]['truth']
        result['risk_channel_substitutions'].append(dict(objective_side=objective_side,
            channels=''.join(channels), **choice, selected_cached_truth=actual,
            certificate_status='unknown_for_altered_ranking'))
    return result


def load_sealed(plan):
    """The only entry to labels: every registered outcome and the global seal exist."""
    output = ROOT/plan['output_dir']
    seal = read(output/'posterior_sealed.json')
    assert seal['status'] == 'sealed' and seal['registered_units'] == plan['registered_units']
    records = {}
    for unit in plan['units']:
        path = output/unit['domain']/f"seed_{unit['seed']}"/unit['arm']/'outcome.json'
        row = read(path)
        assert (row['domain'], row['seed'], row['arm']) == (unit['domain'], unit['seed'], unit['arm'])
        assert row['status'] in ('complete', 'reconstruction_failed', 'registered_prior_search_failure')
        assert row['sealed_utc'] <= seal['sealed_utc']
        if unit['status'] == 'registered_prior_search_failure':
            assert row['status'] == 'registered_prior_search_failure'
        records[unit_key(unit)] = row
    assert len(records) == plan['registered_units'] == len(plan['units'])
    return records, seal


def joined_case(unit, outcome, reference, truth):
    bank = outcome['bank']
    history = [dict(point=bank[i], raw=raw) for i, raw in
               zip(outcome['history_indices'], outcome['history_raw'])]
    assert len(history) == len(outcome['history_raw']) == 13
    labels = policy_labels(history, reference)
    original, transformed, task = outcome['original'], outcome['transformed'], outcome['task']
    moments = outcome['moments']
    bank_index = {tuple(point): i for i, point in enumerate(bank)}
    original_shortlist = {tuple(row['point']): row['shortlist_position']
                          for row in original['terminal']['frozen_shortlist']}
    actual_shortlist = {tuple(row['point']): row['shortlist_position']
                        for row in transformed['frozen_shortlist']}
    shortlist_labels = {row['policy']: row for row in reference['shortlist']}
    guard = task['guard']
    rows = []
    for saved in original['observed']:
        point = tuple(saved['point']); i = bank_index[point]; label = labels[point]
        actual = truth.get(point)
        exposed = shortlist_labels.get(label)
        weights = (moments['policy_weights'][i] if moments['policy_weights'] is not None else moments['weights'])
        row = {k: v for k, v in saved.items() if k != 'point'}
        row.update(policy=label, bank_index=i,
            history_indices=[j for j, point_index in enumerate(outcome['history_indices']) if point_index == i],
            marginal_mean=moments['mean'][i], marginal_variance=moments['variance'][i],
            episode_noise=moments['noise'][i], identity_weights=weights,
            dominant_identity=moments['identity_names'][int(np.argmax(weights))],
            original_shortlist_position=original_shortlist.get(point),
            transformed_shortlist_position=actual_shortlist.get(point),
            safety_attempted=bool(exposed and exposed['safety_attempted']),
            safety_passed=exposed['safety_passed'] if exposed else None,
            actual_certified_deployment=reference['certified'] and reference['deployed'] == label,
            counterfactual_certificate='unknown', cached_truth_available=actual is not None, truth=actual,
            standardized_margin=saved['stochastic_margin_mean']/np.sqrt(max(saved['constraint_epistemic'], 1e-12)),
            objective_error=None if actual is None else saved['objective']-actual['objective'],
            centered_constraint_error=None if actual is None else saved['constraint_mean']-task['tau']-actual['constraint_mean_centered'],
            constraint_noise_error=None if actual is None else saved['constraint_aleatoric']-actual['constraint_sigma']**2,
            stochastic_margin_error=None if actual is None else saved['stochastic_margin_mean']-actual['chance_margin'])
        rows.append(row)
    labels_in_order = [row['policy'] for row in rows]
    objective = [row['objective'] for row in rows]
    probability = [row['probability_violation'] for row in rows]
    original_rank = rank(objective, probability, labels_in_order)
    assert original_rank['order'] == [row['policy'] for row in sorted(rows, key=lambda r: r['posterior_rank'])]
    pre_order = precdf_rank(objective, probability, [row['standardized_margin'] for row in rows], labels_in_order)
    result = dict(**{k: unit[k] for k in ('replicate', 'domain', 'seed', 'arm')},
        status='complete', alpha=task['alpha'], tau=task['tau'], guard=guard,
        paid_observations=13, distinct_observed_policies=len(rows),
        history_policy_labels=[labels[tuple(row['point'])] for row in history],
        original_primary=labels[tuple(original['primary'])],
        transformed_primary=labels[tuple(transformed['primary'])],
        original_shortlist=[labels[tuple(row['point'])] for row in original['terminal']['frozen_shortlist']],
        transformed_shortlist=[labels[tuple(row['point'])] for row in transformed['frozen_shortlist']],
        original_challenger=original['terminal']['objective_challenger'],
        observed_pool_no_eligible=not original_rank['eligible'],
        original_rank=original_rank, pre_cdf_order=pre_order,
        pre_cdf_changes_primary=pre_order[0] != original_rank['primary'],
        pre_cdf_changes_order=pre_order != original_rank['order'],
        parity=outcome['parity'], identity_names=moments['identity_names'],
        covariance_semantics=moments['covariance_semantics'], roundoff_events=moments['roundoff_events'],
        observed=rows, prediction_summary=prediction_summary(rows), reference=reference)
    return result, history, labels


def run(plan):
    output = ROOT/plan['output_dir']
    with attempt(output/'summary', 'join-sealed-terminal-cached-labels') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0, new_truth_function_calls=0,
            posterior_restorations=0, engine_restorations=0, raw_replays=0, new_rng_draws=0, download_bytes=0)
        outcomes, seal = load_sealed(plan)
        label_join_started = datetime.now(timezone.utc).isoformat()
        source = read(ROOT/plan['references']['terminal_diagnosis_file'])
        references = {unit_key(row): row for row in source['cases']}
        assert len(references) == plan['references']['completed_cells']
        truth_cache, histories, joined, labels_by_unit, cases, native = {}, {}, {}, {}, [], []
        for unit in plan['units']:
            key = unit_key(unit); outcome = outcomes[key]
            if outcome['status'] != 'complete':
                cases.append(dict(**{k: unit[k] for k in ('replicate', 'domain', 'seed', 'arm')},
                    status=outcome['status'], reason=outcome.get('reason'), reference=references.get(key)))
                continue
            truth_path = unit['cached_truth_file']
            if truth_path not in truth_cache:
                truth_cache[truth_path] = {tuple(row['point']): row['truth'] for row in read(ROOT/truth_path)}
            case, history, labels = joined_case(unit, outcome, references[key], truth_cache[truth_path])
            cases.append(case); joined[key] = case; histories[key] = history; labels_by_unit[key] = labels
        for unit in plan['units']:
            if unit['arm'] not in (FULL, AFTER):
                continue
            key = unit_key(unit)
            labels = labels_by_unit.get(key)
            if labels is None:
                labels = policy_labels(read(ROOT/unit['search_file'])['history'], references[key])
            saved = read((ROOT/unit['search_file']).with_name('native_terminal.json'))
            native.append(dict(replicate=unit['replicate'], seed=unit['seed'], domain=unit['domain'],
                history_arm=unit['arm'], primary=labels[tuple(saved['primary'])],
                observed=[{k:v for k,v in row.items() if k != 'point'} | dict(policy=labels[tuple(row['point'])])
                          for row in saved['observed']],
                probability_semantics='Saved native expert-mixture probability; no Gaussian reconstruction.'))
        pairs = []
        for replicate, seed in dict.fromkeys((u['replicate'], u['seed']) for u in plan['units']):
            for left, right in ((FULL, AFTER), (AFTER, POINT)):
                a, b = (replicate, seed, left), (replicate, seed, right)
                if a in joined and b in joined:
                    pairs.append(paired_channels(joined[a], joined[b], histories[a], histories[b]))
                else:
                    pairs.append(dict(replicate=replicate, seed=seed, left=left, right=right,
                        terminal_mechanism_attribution=False, status='missing_reconstruction',
                        left_status=outcomes[a]['status'], right_status=outcomes[b]['status']))
        aggregates = []
        for arm in plan['arms']:
            group = [case for case in cases if case['arm'] == arm and case['status'] == 'complete']
            aggregates.append(dict(arm=arm, registered_tasks=plan['registered_geometry_count'],
                reconstructed_tasks=len(group), missing_tasks=plan['registered_geometry_count']-len(group),
                observed_pool_no_eligible=sum(row['observed_pool_no_eligible'] for row in group),
                challenger_subset_no_eligible=sum(row['original_challenger']['eligible_count'] == 0 for row in group),
                precdf_primary_changes=sum(row['pre_cdf_changes_primary'] for row in group),
                precdf_order_changes=sum(row['pre_cdf_changes_order'] for row in group),
                **{name: float(np.mean([row['prediction_summary'][name]['mse'] for row in group
                    if row['prediction_summary'][name] is not None])) if group else None for name in ERRORS}))
        result = dict(status='complete', registered_geometries=plan['registered_geometry_count'],
            registered_units=plan['registered_units'], outcome_counts=dict(Counter(row['status'] for row in cases)),
            posterior_seal=seal, label_join_started_utc=label_join_started,
            cases=cases, arms=aggregates, pairs=pairs, saved_native_observed=native,
            references=dict(registered_cells=plan['references']['registered_cells'], completed_cells=len(references),
                cases=source['cases'], missing_cells=source['missing_cells'],
                arms=reference_summary(source['cases'], [a['name'] for a in plan['reference_terminal_arms']],
                                       plan['registered_geometry_count']),
                undefined_baseline_cells=source['undefined_baseline_cells'], execution_failures=source['execution_failures']),
            channel_order=['centered_risk_mean', 'episode_noise_variance', 'epistemic_variance'],
            limits=plan['limitations']+[
                'Moment swaps are static diagnostics restricted to identical ordered raw histories; no new posterior or certified outcome.',
                'Arm prediction errors average within each reconstructed task before averaging tasks; failed tasks are not zero errors.',
                'No-eligible challenger subsets may exclude the primary; observed-pool eligibility is reported separately.'])
        save(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    run(read(parser.parse_args().manifest))
