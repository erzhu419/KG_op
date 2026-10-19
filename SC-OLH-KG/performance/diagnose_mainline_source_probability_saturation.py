"""Inspect frozen empty-eligible fallbacks without changing the deployed order."""
import argparse
from itertools import combinations

import numpy as np
from scipy.stats import norm

from acquisition.decision_backends import feasible_first_terminal_order
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.summarize_mainline_source_observed_ranking import (
    aggregate, load_frozen_cases, observed_ranking_metrics, scoring_labels,
)

MANIFEST = ROOT/'performance/manifests/mainline_source_risk_channel_attribution_v1.json'
FIELDS = ('no_eligible', 'probabilities_exactly_one', 'minimum_probability_ties',
          'minimum_ties_hide_distinct_margins', 'fallback_choice_changed',
          'empirical_eligibility_change', 'empirical_objective_change')


def diagnose(case, task, arm):
    identity = {key: case[key] for key in ('archive_heldout_family', 'fold', 'family')}
    prediction = case.get('predictions', {}).get(arm, {})
    snapshot = prediction.get('snapshots', {}).get('13', {})
    result = dict(identity, arm=arm, status='failed', metrics=dict.fromkeys(FIELDS))
    if snapshot.get('status') != 'complete' or prediction.get('terminal13') is None:
        return dict(result, failure=prediction.get('failures', case.get('failure')))
    p = np.asarray(snapshot['probability_violation'], dtype=float)[:13]
    objective = np.asarray(snapshot['mean'], dtype=float)[:13, 0]
    order, mode = feasible_first_terminal_order(objective, p, maximum_violation_probability=.5)
    terminal = prediction['terminal13']
    assert order.tolist() == terminal['ordered_indices']
    assert int(order[0]) == terminal['selected_index']
    no_eligible = not np.any(p <= .5)
    ties = np.flatnonzero(p == p.min())
    measured = dict.fromkeys(FIELDS)
    measured.update(no_eligible=int(no_eligible), probabilities_exactly_one=int(np.sum(p == 1)),
                    minimum_probability_ties=len(ties))
    result.update(status='complete', metrics=measured, original_mode=mode,
        original_selected_policy=terminal['selected_policy_id'],
        minimum_probability=float(p.min()), probability=p.tolist(),
        original_minimum_tie_policies=[case['held_policy_ids'][i] for i in ties])
    if arm == 'legacy_native':
        # Saved expert-mixture probability has no single Gaussian pre-CDF margin.
        return dict(result, margin_diagnostic=None)
    mean = np.asarray(snapshot['mean'], dtype=float)[:13, 2]
    noise = np.maximum(np.asarray(snapshot['noise'], dtype=float)[:13, 1], 0.)
    variance = np.maximum(np.asarray(snapshot['variance'], dtype=float)[:13, 2], 0.)
    margin = mean + norm.ppf(1-task['alpha'])*np.sqrt(noise)
    standardized = margin/np.sqrt(np.maximum(variance, 1e-12))
    reconstructed = norm.cdf(standardized)
    np.testing.assert_allclose(reconstructed, p, rtol=0, atol=1e-14)
    measured['minimum_ties_hide_distinct_margins'] = int(np.ptp(standardized[ties]) > 0)
    selected = int(order[0])
    # This is an audit of the strictly monotone CDF's mathematical ordering.
    # It is not passed to the optimizer, verifier or the saved terminal record.
    audit_index = (min(range(13), key=lambda i: (standardized[i], objective[i], i))
                   if no_eligible else selected)
    labels = scoring_labels(task, case['held_policy_ids'])
    original = observed_ranking_metrics(snapshot, terminal, labels, task['alpha'])
    audit_terminal = dict(terminal, selected_index=int(audit_index))
    audited = observed_ranking_metrics(snapshot, audit_terminal, labels, task['alpha'])
    measured.update(fallback_choice_changed=int(audit_index != selected),
        empirical_eligibility_change=audited['selected_empirically_eligible']-original['selected_empirically_eligible'],
        empirical_objective_change=float(labels['mean'][audit_index, 0]-labels['mean'][selected, 0]))
    return dict(result, margin_diagnostic=dict(
        probability_max_error=float(np.max(np.abs(reconstructed-p))),
        margin=margin.tolist(), standardized_margin=standardized.tolist(),
        distinct_pairs_collapsed_by_cdf=sum(p[i] == p[j] and standardized[i] != standardized[j]
                                          for i, j in combinations(range(13), 2)),
        audit_selected_policy=case['held_policy_ids'][audit_index],
        original_empirical_eligible=original['selected_empirically_eligible'],
        audit_empirical_eligible=audited['selected_empirically_eligible'],
        original_empirical_margin=original['selected_empirical_margin'],
        audit_empirical_margin=audited['selected_empirical_margin']))


def run(plan):
    output = ROOT/plan['output_dir']
    with attempt(output/'probability_saturation', 'saved-probability-saturation') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0, new_truth_function_calls=0,
                    posterior_restorations=0, raw_replays=0, new_rng_draws=0, download_bytes=0)
        parent = read(ROOT/plan['parent_manifest'])
        frozen = load_frozen_cases(parent)
        rows = []
        for archive in parent['source_archives']:
            name = archive['heldout_target_family']
            tasks = {t['name']: t for t in read(ROOT/archive['path'])['tasks']}
            for fold in range(parent['folds']):
                record = frozen[name, fold]
                cases = {c['family']: c for c in record['cases']}
                for family in archive['source_families']:
                    case = dict(cases.get(family, dict(predictions={}, failure=record.get('failure'))),
                                archive_heldout_family=name, fold=fold, family=family)
                    rows.extend(diagnose(case, tasks[family], arm) for arm in parent['arms'])
        arms = []
        for arm in parent['arms']:
            selected = [r for r in rows if r['arm'] == arm]
            arms.append(dict(arm=arm, **aggregate(selected, FIELDS),
                by_family=[dict(family=family, **aggregate(
                    [r for r in selected if r['family'] == family], FIELDS))
                    for family in dict.fromkeys(r['family'] for r in selected)]))
        result = dict(status='complete' if all(r['status'] == 'complete' for r in rows)
                      else 'complete_with_missing_forecasts', registered_arm_cases=len(rows),
            rows=rows, arms=arms,
            changed_cases=[r for r in rows if r['metrics']['fallback_choice_changed']],
            interpretation='Pre-CDF audit changes only empty-set fallback ranking, never any frozen prediction or deployment. Native probabilities remain untouched.',
            limits=['Repeated source cases and noisy empirical labels are not independent certificates.',
                    'Collapsed nonminimum probability ties do not establish a changed first choice.',
                    'Source terminal saturation is not automatically present in d1000 deployment.'])
        save(output/'probability_saturation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    run(read(parser.parse_args().manifest))
