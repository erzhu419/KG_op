"""Compare frozen bounded-encoder forecasts with the fixed Stage43 source baseline."""
import argparse
from copy import deepcopy

import numpy as np
from scipy.special import digamma

from baselines.transfer_archive import FrozenTransferArchive
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.summarize_mainline_role_source_holdout import (
    ARMS, aggregate_rows, score_case, summarize as score_frozen_forecasts,
)


MANIFEST = ROOT/'performance/manifests/mainline_bounded_mean_source_v1.json'
HEADS = ('f', 'g', 'eta_g', 'f_difference')


def ordinary_case(case):
    """Remove boundary policy zero only from the unobserved scoring pool."""
    held = np.asarray(case['held_policy_ids'])
    keep = np.r_[0, 1, np.flatnonzero(held[2:] != 0)+2]
    result = dict(case, held_policy_ids=held[keep].tolist())
    result['predictions'] = {
        arm: {key:np.asarray(value)[:, keep].tolist() for key, value in pred.items()}
        for arm, pred in case['predictions'].items()}
    return result


def contrast(before, after):
    result = {}
    for head in HEADS:
        result[head] = {
            'before_mse':before[head]['mse'], 'after_mse':after[head]['mse'],
            'mse_percent_change':100*(after[head]['mse']/before[head]['mse']-1)
                if before[head]['mse'] != 0 else None,
            'delta':{key:after[head][key]-before[head][key] for key in before[head]},
        }
    return result


def aggregate_contrasts(baseline, candidate):
    index = {(r['family'], r['arm'], r['round']):r for r in baseline}
    return [dict(family=r['family'], arm=r['arm'], round=r['round'],
                 **contrast(index[r['family'], r['arm'], r['round']], r))
            for r in candidate]


def propagation_contrasts(aggregate):
    index = {(r['family'], r['arm'], r['round']):r for r in aggregate}
    families = list(dict.fromkeys(r['family'] for r in aggregate))
    result = []
    for family in families:
        for name, arm, t in [('initial', 'point_from_source0', 1),
                             ('late', 'point_after_first', 2)]:
            result.append(dict(family=family, propagation=name, round=t,
                control=arm, **contrast(index[family, arm, t], index[family, 'full', t])))
    return result


def signature(case, arm, t):
    mean = np.asarray(case['predictions'][arm]['mean'])[t]
    return (np.sign(mean[2:, 0]-mean[0, 0]),
            int(case['held_policy_ids'][2+int(np.argmin(mean[2:, 0]))]))


def choice_changes(baseline, candidate, families):
    rows = []
    for key in baseline:
        fold, family = key
        old, new = baseline[key], candidate[key]
        assert old['held_policy_ids'] == new['held_policy_ids']
        comparisons = []
        for version, case in [('baseline', old), ('candidate', new)]:
            for name, control, t in [('initial', 'point_from_source0', 1),
                                      ('late', 'point_after_first', 2)]:
                comparisons.append((version+'_'+name,
                    signature(case, control, t), signature(case, 'full', t)))
        for arm in ARMS:
            for t in range(3):
                comparisons.append((f'candidate_minus_baseline:{arm}:round{t}',
                                     signature(old, arm, t), signature(new, arm, t)))
        for name, before, after in comparisons:
            rows.append(dict(fold=fold, family=family, comparison=name,
                direction_changes=int(np.sum(before[0] != after[0])),
                scored_policy_count=len(before[0]), argmin_before=before[1],
                argmin_after=after[1], argmin_changed=bool(before[1] != after[1])))
    aggregate = []
    names = list(dict.fromkeys(r['comparison'] for r in rows))
    for family in [*families, 'both_families_equal']:
        for name in names:
            selected = [r for r in rows if r['comparison'] == name and
                        (family == 'both_families_equal' or r['family'] == family)]
            aggregate.append(dict(family=family, comparison=name,
                direction_changes=sum(r['direction_changes'] for r in selected),
                scored_policy_count=sum(r['scored_policy_count'] for r in selected),
                argmin_changes=sum(r['argmin_changed'] for r in selected),
                scoring_pool_count=len(selected)))
    return dict(cases=rows, aggregate=aggregate)


def boundary_rows(cases, tasks):
    rows = []
    for (fold, family), case in cases.items():
        held = case['held_policy_ids']
        if 0 not in held[2:]:
            continue
        k, task = held.index(0), tasks[family]
        raw = np.asarray(task.Y_replicates[0]).copy()
        raw[:, 1] -= task.tau
        observed, variance = raw.mean(axis=0), raw.var(axis=0, ddof=1)
        eta_label = np.log(variance)-digamma(1.)
        forecasts = []
        for arm, t in [('full', 0), *((arm, 2) for arm in ARMS)]:
            prediction = case['predictions'][arm]
            mean = np.asarray(prediction['mean'])[t, k]
            latent = np.asarray(prediction['variance'])[t, k]
            noise = np.asarray(prediction['noise'])[t, k]
            total = latent[[0, 2]]+noise/3
            error = mean[[0, 2]]-observed
            forecasts.append(dict(arm=arm, round=t, mean=mean, variance=latent,
                noise=noise, episode_variance_ratio=noise/variance,
                eta_bias=mean[[1, 3]]-eta_label,
                standardized_mean_squared_error=error**2/total,
                dawid_sebastiani_score=np.log(total)+error**2/total))
        rows.append(dict(fold=fold, family=family, policy_id=0,
            raw_f_g_centered=raw, observed_mean=observed,
            observed_episode_variance=variance, observed_unbiased_eta=eta_label,
            forecasts=forecasts))
    return rows


def summarize(plan):
    output = ROOT/plan['output_dir']
    # The caller must freeze all four candidate folds before any held labels load.
    forecast_paths = [output/f'fold_{i}'/'forecasts.json' for i in range(plan['folds'])]
    missing = [str(path) for path in forecast_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Candidate forecast barrier incomplete: '+', '.join(missing))
    candidate_summary = score_frozen_forecasts(plan)
    parent = read(ROOT/plan['parent_manifest'])
    baseline_summary = read(ROOT/parent['output_dir']/'summary.json')
    with attempt(output, 'compare-frozen-bounded-source-forecasts') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0,
                    new_truth_function_calls=0, download_bytes=0)
        raw_cases = {}
        for version, directory in [('baseline', ROOT/parent['output_dir']), ('candidate', output)]:
            raw_cases[version] = {
                (fold['fold'], case['family']):case
                for i in range(plan['folds'])
                for fold in [read(directory/f'fold_{i}'/'forecasts.json')]
                for case in fold['cases']}
        assert raw_cases['baseline'].keys() == raw_cases['candidate'].keys()
        for key, case in raw_cases['baseline'].items():
            other = raw_cases['candidate'][key]
            assert case['held_policy_ids'] == other['held_policy_ids']
            assert case['d'] == other['d'] and case['tau'] == other['tau']
            assert case['adaptation_raw'] == other['adaptation_raw']
        archive = FrozenTransferArchive.load(ROOT/plan['archive'])
        tasks = {task.name:task for task in archive.tasks}
        families = list(tasks)
        views = {}
        for view, drop_boundary in [('all112', False), ('ordinary110', True)]:
            cases = {version:{key:ordinary_case(case) if drop_boundary else case
                              for key, case in values.items()}
                     for version, values in raw_cases.items()}
            aggregates = {}
            for version in cases:
                if not drop_boundary:
                    aggregates[version] = (baseline_summary if version == 'baseline'
                                            else candidate_summary)['aggregate']
                else:
                    scored = [score_case(case, tasks[family], fold)
                              for (fold, family), case in cases[version].items()]
                    aggregates[version] = aggregate_rows(scored, families)
            counts = {version:sum(len(case['held_policy_ids'])-2 for case in values.values())
                      for version, values in cases.items()}
            assert counts['baseline'] == counts['candidate'] == (110 if drop_boundary else 112)
            views[view] = dict(scored_policy_family_pairs=counts['candidate'],
                aggregate=aggregates,
                candidate_minus_baseline=aggregate_contrasts(aggregates['baseline'], aggregates['candidate']),
                propagation={version:propagation_contrasts(values)
                             for version, values in aggregates.items()},
                f_prediction_changes=choice_changes(cases['baseline'], cases['candidate'], families))
        result = dict(parent_manifest=plan['parent_manifest'],
            candidate_summary=str(output.relative_to(ROOT)/'summary.json'),
            views=views, boundary_policy0={version:boundary_rows(cases, tasks)
                                          for version, cases in raw_cases.items()},
            semantics=dict(
                primary='All112 held policy-family pairs remain primary; ordinary110 excludes policy0 only as a diagnostic.',
                weighting='Original four-fold equal weighting within each of two equally weighted families, including ordinary110; not equal weighting of the110 individual pairs.',
                contrasts='After minus before; candidate minus baseline. Initial = full round1 minus point_from_source0 round1; late = full round2 minus point_after_first round2.',
                boundary_heads=['f', 'eta_f', 'g_centered', 'eta_g'],
                boundary_noise_heads=['f', 'g'],
                labels='Three existing raw repeats, g centered by source tau. Independent first-anchor reference uses its two unused repeats.',
                choice='Posterior f signs relative to first anchor and argmin within the unobserved scoring pool; not feasible deployment decisions.',
                scope='Same two familiar source families and overlapping folds; no true feasibility, target results, new simulations, or deployment evidence.'))
        save(output/'comparison.json', result)
    result['analysis_cost'] = deepcopy(cost)
    save(output/'comparison.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    summarize(read(args.manifest))
