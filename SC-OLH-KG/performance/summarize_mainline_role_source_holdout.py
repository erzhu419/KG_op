"""Score frozen source-policy role forecasts against existing independent raw rows."""
import argparse

import numpy as np
from scipy.special import digamma, polygamma

from baselines.transfer_archive import FrozenTransferArchive
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save


ARMS = ('full', 'point_from_source0', 'point_after_first')
MANIFEST = ROOT/'performance/manifests/mainline_role_source_holdout_v1.json'


def mean_metrics(mean, observed, observation_variance):
    error = np.asarray(mean)-observed
    return dict(mse=float(np.mean(error**2)), bias=float(np.mean(error)),
        noise_subtracted_mse=float(np.mean(error**2-observation_variance)))


def moment_metrics(mean, variance, observed, observation_variance):
    """Dawid-Sebastiani score uses predictive moments, not a mixture density."""
    total = np.asarray(variance)+observation_variance
    if np.any(total <= 0) or not np.isfinite(total).all():
        raise ValueError('Nonpositive or nonfinite predictive scoring variance')
    error = np.asarray(mean)-observed
    return dict(mean_metrics(mean, observed, observation_variance),
        standardized_squared_error=float(np.mean(error**2/total)),
        dawid_sebastiani_score=float(np.mean(np.log(total)+error**2/total)),
        mean_epistemic_variance=float(np.mean(variance)),
        mean_observation_variance=float(np.mean(observation_variance)),
        mean_predictive_variance=float(np.mean(total)))


def score_case(case, task, fold):
    held = np.asarray(case['held_policy_ids'], dtype=int)
    raw = np.asarray([task.Y_replicates[i] for i in held[2:]])
    observed = raw.mean(axis=1)
    observed[:, 1] -= task.tau
    sample_variance = raw.var(axis=1, ddof=1)
    label_variance = sample_variance/3
    eta = np.log(sample_variance[:, 1])-digamma(1.)
    eta_label_variance = float(polygamma(1, 1.))
    reference = task.Y_replicates[held[0]][1:, 0]
    anchor = float(reference.mean())
    anchor_variance = float(reference.var(ddof=1)/len(reference))
    difference = observed[:, 0]-anchor
    difference_variance = label_variance[:, 0]+anchor_variance
    scores, choices, signs = [], {}, {}
    for arm in ARMS:
        predictions = case['predictions'][arm]
        mean, variance, noise = (np.asarray(predictions[key])
                                 for key in ('mean', 'variance', 'noise'))
        choices[arm], signs[arm] = [], []
        for t in range(3):
            heads = {}
            for name, h, raw_h in (('f', 0, 0), ('g', 2, 1)):
                # Calibration uses forecast episode noise, while MSE debiasing
                # uses the independent sample-mean variance estimated from raw.
                heads[name] = moment_metrics(mean[t, 2:, h], variance[t, 2:, h],
                    observed[:, raw_h], noise[t, 2:, raw_h]/3)
                heads[name].update(mean_metrics(mean[t, 2:, h], observed[:, raw_h],
                                                label_variance[:, raw_h]))
                heads[name]['observed_mean_variance'] = float(label_variance[:, raw_h].mean())
                heads[name]['predicted_episode_variance'] = float(noise[t, 2:, raw_h].mean())
                heads[name]['observed_episode_variance'] = float(sample_variance[:, raw_h].mean())
                heads[name]['pooled_episode_variance_ratio'] = (
                    heads[name]['predicted_episode_variance']/heads[name]['observed_episode_variance'])
            heads['eta_g'] = moment_metrics(mean[t, 2:, 3], variance[t, 2:, 3],
                                           eta, eta_label_variance)
            predicted_difference = mean[t, 2:, 0]-mean[t, 0, 0]
            direction = np.sign(predicted_difference)
            selected = int(np.argmin(mean[t, 2:, 0]))
            choices[arm].append(int(held[2+selected]))
            signs[arm].append(direction)
            scores.append(dict(arm=arm, round=t, **heads,
                f_difference=dict(mean_metrics(predicted_difference, difference, difference_variance),
                    direction_agreement=float(np.mean(predicted_difference*difference > 0)),
                    predicted_argmin_policy_id=choices[arm][-1])))
    contrasts = []
    for control in ARMS[1:]:
        contrasts.append(dict(control=control, round=2,
            direction_changes=int(np.sum(signs['full'][2] != signs[control][2])),
            scoring_argmin_changed=bool(choices['full'][2] != choices[control][2])))
    return dict(fold=fold, family=task.name, scored_policy_ids=held[2:],
        anchor_policy_id=int(held[0]), independent_anchor_mean=anchor,
        independent_anchor_mean_variance=anchor_variance,
        observed_mean=observed, observed_mean_variance=label_variance,
        observed_eta_g=eta, scores=scores, final_contrasts=contrasts)


def aggregate_rows(cases, families):
    aggregate = []
    for family in [*families, 'both_families_equal']:
        selected = [r for r in cases if family == 'both_families_equal' or r['family'] == family]
        for arm in ARMS:
            for t in range(3):
                values = [next(v for v in r['scores'] if v['arm']==arm and v['round']==t)
                          for r in selected]
                metrics = {}
                for name in ('f', 'g', 'eta_g', 'f_difference'):
                    metrics[name] = {key:float(np.mean([v[name][key] for v in values]))
                        for key in values[0][name] if key != 'predicted_argmin_policy_id'}
                for name in ('f', 'g'):
                    metrics[name]['pooled_episode_variance_ratio'] = (
                        metrics[name]['predicted_episode_variance']/metrics[name]['observed_episode_variance'])
                aggregate.append(dict(family=family, arm=arm, round=t, **metrics))
    return aggregate


def summarize(plan):
    output = ROOT/plan['output_dir']
    # All forecasts must exist before any score labels are opened.
    folds = [read(output/f'fold_{i}'/'forecasts.json') for i in range(plan['folds'])]
    for fold in folds:
        for case in fold['cases']:
            assert len(case['held_policy_ids']) == 16
            for arm in ARMS:
                for key, shape in (('mean', (3,16,4)), ('variance', (3,16,4)), ('noise', (3,16,2))):
                    values = np.asarray(case['predictions'][arm][key])
                    assert values.shape == shape and np.isfinite(values).all(), (arm,key)
    with attempt(output, 'score-frozen-role-source-forecasts') as cost:
        cost.update(new_source_fits=0, new_target_initial_calls=0,
                    new_truth_function_calls=0, download_bytes=0)
        archive = FrozenTransferArchive.load(ROOT/plan['archive'])
        by_name = {task.name:task for task in archive.tasks}
        cases = [score_case(case, by_name[case['family']], fold['fold'])
                 for fold in folds for case in fold['cases']]
        comparisons = []
        for control in ARMS[1:]:
            values = [next(v for v in case['final_contrasts'] if v['control']==control) for case in cases]
            comparisons.append(dict(control=control,
                final_direction_changes=sum(v['direction_changes'] for v in values),
                final_scoring_argmin_changes=sum(v['scoring_argmin_changed'] for v in values)))
        result = dict(cases=cases, aggregate=aggregate_rows(cases, list(by_name)),
            final_contrasts=comparisons, source_family_count=len(by_name),
            folds=plan['folds'], scored_policy_family_pairs=sum(len(r['scored_policy_ids']) for r in cases),
            metric_semantics=dict(
                mean_noise_correction='Subtract independent held-policy sample variance / 3; negative values retained.',
                mean_predictive_variance='Latent mean variance plus predicted episode variance / 3.',
                eta_label='log(sample variance) - digamma(1), with trigamma(1) label variance for three Gaussian repeats.',
                score='Dawid-Sebastiani log(V) + squared error / V; a moment score, not the exact mixture log density.',
                objective_reference='First anchor two unused raw episodes; its update raw is excluded.',
                weighting='Four folds equally within each of two source families; both-families aggregate gives each family equal weight.'),
            interpretation='Frozen role-law source-policy diagnostic, not task-independent uncertainty, true chance feasibility, or target deployment evidence. Folds and shared physical policies are dependent.')
        save(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(MANIFEST))
    args = parser.parse_args()
    summarize(read(args.manifest))
