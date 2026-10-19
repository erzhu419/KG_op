"""Describe saved source noise moments without fitting or posterior restoration."""
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'paper_artifacts/mainline_role_source_holdout_v1'
MOMENTS = ('exp_mean_eta', 'matched_gaussian_noise', 'saved_noise', 'observed_noise')


def read(path):
    return json.loads(path.read_text())


def aggregate(rows, **labels):
    sums = {key:float(sum(row[key] for row in rows)) for key in MOMENTS}
    largest = sorted(rows, key=lambda row:row['saved_noise'], reverse=True)[:3]
    return dict(labels, policy_count=len(rows), means={key:value/len(rows) for key,value in sums.items()},
        pooled_ratios=dict(saved_over_observed=sums['saved_noise']/sums['observed_noise'],
            exp_mean_over_observed=sums['exp_mean_eta']/sums['observed_noise'],
            gaussian_over_observed=sums['matched_gaussian_noise']/sums['observed_noise'],
            saved_over_exp_mean=sums['saved_noise']/sums['exp_mean_eta'],
            saved_over_gaussian=sums['saved_noise']/sums['matched_gaussian_noise']),
        top_saved_noise=[dict(row, fraction_of_group_saved_noise=row['saved_noise']/sums['saved_noise'])
                         for row in largest])


def main():
    started, wall = time.process_time(), time.perf_counter()
    summary = read(OUTPUT/'summary.json')
    forecasts = [read(OUTPUT/f'fold_{i}'/'forecasts.json') for i in range(4)]
    observed_cases = {(case['fold'],case['family']):case for case in summary['cases']}
    rows = []
    for fold in forecasts:
        for case in fold['cases']:
            observed = observed_cases[fold['fold'],case['family']]
            assert case['held_policy_ids'][2:] == observed['scored_policy_ids']
            prediction = case['predictions']['full']
            mean, variance, noise = (np.asarray(prediction[key]) for key in ('mean','variance','noise'))
            observed_variance = 3*np.asarray(observed['observed_mean_variance'])
            for head, h, eta in (('f',0,1), ('g',1,3)):
                for t in range(3):
                    for j,policy in enumerate(observed['scored_policy_ids']):
                        m, v = float(mean[t,j+2,eta]), float(variance[t,j+2,eta])
                        rows.append(dict(fold=fold['fold'],family=case['family'],head=head,round=t,
                            policy_id=policy,mean_eta=m,variance_eta=v,
                            exp_mean_eta=float(np.exp(m)),matched_gaussian_noise=float(np.exp(m+v/2)),
                            saved_noise=float(noise[t,j+2,h]),observed_noise=float(observed_variance[j,h])))
    families = list(dict.fromkeys(row['family'] for row in rows))
    by_case, by_family, fourth_fold = [], [], []
    for family in families:
        for head in ('f','g'):
            for t in range(3):
                group = [r for r in rows if (r['family'],r['head'],r['round']) == (family,head,t)]
                by_family.append(aggregate(group,family=family,head=head,round=t))
                for fold in range(4):
                    by_case.append(aggregate([r for r in group if r['fold']==fold],
                                             family=family,fold=fold,head=head,round=t))
                fourth_fold.append(dict(family=family,head=head,round=t,
                    fractions={key:sum(r[key] for r in group if r['fold']==3)/sum(r[key] for r in group)
                               for key in MOMENTS}))
    pooled = [aggregate([r for r in rows if (r['head'],r['round'])==(head,t)],
                        family='both_families_equal',head=head,round=t)
              for head in ('f','g') for t in range(3)]
    result = dict(status='complete',arm='full',scored_policy_family_pairs=112,
        source_files=['summary.json']+[f'fold_{i}/forecasts.json' for i in range(4)],
        rows=rows,by_case=by_case,by_family=by_family,pooled=pooled,fold3_contributions=fourth_fold,
        interpretation='exp(mean eta) is the exponentiated log-variance mean. exp(mean eta + variance eta / 2) '
            'is the noise moment of a single Gaussian matched to the saved first two eta moments; it is not '
            'the actual mixture. Its difference from saved E[exp(eta)] describes higher-moment effects, '
            'without identifying an individual node or identity. Observed noise is the independent held-policy '
            'three-repeat sample variance. All cases and policies retained; no correction selected.',
        cost=dict(cpu_seconds=time.process_time()-started,wall_seconds=time.perf_counter()-wall,
            new_source_calls=0,new_target_initial_calls=0,new_target_search_calls=0,new_verification_calls=0,
            new_truth_function_calls=0,new_source_fits=0,source_model_restorations=0,download_bytes=0),
        accounting='Standalone descriptive JSON analysis; no modification of completed runner ledger.')
    (OUTPUT/'noise_moments.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cost=result['cost'],pooled=[{k:v for k,v in r.items() if k!='top_saved_noise'}
                                                     for r in pooled]),indent=2))


if __name__ == '__main__':
    main()
