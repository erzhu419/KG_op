"""Read sealed target moments/labels; separate prediction error from decision value."""
from collections import Counter
from math import sqrt
from statistics import mean, median

from scipy.stats import norm

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

OUTPUT = ROOT/'paper_artifacts/mainline_target_terminal_risk_reconstruction_v1'
ARMS = ('legacy_role_full','legacy_role_after_init_point_only','legacy_role_point_only')
METRICS = ('risk_mean_mse','noise_quantile_mse','margin_mse','risk_mean_bias',
           'noise_quantile_bias','margin_bias','epistemic_mean','mean_error_squared_over_epistemic',
           'predicted_noise_mean','true_noise_mean','noise_ratio_median','eligibility_errors',
           'excluded_true_feasible','eligible_true_infeasible')


def aggregate(rows, fields, **identity):
    values = {}
    for field in fields:
        known = [r[field] for r in rows if r.get(field) is not None]
        values[field] = dict(registered=len(rows), available=len(known),
            mean=mean(known) if len(known) == len(rows) and known else None,
            conditional_mean=mean(known) if known else None,
            conditional_median=median(known) if known else None)
    return dict(identity, metrics=values)


def detail(case, row):
    truth = row['truth']
    z = float(norm.ppf(1-case['alpha']))
    nq = z*(sqrt(row['constraint_aleatoric'])-truth['constraint_sigma'])
    return dict(seed=case['seed'],domain=case['domain'],arm=case['arm'],policy=row['policy'],
        rank=row['posterior_rank'],eligible=row['eligible'],probability=row['probability_violation'],
        true_feasible=truth['true_feasible'],true_margin=truth['chance_margin'],
        risk_mean_error=row['centered_constraint_error'],noise_quantile_error=nq,
        margin_error=row['stochastic_margin_error'],
        predicted_noise=row['constraint_aleatoric'],true_noise=truth['constraint_sigma']**2,
        noise_ratio=row['constraint_aleatoric']/truth['constraint_sigma']**2,
        epistemic=row['constraint_epistemic'],
        objective=row['objective'],true_objective=truth['objective'],
        objective_noise=row['episode_noise'][0],eta_f=row['marginal_mean'][1],
        eta_f_variance=row['marginal_variance'][1],eta_g=row['marginal_mean'][3],
        eta_g_variance=row['marginal_variance'][3],dominant_identity=row['dominant_identity'],
        identity_weights=row['identity_weights'],transformed_shortlist_position=row['transformed_shortlist_position'],
        safety_attempted=row['safety_attempted'],safety_passed=row['safety_passed'],
        actual_certified_deployment=row['actual_certified_deployment'])


def diagnose(summary):
    assert summary['status'] == 'complete' and summary['posterior_seal']['status'] == 'sealed'
    cases, point_rows, good_new = [], [], []
    for case in summary['cases']:
        base = {k:case[k] for k in ('replicate','domain','seed','arm','status')}
        if case['status'] != 'complete':
            cases.append(dict(base, **{m:None for m in METRICS}, C=None,R=None,G=None))
            continue
        known = [r for r in case['observed'] if r['cached_truth_available']]
        assert len(known) == len(case['observed'])  # Do not silently change point denominators.
        points = [detail(case,r) for r in known]
        for r in points:
            assert abs(r['margin_error']-r['risk_mean_error']-r['noise_quantile_error']-case['guard']) < 1e-10
        point_rows.extend(points)
        metrics = {kind+'_mse':mean(r[name]**2 for r in points) for kind,name in
            (('risk_mean','risk_mean_error'),('noise_quantile','noise_quantile_error'),('margin','margin_error'))}
        metrics.update({kind+'_bias':mean(r[name] for r in points) for kind,name in
            (('risk_mean','risk_mean_error'),('noise_quantile','noise_quantile_error'),('margin','margin_error'))})
        ref = case['reference']
        metrics.update(epistemic_mean=mean(r['epistemic'] for r in points),
            mean_error_squared_over_epistemic=mean(r['risk_mean_error']**2/max(r['epistemic'],1e-12) for r in points),
            predicted_noise_mean=mean(r['predicted_noise'] for r in points),
            true_noise_mean=mean(r['true_noise'] for r in points),
            noise_ratio_median=median(r['noise_ratio'] for r in points),
            eligibility_errors=sum(r['eligible'] != r['true_feasible'] for r in points),
            excluded_true_feasible=sum(not r['eligible'] and r['true_feasible'] for r in points),
            eligible_true_infeasible=sum(r['eligible'] and not r['true_feasible'] for r in points))
        cases.append(dict(base,**metrics,C=int(ref['certified']),R=ref['actual_safe_reward'],
            G=ref['best_initial_quality_increment'],observed_policies=len(points),
            no_eligible=case['observed_pool_no_eligible'],deployed=ref['deployed'],
            identity_names=case['identity_names'],
            mean_identity_weights=[mean(r['identity_weights'][i] for r in points)
                                   for i in range(len(case['identity_names']))],
            dominant_identity_counts=dict(Counter(r['dominant_identity'] for r in points))))
        by_policy = {r['policy']:r for r in points}
        for item in (ref['good_new_policies'] if ref['baseline_defined'] else []):
            good_new.append(dict(by_policy[item['policy']], disposition=item['disposition'],
                gain_over_best_initial=item['quality_gain_over_best_initial'],
                better_than_certified_deployment=item['better_than_certified_deployment'],
                quality_difference_to_certified_deployment=item['quality_difference_to_certified_deployment']))
    index = {(c['seed'],c['arm']):c for c in cases}
    pairs = []
    for p in summary['pairs']:
        left,right = index[p['seed'],p['left']],index[p['seed'],p['right']]
        delta = {m:left.get(m)-right.get(m) if left.get(m) is not None and right.get(m) is not None else None
                 for m in METRICS+('C','R','G')}
        pairs.append(dict(seed=p['seed'],domain=left['domain'],left=p['left'],right=p['right'],
            same_ordered_raw_history=p.get('same_ordered_raw_history',False),
            terminal_mechanism_attribution=p['terminal_mechanism_attribution'],
            left_status=left['status'],right_status=right['status'],**delta))
    aggregates = []
    for arm in ARMS:
        for family in [None]+sorted({c['domain'] for c in cases}):
            selected = [c for c in cases if c['arm']==arm and (family is None or c['domain']==family)]
            aggregates.append(aggregate(selected,METRICS+('C','R','G'),arm=arm,domain=family))
    contrast_summaries = []
    for left,right in ((ARMS[0],ARMS[1]),(ARMS[1],ARMS[2])):
        group = [p for p in pairs if (p['left'],p['right'])==(left,right)]
        alignment = {}
        for error in ('risk_mean_mse','noise_quantile_mse','margin_mse'):
            complete = [p for p in group if p[error] is not None and p['G'] is not None]
            counts = Counter(('error_lower' if p[error]<0 else 'error_higher' if p[error]>0 else 'error_tied',
                              'G_higher' if p['G']>1e-15 else 'G_lower' if p['G'] < -1e-15 else 'G_tied')
                             for p in complete)
            alignment[error] = dict(registered=24,available=len(complete),
                counts={'/'.join(k):v for k,v in counts.items()})
        contrast_summaries.append(dict(aggregate(group,METRICS+('C','R','G'),left=left,right=right),
            same_history_count=sum(p['same_ordered_raw_history'] for p in group),error_gain_alignment=alignment))
    extremes = []
    for arm in ARMS:
        group = [r for r in point_rows if r['arm']==arm]
        for metric in ('predicted_noise','noise_ratio','epistemic','objective_noise'):
            extremes.append(dict(metric=metric,**max(group,key=lambda r:r[metric])))
    focus = [r for r in point_rows if r['seed'] in (2026103011,2026103016,2026103023)
             and r['arm'] in ARMS[:2]]
    return dict(status='complete',registered_geometries=24,registered_role_units=72,
        completed_role_units=sum(c['status']=='complete' for c in cases),
        cases=cases,aggregates=aggregates,pairs=pairs,contrasts=contrast_summaries,
        maximum_rows=extremes,good_new_policy_decisions=good_new,
        undefined_good_new_reference_units=[{k:c[k] for k in ('domain','seed','arm')}
            for c in cases if c['status']=='complete' and c['G'] is None],
        same_history_changed_primary_cases=focus,
        interpretation=['All per-task error metrics average over that arm own observed pool, then tasks receive equal weight. '
            'Different histories/pools are descriptive contrasts; exact-history terminal substitutions remain in the parent summary.',
            'Noise error is also reported on the decision scale z*(predicted sigma-true sigma). '
            'Its sum with centered mean error and the guard equals the chance-margin error.',
            'Mean-error squared divided by epistemic variance is an observed-point residual diagnostic, '
            'not an independent frequentist coverage or calibration test.',
            'G retains the best-feasible-initial reference: Queue3009 is undefined; POINT3018 remains missing. '
            'R and C refer only to actual completed verification. Unverified alternative certificates remain unknown.',
            'This final observed-pool analysis cannot exclude the earlier source failure on never-observed support-boundary points. '
            'No new model correction, posterior call, or target outcome is created.'])


def main():
    with attempt(OUTPUT/'risk_error_diagnostic','saved-target-risk-error-analysis') as cost:
        cost.update(new_target_initial_calls=0,new_source_fits=0,new_truth_function_calls=0,
            engine_restorations=0,posterior_restorations=0,raw_replays=0,new_rng_draws=0,download_bytes=0)
        result = diagnose(read(OUTPUT/'summary.json'))
        save(OUTPUT/'risk_error_diagnostic.json',result)
    print({'status':result['status'],'completed':result['completed_role_units'],'cpu_seconds':cost['cpu_seconds']})


if __name__ == '__main__':
    main()
