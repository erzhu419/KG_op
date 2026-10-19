"""Attribute frozen query shifts and terminal omissions from saved JSON/NPZ."""
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import json

import numpy as np
from scipy.stats import norm

from performance.diagnose_mainline_role_choice import score_choice
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save

MANIFEST = ROOT/'performance/manifests/mainline_propagation_attribution_v1.json'
FULL = 'legacy_role_full'
LOCAL = 'legacy_role_after_init_point_only'
RISK = ('risk_mean', 'noise_margin', 'risk_draw')
OBJECTIVE = ('objective_mean', 'objective_draw')


def components(score, z):
    a = lambda key: np.asarray(score[key], dtype=float)
    result = dict(risk_mean=a('constraint_mean'),
        noise_margin=z*np.sqrt(np.maximum(a('constraint_aleatoric'), 1e-12)),
        risk_draw=a('sampled_constraint')-a('constraint_mean'),
        objective_mean=a('objective_mean'),
        objective_draw=a('sampled_objective')-a('objective_mean'))
    margin = sum(result[k] for k in RISK)
    np.testing.assert_allclose(margin, a('sampled_margin'), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(result['risk_mean']+result['noise_margin'],
        a('stochastic_margin_mean'), rtol=1e-11, atol=1e-11)
    return result


def analyze_queries(plan):
    source = ROOT/plan['source_output_dir']
    parent = read(ROOT/plan['source_manifest'])
    z = float(norm.ppf(1-parent['task']['alpha']))
    rows, parity_errors, gap_errors = [], [], []
    modes = [('risk_' + '+'.join(subset), subset)
             for n in range(1, 4) for subset in combinations(RISK, n)]
    modes += [('objective_mean', ('objective_mean',)),
              ('objective_draw', ('objective_draw',)),
              ('all_objective', OBJECTIVE), ('all_components', RISK+OBJECTIVE)]
    for seed in plan['target_seeds']:
        directory = source/f'seed_{seed}'
        truth = {tuple(r['point']): r['truth'] for r in read(directory/'truth.json')}
        neighbors = {tuple(p): i for i, p in enumerate(np.load(directory/'neighbors.npz')['points'])}
        bundles = {}
        for arm in (FULL, LOCAL):
            case = directory/arm
            bundles[arm] = dict(audit=read(case/'query_audit.json'),
                points=[tuple(p) for p in np.load(case/'query_policies.npz')['points']],
                history=read(case/'search.json')['history'])
        full_ids = {point: i for i, point in enumerate(bundles[FULL]['points'])}
        initial = truth[tuple(bundles[FULL]['history'][0]['point'])]
        first_divergence = None
        for step in range(3):
            states = {}
            for arm, bundle in bundles.items():
                stage = bundle['audit']['stages'][step]
                points = [bundle['points'][i] for i in stage['action_ids']]
                index, total, mode, count = score_choice(
                    stage['score']['sampled_objective'], stage['score']['sampled_margin'])
                error = float(np.max(abs(total-np.asarray(stage['score']['total']))))
                assert error <= 1e-12 and index == stage['selected_index']
                assert mode == stage['score']['sampled_feasibility_mode']
                assert count == stage['score']['sampled_feasible_count']
                assert points[index] == tuple(bundle['history'][stage['paid_prefix']]['point'])
                parity_errors.append(error)
                states[arm] = dict(stage=stage, points=points, selected=points[index],
                    components=components(stage['score'], z), mode=mode, feasible_count=count)
            f, a = states[FULL], states[LOCAL]
            prefix = f['stage']['paid_prefix']
            assert prefix == a['stage']['paid_prefix'] == 10+step
            assert set(f['points']) == set(a['points']), 'Physical pools differ; paired swap requires alignment'
            alignment = [a['points'].index(p) for p in f['points']]
            c = {FULL: f['components'], LOCAL: {k: v[alignment] for k, v in a['components'].items()}}
            same_history = bundles[FULL]['history'][:prefix] == bundles[LOCAL]['history'][:prefix]
            changed = f['selected'] != a['selected']
            if changed and first_divergence is None:
                first_divergence = prefix
            if step == 0:
                assert same_history and f['points'] == a['points'] and not changed
                for key in c[FULL]:
                    np.testing.assert_allclose(c[FULL][key], c[LOCAL][key], rtol=1e-10, atol=1e-10)

            def policy(point):
                actual = truth.get(point)
                return dict(full_bank_id=full_ids[point], neighbor_index=neighbors.get(point),
                    truth=actual, feasible_improving=(None if actual is None else
                        bool(actual['true_feasible'] and actual['objective'] < initial['objective']-1e-12)))

            def choose(values):
                index, _, mode, count = score_choice(sum(values[k] for k in OBJECTIVE),
                                                      sum(values[k] for k in RISK))
                return dict(selected=policy(f['points'][index]), mode=mode, feasible_count=count)

            swapped = []
            for receiver, donor in ((FULL, LOCAL), (LOCAL, FULL)):
                for name, subset in modes:
                    values = {k: c[donor][k] if k in subset else c[receiver][k] for k in c[receiver]}
                    result = choose(values)
                    result.update(receiver=receiver, donor=donor, substitution=name,
                        selected_changed=result['selected']['full_bank_id'] != full_ids[states[receiver]['selected']],
                        donor_choice_recovered=result['selected']['full_bank_id'] == full_ids[states[donor]['selected']])
                    if name == 'all_components':
                        assert result['donor_choice_recovered'], 'Aligned all-component scores changed original action'
                    swapped.append(result)

            pair = []
            for point in dict.fromkeys([f['selected'], a['selected']]):
                index = f['points'].index(point)
                records = {}
                for arm in (FULL, LOCAL):
                    values = {k: float(v[index]) for k, v in c[arm].items()}
                    local_index = states[arm]['points'].index(point)
                    score = states[arm]['stage']['score']
                    values.update(risk_epistemic=float(score['constraint_epistemic'][local_index]),
                        objective_epistemic=float(score['objective_epistemic'][local_index]),
                        previously_observed=point in {tuple(r['point']) for r in bundles[arm]['history'][:prefix]},
                        sampled_margin=sum(values[k] for k in RISK),
                        sampled_objective=sum(values[k] for k in OBJECTIVE))
                    if point in truth:
                        values.update(risk_mean_bias=values['risk_mean']-truth[point]['constraint_mean_centered'],
                            noise_margin_bias=values['noise_margin']-z*truth[point]['constraint_sigma'])
                    records[arm] = values
                pair.append(dict(**policy(point), arms=records,
                    full_minus_local={k: records[FULL][k]-records[LOCAL][k] for k in RISK+OBJECTIVE}))
            fi, ai = (f['points'].index(states[arm]['selected']) for arm in (FULL, LOCAL))
            gaps = {arm: {k: float(v[ai]-v[fi]) for k, v in c[arm].items()} for arm in (FULL, LOCAL)}
            gap_delta = {k: gaps[FULL][k]-gaps[LOCAL][k] for k in RISK+OBJECTIVE}
            actual_f = np.asarray(f['stage']['score']['sampled_margin'])
            actual_a = np.asarray(a['stage']['score']['sampled_margin'])[alignment]
            direct = (actual_f[ai]-actual_f[fi])-(actual_a[ai]-actual_a[fi])
            gap_error = abs(sum(gap_delta[k] for k in RISK)-direct)
            assert gap_error < 1e-11, 'Risk components do not reconstruct the saved pair margin gap'
            gap_errors.append(float(gap_error))
            preceding = None
            if step and same_history:
                observation = bundles[FULL]['history'][prefix-1]
                point = tuple(observation['point']); actual = truth[point]
                previous = bundles[FULL]['audit']['stages'][step-1]
                j = previous['selected_index']; score = previous['score']
                raw = observation['raw']
                preceding = dict(policy=policy(point), raw=raw,
                    risk_prediction_before=score['constraint_mean'][j],
                    risk_raw_residual=raw[1]-score['constraint_mean'][j],
                    risk_model_error=actual['constraint_mean_centered']-score['constraint_mean'][j],
                    risk_observation_noise=raw[1]-actual['constraint_mean_centered'],
                    objective_raw_residual=raw[0]-score['objective_mean'][j])
            rows.append(dict(seed=seed, paid_prefix=prefix, same_paid_history=same_history,
                same_ordered_pool=f['points']==a['points'], same_physical_pool=True,
                query_changed=changed, first_divergence=changed and prefix==first_divergence,
                scope='same_history' if same_history else 'different_adaptive_histories',
                full=dict(selected=policy(f['selected']), mode=f['mode'], feasible_count=f['feasible_count']),
                local=dict(selected=policy(a['selected']), mode=a['mode'], feasible_count=a['feasible_count']),
                selected_pair=pair, selected_pair_local_minus_full_gaps=gaps,
                selected_pair_gap_full_minus_local=gap_delta,
                preceding_shared_observation=preceding, substitutions=swapped))
    first = [r for r in rows if r['first_divergence']]
    assert all(r['same_paid_history'] and r['same_ordered_pool'] for r in first)
    counts = []
    for receiver in (FULL, LOCAL):
        for name, _ in modes:
            values = [next(v for v in row['substitutions'] if v['receiver']==receiver and
                           v['substitution']==name) for row in first]
            counts.append(dict(receiver=receiver, substitution=name, tasks=len(first),
                donor_choice_recovered=sum(v['donor_choice_recovered'] for v in values),
                changed=sum(v['selected_changed'] for v in values),
                recovery_seeds=[row['seed'] for row, v in zip(first, values) if v['donor_choice_recovered']]))
    return dict(status='complete', cases=rows, first_divergences=[dict(seed=r['seed'],
        paid_prefix=r['paid_prefix'], full=r['full'], local=r['local']) for r in first],
        first_divergence_substitutions=counts,
        summary=dict(task_count=len(plan['target_seeds']), paired_states=len(rows),
            query_changes_by_prefix={str(p): sum(r['query_changed'] for r in rows if r['paid_prefix']==p)
                                     for p in (10,11,12)},
            first_divergence_modes=dict(Counter(r['full']['mode']+' / '+r['local']['mode'] for r in first))),
        reconstruction=dict(recorded_choices=len(parity_errors), max_score_error=max(parity_errors),
            max_pair_risk_decomposition_error=max(gap_errors)),
        limits=plan['limits'])


def main():
    plan = read(MANIFEST); output = ROOT/plan['output_dir']
    from performance.diagnose_mainline_terminal_omission import analyze
    with attempt(output, 'saved-propagation-attribution', 'FactorShockStatePolicyRZDT1') as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0,
            new_truth_function_calls=0, posterior_restorations=0, download_bytes=0)
        queries = analyze_queries(plan)
        save(output/'queries.json', queries)
        terminal = analyze(plan)
        save(output/'terminal.json', terminal)
    attempts = [json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines()]
    fields = ('cpu_seconds','wall_seconds','new_source_calls','new_target_initial_calls',
              'new_target_search_calls','new_verification_calls','new_source_fits',
              'new_truth_function_calls','posterior_restorations','download_bytes')
    save(output/'completion.json', dict(status='complete',
        cost={k:sum(r.get(k,0) for r in attempts) for k in fields}, attempts=len(attempts),
        failed_attempts=[r for r in attempts if r['status']!='completed'],
        reconstruction=queries['reconstruction'], completed_utc=datetime.now(timezone.utc).isoformat()))
    plan['status']='complete'; save(MANIFEST, plan)
    print(json.dumps(queries['summary']), flush=True)


if __name__ == '__main__':
    main()
