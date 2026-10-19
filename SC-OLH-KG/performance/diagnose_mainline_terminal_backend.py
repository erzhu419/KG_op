"""Compare frozen role/native terminal decisions on identical paid histories."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from performance.diagnose_mainline_role_terminal import summarize as trace_saved
from performance.mainline_legacy_native import LegacyNativeAlgorithm, SearchFrozen
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save

MANIFEST = ROOT / 'performance/manifests/mainline_terminal_backend_v1.json'
NATIVE = 'legacy_native_neighbors'


def signature(frozen):
    terminal = frozen['terminal']
    return dict(primary=frozen['primary'], shortlist=[
        {key: row.get(key) for key in ('point', 'shortlist_position', 'shortlist_role',
                                     'posterior_rank', 'objective_incumbent')}
        for row in terminal['frozen_shortlist']],
        budgets=terminal['candidate_verification_budgets'],
        incumbent_position=terminal['objective_incumbent_position'])


def challenger_error(actual, saved):
    actual, saved = actual['terminal']['objective_challenger'], saved['terminal']['objective_challenger']
    assert actual['status'] == saved['status'] and actual['eligible_count'] == saved['eligible_count']
    errors = []
    for key in ('selected_posterior_objective', 'selected_probability_violation'):
        if key in saved:
            np.testing.assert_allclose(actual[key], saved[key], rtol=1e-9, atol=1e-11)
            errors.append(abs(actual[key]-saved[key]))
    return max(errors, default=0.)


def freeze(engine, native):
    """The original protocol with zero verification budget, before any truth join."""
    assert engine.defer_terminal and not engine.config.truth_pool_diagnostics
    assert getattr(engine, '_decision_context', 0) == 0
    # The saved frontend already contains native models updated with this history.
    recommend = (lambda pool: LegacyNativeAlgorithm._decision_backend_terminal_recommendation(engine, pool)) if native else engine._decision_backend_terminal_recommendation
    verify = (lambda point: LegacyNativeAlgorithm._terminal_verification_protocol(engine, point)) if native else engine._terminal_verification_protocol
    primary, details = recommend(engine._last_terminal_pool)
    assert engine._posterior_dominance_terminal_recommendation() is None
    ranked = list(engine._last_terminal_bayes_ranked_points)
    try:
        verify(primary)
    except SearchFrozen:
        pass
    else:
        raise AssertionError('Terminal must stop before independent verification')
    pool = engine._terminal_action_pool(engine._last_terminal_pool, observations=engine.observations)
    assert set(pool) == {p for p, _ in engine.history}
    with ExitStack() as context:
        if not native:
            context.enter_context(engine._posterior_decision())
        components = engine._terminal_bayes_risk_components(
            engine.gpr, engine.variance_model, pool, task_ensemble=engine.task_ensemble,
            risk_penalty=engine.config.decision_risk_penalty)
    extra = {}
    if not native:
        views, noise = engine._posterior_views(pool)
        mean = views[1].posterior_mean_many(pool)
        epi = engine._constraint_decision_epistemic_many(views[1], pool)
        aleatoric = noise.predict_variance_many(1, pool, engine.problem)
        guard = max(float(engine.problem.pilot_constraint_guard()), 0.) if hasattr(engine.problem, 'pilot_constraint_guard') else 0.
        margin = mean + guard + norm.ppf(1-engine.problem.alpha)*np.sqrt(aleatoric)-engine.problem.tau
        # Detect a centering/noise-view mismatch before interpreting the .5 gate.
        np.testing.assert_allclose(norm.cdf(margin/np.sqrt(np.maximum(epi, 1e-12))),
                                   components['probability_violation'], rtol=1e-10, atol=1e-12)
        extra = dict(constraint_mean=mean, constraint_epistemic=epi,
                     constraint_aleatoric=aleatoric, stochastic_margin_mean=margin)
    fields = ('objective', 'objective_variance', 'risk', 'expected_violation', 'probability_violation')
    records = []
    for i, point in enumerate(pool):
        record = dict(point=list(point), posterior_rank=ranked.index(point)+1,
                      **{name: float(components[name][i]) for name in fields},
                      **{name: float(values[i]) for name, values in extra.items()})
        record['eligible'] = record['probability_violation'] <= engine.config.terminal_objective_challenger_max_violation_probability
        records.append(record)
    return dict(**engine.frozen_terminal, observed=records,
                feasibility_mode=details['decision_backend_terminal_feasibility_mode'],
                pool_source=details['decision_backend_terminal_pool_source'],
                risk_source=components['constraint_posterior_source'])


def replay(plan, parent, seed, arm, bundle):
    source = ROOT / plan['input_dir'] / f'seed_{seed}' / arm
    output = ROOT / plan['output_dir'] / f'seed_{seed}' / arm
    with attempt(output, 'same-history-terminal', parent['domain'], arm) as cost:
        cost.update(seed=seed, new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        saved = read(source / 'search.json')
        engine = load_pickle(source / 'search_engine.pkl')
        before = [dict(point=list(p), raw=np.asarray(y).tolist()) for p, y in engine.history]
        assert before == saved['history']
        if arm != NATIVE:
            engine.backend_bundle = bundle
        with ExitStack() as context:
            for obj, name in ((engine.problem, 'simulate'), (engine, '_true_chance_margin'),
                              (engine, '_evaluate_recommendation')):
                context.enter_context(patch.object(obj, name, side_effect=AssertionError('Frozen-history diagnosis cannot simulate or evaluate truth')))
            original = freeze(engine, native=arm == NATIVE)
            # Catch wrong restoration or dispatch before attributing a backend difference.
            assert signature(original) == signature(saved)
            error = challenger_error(original, saved)
            native = original if arm == NATIVE else freeze(engine, native=True)
        assert before == [dict(point=list(p), raw=np.asarray(y).tolist()) for p,y in engine.history]
        for result in (original, native):
            assert result['terminal']['candidate_verification_budgets'] == [80,128,128]
            assert result['terminal']['max_verification_budget'] == 352
        save(output / 'decisions.json', dict(seed=seed, history_arm=arm,
            original=original, native=native, saved_decision_parity=True,
            saved_challenger_max_error=error,
            truth_used=False, verification_performed=False,
            frozen_utc=datetime.now(timezone.utc).isoformat()))
        print(seed, arm, 'terminal frozen', flush=True)


def summarize(plan):
    source, output = ROOT/plan['input_dir'], ROOT/plan['output_dir']
    # All model decisions must exist before joining previously cached target truth.
    decisions = [read(output/f'seed_{seed}'/arm/'decisions.json')
                 for seed in plan['target_seeds'] for arm in plan['history_arms']]
    trace_path = output/'saved_terminal_trace.json'
    trace = read(trace_path) if trace_path.exists() else trace_saved(source)
    if not trace_path.exists():
        save(trace_path, trace)
    traces = {(r['seed'], r['arm']): r for r in trace['cases']}
    cases = []
    for decision in decisions:
        seed, arm = decision['seed'], decision['history_arm']
        task = source/f'seed_{seed}'
        saved = read(task/arm/'search.json')
        decision['saved_challenger_max_error'] = challenger_error(decision['original'], saved)
        save(output/f'seed_{seed}'/arm/'decisions.json', decision)
        truth = {tuple(r['point']): r['truth'] for r in read(task/'truth.json')}
        neighbors = {tuple(p): i for i,p in enumerate(np.load(task/'neighbors.npz')['points'])}
        history = saved['history']
        initial = tuple(history[0]['point'])
        initial_keys = [tuple(r['point']) for r in history[:10]]
        def label(point):
            point = tuple(point)
            if point in neighbors:
                return f'neighbor:{neighbors[point]}'
            if point in initial_keys:
                return f'initial:{initial_keys.index(point)}'
            return f'history:{next(i for i,r in enumerate(history) if tuple(r["point"])==point)}'
        views = {}
        for name in ('original', 'native'):
            frozen = decision[name]
            terminal = frozen['terminal']
            records = []
            for record in frozen['observed']:
                point = tuple(record['point'])
                actual = truth.get(point)
                position = next((r['shortlist_position'] for r in terminal['frozen_shortlist'] if tuple(r['point']) == point), None)
                records.append({k:v for k,v in record.items() if k!='point'} | dict(
                    policy=label(point), truth=actual, cached_truth_available=actual is not None,
                    shortlist_position=position,
                    feasible_improving=(actual['true_feasible'] and actual['objective'] < truth[initial]['objective']-1e-12) if actual is not None else None,
                    objective_difference_to_initial=record['objective']-next(r['objective'] for r in frozen['observed'] if tuple(r['point'])==initial)))
            views[name] = dict(primary=label(frozen['primary']),
                shortlist=[label(r['point']) for r in terminal['frozen_shortlist']],
                shortlist_roles=[r['shortlist_role'] for r in terminal['frozen_shortlist']],
                challenger_status=terminal['objective_challenger']['status'],
                eligible_count=terminal['objective_challenger']['eligible_count'],
                feasibility_mode=frozen['feasibility_mode'], observed=records)
        improving = [r for r in views['original']['observed'] if r['feasible_improving']]
        cases.append(dict(seed=seed, history_arm=arm, **views,
            primary_changed=views['original']['primary']!=views['native']['primary'],
            shortlist_changed=views['original']['shortlist']!=views['native']['shortlist'],
            native_first_feasible_improving=next(r['feasible_improving'] for r in views['native']['observed'] if r['policy']==views['native']['shortlist'][0]),
            original_first_feasible_improving=next(r['feasible_improving'] for r in views['original']['observed'] if r['policy']==views['original']['shortlist'][0]),
            discovered_improving_policies=[r['policy'] for r in improving],
            saved_certified=traces[seed,arm]['certified'], saved_deployed=traces[seed,arm]['deployed'],
            saved_query_dispositions=traces[seed,arm]['distinct_query_policies'],
            native_deployment_status='not_verified_this_stage'))
    arms = []
    for arm in plan['history_arms']:
        rows = [r for r in cases if r['history_arm']==arm]
        arms.append(dict(history_arm=arm, cases=len(rows),
            **{field: sum(r[field] is True for r in rows) for field in ('primary_changed','shortlist_changed',
                'native_first_feasible_improving','original_first_feasible_improving')},
            discovered_improving_policy_task_pairs=sum(len(r['discovered_improving_policies']) for r in rows),
            **{name+'_first_'+metric: sum(next(
                (p['truth'] is None if metric=='truth_unavailable' else p['truth'] is not None and p['truth']['true_feasible'])
                for p in r[name]['observed'] if p['policy']==r[name]['shortlist'][0]) for r in rows)
                for name in ('original','native') for metric in ('feasible','truth_unavailable')}))
    result = dict(status='complete', cases=cases, arms=arms,
        certificate_replay=trace['certificate_replay'],
        original_decision_parity_cases=len(decisions),
        saved_challenger_max_error=max(r['saved_challenger_max_error'] for r in decisions),
        scope='Same paid histories; cached truth joined after all decisions. No new deployment or confirmation claim.')
    save(output/'diagnosis.json', result)
    return result


def main():
    plan = read(MANIFEST)
    parent = read(ROOT/plan['parent_manifest'])
    output = ROOT/plan['output_dir']
    pending = [(seed,arm) for seed in plan['target_seeds'] for arm in plan['history_arms']
               if not (output/f'seed_{seed}'/arm/'decisions.json').exists()]
    if pending:
        with attempt(output, 'load-existing-source-model', parent['domain']) as cost:
            cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0,
                        source_model_restorations=1)
            bundle = load_pickle(ROOT/parent['frozen_assets']['source_models_file'])
        for seed,arm in pending:
            replay(plan, parent, seed, arm, bundle)
    with attempt(output, 'summarize-cached-evidence', parent['domain']) as cost:
        cost.update(new_source_fits=0, new_truth_function_calls=0, download_bytes=0)
        result = summarize(plan)
    ledger = [json.loads(line) for path in output.rglob('attempts.jsonl') for line in path.read_text().splitlines()]
    fields = ('cpu_seconds','wall_seconds','new_source_calls','new_source_fits',
              'new_target_search_calls','new_verification_calls','new_truth_function_calls','download_bytes')
    save(output/'completion.json', dict(status='complete', cases=len(result['cases']),
        cost={k:sum(r.get(k,0) for r in ledger) for k in fields},
        attempts=len(ledger), failed_attempts=sum(r['status']!='completed' for r in ledger),
        preliminary_read_only_diagnostics=dict(timing='unmetered', native_terminal_smoke_cases=3,
            pure_saved_raw_replays=3, new_calls=0), new_deployment_claims=0))
    print(json.dumps(result['arms']), flush=True)


if __name__ == '__main__':
    main()
