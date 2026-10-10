"""Refine queried guarded actions by an equivalent conditional mean integral."""

import copy
import signal
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from core.conditional_raw_reference import conditional_raw_coordinates
from core.conditional_terminal_kg import ConditionalTerminalActions, risk_coordinates_from_moments, guarded_risk_statistics
from core.positive_mixture_lookahead import condition_positive_raw
from core.positive_raw_terminal import raw_terminal_component
from core.raw_likelihood_lookahead import risk_terminal_statistics
from core.stratified_raw_lookahead import stratified_record_weights, summarize_stratified_futures
from performance.complete_historical_source_heads import ROOT, read, save, cpu_timeout
from performance.connect_historical_learned_prior_kg import build_connection
from representation.vectorized_joint_function_risk import raw_fantasy_moments, apply_raw_moments

PLAN = 'performance/manifests/historical_primary_mean_terminal_v1.json'


def restore_future(model, query, labels):
    future = copy.deepcopy(model); scores = future.update_raw(query, labels)['head_log_scores']
    return future, scores


def validate_anchors(model, terminal, actions, bank, spec):
    started = time.process_time(); rows = []
    action = next(a for a in actions if a['kind'] == 'guarded' and a['primary'] == spec['candidate']
        and a['budgets'] == [130, 130, 83])
    nodes = qmc.Sobol(7, scramble=True, seed=spec['seed']).random_base2(spec['anchor_conditional_power'])
    full_nodes = qmc.Sobol(7, scramble=True, seed=spec['seed']).random_base2(spec['anchor_full_joint_power'])
    ix = (2*np.asarray(action['shortlist'])[:, None]+np.arange(2)).ravel()
    for anchor in spec['anchors']:
        record = next(r for r in bank if (r['origin_identity'], r['identity_fantasy_position']) ==
            (anchor['origin_identity'], anchor['fantasy_position']))
        future, scores = restore_future(model, spec['candidate'], record['labels'])
        for identity in spec['utility_identities']:
            prior, updated = model.components[identity][1], future.components[identity][1]
            eta, mean, covariance, weights, _ = conditional_raw_coordinates(prior, updated, action['shortlist'],
                nodes, record['labels'][1], scores[identity][1], query=spec['candidate'])
            coordinates = risk_coordinates_from_moments(eta, mean, covariance, np.arange(len(nodes)), nodes,
                mean_normals=ndtri(nodes[:, 4:6]), integrated_position=spec['integrated_mean_position'])
            actual, cost = guarded_risk_statistics(coordinates, action['incumbent_position'], action['budgets'],
                terminal.protocol, weights, integrated_position=spec['integrated_mean_position'])
            # Exact three-policy marginal avoids copying unrelated16D covariances
            # into every262144-node historical reference draw.
            marginal_prior, marginal_updated = copy.copy(prior), copy.copy(updated)
            for source, marginal in ((prior, marginal_prior), (updated, marginal_updated)):
                marginal.means = source.means[:, ix]; marginal.covariances = source.covariances[:, ix][:, :, ix]
            draws, full_weights, _ = condition_positive_raw(marginal_prior, marginal_updated, 0, [0, 1, 2],
                full_nodes, record['labels'][1], scores[identity][1])
            expected, expected_cost = risk_terminal_statistics(draws.reshape(1, len(full_nodes), 3, 2),
                full_weights[None], action['incumbent_position'], action['budgets'], terminal.protocol)
            error = float(np.max(abs(actual-expected[0])))
            rows.append({**anchor, 'evaluated_identity': identity, 'maximum_branch_error': error,
                'expected_cost_difference': float(cost-expected_cost[0]),
                'branch_reference_gate_passed': error <= spec['maximum_anchor_branch_error']})
            print('primary mean anchor', anchor, identity, 'error', error, flush=True)
    return {'rows': rows, 'all_anchors_equivalent': all(r['branch_reference_gate_passed'] for r in rows),
        'cpu_seconds': time.process_time()-started}


def run(plan=PLAN):
    spec = read(plan); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out/'results.json'; checkpoint_path = out/'futures.json'; anchor_path = out/'anchors.json'
    previous = read(str(path.relative_to(ROOT))) if path.exists() else {}
    if previous.get('all_registered_futures_refined'):
        return previous
    tests = read(str((out/'unit_tests.json').relative_to(ROOT))); assert tests['successful']
    parent = read(spec['completed_results']); assert parent['all_registered_rules_complete']
    old_plan = read(parent['execution_plan'])
    for key in ('connection_plan', 'terminal_results', 'layout', 'candidate', 'seed', 'main_origins',
            'utility_identities', 'raw_draw_powers', 'projected_draw_power', 'tail_counts',
            'complement_prefix_count', 'cached_predictive_power', 'low_constraint_probability'):
        assert spec[key] == old_plan[key]
    started = time.process_time(); rows = []; stopped = False
    bank = read(spec['completed_futures']); originals = bank['records']; records = copy.deepcopy(originals)
    if checkpoint_path.exists():
        records = read(str(checkpoint_path.relative_to(ROOT)))['records']
    _, _, models = build_connection(spec['connection_plan'])
    state, model, old_terminal, remaining = next(r for r in models if r[0]['layout'] == spec['layout'])
    np.testing.assert_array_equal(model.weights, bank['source_weights'])
    subset = next(a for a in bank['actions'] if a['kind'] == 'guarded' and a['primary'] == spec['candidate'])
    terminal = ConditionalTerminalActions([subset['shortlist']], [subset['incumbent_position']], old_terminal.protocol)
    main = [r for r in records if r['origin_identity'] in spec['main_origins']]
    pending = [r for r in main if not r.get('primary_mean_refined')]
    anchors = read(str(anchor_path.relative_to(ROOT))) if anchor_path.exists() else {}
    def checkpoint():
        save(checkpoint_path, {**bank, 'execution_plan': plan, 'records': records})
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, max(.001, spec['maximum_single_thread_cpu_seconds']-(time.process_time()-started)))
    try:
        if not anchors:
            anchors = validate_anchors(model, terminal, bank['actions'], originals, spec)
            save(anchor_path, anchors)
        assert anchors['all_anchors_equivalent']
        if pending:
            labels = np.asarray([r['labels'] for r in pending])
            cache = [[raw_fantasy_moments(field, spec['candidate'], labels[:, head]) for head, field in enumerate(pair)]
                for pair in model.components]
        for position, record in enumerate(pending):
            tick = time.process_time(); future = copy.deepcopy(model)
            scores = [[apply_raw_moments(field, spec['candidate'], cache[i][head][0][position], cache[i][head][1])
                for head, field in enumerate(pair)] for i, pair in enumerate(future.components)]
            future._record_scores(spec['candidate'], scores, 1)
            np.testing.assert_allclose(future.weights, record['shared_identity_weights'], atol=1e-15, rtol=1e-12)
            refined = copy.deepcopy(record['raw']); omitted_mass = float(sum(future.weights[i]
                for i in range(len(future.components)) if i not in spec['utility_identities']))
            maximum_replay_error = 0.
            for power in spec['raw_draw_powers']:
                nodes = qmc.Sobol(7, scramble=True, seed=spec['seed']).random_base2(power)
                old_values = []; delta = []; delta_cost = []
                for identity in spec['utility_identities']:
                    args = (model.components[identity], future.components[identity], terminal, remaining-1,
                        spec['candidate'], record['labels'], scores[identity], nodes)
                    old, old_cost, actions, _ = raw_terminal_component(*args)
                    new, new_cost, new_actions, _ = raw_terminal_component(*args, guarded_mean_position=spec['integrated_mean_position'])
                    assert actions == new_actions
                    np.testing.assert_array_equal(new[3:], old[3:])
                    old_values.append(old); delta.append(new-old); delta_cost.append(new_cost-old_cost)
                indices = [bank['actions'].index(a) for a in actions]; affected = indices[:3]
                main_weights = future.weights[spec['utility_identities']]
                value = np.asarray(record['raw'][str(power)]['values']); cost = np.asarray(record['raw'][str(power)]['costs'])
                replay = value[indices]-main_weights @ old_values
                assert replay.min() >= -1e-12 and replay.max() <= omitted_mass+1e-12
                maximum_replay_error = max(maximum_replay_error, float(np.max(abs(replay))))
                value[affected] += (main_weights @ delta)[:3]
                cost[affected] += (main_weights @ delta_cost)[:3]
                assert np.all(value >= -1e-12) and np.all(value <= 1.+1e-12)
                refined[str(power)].update(values=value.tolist(), costs=cost.tolist(),
                    primary_mean_refined_action_indices=affected, primary_mean_integrated_position=spec['integrated_mean_position'])
            record.update(raw=refined, primary_mean_refined=True, primary_refinement_cpu_seconds=time.process_time()-tick,
                subset_replay_error=maximum_replay_error)
            checkpoint()
            print('primary mean refined', sum(r.get('primary_mean_refined', False) for r in records), '/', len(main),
                'CPU', round(time.process_time()-started, 2), flush=True)
        terminal_rows = read(spec['terminal_results'])['rows']
        before, after = [next(r for r in terminal_rows if (r['layout'], r['remaining_calls'], r['power']) ==
            (spec['layout'], budget, spec['projected_draw_power'])) for budget in (remaining, remaining-1)]
        baseline = before['best_reward'], after['best_reward'], after['best_action_index']
        uniforms = qmc.Sobol(2, scramble=True, seed=spec['seed']+1).random_base2(spec['cached_predictive_power'])
        rare_mass = float(sum(model.weights[i] for i in range(len(model.components)) if i not in spec['main_origins']))
        for count in spec['tail_counts']:
            selected, weights, regions = stratified_record_weights(records, model.weights, uniforms, spec['main_origins'],
                spec['low_constraint_probability'], count, spec['complement_prefix_count'])
            current = summarize_stratified_futures(selected, weights, regions, bank['actions'], baseline, spec['raw_draw_powers'], rare_mass)
            contrasts = [np.asarray([max(r['raw'][str(p)]['values'])-r['raw'][str(p)]['values'][baseline[-1]] for r in selected])
                for p in spec['raw_draw_powers']]
            for row in current:
                row['tail_fantasies_per_main_origin'] = count
                row['weighted_absolute_terminal_contrast_difference'] = float(weights @ abs(contrasts[-1]-contrasts[0]))
                preceding = next((r for r in reversed(rows) if r['raw_draw_power'] == row['raw_draw_power']), None)
                if preceding:
                    row['nested_tail_adaptation_difference'] = abs(row['raw_adaptation_value']-preceding['raw_adaptation_value'])
            rows.extend(current)
        assert model.paid_episodes == len(state['paid_observations'])
    except TimeoutError:
        stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    result = {'execution_plan': plan, 'rows': rows, 'all_registered_futures_refined': bool(rows),
        'refined_futures': sum(r.get('primary_mean_refined', False) for r in records), 'total_existing_futures': len(records),
        'cpu_seconds': previous.get('cpu_seconds', 0.)+time.process_time()-started, 'cpu_budget_stopped': stopped,
        'actual_source_anchors_equivalent': anchors.get('all_anchors_equivalent', False),
        'absolute_terminal_contrast_gate_passed': bool(rows) and rows[-1]['weighted_absolute_terminal_contrast_difference'] <=
            spec['maximum_weighted_absolute_terminal_contrast_difference'],
        'nested_tail_gate_passed': bool(rows) and rows[-1]['nested_tail_adaptation_difference'] <= spec['maximum_nested_fantasy_adaptation_difference'],
        'identity_remainder_gate_passed': bool(rows) and all(r['identity_and_rare_origin_error_bound'] <=
            spec['maximum_identity_integration_error_bound'] for r in rows),
        'all_source_components_and_shared_weights_retained': True, 'projected_choices_frozen': True,
        'unaffected_action_count': 37, 'new_fantasies': 0, 'cost': spec['cost'],
        'independent_confirmation': False, 'decision_ranking_certified': False,
        'limitations': 'Only three queried-primary guarded actions in dominant-origin futures are numerically '
            'refined. Their old low-weight contributions and all other actions stay cached. Adjacent rules are '
            'diagnostics;earlier paid projection and independent source-prior advantage remain unresolved.'}
    save(path, result); return result


if __name__ == '__main__':
    r = run()
    print({k: r[k] for k in ('all_registered_futures_refined', 'refined_futures', 'absolute_terminal_contrast_gate_passed',
        'nested_tail_gate_passed', 'cpu_seconds', 'cpu_budget_stopped')})
