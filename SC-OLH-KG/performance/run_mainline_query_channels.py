"""Frozen-path query-channel contrasts using saved raw observations and TS draws.

The own-point channel starts after the shared complete ten-row initialization.
Mean/noise contrasts preserve the full query covariance. The complete control
uses own-point marginals and the initialized correlation matrix; this is a
diagnostic moment control, not a new native joint posterior or optimizer run.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from acquisition.decision_backends import _EPS, _joint_gpr_draw
from performance.mainline_legacy_backend import posterior_views
from performance.mainline_legacy_source_models import restore_moment_state
from performance.run_mainline_legacy_backend_bridge import (
    ROOT, attempt, load_pickle, read, save,
)


MODELS = {'same_source_gp': 'source_common_task_rbf',
          'current_structural_mixture': 'source_structural_mixture'}


def case_dir(output, seed, arm):
    return output / f'seed_{seed}' / arm


def same(actual, expected, *, name, rtol=1e-8, atol=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.


def point_variances(state):
    n = len(state['mean'])
    return state['cov'][np.arange(n), :, np.arange(n), :].diagonal(axis1=1, axis2=2)


def compact(state, indices):
    return {key: value[indices] for key, value in (
        ('mean', state['mean']), ('variance', point_variances(state)),
        ('noise', state['noise']))}


def constrained_ts(draw_f, draw_g, noise_g, problem, action_ids):
    """The original constrained-TS branch, including its finite tie rules."""
    draw_f, draw_g = np.asarray(draw_f), np.asarray(draw_g)
    margin = draw_g + norm.ppf(1 - problem.alpha) * np.sqrt(
        np.maximum(np.asarray(noise_g), _EPS)) - problem.tau
    feasible = margin <= 0.
    if np.any(feasible):
        total = np.full(len(action_ids), -1e300, dtype=float)
        total[feasible] = -draw_f[feasible]
        mode = 'feasible_objective'
    else:
        objective_scale = max(float(np.max(np.abs(draw_f))), 1.)
        total = -margin - np.finfo(float).eps * draw_f / objective_scale
        mode = 'minimum_violation'
    selected = int(np.argmax(total))
    return {'total': total, 'sampled_objective': draw_f,
            'sampled_constraint': draw_g, 'sampled_margin': margin,
            'sampled_feasible_count': int(np.sum(feasible)),
            'sampled_feasibility_mode': mode, 'selected_index': selected,
            'selected_id': int(action_ids[selected])}


def joint_draws(bank, candidates, state, tau, rng_state):
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(rng_state)
    views, _ = posterior_views(bank, state, tau)
    # Keep both complete-bank and residual draws, in the original head order.
    return (_joint_gpr_draw(views[0], candidates, rng),
            _joint_gpr_draw(views[1], candidates, rng))


def fixed_correlation_state(anchor, own):
    result = {key: np.array(own[key], copy=True) for key in ('mean', 'cov', 'noise')}
    before, after = point_variances(anchor), point_variances(own)
    # Only these response-head blocks are read by the deployed Gaussian TS.
    for head in (0, 2):
        if np.any(before[:, head] <= 0) or np.any(after[:, head] < 0):
            raise ValueError('Nonpositive anchor or negative own response variance')
        scale = np.sqrt(after[:, head] / before[:, head])
        result['cov'][:, head, :, head] = (
            scale[:, None] * anchor['cov'][:, head, :, head] * scale[None, :])
    return result


def run_case(plan, seed, arm):
    if seed not in plan['target_seeds'] or arm not in MODELS:
        raise ValueError('Unregistered query-channel case')
    output, parent = ROOT / plan['output_dir'], ROOT / plan['parent_output']
    directory = case_dir(output, seed, arm)
    with attempt(directory, 'query-channels', plan['domain'], arm) as cost:
        cost.update(seed=seed, new_target_initial_calls=0, new_truth_function_calls=0,
                    new_source_fits=0, download_bytes=0, saved_raw_rows_replayed=0,
                    full_restorations=0, anchor_restorations=0,
                    own_point_restorations=0, query_stages_completed=0)
        destination = directory / 'channels.json'
        if destination.exists():
            cost['reused'] = True
            return read(destination)
        recovery = read(directory / 'recovery.json')
        assert recovery['status'] == 'complete' and recovery['all_raw_order_parity']
        source = case_dir(parent, seed, arm)
        search, audit = read(source / 'search.json'), read(source / 'query_audit.json')
        points = [tuple(map(int, row)) for row in
                  np.load(source / 'query_policies.npz')['points']]
        point_ids = {point: i for i, point in enumerate(points)}
        bundle = load_pickle(ROOT / plan['frozen_assets']['source_models_file'])
        stop = load_pickle(case_dir(parent, seed, 'initialization_only') / 'search_engine.pkl')
        problem, history = stop.problem, search['history']
        assert len(history) == 13 and len(audit['stages']) == len(recovery['stages']) == 3
        assert [list(point) for point, _ in stop.history] == [row['point'] for row in history[:10]]
        same([raw for _, raw in stop.history], [row['raw'] for row in history[:10]],
             name='shared initialization raw', rtol=0., atol=0.)

        def restore(bank, rows, category):
            cost['source_model_restorations'] += 1
            cost[f'{category}_restorations'] += 1
            cost['saved_raw_rows_replayed'] += len(rows)
            return restore_moment_state(bundle, MODELS[arm], problem, bank, rows)

        stages = []
        with ExitStack() as stack:
            for name in ('simulate', 'true_objective', 'true_constraint_mean', 'true_sigma'):
                stack.enter_context(patch.object(problem, name,
                    side_effect=AssertionError(f'{name} is outside frozen query diagnosis')))
            for number, (saved, recovered) in enumerate(zip(audit['stages'], recovery['stages'], strict=True)):
                count = saved['paid_prefix']
                assert count == 10 + number == recovered['paid_prefix']
                assert recovered['full_score_parity'] and recovered['candidate_order_route_selection_parity']
                action_ids = saved['scored_ids']
                assert action_ids == saved['action_ids']
                candidates = [points[i] for i in action_ids]
                prefix = history[:count]
                bank = list(dict.fromkeys([tuple(row['point']) for row in prefix] + candidates))
                positions = {point: i for i, point in enumerate(bank)}
                action_indices = np.asarray([positions[point] for point in candidates])
                anchor = restore(bank, history[:10], 'anchor')
                full = anchor if count == 10 else restore(bank, prefix, 'full')
                own = {key: np.array(anchor[key], copy=True) for key in ('mean', 'cov', 'noise')}
                for point in dict.fromkeys(tuple(row['point']) for row in prefix[10:]):
                    rows = history[:10] + [row for row in prefix[10:] if tuple(row['point']) == point]
                    small = list(dict.fromkeys([tuple(row['point']) for row in history[:10]] + [point]))
                    local = restore(small, rows, 'own_point')
                    index, local_index = positions[point], small.index(point)
                    own['mean'][index] = local['mean'][local_index]
                    own['noise'][index] = local['noise'][local_index]
                    own['cov'][index, :, index, :] = local['cov'][local_index, :, local_index, :]

                score = saved['score']
                parity = {}
                for field, actual in (
                    ('objective_mean', full['mean'][action_indices, 0]),
                    ('constraint_mean', full['mean'][action_indices, 2] + problem.tau),
                    ('objective_epistemic', point_variances(full)[action_indices, 0]),
                    ('constraint_epistemic', point_variances(full)[action_indices, 2]),
                    ('constraint_aleatoric', full['noise'][action_indices, 1]),
                ):
                    parity[field] = same(actual, score[field], name=field)
                rng_state = recovered['rng_state']
                drawn = joint_draws(bank, candidates, full, problem.tau, rng_state)
                for field, actual in zip(('sampled_objective', 'sampled_constraint'), drawn):
                    parity[field] = same(actual, score[field], name=field)

                full_f, full_g = (np.asarray(score[key]) for key in
                                  ('sampled_objective', 'sampled_constraint'))
                local_f = full_f - full['mean'][action_indices, 0] + own['mean'][action_indices, 0]
                # Both saved draw and saved constraint_mean are raw (include tau).
                local_g = full_g - full['mean'][action_indices, 2] + own['mean'][action_indices, 2]
                noise_full, noise_own = full['noise'][action_indices, 1], own['noise'][action_indices, 1]
                controls = {}
                for name, f, g, noise in (
                    ('full', full_f, full_g, noise_full),
                    ('no_cross_noise', full_f, full_g, noise_own),
                    ('no_cross_mean', local_f, local_g, noise_full),
                    ('no_cross_mean_noise', local_f, local_g, noise_own),
                ):
                    controls[name] = constrained_ts(f, g, noise, problem, action_ids)
                for field in ('total', 'sampled_margin'):
                    parity[field] = same(controls['full'][field], score[field], name=field,
                                         rtol=1e-12, atol=1e-12)
                assert controls['full']['selected_index'] == saved['selected_index']
                assert controls['full']['selected_id'] == saved['selected_id']
                assert controls['full']['sampled_feasibility_mode'] == score['sampled_feasibility_mode']
                for name, state in (
                    ('no_cross_all_fixed_correlation', fixed_correlation_state(anchor, own)),
                    ('frozen10', anchor),
                ):
                    f, g = joint_draws(bank, candidates, state, problem.tau, rng_state)
                    controls[name] = constrained_ts(f, g, state['noise'][action_indices, 1], problem, action_ids)
                if count == 10:
                    for name, control in controls.items():
                        for field in ('total', 'sampled_objective', 'sampled_constraint', 'sampled_margin'):
                            parity[f'initial_{name}_{field}'] = same(control[field], score[field], name=name)
                        assert control['selected_id'] == saved['selected_id']
                stages.append({'iteration': saved['iteration'], 'paid_prefix': count,
                    'action_ids': action_ids, 'action_sources': saved['action_sources'],
                    'bank_ids': [point_ids[point] for point in bank],
                    'observed_ids': saved['observed_ids'], 'actual_selected_id': saved['selected_id'],
                    'full': compact(full, action_indices), 'anchor': compact(anchor, action_indices),
                    'own': compact(own, action_indices), 'controls': controls,
                    'parity_max_errors': parity, 'full_draw_and_choice_parity': True})
                cost['query_stages_completed'] += 1
                print(f'channels {seed} {arm} query {number + 1}/3', flush=True)

        result = {'status': 'complete', 'seed': seed, 'arm': arm,
            'parent_output': plan['parent_output'], 'policy_bank_file': str((source / 'query_policies.npz').relative_to(ROOT)),
            'stages': stages, 'query_stage_count': len(stages),
            'all_full_draws_and_choices_match': True, 'initial10_controls_match_full': True,
            'constraint_mean_convention': 'Moment means are centered; saved scorer means and TS draws include tau.',
            'scope': 'Fixed actual histories and candidate order; no alternate queries, observations or deployments executed.',
            'control_definition': {
                'own': 'Complete shared initial10, followed only by each policy own post-initial raw observations.',
                'mean_noise': 'Original full covariance and actual TS draw residual retained.',
                'no_cross_all_fixed_correlation': 'Own-point marginals with initialized response correlations; diagnostic moment control, not native joint posterior.',
                'frozen10': 'Complete initialized means, covariance and noise; actual same-query RNG.'},
            'truth_used': False, 'completed_utc': datetime.now(timezone.utc).isoformat()}
        save(destination, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='run', choices=('run',))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--arm', choices=tuple(MODELS), required=True)
    args = parser.parse_args()
    run_case(read(args.manifest), args.seed, args.arm)


if __name__ == '__main__':
    main()
