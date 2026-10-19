"""Frozen-history FactorShock posterior attribution; no optimizer or refit."""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from acquisition.decision_backends import feasible_first_terminal_order
from performance.mainline_legacy_source_models import restore_model
from performance.mainline_moment_policy import moment_state
from performance.mainline_point_only import PointOnlyResponseMixture
from performance.run_mainline_legacy_backend_bridge import load_pickle, read, save

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'paper_artifacts/mainline_factorshock_diagnosis_v1'
ARMS = ('same_source_gp', 'current_structural_mixture')


def compact(state):
    n = len(state['mean'])
    return {'mean': state['mean'], 'noise': state['noise'],
            'epistemic_variance': np.stack([state['cov'][i, :, i, :] for i in range(n)])}


def decision(snapshot, observed, alpha=.05):
    ids = np.asarray(list(dict.fromkeys(observed)), dtype=int)
    mean = np.asarray(snapshot['mean']); noise = np.asarray(snapshot['noise'])
    epistemic = np.asarray(snapshot['epistemic_variance'])[:, 2, 2]
    margin = mean[:, 2]+norm.ppf(1-alpha)*np.sqrt(noise[:, 1])
    standardized = margin/np.sqrt(np.maximum(epistemic, 1e-12))
    probability = norm.cdf(standardized)
    result = {'chance_margin': margin, 'standardized_margin': standardized,
              'violation_probability': probability, 'observed_indices': ids}
    if len(ids):
        order, mode = feasible_first_terminal_order(mean[ids, 0], probability[ids],
                                                    maximum_violation_probability=.5)
        result.update(selected=int(ids[order[0]]), mode=mode,
                      ranked_indices=ids[order], predicted_feasible_count=int(np.sum(probability[ids] <= .5)))
    return result


def score_truth(output, inputs, cost):
    path = output/'truth.json'
    if path.exists():
        cost['truth_reused'] = True
        return read(path)
    engine = load_pickle(ROOT/inputs['search_engine_files']['legacy_native_v69'])
    rows = []
    for point in inputs['bank']:
        f = engine.problem.true_objective(point); cost['truth_function_calls'] += 1
        g = engine.problem.true_constraint_mean(point)-engine.problem.tau; cost['truth_function_calls'] += 1
        sigma = np.asarray(engine.problem.true_sigma(point)); cost['truth_function_calls'] += 1
        rows.append({'f': f, 'g_centered': g, 'episode_sigma': sigma,
                     'chance_margin': g+norm.ppf(1-engine.problem.alpha)*sigma[1]})
    result = {'rows': rows, 'unique_scored_policies': len(rows)}
    save(path, result)
    return result


def replay(output, inputs, arm, cost):
    path = output/f'{arm}.json'
    if path.exists():
        cost['reused'] = True
        return read(path)
    engine = load_pickle(ROOT/inputs['search_engine_files'][arm])
    bundle = load_pickle(ROOT/inputs['source_models_file'])
    bank, history = inputs['bank'], inputs['histories'][arm]
    model_name = {'same_source_gp': 'source_common_task_rbf',
                  'current_structural_mixture': 'source_structural_mixture'}[arm]
    result = {'arm': arm, 'snapshots': [], 'updates': [], 'point_only_from_zero': [],
              'point_only_after_initial10': [], 'native_posterior_changed': False}
    with patch.object(engine.problem, 'simulate', side_effect=AssertionError('No new simulation in replay')):
        full = restore_model(bundle, model_name, engine.problem, bank)
        local_zero = PointOnlyResponseMixture.from_model(full)
        initial = compact(moment_state(full))
        result['snapshots'].append({'paid': 0, **initial, 'weights': full.weights.copy(),
                                    'decision': decision(initial, [])})
        local_online = None
        observed = []
        for step, row in enumerate(history, start=1):
            candidate = row['bank_index']
            raw = np.array(row['raw'], dtype=float); raw[1] -= engine.problem.tau
            before = result['snapshots'][-1]
            full.update_raw(candidate, raw)
            local_zero.update_raw(candidate, raw)
            if local_online is not None:
                local_online.update_raw(candidate, raw)
            state = compact(moment_state(full)); observed.append(candidate)
            result['snapshots'].append({'paid': step, **state, 'weights': full.weights.copy(),
                                        'decision': decision(state, observed)})
            result['updates'].append({'paid': step, 'candidate': candidate, 'raw_centered': raw,
                'local_mean_before': before['mean'][candidate, [0, 2]],
                'local_mean_after': state['mean'][candidate, [0, 2]],
                'local_noise_before': before['noise'][candidate],
                'local_noise_after': state['noise'][candidate],
                'cross_mean_changes': state['mean'][:, [0, 2]]-before['mean'][:, [0, 2]]})
            if step == 10:
                local_online = PointOnlyResponseMixture.from_model(full)
                shared = local_online.moment_state()
                np.testing.assert_allclose(shared['mean'], state['mean'], atol=2e-11, rtol=2e-11)
                np.testing.assert_allclose(shared['noise'], state['noise'], atol=2e-11, rtol=2e-11)
            if step in (10, 11, 12, 13):
                for branch, model in [('point_only_from_zero', local_zero),
                                      ('point_only_after_initial10', local_online)]:
                    local = compact(model.moment_state())
                    result[branch].append({'paid': step, **local,
                        'decision': decision(local, observed), 'weights': model.weights})
                unseen = sorted(set(range(len(bank)))-set(observed))
                np.testing.assert_allclose(local_zero.moment_state()['mean'][unseen],
                                           initial['mean'][unseen], atol=2e-11, rtol=0)
                if step > 10:
                    not_updated = sorted(set(range(len(bank)))-{h['bank_index'] for h in history[10:step]})
                    np.testing.assert_allclose(local_online.moment_state()['mean'][not_updated],
                        result['snapshots'][10]['mean'][not_updated], rtol=0, atol=2e-11)
            print(arm, 'replayed', step, 'selected', result['snapshots'][-1]['decision']['selected'], flush=True)
        selected = result['snapshots'][-1]['decision']['selected']
        if selected != inputs['primary_bank_indices'][arm]:
            raise AssertionError('Full replay does not reproduce saved primary')
        previous = read(ROOT/'paper_artifacts/mainline_legacy_backend_bridge_v1/diagnostic_replay.json')
        old = next(r for r in previous['rows'] if r['family']==inputs['domain'] and r['arm']==arm)
        old = old['initial_and_final_actions']['final']
        new = result['snapshots'][-1]
        np.testing.assert_allclose([new['mean'][selected, 0], new['mean'][selected, 2],
            new['decision']['violation_probability'][selected]],
            [old['objective_mean'], old['constraint_mean_centered'], old['posterior_violation_probability']],
            atol=1e-8, rtol=1e-8)
        result['final_primary_and_moment_parity'] = True
        result['point_only_untouched_marginal_parity'] = True
        result['paid_rows_reused'] = full.paid_episodes
    save(path, result)
    return result


def decompose(output, inputs):
    truth = read(output/'truth.json')['rows']
    native = read(output/'native_views.json')['arms']
    true_g = np.array([r['g_centered'] for r in truth])
    true_f = np.array([r['f'] for r in truth])
    true_noise = np.array([r['episode_sigma'][1]**2 for r in truth])
    rows = []
    for arm in ARMS:
        result = read(output/f'{arm}.json')
        final = result['snapshots'][-1]
        observed = final['decision']['observed_indices']
        old = native[arm]['aggregate_moments']
        interventions = {}
        for name in ['unchanged', 'true_objective_mean', 'true_constraint_mean', 'true_constraint_noise',
                     'true_constraint_mean_and_noise', 'common_constraint_epistemic',
                     'old_constraint_mean', 'old_constraint_noise', 'old_constraint_epistemic',
                     'point_only_zero_constraint_noise', 'point_only_online_constraint_noise']:
            snapshot = {k: np.array(final[k], dtype=float) for k in ('mean','noise','epistemic_variance')}
            if name == 'true_objective_mean': snapshot['mean'][:, 0] = true_f
            if name in {'true_constraint_mean', 'true_constraint_mean_and_noise'}: snapshot['mean'][:, 2] = true_g
            if name in {'true_constraint_noise', 'true_constraint_mean_and_noise'}: snapshot['noise'][:, 1] = true_noise
            if name == 'common_constraint_epistemic':
                snapshot['epistemic_variance'][:, 2, 2] = np.median(snapshot['epistemic_variance'][observed, 2, 2])
            if name == 'old_constraint_mean': snapshot['mean'][:, 2] = old['constraint_mean_raw']
            if name == 'old_constraint_noise': snapshot['noise'][:, 1] = old['constraint_episode_variance']
            if name == 'old_constraint_epistemic': snapshot['epistemic_variance'][:, 2, 2] = old['constraint_epistemic_variance']
            if name.startswith('point_only_'):
                branch = 'point_only_from_zero' if name == 'point_only_zero_constraint_noise' else 'point_only_after_initial10'
                snapshot['noise'][:, 1] = np.array(result[branch][-1]['noise'])[:, 1]
            action = decision(snapshot, observed)
            # Gaussian CDF can round distinct upper-tail risks to the same 1.0.
            # Keep the frozen decision; identify when that tie changes its ranking.
            least_z = min(observed, key=lambda i: action['standardized_margin'][i])
            interventions[name] = {**action, 'selected_truth': truth[action['selected']],
                'minimum_standardized_margin_index': int(least_z),
                'cdf_saturation_changes_selection': bool(
                    action['predicted_feasible_count'] == 0 and action['selected'] != least_z)}
        never_observed = sorted(set(range(len(truth)))-set(observed))
        unsampled = []
        for branch in ['snapshots', 'point_only_from_zero', 'point_only_after_initial10']:
            for state in result[branch]:
                if state['paid'] not in (10, 11, 12, 13):
                    continue
                ids = never_observed
                g = np.array(state['mean'])[ids, 2]
                sigma = np.sqrt(np.array(state['noise'])[ids, 1])
                unsampled.append({'branch': branch, 'paid': state['paid'],
                    'constraint_mean_mse': float(np.mean((g-true_g[ids])**2)),
                    'constraint_noise_sd_mse': float(np.mean((sigma-np.sqrt(true_noise[ids]))**2))})
        rows.append({'arm': arm, 'interventions': interventions,
            'old_posterior_same_paid_history_action': native[arm]['observed_selected_bank_id'],
            'point_only_zero_action': result['point_only_from_zero'][-1]['decision'],
            'point_only_online_action': result['point_only_after_initial10'][-1]['decision'],
            'never_observed_bank_indices': never_observed, 'unsampled_prediction_errors': unsampled})
    summary = {'rows': rows, 'scope': 'Fixed-path, observed-only retrospective substitutions; no new policy execution, certification or model selection.'}
    save(output/'decomposition.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('truth', 'replay', 'decompose'))
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    inputs = read(args.output/'inputs.json')
    cost = {'stage': args.stage, 'arm': args.arm, 'status': 'failed', 'truth_function_calls': 0,
            'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_fits': 0, 'download_bytes': 0}
    started, wall = time.process_time(), time.perf_counter()
    try:
        if args.stage == 'truth': score_truth(args.output, inputs, cost)
        elif args.stage == 'replay':
            if args.arm is None: parser.error('replay requires --arm')
            replay(args.output, inputs, args.arm, cost)
        else: decompose(args.output, inputs)
        cost['status'] = 'complete'
    except BaseException as error:
        cost['error'] = repr(error)
        raise
    finally:
        cost.update(cpu_seconds=time.process_time()-started, wall_seconds=time.perf_counter()-wall)
        name = args.arm or args.stage
        with (args.output/f'{name}_attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(cost)+'\n')
    print(json.dumps(cost), flush=True)


if __name__ == '__main__':
    main()
