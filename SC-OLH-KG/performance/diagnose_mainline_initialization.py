"""Replay the original initial10 without new outcomes or source fitting.

Structural components are streamed once through all ten saved raw rows. Only
projected marginal moments are retained at each prefix; the original native
raw likelihood, component rule and shared identity update are unchanged.
"""
import argparse
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from performance.mainline_legacy_source_models import (
    _query_design, restore_model, restore_moment_state,
)
from performance.mainline_moment_policy import moment_state
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.legacy_function_prior import legacy_public_coordinates
from representation.legacy_observable_prior import legacy_exposure_descriptors


MODELS = {'same_source_gp': 'source_common_task_rbf',
          'current_structural_mixture': 'source_structural_mixture'}


def marginal(state):
    count = len(state['mean'])
    variance = state['cov'][np.arange(count), :, np.arange(count), :].diagonal(axis1=1, axis2=2)
    return {'mean': state['mean'], 'variance': variance, 'noise': state['noise']}


def merge(previous, mass, mean, variance, noise):
    if previous is None:
        return mass, mean, variance, noise
    old_mass, old_mean, old_variance, old_noise = previous
    total = np.logaddexp(old_mass, mass)
    a, b = np.exp(old_mass-total), np.exp(mass-total)
    return (total, a*old_mean+b*mean,
            a*old_variance+b*variance+a*b*(old_mean-mean)**2,
            a*old_noise+b*noise)


def stream_head(record, D, E, history, head, batch_size, observer=None):
    """Existing filter; optional observer reads node projections before updates."""
    law, risk = record['law'], record['risk']
    observed = list(dict.fromkeys(index for index, _ in history))
    local = {index: position for position, index in enumerate(observed)}
    n, no, p = len(D), len(observed), max(D.shape[1], E.shape[1])
    compact_n = no+p
    DF = np.vstack([D[observed], np.eye(p, D.shape[1])])
    DE = np.vstack([E[observed], np.eye(p, E.shape[1])])
    LF, LE = np.zeros((n, compact_n)), np.zeros((n, compact_n))
    LF[:, no:no+D.shape[1]], LE[:, no:no+E.shape[1]] = D, E
    unobserved = np.ones(n, dtype=bool)
    for index, position in local.items():
        LF[index] = 0.; LE[index] = 0.
        LF[index, position] = LE[index, position] = 1.
        unobserved[index] = False
    totals = [None]*(len(history)+1)
    roundoff_events = []
    for start in range(0, len(record['mean']), batch_size):
        stop = min(start+batch_size, len(record['mean']))
        cm, em = record['mean'][start:stop], record['eta_mean'][start:stop]
        means = np.stack([cm@DF.T, em@DE.T], axis=2)
        covariance = np.zeros((len(cm), compact_n, 2, compact_n, 2))
        covariance[:, :, 0, :, 0] = np.einsum('np,kpq,mq->knm', DF,
            record['covariance'][start:stop]+law['task'], DF)
        covariance[:, :, 1, :, 1] = DE@(record['eta_covariance']+risk['task'])@DE.T
        covariance[:, np.arange(no), 0, np.arange(no), 0] += law['white']
        covariance[:, np.arange(no), 1, np.arange(no), 1] += risk['white']
        field = JointFunctionRiskPosterior.from_moments(means[0], covariance[0])
        field.means = means.reshape(len(cm), 2*compact_n).copy()
        field.covariances = covariance.reshape(len(cm), 2*compact_n, 2*compact_n)
        field.component_pairs = [(0, 0)]*len(cm)
        logs = record['log_weights'][start:stop]
        mass = float(logsumexp(logs)); field.log_weights = logs-mass
        for prefix in range(len(history)+1):
            if observer is not None:
                observer(prefix, start, mass, field, LF, unobserved, local)
            m, c = field.predict(full_cov=True)
            projected_mean = np.column_stack([LF@m[:, 0], LE@m[:, 1]])
            projected_variance = np.column_stack([
                np.einsum('ni,ij,nj->n', LF, c[:, 0, :, 0], LF)+law['white']*unobserved,
                np.einsum('ni,ij,nj->n', LE, c[:, 1, :, 1], LE)+risk['white']*unobserved])
            eta_mean = field.means[:, 1::2]@LE.T
            eta_variance = np.einsum('ni,kij,nj->kn', LE, field.covariances[:, 1::2, 1::2], LE)
            eta_variance += risk['white']*unobserved
            noise = np.exp(field.log_weights)@np.exp(eta_mean+eta_variance/2)
            totals[prefix] = merge(totals[prefix], mass, projected_mean, projected_variance, noise)
            if prefix < len(history):
                index, raw = history[prefix]
                mass += field.update_batch(local[index], raw[head], None, 1)
        roundoff_events.extend([{**event, 'component_start': start, 'component_stop': stop,
                                  'head': head} for event in getattr(field, 'roundoff_events', [])])
    return totals, roundoff_events


def structural_trajectory(bundle, problem, bank, rows, batch_size=64):
    indices = {point: i for i, point in enumerate(bank)}
    history = [(indices[tuple(row['point'])], [row['raw'][0], row['raw'][1]-problem.tau]) for row in rows]
    public = (legacy_public_coordinates(np.asarray(bank)/problem.L)-bundle['mean_center'])/bundle['mean_scale']
    risk = (bundle['encoder'].transform(legacy_exposure_descriptors(problem, bank, 50))
            -bundle['risk_center'])/bundle['risk_scale']
    totals, identity_masses, head_masses, events = [None]*(len(rows)+1), [], [], []
    for identity, pair, prior_mass in zip(bundle['identities'], bundle['components'], bundle['log_weights']):
        heads = []
        for head, record in enumerate(pair):
            D = _query_design(bundle, identity, head, (public, risk), record['law'])
            E = _query_design(bundle, identity, 3 if head == 0 else 2, (public, risk), record['risk'])
            trajectory, roundoff = stream_head(record, D, E, history, head, batch_size)
            heads.append(trajectory)
            events.extend([{**event, 'identity': identity} for event in roundoff])
        masses = []
        for prefix, (f, g) in enumerate(zip(*heads)):
            mass = float(prior_mass+f[0]+g[0]); masses.append(mass)
            totals[prefix] = merge(totals[prefix], mass, np.column_stack([f[1], g[1]]),
                                    np.column_stack([f[2], g[2]]), np.column_stack([f[3], g[3]]))
        identity_masses.append(masses)
        head_masses.append([[f[0], g[0]] for f, g in zip(*heads)])
        print('identity', identity, 'prefixes', len(rows)+1, flush=True)
    identity_masses, head_masses = np.asarray(identity_masses), np.asarray(head_masses)
    snapshots = []
    for prefix, (mass, mean, variance, noise) in enumerate(totals):
        snapshots.append({'paid_prefix': prefix, 'mean': mean, 'variance': variance, 'noise': noise,
                          'identity_weights': np.exp(identity_masses[:, prefix]-mass)})
    scores = []
    for prefix in range(len(rows)):
        before = identity_masses[:, prefix]-logsumexp(identity_masses[:, prefix])
        increments = head_masses[:, prefix+1]-head_masses[:, prefix]
        scores.append({'heads': logsumexp(before[:, None]+increments, axis=0),
                       'joint': float(logsumexp(before+increments.sum(axis=1)))})
    return snapshots, scores, events


def gp_trajectory(bundle, problem, bank, rows):
    model = restore_model(bundle, MODELS['same_source_gp'], problem, bank)
    indices = {point: i for i, point in enumerate(bank)}
    snapshots, scores = [], []
    for prefix in range(len(rows)+1):
        snapshots.append({'paid_prefix': prefix, **marginal(moment_state(model)),
                          'identity_weights': model.weights.copy()})
        if prefix < len(rows):
            row = rows[prefix]
            record = model.update_raw(indices[tuple(row['point'])], [row['raw'][0], row['raw'][1]-problem.tau])
            scores.append({'heads': record['head_log_scores'][0], 'joint': record['mixture_log_score']})
    return snapshots, scores, []


def query_parity(parent, seed, arm, bank, snapshot):
    directory = parent/f'seed_{seed}'/arm
    audit = read(directory/'query_audit.json')['stages'][0]
    points = [tuple(map(int, point)) for point in np.load(directory/'query_policies.npz')['points']]
    positions = {points[index]: i for i, index in enumerate(audit['action_ids'])}
    common = [(i, positions[point]) for i, point in enumerate(bank) if point in positions]
    b, a = np.asarray(common).T
    pairs = {'objective_mean': snapshot['mean'][b, 0], 'constraint_mean': snapshot['mean'][b, 2],
             'objective_epistemic': snapshot['variance'][b, 0], 'constraint_epistemic': snapshot['variance'][b, 2],
             'constraint_aleatoric': snapshot['noise'][b, 1]}
    errors = {}
    for name, actual in pairs.items():
        expected = np.asarray(audit['score'][name])[a]
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10, err_msg=name)
        errors[name] = float(np.max(np.abs(actual-expected)))
    return {'compared_policies': len(common), 'maximum_absolute_errors': errors}


def run_case(plan, seed, arm):
    if seed not in plan['target_seeds'] or arm not in MODELS:
        raise ValueError('Unregistered initialization case')
    parent, output = ROOT/plan['parent_output'], ROOT/plan['output_dir']
    directory = output/f'seed_{seed}'/arm
    destination = directory/'initialization.json'
    with attempt(directory, 'initialization-trajectory', plan['domain'], arm) as cost:
        cost.update(seed=seed, new_truth_function_calls=0, new_source_fits=0, download_bytes=0,
                    saved_raw_rows_replayed=10, posterior_prefixes=11)
        if destination.exists():
            cost['reused'] = True
            return read(destination)
        source = parent/f'seed_{seed}'
        rows = read(source/'initialization_only/search.json')['history']
        assert len(rows) == 10
        neighbors = [tuple(map(int, point)) for point in np.load(source/'neighbors.npz')['points']]
        initial = [tuple(row['point']) for row in rows]
        bank = list(dict.fromkeys(initial+neighbors)); indices = {point: i for i, point in enumerate(bank)}
        problem = load_pickle(source/'initialization_only/search_engine.pkl').problem
        bundle = load_pickle(ROOT/plan['frozen_assets']['source_models_file'])
        with ExitStack() as stack:
            for name in ('simulate', 'true_objective', 'true_constraint_mean', 'true_sigma'):
                stack.enter_context(patch.object(problem, name, side_effect=AssertionError(name+' is outside this replay')))
            cost['source_model_restorations'] += 1
            trajectory = gp_trajectory if arm == 'same_source_gp' else structural_trajectory
            snapshots, scores, roundoff = trajectory(bundle, problem, bank, rows)
            parity = query_parity(parent, seed, arm, bank, snapshots[-1])
            # Source-zero equivalence detects an incorrect discrepancy projection;
            # initial10 comparison against saved actual queries detects update drift.
            cost['source_model_restorations'] += 1
            reference = restore_moment_state(bundle, MODELS[arm], problem, bank)
            zero_error = {}
            for key, expected in marginal(reference).items():
                np.testing.assert_allclose(snapshots[0][key], expected, rtol=1e-8, atol=1e-10, err_msg='source0 '+key)
                zero_error[key] = float(np.max(np.abs(snapshots[0][key]-expected)))
            np.savez_compressed(directory/'source0_covariance.npz', covariance=reference['cov'])
            endpoint_error = None
            if arm == 'current_structural_mixture' and seed == plan['target_seeds'][0]:
                cost['source_model_restorations'] += 1
                endpoint = restore_moment_state(bundle, MODELS[arm], problem, bank, rows)
                endpoint_error = {}
                for key, expected in {**marginal(endpoint), 'identity_weights': endpoint['weights']}.items():
                    np.testing.assert_allclose(snapshots[-1][key], expected, rtol=1e-8, atol=1e-10,
                                               err_msg='initial10 '+key)
                    endpoint_error[key] = float(np.max(np.abs(snapshots[-1][key]-expected)))
        observations = []
        for prefix, row in enumerate(rows):
            point = indices[tuple(row['point'])]; before, after = snapshots[prefix:prefix+2]
            mean, epistemic = before['mean'][point, [0, 2]], before['variance'][point, [0, 2]]
            noise = before['noise'][point]; centered = np.asarray([row['raw'][0], row['raw'][1]-problem.tau])
            observations.append({'initialization_row': prefix, 'bank_index': point, 'raw': row['raw'],
                'centered_raw': centered, 'predictive_mean': mean, 'predictive_epistemic': epistemic,
                'predictive_noise': noise, 'predictive_total_variance': epistemic+noise,
                'standardized_residual': (centered-mean)/np.sqrt(epistemic+noise),
                'log_predictive': scores[prefix], 'identity_weights_before': before['identity_weights'],
                'identity_weights_after': after['identity_weights']})
        result = {'status': 'complete', 'seed': seed, 'arm': arm, 'bank_count': len(bank),
            'initial_bank_indices': [indices[point] for point in initial],
            'neighbor_bank_indices': [indices[point] for point in neighbors],
            'new_neighbor_indices': [i for i, point in enumerate(neighbors) if point not in set(initial)],
            'heads': ['f', 'eta_f', 'g_centered', 'eta_g'], 'noise_heads': ['f', 'g'],
            'identity_names': bundle['identities'] if arm == 'current_structural_mixture' else [MODELS[arm]],
            'snapshots': snapshots, 'pre_observation': observations,
            'parity': {'source0_maximum_absolute_errors': zero_error, 'initial10_recorded_query': parity,
                       'initial10_full_restore_maximum_absolute_errors': endpoint_error},
            'roundoff_events': roundoff}
        np.savez_compressed(directory/'bank.npz', points=np.asarray(bank, dtype=np.int16))
    result['cost'] = cost
    save(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(ROOT/'performance/manifests/mainline_initialization_audit_v1.json'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--arm', choices=MODELS, required=True)
    args = parser.parse_args()
    result = run_case(read(args.manifest), args.seed, args.arm)
    print({'status': result['status'], 'seed': result['seed'], 'arm': result['arm'],
           'bank_count': result['bank_count'], 'cpu_seconds': result['cost']['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    main()
