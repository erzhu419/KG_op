"""Fixed-law source deletion diagnostics; no target data or source-law refits.

The old 1024 eta states propose each training posterior. Both held-out mean
and variance likelihoods are removed. Weak deletion overlap is reported, not
repaired by fitting to a held-out observation. Two GLS common-mean hypotheses
use training rows only and update eta masses by their training likelihood.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.mainline_joint_risk_source import _source_context, read, save
from representation.physical_risk_prior import log_sample_variance_density

CALIBRATION = 'paper_artifacts/mainline_joint_risk_source_v1/calibration.json'
OUTPUT = 'paper_artifacts/mainline_joint_source_deletion_v1'
MODELS = ('joint_original', 'gls_original_g6_mean', 'gls_augmented_g6_mean4_mean')


def gaussian_log_density(error, covariance):
    factor = cho_factor(covariance, lower=True)
    return float(-.5 * (len(error)*np.log(2*np.pi)
        + 2*np.log(np.diag(factor[0])).sum() + error @ cho_solve(factor, error)))


def normalize(log_weights, shape=None):
    shape = (4, len(log_weights)//4) if shape is None else shape
    logz = float(logsumexp(log_weights))
    weights = np.exp(log_weights-logz)
    halves = log_weights.reshape(shape)
    splits = [float(logsumexp(h)) for h in np.split(halves, 2, axis=1)]
    gap = abs(splits[0]-splits[1])
    ess = float(1/(weights @ weights))
    return weights, logz, {'ESS': ess, 'maximum_mass': float(weights.max()),
        'split_log_mass_gap': gap, 'usable': bool(ess >= 100 and gap <= .1)}


def prepare(pair, context, calibration_row, projection):
    law = pair['law']
    D = np.asarray(law['source_design'])
    H = context['mean_coordinates']-D @ projection
    ac, at = calibration_row['new_variances']
    K = np.asarray(law['source_covariance'])+ac*H @ H.T
    for indices in np.split(np.arange(48), 4):
        K[np.ix_(indices, indices)] += at*H[indices] @ H[indices].T
    eta = pair['eta'].reshape(-1, 48)
    noise = np.exp(eta)/3
    error = context['labels']-law['source_mean']
    F = np.column_stack([D, H])
    precision, alpha, information, score, pf = [], [], [], [], []
    for n in noise:
        factor = cho_factor(K+np.diag(n), lower=True)
        P = cho_solve(factor, np.eye(48))
        a, p = P @ error, P @ F
        precision.append(P); alpha.append(a); pf.append(p)
        information.append(F.T @ p); score.append(F.T @ a)
    return {'K': K, 'F': F, 'error': error, 'eta': eta, 'noise': noise,
        'precision': np.asarray(precision), 'alpha': np.asarray(alpha), 'pf': np.asarray(pf),
        'information': np.asarray(information), 'score': np.asarray(score),
        'log_mass': np.log(calibration_row['new_component_weights'])}


def delete_fold(state, labels, sample_variances, heldout):
    """Precision-block deletion, equivalent to training-only conditioning."""
    n, q = len(state['noise']), len(heldout)
    means, variances, scores = np.empty((n, q)), np.empty((n, q)), np.empty(n)
    information = np.empty_like(state['information'])
    linear = np.empty_like(state['score'])
    mean_loading = np.empty((n, q, state['F'].shape[1]))
    for j, P in enumerate(state['precision']):
        covariance = cho_solve(cho_factor(P[np.ix_(heldout, heldout)], lower=True), np.eye(q))
        error = covariance @ state['alpha'][j, heldout]
        means[j] = labels[heldout]-error
        variances[j] = np.diag(covariance)-state['noise'][j, heldout]
        scores[j] = gaussian_log_density(error, covariance)
        scores[j] += np.sum(log_sample_variance_density(
            state['eta'][j, heldout], sample_variances[heldout], 3))
        loading = state['pf'][j, heldout]
        mean_loading[j] = covariance @ loading
        information[j] = state['information'][j]-loading.T @ mean_loading[j]
        linear[j] = state['score'][j]-loading.T @ error
    log_deleted = state['log_mass']-scores
    weights, logz, overlap = normalize(log_deleted)
    return {'means': means, 'latent_variances': variances,
        'log_weights': log_deleted-logz, 'deletion_log_evidence_ratio': logz,
        'deletion_overlap': overlap, 'weights': weights,
        'mean_loading': mean_loading, 'information': information, 'score': linear}


def gls_candidate(state, fold, heldout, columns):
    train = np.setdiff1d(np.arange(48), heldout)
    expected_noise = fold['weights'] @ state['noise'][:, train]
    factor = cho_factor(state['K'][np.ix_(train, train)]+np.diag(expected_noise), lower=True)
    F = state['F'][train, :columns]
    precision_F = cho_solve(factor, F)
    information = F.T @ precision_F
    beta = np.linalg.pinv(information, rcond=1e-10) @ (F.T @ cho_solve(factor, state['error'][train]))
    delta = np.zeros(state['F'].shape[1]); delta[:columns] = beta
    ratios = fold['score'] @ delta-.5*np.einsum('i,kij,j->k', delta, fold['information'], delta)
    weights, logz, overlap = normalize(fold['log_weights']+ratios)
    return {'means': fold['means']+fold['mean_loading'] @ delta,
        'latent_variances': fold['latent_variances'], 'weights': weights,
        'deletion_log_evidence_ratio': fold['deletion_log_evidence_ratio']+logz,
        'deletion_overlap': fold['deletion_overlap'], 'candidate_overlap': overlap,
        'training_GLS_rank': int(np.linalg.matrix_rank(information)),
        'training_GLS_columns': columns, 'common_mean_increment': delta.tolist(),
        'training_log_evidence_ratio': logz}


def summarize_components(rows, states, outer_weights, heldout, labels, sample_variances):
    outer_log = np.log(outer_weights)+[r['deletion_log_evidence_ratio'] for r in rows]
    outer = np.exp(outer_log-logsumexp(outer_log))
    weights = np.concatenate([o*r['weights'] for o, r in zip(outer, rows)])
    means = np.vstack([r['means'] for r in rows])
    latent = np.vstack([r['latent_variances'] for r in rows])
    noise = np.vstack([s['noise'][:, heldout] for s in states])
    eta = np.vstack([s['eta'][:, heldout] for s in states])
    mean = weights @ means
    latent_variance = weights @ (latent+(means-mean)**2)
    observation_noise = weights @ noise
    variance = latent_variance+observation_noise
    observed, s2 = labels[heldout], sample_variances[heldout]
    total = latent+noise
    pit = weights @ ndtr((observed-means)/np.sqrt(total))
    variance_pit = weights @ -np.expm1(-s2/(3*noise))
    mean_logs = -.5*(np.log(2*np.pi*total)+(observed-means)**2/total)
    variance_logs = log_sample_variance_density(eta, s2, 3)
    return {'mean': mean.tolist(), 'latent_variance': latent_variance.tolist(),
        'observation_mean_noise': observation_noise.tolist(), 'total_mean_variance': variance.tolist(),
        'observed_batch_means': observed.tolist(), 'observed_sample_variances': s2.tolist(),
        'predicted_sample_variances': (3*observation_noise).tolist(),
        'mean_PIT': pit.tolist(), 'sample_variance_PIT': variance_pit.tolist(),
        'mean_log_predictive_density': logsumexp(np.log(weights)[:, None]+mean_logs, axis=0).tolist(),
        'sample_variance_log_predictive_density': logsumexp(np.log(weights)[:, None]+variance_logs, axis=0).tolist(),
        'identity_weights': outer.tolist(),
        'usable': all(r['deletion_overlap']['usable'] and r.get('candidate_overlap', {'usable': True})['usable'] for r in rows),
        'identity_diagnostics': [{k: r[k] for k in ('deletion_overlap', 'candidate_overlap',
            'common_mean_increment', 'training_GLS_rank', 'training_GLS_columns', 'training_log_evidence_ratio') if k in r} for r in rows]}


def aggregate(folds, model, usable_only):
    rows = [f['models'][model] for f in folds if not usable_only or f['models'][model]['usable']]
    if not rows:
        return {'rows': 0, 'folds': 0}
    get = lambda k: np.concatenate([r[k] for r in rows])
    error = get('mean')-get('observed_batch_means')
    ratio = get('observed_sample_variances')/get('predicted_sample_variances')
    return {'rows': len(error), 'folds': len(rows), 'mse': float(np.mean(error**2)),
        'mean_bias_prediction_minus_observation': float(error.mean()),
        'standardized_mean_error_rms': float(np.sqrt(np.mean(error**2/get('total_mean_variance')))),
        'mean_latent_variance': float(get('latent_variance').mean()),
        'mean_observation_noise': float(get('observation_mean_noise').mean()),
        'mean_predictive_95pct_coverage': float(np.mean((get('mean_PIT') >= .025)&(get('mean_PIT') <= .975))),
        'mean_PIT_average': float(get('mean_PIT').mean()),
        'variance_PIT_average': float(get('sample_variance_PIT').mean()),
        'variance_predictive_95pct_coverage': float(np.mean((get('sample_variance_PIT') >= .025)&(get('sample_variance_PIT') <= .975))),
        'mean_observed_to_predicted_sample_variance_ratio': float(ratio.mean()),
        'mean_log_observed_to_predicted_sample_variance_ratio': float(np.log(ratio).mean()),
        'mean_log_predictive_density': float(get('mean_log_predictive_density').mean()),
        'sample_variance_log_predictive_density': float(get('sample_variance_log_predictive_density').mean())}


def run():
    started = time.process_time()
    attempt = {'status': 'failed', 'target_calls': 0, 'source_law_refits': 0}
    try:
        context, calibration = _source_context(), read(CALIBRATION)
        states = [prepare(p, context, r, np.asarray(calibration['projection_coefficients']))
                  for p, r in zip(context['pairs'], calibration['rows'])]
        ids = [calibration['identities'].index(p['identity']) for p in context['pairs']]
        outer = np.asarray(calibration['new_outer_source_weights'])[ids]
        outer /= outer.sum()
        groups = []
        for i, x in enumerate(context['mean_coordinates']):
            match = next((g for g in groups if np.allclose(x, context['mean_coordinates'][g[0]], rtol=0., atol=1e-12)), None)
            if match is None: groups.append([i])
            else: match.append(i)
        schemes = {'leave_task': [list(x) for x in np.split(np.arange(48), 4)], 'leave_policy': groups}
        output = {'calibration': CALIBRATION, 'source_inputs': context['source_inputs'],
            'source_tasks': context['source_tasks'], 'source_rows': 48,
            'source_policy_groups': len(groups), 'retained_identity_names': [p['name'] for p in context['pairs']],
            'method': 'Fixed augmented Gaussian source kernels; exact likelihood deletion on existing eta proposals; GLS mean-only hypotheses refit on training rows and eta masses reweighted by the changed training likelihood.',
            'sample_variance_density_coordinate': 'Bias-corrected log sample variance (the existing log_sample_variance_density includes its Jacobian).',
            'limitations': ['Selector, original prior parameters and two augmented variances used all48 source labels; this is fixed-law deletion diagnosis, not nested heldout source-law training.',
                'Only the two dominant g identities are diagnosed; this is not reconstruction of the full eight-identity deletion posterior.',
                'ESS<100 or split log-mass gap>0.1 marks a fold unusable; its numbers remain exploratory.',
                'GLS uses deletion-posterior expected observation variance; covariance parameters stay fixed and uncertainty in empirical GLS mean fitting is not added.'],
            'schemes': {}}
        for scheme, indices in schemes.items():
            folds = []
            for index, heldout_list in enumerate(indices):
                heldout = np.asarray(heldout_list)
                raw = [delete_fold(s, context['labels'], context['sample_variances'], heldout) for s in states]
                candidates = {MODELS[0]: raw}
                for model, columns in zip(MODELS[1:], (7, 11)):
                    candidates[model] = [gls_candidate(s, r, heldout, columns) for s, r in zip(states, raw)]
                folds.append({'index': index, 'heldout_rows': heldout.tolist(), 'models': {
                    name: summarize_components(rows, states, outer, heldout, context['labels'], context['sample_variances'])
                    for name, rows in candidates.items()}})
            summary = {name: {'all_folds_exploratory': aggregate(folds, name, False),
                'usable_folds': aggregate(folds, name, True)} for name in MODELS}
            output['schemes'][scheme] = {'folds': folds, 'aggregate': summary}
            print(scheme, json.dumps(summary), flush=True)
        output['cost'] = {'cpu_seconds': time.process_time()-started, 'GLS_fits': 4*sum(map(len, schemes.values())),
            'source_law_refits': 0, 'eta_sampler_sweeps': 0, 'new_source_labels': 0, 'new_simulator_calls': 0, 'target_outcomes_used': False}
        save(f'{OUTPUT}/results.json', output)
        attempt['status'] = 'completed'
        return output
    finally:
        attempt['cpu_seconds'] = time.process_time()-started
        path = ROOT/OUTPUT/'attempts.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream: stream.write(json.dumps(attempt)+'\n')


if __name__ == '__main__':
    run()
