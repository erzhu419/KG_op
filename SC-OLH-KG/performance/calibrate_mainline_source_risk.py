"""Source-only one-dimensional calibration of the old g task covariance.

The joint mean4 source law, eta prior, means and common covariance are fixed.
Existing joint eta states are importance proposals for the new likelihood.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.mainline_joint_risk_source import (
    _source_context, read, save, joint_prior_moments, law_with_original_task_scale)
from performance.diagnose_mainline_joint_source import (
    CALIBRATION, OUTPUT as DIAGNOSTIC_OUTPUT, prepare, delete_fold, summarize_components, aggregate)

OUTPUT = 'paper_artifacts/mainline_source_risk_calibration_v1'


class TaskScaleEvidence:
    def __init__(self, law, H, labels, eta, weights, variances):
        D = np.asarray(law['source_design'])
        self.base = joint_prior_moments(law, H, D[:1], H[:1], variances)['source_covariance']
        self.task = block_diag(*[d @ law['task_covariance'] @ d.T for d in np.split(D, 4)])
        self.error = labels-law['source_mean']
        self.noise = np.exp(eta.reshape(-1, len(labels)))/3
        self.log_weights = np.log(weights)
        self.shape = eta.shape[:2]
        self.old_logs = self.evaluate(1., False)[0]

    def evaluate(self, multiplier, gradient=True):
        K = self.base+(multiplier-1)*self.task
        logs, derivatives = [], []
        for n in self.noise:
            factor = cho_factor(K+np.diag(n), lower=True)
            alpha = cho_solve(factor, self.error)
            logs.append(-.5*(len(n)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+self.error @ alpha))
            if gradient:
                derivatives.append(.5*(alpha @ self.task @ alpha-np.trace(cho_solve(factor, self.task))))
        return np.asarray(logs), np.asarray(derivatives)

    def objective(self, value):
        logs, derivatives = self.evaluate(float(value[0]))
        logw = self.log_weights+logs-self.old_logs
        logz = logsumexp(logw)
        return -float(logz)/len(self.error), np.atleast_1d(-np.exp(logw-logz) @ derivatives/len(self.error))

    def summarize(self, multiplier):
        ratio = self.evaluate(multiplier, False)[0]-self.old_logs
        logz = float(logsumexp(self.log_weights+ratio))
        weights = np.exp(self.log_weights+ratio-logz)
        split = []
        for part in (slice(0, self.shape[1]//2), slice(self.shape[1]//2, None)):
            old = self.log_weights.reshape(self.shape)[:, part]
            split.append(float(logsumexp(old+ratio.reshape(self.shape)[:, part])-logsumexp(old)))
        ess, gap = float(1/(weights @ weights)), abs(split[0]-split[1])
        return {'new_component_weights': weights.tolist(), 'log_likelihood_ratios_from_joint': ratio.tolist(),
            'log_source_evidence_ratio_from_joint': logz,
            'importance_diagnostics': {'importance_ESS': ess, 'maximum_component_mass': float(weights.max()),
                'split_log_evidence_ratios': split, 'split_log_evidence_gap': gap,
                'usable': bool(ess >= 100 and gap <= .1 and np.all(weights > 0))}}


def diagnose(context, calibration):
    old = read(f'{DIAGNOSTIC_OUTPUT}/results.json')['schemes']['leave_policy']
    pairs = [dict(p, law=law_with_original_task_scale(p['law'], r['original_task_covariance_multiplier']))
        for p, r in zip(context['pairs'], calibration['rows'])]
    states = [prepare(p, context, r, np.asarray(calibration['projection_coefficients']))
        for p, r in zip(pairs, calibration['rows'])]
    ids = [calibration['identities'].index(p['identity']) for p in pairs]
    outer = np.asarray(calibration['new_outer_source_weights'])[ids]; outer /= outer.sum()
    folds = []
    for prior_fold in old['folds']:
        heldout = np.asarray(prior_fold['heldout_rows'])
        raw = [delete_fold(s, context['labels'], context['sample_variances'], heldout) for s in states]
        result = summarize_components(raw, states, outer, heldout, context['labels'], context['sample_variances'])
        folds.append({'index': prior_fold['index'], 'heldout_rows': heldout.tolist(),
            'models': {'joint_original': prior_fold['models']['joint_original'], 'task_scale_candidate': result},
            'both_usable': prior_fold['models']['joint_original']['usable'] and result['usable']})
    paired = [f for f in folds if f['both_usable']]
    return {'folds': folds, 'paired_usable': {m: aggregate(paired, m, False)
        for m in ('joint_original', 'task_scale_candidate')},
        'all_folds_exploratory': {m: aggregate(folds, m, False)
        for m in ('joint_original', 'task_scale_candidate')},
        'limitation': 'Conditional fixed-law policy deletion: common mean, selector and variance hyperparameters previously used all48 source labels; not independent heldout coverage.'}


def fit_source():
    started = time.process_time()
    attempt = {'status': 'failed', 'new_source_labels': 0, 'new_target_calls': 0}
    try:
        context, previous = _source_context(), read(CALIBRATION)
        rows = []
        for pair, old in zip(context['pairs'], previous['rows']):
            D = np.asarray(pair['law']['source_design'])
            H = context['mean_coordinates']-D @ previous['projection_coefficients']
            evidence = TaskScaleEvidence(pair['law'], H, context['labels'], pair['eta'],
                old['new_component_weights'], old['new_variances'])
            zero = evidence.summarize(1.)
            np.testing.assert_allclose(zero['log_source_evidence_ratio_from_joint'], 0., atol=1e-12, rtol=0.)
            fit = minimize(evidence.objective, np.ones(1), jac=True, method='L-BFGS-B',
                bounds=[(0., None)], options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            summary = evidence.summarize(float(fit.x[0]))
            row = dict(old, **summary)
            cumulative = np.asarray(old['log_likelihood_ratios'])+summary['log_likelihood_ratios_from_joint']
            original_log_mass = np.log(pair['masses'])
            cumulative_split = [float(logsumexp(a+b)-logsumexp(a)) for a, b in zip(
                np.split(original_log_mass, 2, axis=1), np.split(cumulative.reshape(pair['masses'].shape), 2, axis=1))]
            row['importance_diagnostics_from_joint'] = summary['importance_diagnostics'].copy()
            row['importance_diagnostics'] = dict(summary['importance_diagnostics'],
                split_log_evidence_ratios=cumulative_split,
                split_log_evidence_gap=abs(cumulative_split[0]-cumulative_split[1]))
            row['importance_diagnostics']['usable'] = bool(row['importance_diagnostics']['usable']
                and row['importance_diagnostics']['split_log_evidence_gap'] <= .1)
            row.update(original_task_covariance_multiplier=float(fit.x[0]),
                log_likelihood_ratios=cumulative.tolist(),
                task_scale_optimizer_converged=bool(fit.success), task_scale_optimizer_message=str(fit.message),
                task_scale_optimizer_iterations=int(fit.nit), task_scale_optimizer_gradient=fit.jac.tolist(),
                task_scale_zero_boundary=bool(fit.x[0] == 0), zero_change_log_evidence_ratio=zero['log_source_evidence_ratio_from_joint'],
                log_source_evidence_ratio=old['log_source_evidence_ratio']+summary['log_source_evidence_ratio_from_joint'])
            rows.append(row)
            print(pair['name'], json.dumps({k: row[k] for k in ('original_task_covariance_multiplier',
                'log_source_evidence_ratio_from_joint', 'importance_diagnostics')}), flush=True)
        logz = {r['name']: r['log_source_evidence_ratio'] for r in rows}
        logw = np.log(previous['old_outer_source_weights'])+[logz.get('/'.join(i[1:]), 0.) for i in previous['identities']]
        outer = np.exp(logw-logsumexp(logw))
        result = dict(previous, rows=rows, new_outer_source_weights=outer.tolist(),
            parent_joint_calibration=CALIBRATION, adopted=False, status='source_candidate_pending_review',
            source_importance_usable=all(r['task_scale_optimizer_converged'] and r['importance_diagnostics']['usable'] for r in rows),
            task_scale_source_fits=2, task_scale_source_fit_objective='Maximize same raw-source marginal evidence over original g task covariance multiplier >=0; all other joint-law parameters frozen.',
            task_scale_optimizer={'method': 'L-BFGS-B', 'start': [1.], 'bounds': 'nonnegative', 'maxiter': 100,
                'ftol': 1e-12, 'gtol': 1e-8})
        save(f'{OUTPUT}/calibration.json', result)
        diagnostics = diagnose(context, result)
        save(f'{OUTPUT}/source_policy_deletion.json', diagnostics)
        result['source_policy_diagnostic'] = f'{OUTPUT}/source_policy_deletion.json'
        result['task_scale_fit_and_diagnostic_cpu_seconds'] = time.process_time()-started
        save(f'{OUTPUT}/calibration.json', result)
        print('paired_usable', json.dumps(diagnostics['paired_usable']), flush=True)
        attempt['status'] = 'completed_usable' if result['source_importance_usable'] else 'importance_unusable'
        return result
    except Exception as error:
        attempt['exception'] = repr(error)
        raise
    finally:
        attempt['cpu_seconds'] = time.process_time()-started
        path = ROOT/OUTPUT/'attempts.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream: stream.write(json.dumps(attempt)+'\n')


if __name__ == '__main__':
    fit_source()
