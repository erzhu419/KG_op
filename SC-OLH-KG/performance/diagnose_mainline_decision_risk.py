"""Post-verification terminal attribution; never fit or deploy oracle edits."""

import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.stats import norm

from core.guarded_deployment_kg import conditional_safety_power
from core.legacy_joint_kg import legacy_joint_utility
from performance.run_legacy_related_inventory_source import problem_for

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'paper_artifacts/mainline_d1000_joint_risk_v1'
OUTPUT = ROOT/'paper_artifacts/mainline_decision_risk_attribution_v1'
FOCUS = (83, 102, 84, 98)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def interventions(mean, covariance, truth):
    """Eight factorial edits and four covariance-only sensitivity edits."""
    for mask in range(8):
        altered_mean, altered_cov = mean.copy(), covariance.copy()
        names = []
        if mask & 1:
            altered_mean[:, 0] = truth[:, 2]; names.append('true_g')
        if mask & 2:
            altered_mean[:, 1] = truth[:, 3]; names.append('true_eta')
        if mask & 4:
            altered_cov[:] = 0; names.append('risk_point')
        yield '+'.join(names) or 'baseline', altered_mean, altered_cov
    for name in ('zero_cross', 'g_point', 'eta_point', 'risk_cov_half'):
        altered_cov = covariance.copy()
        if name == 'risk_cov_half':
            altered_cov *= .5
        else:
            altered_cov[:, 0, 1] = altered_cov[:, 1, 0] = 0
            if name == 'g_point':
                altered_cov[:, 0, 0] = 0
            if name == 'eta_point':
                altered_cov[:, 1, 1] = 0
        yield name, mean.copy(), altered_cov


def attribute(row, truth, feasible, protocol, gp_primary):
    mean = np.asarray(row['snapshot']['mean'])
    covariance = np.asarray(row['snapshot']['point_cov'])
    baseline_primary = row['action']['primary']
    focus = sorted(set(FOCUS) | {baseline_primary, gp_primary})
    replicates = row['action']['budgets'][0]
    terminal_protocol = dict(protocol, candidate_delta=protocol['familywise_delta'],
                             verification_replicates=replicates)
    true_power = conditional_safety_power(truth[:, 2], np.exp(truth[:, 3]/2),
        replicates, protocol['chance_alpha'], protocol['familywise_delta'])
    quality = np.clip(1-truth[:, 0]/protocol['objective_reference'], 0, 1)
    safe_reward = feasible*quality*true_power
    scenarios, errors, vectors = [], {}, []
    for name, risk_mean, risk_covariance in interventions(mean[:, 2:4], covariance[:, 2:4, 2:4], truth):
        # f mean and f variance always come directly from the original snapshot.
        utility, power = legacy_joint_utility(mean[:, 0], covariance[:, 0, 0],
                                              risk_mean, risk_covariance, terminal_protocol)
        if name == 'baseline':
            for key, actual in [('utilities', utility), ('certificate_powers', power)]:
                expected = np.asarray(row[key])
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
                errors[key] = float(np.max(abs(actual-expected)))
            assert int(utility.argmax()) == baseline_primary
            np.testing.assert_allclose(safe_reward[baseline_primary],
                row['scores']['conditional_safe_reward'], atol=1e-12, rtol=1e-12)
        primary = int(utility.argmax())
        rank = np.argsort(np.argsort(-utility, kind='stable'), kind='stable')+1
        vectors.append((rank, utility, power))
        scenarios.append({'intervention': name, 'primary': primary,
            'changed_recommendation': primary != baseline_primary,
            'true_feasible': bool(feasible[primary]), 'conditional_safe_reward': float(safe_reward[primary]),
            'predicted_utility': float(utility[primary]), 'certificate_power': float(power[primary]),
            'original_primary_power': float(power[baseline_primary]),
            'original_primary_rank': int(rank[baseline_primary]),
            'focus': [[i, int(rank[i]), float(utility[i]), float(power[i])] for i in focus]})
    focus = sorted(set(focus) | {s['primary'] for s in scenarios})
    for scenario, (rank, utility, power) in zip(scenarios, vectors):
        scenario['focus'] = [[i, int(rank[i]), float(utility[i]), float(power[i])] for i in focus]
    moments = []
    for i in focus:
        gv, ev, cross = covariance[i, 2, 2], covariance[i, 3, 3], covariance[i, 2, 3]
        moments.append({'index': i, 'true_feasible': bool(feasible[i]),
            'g_mean': float(mean[i, 2]), 'true_g': float(truth[i, 2]),
            'g_signed_error': float(mean[i, 2]-truth[i, 2]), 'g_posterior_sd': float(np.sqrt(gv)),
            'eta_mean': float(mean[i, 3]), 'true_eta': float(truth[i, 3]),
            'eta_signed_error': float(mean[i, 3]-truth[i, 3]), 'eta_posterior_sd': float(np.sqrt(ev)),
            'g_eta_covariance': float(cross), 'g_eta_correlation': float(cross/np.sqrt(gv*ev)),
            'true_sigma': float(np.exp(truth[i, 3]/2)),
            'gaussian_moment_rms_sigma': float(np.exp(mean[i, 3]/2+ev/4))})
    return {'task_index': row['task_index'], 'arm': row['arm'], 'policy': row['policy'],
        'verification_replicates': replicates, 'baseline_primary': baseline_primary,
        'baseline_actual_safe_reward': row['scores']['actual_safe_reward'],
        'baseline_parity_errors': errors, 'focus_moments': moments, 'scenarios': scenarios}


def summarize(rows):
    groups = []
    for arm in sorted({r['arm'] for r in rows}):
        for policy in sorted({r['policy'] for r in rows}):
            subset = [r for r in rows if r['arm'] == arm and r['policy'] == policy]
            scenarios = []
            for j, scenario in enumerate(subset[0]['scenarios']):
                selected = [r['scenarios'][j] for r in subset]
                scenarios.append({'intervention': scenario['intervention'],
                    'primaries': [s['primary'] for s in selected],
                    'true_feasible_count': sum(s['true_feasible'] for s in selected),
                    'changed_recommendations': sum(s['changed_recommendation'] for s in selected),
                    'mean_conditional_safe_reward': float(np.mean([s['conditional_safe_reward'] for s in selected])),
                    'mean_original_primary_power': float(np.mean([s['original_primary_power'] for s in selected]))})
            groups.append({'arm': arm, 'policy': policy, 'scenarios': scenarios})
    return {'source_results': str(SOURCE/'results.json'), 'rows': len(rows), 'groups': groups,
        'maximum_baseline_parity_error': max(v for r in rows for v in r['baseline_parity_errors'].values()),
        'scope': 'Post-verification terminal diagnostics on 16 frozen snapshots of four already opened geometries. '
            'All f moments, candidate sets, budgets and historical 9-eta/12-SD terminal quadrature stay fixed. '
            'Oracle means edit every candidate only for attribution. These are not fitted models, '
            'new optimizer arms, adaptive trajectories or independently verified rewards.',
        'interpretation': 'risk_point removes both latent g/eta posterior variances and their covariance; '
            'it retains aleatoric episode noise exp(eta). g_point/eta_point remove one latent variance '
            'and the cross term; zero_cross removes correlation alone. These are sensitivity endpoints, '
            'not estimates of true epistemic covariance. Mean-only oracle edits preserve uncertainty. '
            'Conditional safe rewards use the unchanged historical scorer; actual rewards exist only for baseline.',
        'focus_columns': ['global_index', 'utility_rank', 'utility', 'certificate_power']}


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ledger = OUTPUT/'attempts.jsonl'
    previous_attempts = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    started = time.process_time(); attempt = {}
    try:
        results, frozen = read(SOURCE/'results.json'), read(SOURCE/'frozen_inputs.json')
        assert results['all_recommendations_frozen_before_verification']
        assert results['all_verification_finished_before_truth_scoring']
        assert len(results['rows']) == 16
        protocol, truths = frozen['terminal_protocol'], {}
        for task_index, task in enumerate(frozen['tasks']):
            problem = problem_for(frozen['problem_plan'], task)
            truth = []
            for policy in frozen['bank']:
                policy = tuple(policy); sigma = problem.true_sigma(policy)
                truth.append([problem.true_objective(policy), np.log(sigma[0]**2),
                    problem.true_constraint_mean(policy)-problem.tau, np.log(sigma[1]**2)])
            truth = np.asarray(truth)
            feasible = truth[:, 2]+norm.ppf(1-protocol['chance_alpha'])*np.exp(truth[:, 3]/2) <= 0
            truths[task_index] = truth, feasible
        rows = []
        for row in results['rows']:
            gp = next(r for r in results['rows'] if r['task_index'] == row['task_index']
                and r['policy'] == row['policy'] and r['arm'] == 'source_common_task_rbf')
            rows.append(attribute(row, *truths[row['task_index']], protocol, gp['action']['primary']))
        save(OUTPUT/'details.json', {'rows': rows})
        summary = summarize(rows)
        summary['cost'] = {'cpu_seconds': time.process_time()-started, 'new_physical_calls': 0,
            'new_source_fits': 0, 'model_restorations': 0, 'opened_task_truth_recomputations': 416,
            'terminal_evaluations': len(rows)*12}
        summary['cost']['total_attempt_cpu_seconds'] = sum(a['cpu_seconds'] for a in previous_attempts)+time.process_time()-started
        summary['cost']['previous_failed_attempts'] = sum(a['status'] == 'failed' for a in previous_attempts)
        save(OUTPUT/'summary.json', summary)
        attempt['status'] = 'completed'
        print(json.dumps({'cost': summary['cost'], 'parity': summary['maximum_baseline_parity_error'],
            'source_groups': [g for g in summary['groups'] if g['arm'] == 'source_structural_mixture']}), flush=True)
    except Exception:
        attempt.update(status='failed', traceback=traceback.format_exc()); raise
    finally:
        attempt['cpu_seconds'] = time.process_time()-started
        with ledger.open('a') as stream:
            stream.write(json.dumps(attempt)+'\n')


if __name__ == '__main__':
    run()
