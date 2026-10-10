"""Refine one cached query; restore fields without refitting source priors."""

import sys
import time
import signal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.budgeted_guarded_kg import BudgetedTerminalActions
from core.raw_likelihood_lookahead import likelihood_query_lookahead
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture

PLAN = 'performance/manifests/historical_raw_lookahead_precision_v1.json'


def cpu_timeout(signum, frame):
    raise TimeoutError('registered single-thread CPU budget reached inside lookahead')


def run():
    started = time.process_time(); spec = read(PLAN)
    state = next(r for r in read(spec['states'])['rows'] if r['layout'] == spec['layout'])
    old = next(r for r in read(spec['reference'])['rows'] if r['candidate'] == spec['query'])
    posterior = state['posterior']
    model = JointResponseMixture(posterior['names'], [(JointFunctionRiskPosterior.from_moments(r['f_mean'], r['f_covariance']),
        JointFunctionRiskPosterior.from_moments(r['g_mean'], r['g_covariance'])) for r in posterior['components']])
    model.log_weights = np.log(posterior['weights']); model.paid_episodes = posterior['paid_episodes']
    guarded = [a for a in state['terminal_actions'] if a['kind'] == 'guarded']
    # Each of three allocations repeats the same shortlist/incumbent table.
    count = len(guarded)//3
    terminal = BudgetedTerminalActions([a['shortlist'] for a in guarded[:count]],
        [a['incumbent_position'] for a in guarded[:count]], state['terminal_protocol'])
    protocol = read(spec['connection_plan'])
    rows = []
    for draw_power, fantasy_power in spec['rules']:
        remaining = spec['maximum_single_thread_cpu_seconds'] - (time.process_time()-started)
        if remaining <= 0:
            break
        previous_handler = signal.signal(signal.SIGPROF, cpu_timeout)
        signal.setitimer(signal.ITIMER_PROF, remaining)
        try:
            row = likelihood_query_lookahead(model, terminal, protocol['total_target_call_cap']-model.paid_episodes,
                spec['query'], draw_power=draw_power, fantasy_power=fantasy_power, seed=protocol['seed'])
        except TimeoutError:
            signal.setitimer(signal.ITIMER_PROF, 0.)
            saved = read(spec['output']) if rows else {'execution': spec, 'rows': []}
            saved.update(stopped_at_cpu_budget=True, all_registered_rules_complete=False,
                cpu_seconds=time.process_time()-started)
            save(ROOT/spec['output'], saved)
            break
        finally:
            signal.setitimer(signal.ITIMER_PROF, 0.)
            signal.signal(signal.SIGPROF, previous_handler)
        if not rows:
            for key in ('stop_expected_reward', 'net_reward_of_paid_episode', 'likelihood_tower_drift', 'adaptation_value'):
                np.testing.assert_allclose(row[key], old[key], atol=1e-12, rtol=0.)
        rows.append(row)
        checks = {'cached_base_rule_reproduced': True,
            'last_absolute_tower_drift': abs(row['likelihood_tower_drift']),
            'last_maximum_identity_martingale_residual': max(abs(v) for v in row['model_weight_martingale_residual']),
            'last_log_evidence_max_error': row['predictive_log_evidence_max_error']}
        passed = (checks['last_absolute_tower_drift'] <= spec['precision']['maximum_absolute_tower_drift']
            and checks['last_maximum_identity_martingale_residual'] <= spec['precision']['maximum_identity_martingale_residual']
            and checks['last_log_evidence_max_error'] <= spec['precision']['maximum_log_evidence_error'])
        save(ROOT/spec['output'], {'execution': spec, 'rows': rows, 'checks': checks,
            'precision_gate_passed': bool(passed), 'all_registered_rules_complete': len(rows)==len(spec['rules']),
            'cpu_seconds': time.process_time()-started,
            'cost': {'new_prior_fits': 0, 'new_paid_updates': 0, 'new_simulator_calls': 0, 'new_verifier_calls': 0},
            'no_model_adopted': True, 'independent_confirmation': False})
        print('precision', draw_power, fantasy_power, 'net', row['net_reward_of_paid_episode'],
            'adaptation', row['adaptation_value'], 'tower', row['likelihood_tower_drift'], 'pass', passed, flush=True)


if __name__ == '__main__':
    run()
