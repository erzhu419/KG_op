"""Run the registered complete fixed-response source-risk reuse batch serially."""

import time

from performance.complete_historical_source_heads import ROOT, read, save
from performance.reuse_historical_source_risk_states import run as risk_run

PLAN = 'performance/manifests/historical_source_risk_state_reuse_batch_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    path = out / 'results.json'
    if path.exists() and read(str(path.relative_to(ROOT))).get('all_registered_cases_complete'):
        return read(str(path.relative_to(ROOT)))
    started = time.process_time(); rows = []
    for item in spec['cases']:
        if item.get('reused_risk_history'):
            history = [read(p) for p in item['reused_risk_history']]
            result = history[-1]; result_path = item['reused_risk_history'][-1]
        else:
            result = risk_run(item['plan']); history = [result]
            result_path = read(item['plan'])['output'] + '/results.json'
        row = {**item['case'], 'risk_results': result_path, 'status': result['status'],
            'conditional_risk_stationarity_passed': result['conditional_risk_stationarity_passed'],
            'final_risk_scales': result['final_risk_scales'],
            'fixed_response_scales': result['fixed_response_scales'],
            'accepted_risk_updates': sum(r['accepted_risk_updates'] for r in history),
            'sampling_batches': sum(len(r['sampling_batches']) for r in history),
            'recorded_CPU_seconds': sum(r['cpu_seconds'] for r in history),
            'completed_candidate_MCMC_sweeps': sum(r['completed_candidate_MCMC_sweeps'] for r in history),
            'final_score_summaries': result['final_score_summaries'],
            'completed_pilot_reused': bool(item.get('reused_risk_history'))}
        rows.append(row)
        phase = {'execution_plan': plan, 'rows': rows,
            'all_registered_cases_complete': len(rows) == len(spec['cases']),
            'registered_cases': len(spec['cases']),
            'conditional_risk_stationarity_cases_passed': sum(r['conditional_risk_stationarity_passed'] for r in rows),
            'accepted_risk_updates': sum(r['accepted_risk_updates'] for r in rows),
            'sampling_batches': sum(r['sampling_batches'] for r in rows),
            'recorded_risk_learning_CPU_seconds': sum(r['recorded_CPU_seconds'] for r in rows),
            'completed_candidate_MCMC_sweeps': sum(r['completed_candidate_MCMC_sweeps'] for r in rows),
            'execution_CPU_seconds': time.process_time() - started,
            'source_registry_replaced': False, 'joint_source_learning_converged': False,
            'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0,
            'server_jobs': 0, 'original_source_cases_resampled': 0,
            'no_model_adopted': True, 'independent_confirmation': False,
            'limitations': ['Fixed response covariance within each case. All final response posteriors and evidence normalizers need their own certification.',
                'Numerical stopping and sampling economy do not establish independent decision value.']}
        save(path, phase)
        print('risk reuse batch', len(rows), '/', len(spec['cases']),
            'conditional stable', phase['conditional_risk_stationarity_cases_passed'], flush=True)
    return phase


if __name__ == '__main__':
    run()
