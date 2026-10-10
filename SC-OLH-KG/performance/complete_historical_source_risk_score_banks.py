"""Complete the registered sixth-law risk-statistics batch in one serial process."""

import time

from performance.complete_historical_source_heads import ROOT, read, save
from performance.extend_historical_source_risk_score_bank import run as sample_bank, KEY

PLAN = 'performance/manifests/historical_source_risk_score_banks_v1.json'


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    started = time.process_time(); cases = read(spec['source_learning'])['evidence_rows']; rows = []
    existing = {tuple(r[k] for k in KEY): r['results'] for r in spec['reused_banks']}
    fresh = read(spec['fresh_results'])['rows']
    for case in cases:
        key = {k: case[k] for k in KEY}; identity = tuple(key.values())
        if identity in existing:
            path = existing[identity]; bank = read(path)
        else:
            if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
                break
            suffix = '_'.join(identity)
            manifest = f'performance/manifests/historical_source_risk_score_bank_batch_{suffix}.json'
            path = spec['output'] + '/bank_' + suffix + '/results.json'
            matches = any(all(r[k] == key[k] for k in KEY) for r in fresh)
            one = {'status': 'registered before fixed sixth-law source-risk statistics', 'case': key,
                'source_plan': spec['source_plan'], 'source_learning': spec['source_learning'],
                'sampler': {**spec['sampler'], 'seed': spec['existing_fresh_seed'] if matches else spec['sampler']['seed']},
                'prefix_sweeps_per_chain': 4096, 'prefix_stride': 16,
                'maximum_single_thread_cpu_seconds': spec['maximum_case_single_thread_cpu_seconds'],
                'stationarity_threshold': spec['stationarity_threshold'],
                'scope': spec['scope'], 'output': str((ROOT / path).parent.relative_to(ROOT))}
            if matches:
                one['fresh_results'] = spec['fresh_results']
            if not (ROOT / manifest).exists():
                save(ROOT / manifest, one)
            bank = sample_bank(manifest)
        for name in ('f_common_task_scales', 'eta_common_task_scales'):
            assert bank['current_candidate_parameters'][name] == case[name]
        rows.append({**key, 'results': path,
            'risk_score_precision_passed': bank['score_95_MC_halfwidth_below_stationarity_threshold'],
            'scaled_scores': bank['score_summaries']['full']['scaled_scores'],
            'scaled_95_MC_halfwidth': bank['score_summaries']['full']['scaled_95_MC_halfwidth'],
            'reused_existing_bank': identity in existing,
            'cpu_seconds': bank['cpu_seconds'],
            'new_candidate_MCMC_sweeps_executed': bank['new_candidate_MCMC_sweeps_executed']})
        result = {'execution_plan': plan, 'rows': rows, 'all_registered_cases_complete': len(rows) == 32,
            'risk_score_precision_cases_passed': sum(r['risk_score_precision_passed'] for r in rows),
            'cpu_seconds_this_execution': time.process_time() - started,
            'recorded_new_bank_CPU_seconds': sum(r['cpu_seconds'] for r in rows if not r['reused_existing_bank']),
            'new_candidate_MCMC_sweeps_executed': sum(r['new_candidate_MCMC_sweeps_executed'] for r in rows
                if not r['reused_existing_bank']),
            'original_source_cases_resampled': 0, 'KG_components_changed': 0,
            'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0,
            'server_jobs': 0, 'source_learning_converged': False, 'independent_confirmation': False}
        save(out / 'results.json', result)
        print('risk statistics', len(rows), '/32 precise', result['risk_score_precision_cases_passed'], flush=True)
    return read(str((out / 'results.json').relative_to(ROOT)))


if __name__ == '__main__':
    run()
