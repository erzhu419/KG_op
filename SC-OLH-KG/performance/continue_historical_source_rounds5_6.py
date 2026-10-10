"""Registered serial source iterations using the existing learning/precision stages."""

import time

from performance.complete_historical_source_heads import ROOT, read, save
from performance.continue_historical_raw_source_learning import run as learn
from performance.retain_historical_stationary_source_heads import run as retain
from performance.recondition_historical_changed_candidates import run as condition
from performance.complete_historical_source_learning_round2 import run as complete
from performance.replay_historical_source_learning_round3 import run as replay

PLAN = 'performance/manifests/historical_source_learning_rounds5_6_v1.json'
KEY = ('layout', 'selector', 'support', 'treatment', 'head')


def register(path, spec):
    if not (ROOT / path).exists():
        save(ROOT / path, spec)
    return path


def run(plan=PLAN):
    batch = read(plan); (ROOT / batch['output']).mkdir(parents=True, exist_ok=True)
    started = time.process_time(); summaries = []
    previous_learning, previous_precision = batch['previous_learning'], batch['previous_precision']
    learning_template = read('performance/manifests/historical_raw_source_amplitude_learning_v4.json')
    retain_template = read('performance/manifests/historical_stationary_source_continuation_v2.json')
    fresh_template = read('performance/manifests/historical_changed_candidate_reconditioning_v4.json')
    local_template = read('performance/manifests/historical_source_learning_round4_completion_v1.json')
    replay_template = read('performance/manifests/historical_source_learning_round4_replay.json')
    status = 'registered source iterations complete'
    for item in batch['iterations']:
        n = item['round']; base = f'paper_artifacts/historical_source_learning_round{n}'
        if time.process_time() - started >= batch['maximum_single_thread_cpu_seconds']:
            status = 'registered batch CPU budget reached'; break
        proposed = f'paper_artifacts/historical_raw_source_amplitude_learning_v{n}'
        risk_statistics = ({'risk_sufficient_statistics': item['risk_sufficient_statistics']}
            if 'risk_sufficient_statistics' in item else {})
        learning_plan = register(f'performance/manifests/historical_raw_source_amplitude_learning_v{n}.json',
            {**learning_template, 'status': f'registered before source iteration{n}',
                'previous_learning': previous_learning, 'previous_precision': previous_precision,
                'output': proposed, 'scope': batch['scope'], **risk_statistics})
        proposal = learn(learning_plan)
        failures = [r for r in proposal['rows'] if not r['source_Mstep_gate_passed']]
        if not proposal['all_registered_cases_complete'] or any(r['head'] not in ('f', 'g')
                or r['consecutive_source_stationarity_iterations'] < 2 for r in failures):
            status = f'iteration{n} source Q requires precision work'
            summaries.append({'round': n, 'learning_results': proposed + '/results.json',
                'status': status, 'failed_Q_heads': [{**{k:r[k] for k in KEY},
                    'Q_improvements':r['full_and_split_Q_improvements']} for r in failures]})
            break
        chosen_plan = register(f'performance/manifests/historical_source_learning_round{n}_selection.json',
            {**retain_template, 'status': f'registered before iteration{n} stationary selection',
                'previous_learning': previous_learning, 'previous_precision': previous_precision,
                'proposed_learning': proposed + '/results.json',
                'proposed_posterior_precision': proposed + '/posterior_precision.json',
                'stationary_cases': [{k:r[k] for k in KEY} for r in failures], 'output': base,
                'scope': 'Retain only failed mean-head proposals after two registered stationary iterations;preserve failures and require positive full/split joint Q. Reuse other candidates.'})
        step = retain(chosen_plan); post = read(base + '/posterior_precision.json')
        fresh_output = f'paper_artifacts/historical_changed_candidate_reconditioning_v{n}'
        fresh_plan = register(f'performance/manifests/historical_changed_candidate_reconditioning_v{n}.json',
            {**fresh_template, 'status': f'registered before iteration{n} actual HOLD conditioning',
                'learning_results': base + '/results.json', 'previous_posterior_precision': base + '/posterior_precision.json',
                'cases': [{k:r[k] for k in (*KEY, 'source_case')} for r in post['rows']
                    if not r['new_source_posterior_precision_passed']],
                'sampler': {**fresh_template['sampler'], 'seed': item['seed']}, 'output': fresh_output,
                'scope': 'Condition only actual current-law HOLD cases;reuse passed mixtures. Fixed seed and historical sampler;reverse-original evidence stays separate from adjacent evidence.'})
        fresh = condition(fresh_plan); functional_cases = []
        for row in fresh['rows']:
            if row['posterior_functional_precision_passed']:
                continue
            sites = [i for i,v in enumerate(row['half_variance_relative_differences']) if v > .15]
            if sites:
                functional_cases.append({**{k:row[k] for k in KEY}, 'sites': sites,
                    'axis': max(sites, key=lambda i:row['half_variance_relative_differences'][i])})
        local_output = f'paper_artifacts/historical_source_learning_round{n}_completion_v1'
        local_plan = register(f'performance/manifests/historical_source_learning_round{n}_completion_v1.json',
            {**local_template, 'status': f'registered before iteration{n} actual functional repair',
                'previous_learning': previous_learning, 'learning_results': base + '/results.json',
                'fresh_results': fresh_output + '/results.json', 'functional_cases': functional_cases,
                'output': local_output, 'scope': 'Same-law saved residuals:adjacent source evidence and conditional integrals only at actual failed variance sites. No further sampling or fitting.'})
        local = complete(local_plan)
        replay_plan = register(f'performance/manifests/historical_source_learning_round{n}_replay.json',
            {**replay_template, 'step_name': f'Source iteration{n}', 'base': base,
                'proposal': proposed + '/results.json', 'learning': base + '/results.json',
                'fresh': fresh_output + '/results.json', 'local': local_output + '/results.json',
                'unit_tests': batch.get('unit_tests', replay_template['unit_tests']),
                'cost_results': [proposed + '/results.json', base + '/results.json',
                    fresh_output + '/results.json', local_output + '/results.json']})
        phase = replay(replay_plan)
        summaries.append({'round': n, 'phase_results': base + '/phase_results.json',
            'cases_passed': phase['source_posterior_precision_cases_passed'],
            'complete_identities': phase['complete_learned_source_identities'],
            'new_conditioned_cases': len(fresh['rows']), 'local_functional_cases': len(local['functional_repairs']),
            'cpu_seconds': phase['recorded_single_thread_cpu_seconds'],
            'covariance_changes': phase['source_covariance_change_by_head']})
        if phase['source_posterior_precision_cases_passed'] != 32:
            status = f'iteration{n} source posterior precision requires continuation'; break
        previous_learning, previous_precision = base + '/results.json', base + '/phase_results.json'
    result = {'execution_plan': plan, 'status': status, 'iterations': summaries,
        'all_registered_iterations_conditioned': len(summaries) == len(batch['iterations'])
            and all(r.get('cases_passed') == 32 for r in summaries),
        'cpu_seconds_this_execution': time.process_time() - started,
        'latest_learning': previous_learning, 'latest_certified_precision': previous_precision,
        'new_source_labels': 0, 'new_target_simulator_calls': 0, 'new_verifier_calls': 0,
        'server_jobs': 0, 'original_source_cases_unchanged': True,
        'no_model_adopted': True, 'independent_confirmation': False}
    save(ROOT / batch['output'] / 'results.json', result)
    print(status, 'completed stages', len(summaries), 'CPU', round(result['cpu_seconds_this_execution'], 2), flush=True)
    return result


if __name__ == '__main__':
    run()
