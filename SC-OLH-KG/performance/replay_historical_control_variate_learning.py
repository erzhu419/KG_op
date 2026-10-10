"""Replay complete action precision and source-only learning from saved cases."""

from performance.complete_historical_source_heads import ROOT,IDENTITY,read,save
from performance.compare_historical_control_variate_actions import run as compare,refine
from performance.learn_historical_raw_source_amplitudes import run as learn
from performance.check_historical_learned_source_posterior import run as condition,run_retained_baseline


def run():
    learning=learn();posterior=condition();retention=run_retained_baseline()
    comparison=compare();precision=refine()
    tests=read('paper_artifacts/historical_control_variate_action_comparison_v1/unit_tests.json')
    passed={tuple(r[k] for k in IDENTITY)+(r['head'],) for r in posterior['rows']
        if r['new_source_posterior_precision_passed']}
    identities={key[:-1] for key in passed}
    complete=sum(key+('f',) in passed and key+('g',) in passed for key in identities)
    result={'status':'complete12 action integration passed;source amplitude candidates remain HOLD'
            if precision['actions_precision_passed']==12 else 'complete12 actions evaluated;numerical precision remains HOLD',
        'action_comparison':'paper_artifacts/historical_control_variate_action_comparison_v1/results.json',
        'action_refinement':'paper_artifacts/historical_control_variate_action_precision_v2/results.json',
        'registered_actions_complete':comparison['all12_registered_actions_complete'],
        'action_precision_cases_passed':precision['actions_precision_passed'],
        'provisional_comparison':precision['comparison'],
        'source_amplitude_learning':'paper_artifacts/historical_raw_source_amplitude_learning_v1/results.json',
        'source_Mstep_Q_cases_passed':learning['source_Mstep_gates_passed'],
        'source_evidence_ratio_cases_passed':learning['evidence_ratio_precision_cases_passed'],
        'learned_source_posterior_precision':'paper_artifacts/historical_raw_source_amplitude_learning_v1/posterior_precision.json',
        'learned_source_posterior_cases_passed':posterior['cases_passed'],
        'complete_learned_source_identities':complete,
        'paired_retained_precision':'paper_artifacts/historical_raw_source_amplitude_learning_v1/retained_baseline_precision.json',
        'old_retained_precision_cases_passed':retention['old_retained_precision_cases_passed'],
        'new_retained_precision_cases_passed':retention['new_retained_precision_cases_passed'],
        'old_pass_new_fail_cases':retention['old_pass_new_fail_cases'],
        'old_fail_new_fail_cases':retention['old_fail_new_fail_cases'],
        'original_source32_PASS_preserved':True,'original_source_cases_resampled':0,
        'test_record':'paper_artifacts/historical_control_variate_action_comparison_v1/unit_tests.json',
        'tests_passed':tests['passed'],'tests_run':tests['tests_run'],
        'batch_equivalence':'paper_artifacts/historical_control_variate_action_comparison_v1/batch_equivalence.json',
        'cpu_seconds_first_execution':sum(x['cpu_seconds'] for x in (learning,posterior,retention,comparison,precision)),
        'cost':learning['cost'],'scientific_model_ranking_authorized':False,
        'no_model_adopted':True,'independent_confirmation':False,
        'next_work':{'source_learning':'Use the paired retained-state diagnostic to repair changed-law normalizers and posterior functionals with historical conditional integration. Original passed source chains remain unchanged. Then iterate the source-only M-step and compute comparable identity evidence before any new prior enters target evaluation.',
            'decision_validation':'Run the historical completeGP controls under this same raw-source,paid-update and terminal contract after numerical source learning closes. Acquisition diagnostics on two opened states do not establish independent deployment gains.'},
        'limitations':precision['limitations']+learning['limitations']}
    save(ROOT/'paper_artifacts/historical_control_variate_action_comparison_v1/phase_results.json',result)
    print(result['status'],flush=True);return result


if __name__=='__main__':run()
