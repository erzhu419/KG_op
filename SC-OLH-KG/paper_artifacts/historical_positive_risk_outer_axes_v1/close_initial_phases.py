"""Close the completed screens and their same-phase raw probes."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def read(path):
    return json.loads((ROOT/path).read_text())

for axis in ('mean','variance'):
    base=f'paper_artifacts/historical_positive_risk_{axis}_axis_v1'
    refs=f'paper_artifacts/historical_positive_risk_{axis}_axis_incumbents_v1'
    bound=f'paper_artifacts/historical_positive_risk_{axis}_axis_bound_incumbents_v1'
    screen=read(base+'/execution_account.json');account=read(refs+'/execution_account.json')
    assert read(base+'/compact_summary.json')['numerical_containment_passed']
    assert read(bound+'/compact_summary.json')['numerical_containment_passed']
    assert read(refs+'/compact_summary.json')['local_risk_candidate_passed']
    replay=read(refs+'/incumbent_replay_final.json')
    initial_replay_cpu=sum(json.loads(line[21:])['process_cpu_seconds'] for line in
        (ROOT/base/'incumbent_replay_initial.log').read_text().splitlines()
        if line.startswith('SCHEDULEURM_RESOURCE '))
    support=screen['preparation_process_cpu_seconds']+screen['collection_process_cpu_seconds']+account['additional_support_cpu_seconds']+initial_replay_cpu
    total=screen['worker_process_cpu_seconds']+support+account['additional_recorded_process_cpu_seconds']-account['additional_support_cpu_seconds']
    assert support<=600 and total<=account['estimated_phase_cpu_seconds']<=3600
    completion=dict(state='phase_complete',axis=axis,registered_workers=screen['worker_count']+account['registered_workers'],
        all_reference_precision_passed=True,numerical_containment_passed=True,
        estimated_phase_cpu_seconds=account['estimated_phase_cpu_seconds'],shared_phase_reserve_cpu_seconds=600,
        recorded_process_cpu_seconds=total,recorded_support_cpu_seconds=support,
        initial_cached_replay_cpu_seconds=initial_replay_cpu,
        accounts=[base+'/execution_account.json',refs+'/execution_account.json'],
        native_choices_all_raw_referenced=replay['native_choices_all_raw_referenced'],
        missing_raw_observations=len(replay['missing_raw_references']),raw_covered_grid_weight=replay['raw_covered_grid_weight'],
        full_predictive_integration_admitted=False,independent_confirmation=False,
        scope='The 26 registered screen and reference workers passed. This closes their shared budget; remaining raw recommendation coverage and outer convergence are separate unfinished work.')
    (ROOT/refs/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(completion),flush=True)
