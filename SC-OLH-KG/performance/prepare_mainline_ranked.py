"""Replace arbitrary inverse-pool thinning with frozen-source utility ranking."""

import time
import traceback

import numpy as np

from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_cross_dimension_prior import query_coordinates
from performance.mainline_ranked_candidates import build_ranked_shared_bank
from performance.run_mainline_two_round73 import ROOT, context, read, save, timestamp
from performance.run_legacy_related_inventory_source import problem_for

PLAN = 'performance/manifests/mainline_d1000_ranked_v1.json'
OUTPUT = 'paper_artifacts/mainline_d1000_ranked_v1'
PREVIOUS = 'performance/manifests/mainline_d1000_dynamic_v1.json'


def run():
    old = read(PREVIOUS)
    frozen = read(old['output']+'/frozen_inputs.json')
    reuse = read(old['reuse_plan']); pilot = context(reuse)[0]
    provider = no_outcomes(problem_for(frozen['problem_plan'], frozen['tasks'][0]))
    bridge = build_ranked_shared_bank(provider, frozen['bank'][:73], seed=2026101520,
        reuse=reuse, pilot=pilot, protocol=frozen['terminal_protocol'], cache_dir=ROOT/OUTPUT/'ranking_cache')
    coordinates = query_coordinates(old['layout'], provider, bridge['bank'])
    previous_coordinates = read(old['query_coordinates'])
    error = 0.0
    for key in ('mean', 'risk'):
        error = max(error, float(np.max(abs(np.asarray(coordinates[key][:73])-
                                            np.asarray(previous_coordinates[key][:73])))))
        np.testing.assert_allclose(coordinates[key][:73], previous_coordinates[key][:73], atol=1e-12, rtol=1e-12)
    spec = dict(old, status='Frozen-source ranking; cached initialization reused; new search and verification streams.',
        output=OUTPUT, cross_dimension_inputs=OUTPUT+'/bridge_inputs.json',
        query_coordinates=OUTPUT+'/query_coordinates.json', candidate_count=len(bridge['bank']),
        initialization_cache=old['output'], maximum_physical_calls=5760,
        search_seed=2026101533, verification_seed=2026101534, bootstrap_seed=2026101535,
        target_tasks=[dict(t, task_id=f'original_d1000_ranked_seed{i:02d}') for i,t in enumerate(old['target_tasks'])],
        dynamic_candidates=bridge['dynamic_candidates'],
        evaluation_scope='Same nominal d1000 problem and four cached initialization pairs as the completed dynamic pilot. Original public/inverse pools with current frozen-source utility ranking; a development comparison, not independent confirmation or restoration of the old latent ranker.',
        bank='Original73 prefix and unchanged public proposal streams. Each round ranks the original256 inverse pool using source-only structural and same-source GP terminal utility at358 calls. Alternate3 unique nominees per model. No target observation or truth enters shared proposal ranking.',
        randomness='Reuse exactly the preceding four initialization pairs, counted as2 logical calls per arm and0 new physical calls. Proposal seed2026101520 unchanged. Fresh shared-by-seed verification and own-model search streams.',
        accounting='Zero new source fits;8 saved initialization episodes replayed,16 new search+5744 new verification=5760 new physical calls,5792 logical. Keep previous experiment/diagnostic accounts and every ranking attempt. Source-only marginal scoring uses small blocks, no full-pool joint covariance.')
    for path, value in [(PLAN, spec), (spec['cross_dimension_inputs'], bridge), (spec['query_coordinates'], coordinates)]:
        target = ROOT/path
        if target.exists():
            assert read(path) == value, 'An existing frozen ranked plan differs'
        else:
            save(target, value)
    audit = {'frozen_at': timestamp(), 'candidate_count': len(bridge['bank']),
             'round_end_indices': bridge['dynamic_candidates']['round_end_indices'],
             'original_prefix_coordinate_max_error': error,
             'source_law_optimizations': 0, 'gp_optimizations': 0,
             'new_simulator_calls': 0, 'target_truth_calls': 0,
             'reused_initialization_episodes': 8,
             'response_scale_audit_reused': old['dimension_audit']}
    save(ROOT/OUTPUT/'preparation_audit.json', audit)
    return audit


if __name__ == '__main__':
    started, wall = time.process_time(), time.monotonic()
    attempt = {'started_at': timestamp(), 'stage': 'ranked_preparation'}
    try:
        print(run(), flush=True)
        attempt['status'] = 'completed'
    except Exception:
        attempt.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        attempt.update(cpu_seconds=time.process_time()-started, wall_seconds=time.monotonic()-wall)
        folder = ROOT/OUTPUT; folder.mkdir(parents=True, exist_ok=True)
        with (folder/'preparation_attempts.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps(attempt)+'\n')
        if attempt['status'] == 'completed':
            audit = read(OUTPUT+'/preparation_audit.json')
            attempts = [json.loads(line) for line in (folder/'preparation_attempts.jsonl').read_text().splitlines()]
            audit['cpu_seconds'] = sum(row['cpu_seconds'] for row in attempts)
            save(folder/'preparation_audit.json', audit)
