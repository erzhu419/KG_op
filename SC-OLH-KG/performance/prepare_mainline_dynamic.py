"""Freeze source-only candidate expansion; reuse d1000 source laws and units."""

import time

import numpy as np

from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_cross_dimension_prior import query_coordinates, restore_cross_dimension_prior
from performance.mainline_dynamic_candidates import build_shared_dynamic_bank
from performance.run_mainline_two_round73 import ROOT, context, read, save, timestamp
from performance.run_legacy_related_inventory_source import problem_for


PLAN = 'performance/manifests/mainline_d1000_dynamic_v1.json'
OUTPUT = 'paper_artifacts/mainline_d1000_dynamic_v1'
PREVIOUS = 'performance/manifests/mainline_d1000_two_round_v1.json'


def prefix_replay_check(spec, coordinates, pilot, reuse):
    """Adding an unseen policy must preserve the old joint marginal/update."""
    old_indices = [45, 10, 71]
    small = {k: np.asarray(coordinates[k])[old_indices].tolist() for k in ('mean', 'risk')}
    larger = {k: np.asarray(coordinates[k])[old_indices + [73]].tolist() for k in ('mean', 'risk')}
    paid = read('paper_artifacts/mainline_d1000_two_round_v1/task00/initialization.json')
    records = []
    for arm in spec['models']:
        before, _ = restore_cross_dimension_prior(reuse, pilot, spec['layout'], arm, small)
        after, _ = restore_cross_dimension_prior(reuse, pilot, spec['layout'], arm, larger)
        error = 0.0
        for step in range(3):
            bm, bc = before.predict_joint(); am, ac = after.predict_joint()
            differences = [np.max(abs(bm-am[:3])), np.max(abs(bc-ac[:3, :, :3, :])),
                           np.max(abs(before.weights-after.weights))]
            error = max(error, *map(float, differences))
            np.testing.assert_allclose(bm, am[:3], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(bc, ac[:3, :, :3, :], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(before.weights, after.weights, atol=1e-10, rtol=1e-10)
            if step < 2:
                raw = paid['paired_observations'][step]
                value = [raw[0], raw[1]-paid['tau']]
                before.update_raw(step, value); after.update_raw(step, value)
        records.append({'arm': arm, 'maximum_prefix_moment_or_weight_error': error,
                        'replayed_existing_episodes_per_model': 2, 'new_physical_calls': 0})
    return records


def run():
    started = time.process_time()
    old = read(PREVIOUS)
    frozen = read(old['output'] + '/frozen_inputs.json')
    problem = no_outcomes(problem_for(frozen['problem_plan'], frozen['tasks'][0]))
    bridge = build_shared_dynamic_bank(problem, frozen['bank'], seed=2026101520)
    coordinates = query_coordinates(old['layout'], problem, bridge['bank'])
    reference = read(old['query_coordinates'])
    coordinate_error = 0.0
    for key in ('mean', 'risk'):
        np.testing.assert_allclose(coordinates[key][:73], reference[key], atol=1e-12, rtol=1e-12)
        coordinate_error = max(coordinate_error, float(np.max(abs(
            np.asarray(coordinates[key][:73])-np.asarray(reference[key])))))
    spec = dict(old, status='Frozen before new target observations; source-only shared proposals, two releases.',
        output=OUTPUT, cross_dimension_inputs=OUTPUT+'/bridge_inputs.json',
        query_coordinates=OUTPUT+'/query_coordinates.json',
        candidate_count=len(bridge['bank']), task_seed=2026101521,
        initialization_seed=2026101522, search_seed=2026101523,
        verification_seed=2026101524, bootstrap_seed=2026101525,
        target_tasks=[dict(t, task_id=f'original_d1000_dynamic_seed{i:02d}') for i,t in enumerate(old['target_tasks'])],
        evaluation_scope='Original nominal d1000 Inventory, four fresh observation seeds, same frozen sources and mean4/risk10. Shared source-only generator components with two scheduled releases, not the full adaptive V51 generator.',
        bank='Original73 prefix plus two deterministic source-only proposal batches. Deduplicate integer policies; no target truth, observations or online-selected indices enter proposal generation. All four arms share the same final union.',
        selection='Same fixed order3 moment KG and actual native-mixture raw updates. Round1 query/terminal sees release1 prefix; round2 sees full union. Stop chooses from full union using initialization only. Full union is restored once; latent future-point predictions never enter round1 selection.',
        dynamic_candidates=bridge['dynamic_candidates'],
        mechanism_diagnostic=old['mechanism_diagnostic']+' Also freeze a terminal intervention on new generated unsampled marginals only; score new-point initial/final and prerelease propagation after all verifiers.',
        randomness='Fresh seeds separate from previous fixed73 development. Same source-only proposal union across four repetitions; common init per repetition and independent search/verification streams as before.')
    reuse = read(spec['reuse_plan']); pilot = context(reuse)[0]
    parity = prefix_replay_check(spec, coordinates, pilot, reuse)
    audit = {'frozen_at': timestamp(), 'candidate_count': len(bridge['bank']),
             'original_prefix_coordinate_max_error': coordinate_error, 'prefix_replay': parity,
             'source_law_optimizations': 0, 'gp_optimizations': 0,
             'new_simulator_calls': 0, 'target_truth_calls': 0,
             'cpu_seconds': time.process_time()-started,
             'response_scale_audit_reused': old['dimension_audit']}
    for path, value in [(PLAN, spec), (spec['cross_dimension_inputs'], bridge),
                        (spec['query_coordinates'], coordinates), (OUTPUT+'/preparation_audit.json', audit)]:
        target = ROOT/path
        if target.exists() and path != OUTPUT+'/preparation_audit.json':
            assert read(path) == value, 'An existing frozen dynamic plan differs'
        else:
            save(target, value)
    print({'plan': PLAN, 'candidates': len(bridge['bank']), 'releases': bridge['dynamic_candidates']['round_end_indices'],
           'prefix_replay': parity, 'cpu_seconds': audit['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
