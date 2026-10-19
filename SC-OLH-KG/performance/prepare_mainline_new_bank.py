"""Register one new proposal seed and rebuild the shared source-ranked bank.

Reuse the existing generator, source laws, encoders and nomination rule. The
old task is only a public-schema provider with all outcomes disabled. Bank and
representation overlap are reported after generation and never select a seed.
"""

import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_cross_dimension_prior import query_coordinates
from performance.mainline_ranked_candidates import build_ranked_shared_bank
from performance.run_mainline_two_round73 import context, read, save, timestamp
from performance.run_legacy_related_inventory_source import problem_for

SEED = 2026101620
OUTPUT = 'paper_artifacts/mainline_d1000_new_bank_v1'
PREVIOUS = 'paper_artifacts/mainline_d1000_frozen_query_v1/frozen_inputs.json'


def register(out):
    path = out / 'registration.json'
    if path.exists():
        record = read(str(path))
        assert record['proposal_seed'] == SEED and record['previous_frozen_inputs'] == PREVIOUS
        return record
    manifests = sorted((ROOT / 'performance/manifests').rglob('*.json'))
    used = [str(path.relative_to(ROOT)) for path in manifests if str(SEED) in path.read_text()]
    if used:
        raise ValueError(f'Proposal seed was already present in manifests: {used}')
    record = {'registered_at': timestamp(), 'proposal_seed': SEED,
        'previous_frozen_inputs': PREVIOUS, 'existing_manifest_count_checked': len(manifests),
        'existing_seed_occurrences': used, 'expected_original_prefix': 73,
        'expected_round_end_indices': [89, 104],
        'generator': 'performance.mainline_ranked_candidates.build_ranked_shared_bank',
        'ranking': 'Unchanged two source laws; 256 inverse-pool points per round; 358 verification calls; alternating 3+3 unique nominations.',
        'selection_rule': 'One registered seed. Do not reseed or fill unexpected counts using overlap or target outcomes.'}
    save(path, record)
    print(json.dumps({'stage': 'registered', 'proposal_seed': SEED, 'path': str(path)}), flush=True)
    return record


def run():
    started, wall = time.process_time(), time.monotonic()
    out = ROOT / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    attempt = {'stage': 'new_bank_preparation', 'status': 'failed', 'phase': 'registration',
        'started_at': timestamp(), 'proposal_seed': SEED, 'new_simulator_calls': 0,
        'new_target_truth_calls': 0, 'new_source_labels': 0, 'new_source_fits': 0,
        'source_law_optimizations': 0, 'gp_optimizations': 0, 'downloads': 0}
    try:
        registration = register(out)
        frozen = read(PREVIOUS)
        old = frozen['plan']
        reuse = read(old['reuse_plan'])
        pilot = context(reuse)[0]
        provider = no_outcomes(problem_for(frozen['problem_plan'], frozen['tasks'][0]))
        old_bank = np.asarray(frozen['bank'], dtype=int)
        assert old_bank.shape == (104, 1000)
        attempt['phase'] = 'source_only_bank_ranking'
        bridge = build_ranked_shared_bank(provider, old_bank[:73].tolist(), seed=SEED,
            reuse=reuse, pilot=pilot, protocol=frozen['terminal_protocol'], cache_dir=out / 'ranking_cache')
        save(out / 'bridge_inputs.json', bridge)
        bank = np.asarray(bridge['bank'], dtype=int)
        release = bridge['dynamic_candidates']['round_end_indices']
        preliminary = {'registered_at': registration['registered_at'], 'prepared_at': timestamp(),
            'proposal_seed': SEED, 'previous_frozen_inputs': PREVIOUS,
            'candidate_count': len(bank), 'round_end_indices': release,
            'original_prefix_integer_equal': bool(np.array_equal(bank[:73], old_bank[:73])),
            'ranking_source_connection_plan': pilot['source_connection_plan'],
            'ranking_frozen_gp_fit': reuse['gp_fit'], 'ranking_target_observations_used': False,
            'source_law_optimizations': 0, 'gp_optimizations': 0,
            'new_source_fits': 0, 'new_simulator_calls': 0, 'new_target_truth_calls': 0, 'downloads': 0}
        save(out / 'bank_preparation_audit.json', preliminary)
        if bank.shape != (104, 1000) or release != [89, 104]:
            raise ValueError(f'Unexpected bank count/releases: {bank.shape}, {release}; no resampling performed')
        np.testing.assert_array_equal(bank[:73], old_bank[:73])
        attempt['phase'] = 'coordinates_and_overlap_audit'
        coordinates = query_coordinates(old['layout'], provider, bridge['bank'], pilot['source_connection_plan'])
        previous_coordinates = read(old['query_coordinates'])
        prefix_errors = {}
        for key in ('mean', 'risk'):
            actual, expected = np.asarray(coordinates[key][:73]), np.asarray(previous_coordinates[key][:73])
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.)
            prefix_errors[key] = float(np.max(abs(actual - expected)))
        save(out / 'query_coordinates.json', coordinates)
        old_lookup = {tuple(point): index for index, point in enumerate(old_bank)}
        new_lookup = {tuple(point): index for index, point in enumerate(bank)}
        old_features = np.column_stack([previous_coordinates['mean'], previous_coordinates['risk']])
        new_features = np.column_stack([coordinates['mean'], coordinates['risk']])
        assert old_features.shape == new_features.shape == (104, 14)
        rows = []
        for index in range(73, len(bank)):
            distances = np.max(abs(old_features - new_features[index]), axis=1)
            mean_distances = np.max(abs(old_features[:, :4] - new_features[index, :4]), axis=1)
            risk_distances = np.max(abs(old_features[:, 4:] - new_features[index, 4:]), axis=1)
            rows.append({'new_index': index,
                'birth_round': bridge['dynamic_candidates']['origins'][index]['birth_round'],
                'physical_match_old_index': old_lookup.get(tuple(bank[index])),
                'equivalent_old_representation_indices': np.flatnonzero(distances <= 1e-12).tolist(),
                'equivalent_old_mean4_indices': np.flatnonzero(mean_distances <= 1e-12).tolist(),
                'equivalent_old_risk10_indices': np.flatnonzero(risk_distances <= 1e-12).tolist(),
                'nearest_old_representation_index': int(np.argmin(distances)),
                'nearest_old_representation_maxabs_distance': float(np.min(distances)),
                'nearest_old_mean4_maxabs_distance': float(np.min(mean_distances)),
                'nearest_old_risk10_maxabs_distance': float(np.min(risk_distances)),
                'old83_representation_maxabs_distance': float(distances[83]),
                'old98_representation_maxabs_distance': float(distances[98]),
                'old83_mean4_maxabs_distance': float(mean_distances[83]),
                'old98_mean4_maxabs_distance': float(mean_distances[98]),
                'old83_risk10_maxabs_distance': float(risk_distances[83]),
                'old98_risk10_maxabs_distance': float(risk_distances[98])})
        audit = {**preliminary, 'original_prefix_coordinate_max_errors': prefix_errors,
            'appended_count': len(rows), 'physical_overlap_total': sum(tuple(p) in old_lookup for p in bank),
            'appended_physical_overlap_count': sum(r['physical_match_old_index'] is not None for r in rows),
            'appended_representation_overlap_count_at_1e_minus12': sum(bool(r['equivalent_old_representation_indices']) for r in rows),
            'appended_mean4_overlap_count_at_1e_minus12': sum(bool(r['equivalent_old_mean4_indices']) for r in rows),
            'appended_risk10_overlap_count_at_1e_minus12': sum(bool(r['equivalent_old_risk10_indices']) for r in rows),
            'appended_internal_equivalent_pairs': {name: [[i, j] for i in range(73, 104)
                for j in range(i + 1, 104) if np.max(abs(new_features[i, columns] - new_features[j, columns])) <= 1e-12]
                for name, columns in [('mean4', slice(0, 4)), ('risk10', slice(4, 14)), ('joint14', slice(0, 14))]},
            'old_selected_policy_new_indices': {str(index): new_lookup.get(tuple(old_bank[index])) for index in (83, 98)},
            'representation_distance': 'Maximum absolute difference in concatenated frozen mean4 and risk10; equivalence at absolute tolerance 1e-12.',
            'appended_policy_audit': rows,
            'frozen_encoder_reconstructions': 2,
            'encoder_note': 'Existing query_coordinates reconstructs the same 32-row frozen encoder for pools and final bank; no source-law or GP refitting.',
            'provider_note': 'Previous task0 public schema only; no_outcomes disables simulation and truth. No new geometry or target observations are read.',
            'ranking_note': 'Reuses the original ranked-bank source laws exactly, without changing ranking to the later task-risk calibration.',
            'overlap_role': 'Descriptive audit only. The registered seed and returned bank are retained regardless of overlap.'}
        save(out / 'bank_preparation_audit.json', audit)
        attempt.update(status='completed', phase='completed')
        print(json.dumps({key: audit[key] for key in ('candidate_count', 'round_end_indices',
            'physical_overlap_total', 'appended_physical_overlap_count',
            'appended_representation_overlap_count_at_1e_minus12', 'old_selected_policy_new_indices')}), flush=True)
        return audit
    except Exception:
        attempt['traceback'] = traceback.format_exc()
        raise
    finally:
        attempt.update(cpu_seconds=time.process_time() - started, wall_seconds=time.monotonic() - wall)
        ledger = out / 'bank_preparation_attempts.jsonl'
        with ledger.open('a') as handle:
            handle.write(json.dumps(attempt, allow_nan=False) + '\n')
        if attempt['status'] == 'completed':
            audit = read(str(out / 'bank_preparation_audit.json'))
            attempts = [json.loads(line) for line in ledger.read_text().splitlines()]
            audit['preparation_cost'] = {'attempt_count': len(attempts),
                'failed_attempt_count': sum(row['status'] != 'completed' for row in attempts),
                'cpu_seconds': sum(row['cpu_seconds'] for row in attempts),
                'wall_seconds': sum(row['wall_seconds'] for row in attempts)}
            save(out / 'bank_preparation_audit.json', audit)


if __name__ == '__main__':
    run()
