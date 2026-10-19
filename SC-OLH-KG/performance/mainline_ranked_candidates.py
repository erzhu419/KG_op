"""Original inverse proposal pools ranked by the current frozen source laws.

This retains the existing public proposals and source-template inverse pools,
but replaces arbitrary tail selection with balanced source-mixture/GP terminal
utility nominations. It does not restore the old latent-coordinate ranker.
All ranking is source-only; target observations and outcomes are not inputs.
"""

import json
from pathlib import Path
import time

import numpy as np

from core.candidates import latin_hypercube_candidates, random_candidates
from core.designs import next_sobol_integer_candidate
from performance.mainline_cross_dimension_prior import query_coordinates, restore_cross_dimension_prior
from performance.mainline_dynamic_candidates import _source_templates
from performance.mainline_moment_policy import moment_state, terminal_action
from representation.meta_prior import LearnedMetaPrior


ARMS = ('source_structural_mixture', 'source_common_task_rbf')
CHUNK_SIZE = 16
REMAINING = 358


def _proposal_rounds(problem, original_bank, templates, seed):
    """Replay the original dynamic driver's seeds and full inverse pools."""
    canonical = next_sobol_integer_candidate(problem, int(seed) + 470003,
        observed=[original_bank[45], original_bank[10]])
    rounds = []
    for number in (1, 2):
        def rng(stream):
            return np.random.default_rng(np.random.SeedSequence([seed, number, stream]))

        source_seed = int(np.random.SeedSequence([seed, number, 3]).generate_state(1)[0])
        prior = LearnedMetaPrior(seed=source_seed)
        prior.alignment_profile_templates = templates
        pool = prior._alignment_inverse_pool(problem, np.random.default_rng(source_seed), 256)
        public = [(canonical, 'canonical_sobol')]
        public += [(p, 'lhs') for p in latin_hypercube_candidates(problem, 4, rng(1))]
        public += [(p, 'random') for p in random_candidates(problem, 5, rng(2))]
        rounds.append({'round': number, 'public': public, 'pool': pool,
                       'source_pool_seed': source_seed})
    return rounds


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
    temporary.replace(path)


def _score_pool(coordinates, *, reuse, pilot, protocol, cache_dir, round_number, source_seed):
    """Project at most 16 policies at once; terminal utility uses marginals only."""
    count = len(coordinates['mean'])
    scores = {arm: [] for arm in ARMS}
    powers = {arm: [] for arm in ARMS}
    chunks = []
    for start in range(0, count, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, count)
        block = {key: coordinates[key][start:stop] for key in ('mean', 'risk')}
        contract = {'round': round_number, 'source_pool_seed': int(source_seed),
                    'start': start, 'stop': stop, 'layout': '5', 'coordinates': block,
                    'source_connection_plan': pilot['source_connection_plan'],
                    'gp_fit': reuse['gp_fit'], 'training_tasks': pilot['training_tasks'],
                    'remaining': REMAINING, 'terminal_protocol': protocol}
        path = Path(cache_dir) / f'round{round_number:02d}' / f'chunk{start:03d}.json'
        cached = path.exists()
        if cached:
            record = json.loads(path.read_text())
            if record['contract'] != contract:
                raise ValueError(f'Frozen ranking cache inputs differ: {path}')
        else:
            started = time.process_time()
            estimates = {}
            for arm in ARMS:
                model, diagnostics = restore_cross_dimension_prior(reuse, pilot, '5', arm, block)
                action = terminal_action(moment_state(model), REMAINING, protocol)
                estimates[arm] = {'utilities': action['utilities'],
                                  'certificate_powers': action['certificate_powers'],
                                  'source_projection': diagnostics}
                del model
            record = {'contract': contract, 'estimates': estimates,
                      'cpu_seconds': time.process_time() - started}
            _save(path, record)
        for arm in ARMS:
            estimate = record['estimates'][arm]
            scores[arm].extend(estimate['utilities'])
            powers[arm].extend(estimate['certificate_powers'])
        chunks.append({'path': str(path), 'start': start, 'stop': stop,
                       'cpu_seconds': record['cpu_seconds'], 'reused': cached})
    for arm in ARMS:
        if (len(scores[arm]) != count or len(powers[arm]) != count
                or not np.isfinite(scores[arm]).all() or not np.isfinite(powers[arm]).all()):
            raise ValueError('Frozen source ranking returned incomplete or nonfinite marginals')
    return {'scores': scores, 'powers': powers, 'chunks': chunks,
            'scoring_cpu_seconds_including_reused_chunks': sum(c['cpu_seconds'] for c in chunks)}


def _balanced_nominations(pool, scores, seen):
    """Three alternating unique nominations per arm; ties use pool index."""
    ranked = {arm: sorted(range(len(pool)), key=lambda i: (-scores[arm][i], i)) for arm in ARMS}
    cursors = dict.fromkeys(ARMS, 0)
    used = set(seen)
    selected = []
    for _ in range(3):
        for arm in ARMS:
            while cursors[arm] < len(pool):
                position = cursors[arm]
                index = ranked[arm][position]
                cursors[arm] += 1
                point = tuple(pool[index])
                if point not in used:
                    used.add(point)
                    selected.append((index, arm, position + 1))
                    break
            else:
                raise ValueError('Original inverse pool cannot fill the registered unique quota')
    return selected


def build_ranked_shared_bank(problem, original_bank, *, seed=2026101520,
                            reuse, pilot, protocol, cache_dir):
    """Freeze a common two-release bank without target data or new source fits."""
    bank = [tuple(map(int, point)) for point in original_bank]
    if len(bank) != 73 or len(set(bank)) != 73 or any(len(p) != problem.d for p in bank):
        raise ValueError('Expected the unique original73 bank at the actual policy dimension')
    templates, _, source = _source_templates()
    proposals = _proposal_rounds(problem, bank, templates, seed)
    all_pool_points = [point for proposal in proposals for point in proposal['pool']]
    coordinates = query_coordinates('5', problem, all_pool_points, pilot['source_connection_plan'])
    origins = [{'birth_round': 0, 'route': 'original_bank', 'original_index': i} for i in range(73)]
    seen = set(bank)
    ends, rounds = [], []
    offset = 0
    for proposal in proposals:
        number, pool = proposal['round'], proposal['pool']
        block = {key: coordinates[key][offset:offset + len(pool)] for key in ('mean', 'risk')}
        offset += len(pool)
        ranking = _score_pool(block, reuse=reuse, pilot=pilot, protocol=protocol,
            cache_dir=cache_dir, round_number=number, source_seed=proposal['source_pool_seed'])
        accepted, duplicates = {}, {}

        def append(point, info):
            point, route = tuple(map(int, point)), info['route']
            if point in seen:
                duplicates[route] = duplicates.get(route, 0) + 1
                return
            seen.add(point)
            bank.append(point)
            origins.append({'birth_round': number, **info})
            accepted[route] = accepted.get(route, 0) + 1

        for point, route in proposal['public']:
            append(point, {'route': route})
        nominations = _balanced_nominations(pool, ranking['scores'], seen)
        for index, arm, rank in nominations:
            append(pool[index], {'route': 'source_ranked_inverse_pool',
                'source_route': 'LearnedMetaPrior._alignment_inverse_pool',
                'source_pool_index': index, 'nominated_by': arm, 'rank_within_arm': rank,
                'utilities': {a: ranking['scores'][a][index] for a in ARMS},
                'certificate_powers': {a: ranking['powers'][a][index] for a in ARMS}})
        ends.append(len(bank))
        rounds.append({'round': number, 'proposed_count': len(proposal['public']) + len(nominations),
            'added_by_route': accepted, 'duplicates_by_route': duplicates,
            'source_pool_seed': proposal['source_pool_seed'], 'source_pool_count': len(pool),
            'ranking_scores': [{'pool_index': i,
                'utilities': {a: ranking['scores'][a][i] for a in ARMS},
                'certificate_powers': {a: ranking['powers'][a][i] for a in ARMS}}
                for i in range(len(pool))],
            'score_chunks': [{k: v for k, v in chunk.items() if k != 'reused'}
                             for chunk in ranking['chunks']],
            'scoring_cpu_seconds_including_reused_chunks': ranking['scoring_cpu_seconds_including_reused_chunks']})
    if len(bank) > 104:
        raise ValueError('Shared expansion exceeded its registered 104-policy cap')
    return {'bank': [list(p) for p in bank], 'dynamic_candidates': {
        'original_count': 73, 'round_end_indices': ends, 'origins': origins,
        'seed': int(seed), 'target_dimension': int(problem.d), 'rounds': rounds,
        'source': source, 'source_inverse_pool_size': 256,
        'ranking': 'Current frozen-law marginal terminal utility; source mixture and GP alternate three unique nominations each',
        'ranking_arms': list(ARMS), 'ranking_verification_replicates': REMAINING,
        'ranking_chunk_size': CHUNK_SIZE,
        'ranking_scope': 'All original inverse-pool routes; exclude original bank, public proposals and earlier accepted points',
        'source_connection_plan': pilot['source_connection_plan'], 'frozen_gp_fit': reuse['gp_fit'],
        'coordinate_reconstruction': coordinates['diagnostics'],
        'canonical_rule': 'same first unobserved Sobol point relative to shared initialization45/10 in both rounds',
        'omitted_original_routes': ['old fitted latent-distance ranker', 'posterior expert allocation',
                                    'constraint-uncertainty ranking', 'online feedback to shared proposals'],
        'target_observations_used': False, 'target_truth_used': False,
        'new_source_fits': 0, 'simulation_calls': 0}}
