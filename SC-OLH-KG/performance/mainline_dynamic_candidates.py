"""Shared public/source-only candidate expansion using original V51 routes.

Reuse Sobol/LHS/random and the original risk-aligned inverse-pool source
profile perturbation. Keep all source laws and encoders unchanged. The old
latent-distance ranker, posterior expert allocation, uncertainty ranking and
adaptive online proposal feedback are deliberately outside this shared driver.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.candidates import latin_hypercube_candidates, random_candidates
from core.designs import next_sobol_integer_candidate
from representation.meta_prior import LearnedMetaPrior


CONNECTION = 'performance/manifests/historical_learned_prior_kg_connection_v1.json'
ORIGINAL = 'paper_artifacts/mainline_d1000_two_round_v1/original_bank.json'


def _read(path):
    return json.loads((ROOT / path).read_text())


def _source_templates():
    """Use only the 32 existing encoder rows, a subset of the source-law data."""
    connection = _read(CONNECTION)
    old = _read(connection['legacy_connection_plan'])
    path = old['query_risk_inputs_by_layout']['5']
    risk, public = _read(path), _read(old['public_inputs'])
    level = public['plan']['task_distribution']['L']
    alpha = public['plan']['task_distribution']['alpha']
    templates, row_origins = [], []
    for task_id, sites, task in zip(risk['training_task_ids'],
                                    risk['source_site_indices_by_task'], risk['source_tasks']):
        if task['name'] != public['tasks'][task_id]['task_id'] or len(sites) != len(task['response']):
            raise ValueError('Frozen source task/site mapping differs from its responses')
        for row, (site, response, noise) in enumerate(zip(sites, task['response'], task['noise'])):
            # The frozen source records are means of three raw replicates.
            # noise[g] is their observed sample variance divided by three.
            sigma = float(np.sqrt(max(0.0, 3 * noise[1])))
            margin = float(response[1] + ndtri(1 - alpha) * sigma)
            templates.append({'profile': np.asarray(public['bank'][site]) / level,
                              'domain': task['name'], 'feasible': margin <= 0,
                              'scaled_margin': margin / max(sigma, 1e-8)})
            row_origins.append({'source_task': task['name'], 'source_row': row,
                                'source_public_bank_index': int(site)})
    if len(templates) != 32:
        raise ValueError('This fixed layout5 proposal contract uses 32 existing source rows')
    return templates, row_origins, {'source_inputs': path,
        'source_public_inputs': old['public_inputs'],
        'source_task_ids': [t['name'] for t in risk['source_tasks']],
        'source_site_indices_by_task': risk['source_site_indices_by_task'],
        'proposal_source_rows': len(templates),
        'source_scope': '32 frozen encoder rows; a subset of the unchanged 48-row source-law training data'}


def _source_refinements(problem, templates, row_origins, seed):
    """Select six source-perturbation outputs of the existing inverse pool."""
    prior = LearnedMetaPrior(seed=int(seed))
    prior.alignment_profile_templates = templates
    pool_size = 256
    pool = prior._alignment_inverse_pool(problem, np.random.default_rng(seed), pool_size)
    # At most 32 template replays + 32 universal + 6 endpoints/ramps +
    # 64 random rows precede the original while loop. Its even row indices
    # execute source-profile + five-knot Normal(0,.10) perturbations.
    prefix_upper_bound = len(templates) + 32 + 6 + pool_size // 4
    indices = list(range(pool_size - 12, pool_size, 2))
    if len(pool) != pool_size or min(indices) < prefix_upper_bound:
        raise ValueError('Original inverse pool no longer exposes the registered source tail')
    return [(tuple(pool[i]), {'source_route': 'LearnedMetaPrior._alignment_inverse_pool',
                             'source_pool_index': i,
                             **row_origins[i % len(templates)]}) for i in indices]


def build_shared_dynamic_bank(problem, original_bank, *, seed):
    """Return one source-only two-round schedule shared by every target seed/arm.

    The canonical Sobol point is anchored to the unchanged shared initial
    coordinates 45/10, so its second appearance is intentionally deduplicated.
    No target observation values or posterior objects enter this API.
    """
    bank = [tuple(map(int, point)) for point in original_bank]
    if len(bank) != 73 or len(set(bank)) != 73 or any(len(p) != problem.d for p in bank):
        raise ValueError('Expected the unique original73 bank at the actual policy dimension')
    templates, row_origins, source = _source_templates()
    origins = [{'birth_round': 0, 'route': 'original_bank', 'original_index': i}
               for i in range(len(bank))]
    seen = set(bank)
    ends, rounds = [], []
    canonical = next_sobol_integer_candidate(problem, int(seed) + 470003,
                                             observed=[bank[45], bank[10]])
    for round_number in (1, 2):
        def rng(stream):
            return np.random.default_rng(np.random.SeedSequence([seed, round_number, stream]))

        source_seed = int(np.random.SeedSequence([seed, round_number, 3]).generate_state(1)[0])
        proposed = [(canonical, {'route': 'canonical_sobol'})]
        proposed += [(p, {'route': 'lhs'}) for p in latin_hypercube_candidates(problem, 4, rng(1))]
        proposed += [(p, {'route': 'random'}) for p in random_candidates(problem, 5, rng(2))]
        proposed += [(p, {'route': 'source_profile_refinement', **info})
                     for p, info in _source_refinements(problem, templates, row_origins, source_seed)]
        accepted, duplicates = {}, {}
        for point, info in proposed:
            point = tuple(map(int, point))
            route = info['route']
            if point in seen:
                duplicates[route] = duplicates.get(route, 0) + 1
                continue
            seen.add(point)
            bank.append(point)
            origins.append({'birth_round': round_number, **info})
            accepted[route] = accepted.get(route, 0) + 1
        if accepted.get('source_profile_refinement', 0) != 6:
            raise ValueError('Registered six source refinements did not produce six new policies')
        ends.append(len(bank))
        rounds.append({'round': round_number, 'proposed_count': len(proposed),
                       'added_by_route': accepted, 'duplicates_by_route': duplicates,
                       'source_pool_seed': source_seed})
    return {'bank': [list(p) for p in bank], 'dynamic_candidates': {
        'original_count': 73, 'round_end_indices': ends, 'origins': origins,
        'seed': int(seed), 'target_dimension': int(problem.d), 'rounds': rounds,
        'source': source, 'source_inverse_pool_size': 256,
        'source_refinement_indices': list(range(244, 256, 2)),
        'source_perturbation': 'original five-knot Normal(0,.10); original every-fifth-row head randomization retained',
        'canonical_rule': 'same first unobserved Sobol point relative to shared initialization45/10 in both rounds',
        'omitted_original_routes': ['posterior expert allocation', 'latent-distance source ranker',
                                    'constraint-uncertainty ranking', 'online feedback to shared proposals'],
        'target_observations_used': False, 'target_truth_used': False,
        'new_source_fits': 0, 'simulation_calls': 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--original-bank', default=ORIGINAL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from performance.diagnose_legacy_risk_transfer import no_outcomes
    from performance.benchmark_lodo_meta_prior import build_scalarized_problem
    original = _read(args.original_bank)
    spec = _read(original['lineage']['legacy_plan'])
    problem = no_outcomes(build_scalarized_problem('InventorySupplyChain', spec['target_dimension'],
                          spec['L'], spec['sigma'], spec['alpha'], spec['weights']))
    result = build_shared_dynamic_bank(problem, original['bank'], seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'candidate_count': len(result['bank']),
                      'round_end_indices': result['dynamic_candidates']['round_end_indices']}))


if __name__ == '__main__':
    main()
