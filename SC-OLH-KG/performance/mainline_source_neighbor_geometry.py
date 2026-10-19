"""Freeze source held-policy neighbors from public inputs before label scoring."""
from pathlib import Path

import numpy as np

from performance.run_mainline_legacy_backend_bridge import read
from representation.legacy_function_prior import legacy_public_coordinates

PREFIXES = (10, 13)


def public_coordinates(integer_points, level=100):
    """The existing ten public coordinates; no fitted centering or rescaling."""
    return legacy_public_coordinates(np.asarray(integer_points, dtype=float) / level)


def nearest_pairs(coordinates, held_policy_ids):
    """Stage61's one-neighbor union rule, now in the fixed public10 space."""
    coordinates = np.asarray(coordinates, dtype=float)
    distance = np.mean((coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    edges = {}
    nodes = []
    for i, j in enumerate(nearest):
        j = int(j)
        edges.setdefault(tuple(sorted((i, j))), []).append(i)
        nodes.append(dict(i=i, policy_id=int(held_policy_ids[i]),
            coordinates=coordinates[i].tolist(), nearest_j=j,
            nearest_distance_squared=float(distance[i, j]),
            nearest_tied_positions=np.flatnonzero(distance[i] == distance[i, j]).tolist(),
            observed={str(prefix): i < prefix for prefix in PREFIXES}))
    pairs = []
    for (i, j), selected_from in sorted(edges.items()):
        observed = {}
        for prefix in PREFIXES:
            group = ('observed_observed' if j < prefix else
                     'observed_unobserved' if i < prefix else 'unobserved_unobserved')
            observed[str(prefix)] = dict(i=i < prefix, j=j < prefix, pair_group=group)
        pairs.append(dict(i=i, j=j, policy_ids=[int(held_policy_ids[i]), int(held_policy_ids[j])],
            distance_squared=float(distance[i, j]), selected_from=selected_from, observed=observed))
    counts = {str(prefix): {group: sum(p['observed'][str(prefix)]['pair_group'] == group for p in pairs)
        for group in ('observed_observed', 'observed_unobserved', 'unobserved_unobserved')}
        for prefix in PREFIXES}
    return dict(nodes=nodes, pairs=pairs, prefix_counts=counts)


def build_fixed_pairs(root, plan):
    """Read archive containers but use only geometry, never responses or forecasts.

    The scheduler coordinator saves this complete24-group return value before
    entering its separate source-label scoring phase.
    """
    root = Path(root)
    source_path = plan['inputs']['source_manifest']
    source = read(root / source_path)
    groups = []
    for archive in source['source_archives']:
        payload = read(root / archive['path'])
        tasks = {task['name']: task for task in payload['tasks']}
        # These legacy archives do not serialize L; their frozen physical unit is100.
        level = float(archive.get('L', payload.get('L', 100)))
        for fold in range(source['folds']):
            split_path = (Path(source['output_dir']) / archive['heldout_target_family'] /
                          f'fold_{fold}' / 'split.json')
            held = read(root / split_path)['held']
            for family in archive['source_families']:
                points = np.asarray(tasks[family]['X_integer'], dtype=float)[held]
                fixed = nearest_pairs(public_coordinates(points, level), held)
                groups.append(dict(archive_heldout_family=archive['heldout_target_family'],
                    fold=fold, family=family, d=int(points.shape[1]), L=level,
                    held_policy_ids=list(held), archive_file=archive['path'],
                    split_file=str(split_path), **fixed))
    totals = {str(prefix): {group: sum(g['prefix_counts'][str(prefix)][group]
        for g in groups) for group in ('observed_observed', 'observed_unobserved', 'unobserved_unobserved')}
        for prefix in PREFIXES}
    return dict(status='complete', source_manifest=source_path, groups=groups,
        summary=dict(groups=len(groups), nodes=sum(len(g['nodes']) for g in groups),
            pairs=sum(len(g['pairs']) for g in groups), prefix_counts=totals,
            source_dimensions=sorted({g['d'] for g in groups})),
        representation=dict(function='representation.legacy_function_prior.legacy_public_coordinates',
            dimensions=10, input='X_integer/L',
            columns='First10 of existing65-column normalized profile map: mean, SD, eight weighted cosines.',
            fixed_scaling='Existing frequency_penalty0.10 and65-column row norm; no new center/scale fit.',
            dimension_semantics='The same midpoint-grid averages and frequency weights apply at d50 and d1000; no sqrt(d) factor.',
            target_reference='performance.mainline_role_prior.role_inputs uses this same public map before frozen source standardization.'),
        rule=dict(k=1, metric='squared Euclidean distance divided by10 on existing normalized public10',
            tie='First held position among equal-distance nonself neighbors.',
            union='Sorted undirected union of each held node nearest-neighbor edge.',
            prefix='The same pairs at10 and13; endpoint observation flags are annotations only.',
            previous_reference='Stage61 used the same k1/union/tie rule in normalized raw physical coordinates.'),
        inputs_used=['source manifest archive/family/fold order', 'archive name/X_integer and L if present',
                     'saved split held policy order'],
        response_fields_used=False, forecast_files_read=False, fitted_scales=0,
        limits=['Public10 proximity is a fixed diagnostic geometry, not the source-standardized kernel distance.',
            'The archive JSON container also stores outcomes; only public geometry fields are used in this phase.',
            'Source families/archives/folds overlap;24groups and repeated pairs are not independent tasks.',
            'Same cross-dimension coordinate semantics do not guarantee equal features for arbitrary quantized policies.',
            'The caller saves all24groups before any scoring phase; no response-dependent pair filtering is performed.'])
