"""Project the frozen source noise geometry onto saved d1000 policy banks.

No fitting, outcome evaluation, or target posterior updates occur here. The
source design parity ties the restored maps to the actual deployed source law.
"""

import numpy as np

from performance.diagnose_mainline_noise_boundary import decompose
from performance.mainline_legacy_source_models import _query_design
from representation.legacy_function_prior import legacy_public_coordinates
from representation.legacy_observable_prior import legacy_exposure_descriptors


def _coordinates(bundle, provider, points):
    points = np.asarray(points)
    descriptor = legacy_exposure_descriptors(provider, points, 50)
    encoded = bundle['encoder'].transform(descriptor)
    public = (legacy_public_coordinates(points / provider.L)
              - bundle['mean_center']) / bundle['mean_scale']
    risk = (encoded - bundle['risk_center']) / bundle['risk_scale']
    return descriptor, encoded, (public, risk)


def _eta_maps(bundle):
    seen = set()
    for identity, pair in zip(bundle['identities'], bundle['components']):
        kind, support, _ = identity.split('/')
        for pair_head, eta_head in ((0, 3), (1, 2)):
            key = (kind, support, eta_head if support == 'head_specific' else -1)
            if key in seen:
                continue
            seen.add(key)
            mass = sum(float(np.exp(w)) for name, w in
                       zip(bundle['identities'], bundle['log_weights'])
                       if name.startswith(f'{kind}/{support}/'))
            yield dict(key=key, identity=identity, eta_head=eta_head,
                       law=pair[pair_head]['risk'], source_identity_mass=mass)


def _source0(bundle, coordinates):
    result = {}
    for head, name in enumerate(('f', 'g')):
        row = decompose(bundle, coordinates, head, [])
        result[name] = {key: row[key] for key in
                        ('mean_eta', 'variance_eta', 'noise', 'variance_parts',
                         'log_noise_parts')}
    return result


def prepare_source_geometry(bundle, archive, providers):
    """Read the complete deployed archive and check all saved eta designs.

    This full 64-policy-per-family archive includes policy0. It is different
    from the 48-policy source fold used in the earlier boundary diagnostic.
    """
    pieces = [_coordinates(bundle, providers[task.name], task.X_integer)
              for task in archive.tasks]
    descriptor = np.vstack([p[0] for p in pieces])
    encoded = np.vstack([p[1] for p in pieces])
    coordinates = tuple(np.vstack([p[2][j] for p in pieces]) for j in range(2))
    errors, maps = {}, []
    for item in _eta_maps(bundle):
        design = _query_design(bundle, item['identity'], item['eta_head'],
                               coordinates, item['law'])
        expected = item['law']['source_design']
        np.testing.assert_allclose(design, expected, rtol=1e-9, atol=1e-10,
                                   err_msg='Restored source eta design mismatch')
        label = '/'.join(map(str, item['key']))
        errors[label] = float(np.max(np.abs(design - expected)))
        basis = bundle['maps'][item['key']]
        kind, support, _ = item['key']
        names = ([basis.library_names_[int(i)] for i in basis.selected_idx_]
                 if kind == 'spectral' else basis.function_names_)
        maps.append(dict(item, training_design_norm=np.linalg.norm(design, axis=1),
                         selected_features=names))
    source0 = _source0(bundle, coordinates)
    policy0, offset = [], 0
    for task in archive.tasks:
        policy0.append(dict(family=task.name, source_row=offset,
            physical_policy_mean=float(np.asarray(task.X_integer[0]).mean()),
            descriptor_second_bin=float(descriptor[offset, 13]),
            encoder_values=encoded[offset],
            eta_design_norm={str(m['key']): float(m['training_design_norm'][offset])
                             for m in maps},
            source0={name: {key: float(source0[name][key][offset]) for key in
                           ('mean_eta', 'variance_eta', 'noise')}
                     for name in ('f', 'g')}))
        offset += len(task.X_integer)
    summary = dict(source_sites=len(encoded), source_task_names=[t.name for t in archive.tasks],
        encoder_source_count=int(bundle['encoder'].source_count),
        encoder_min=encoded.min(axis=0), encoder_max=encoded.max(axis=0),
        encoder_standardized_norm_max=float(np.linalg.norm(coordinates[1], axis=1).max()),
        descriptor_second_bin_range=[float(descriptor[:, 13].min()), float(descriptor[:, 13].max())],
        policy0=policy0, source_design_parity=errors,
        maximum_source_design_error=max(errors.values()))
    return dict(coordinates=coordinates, descriptor=descriptor, encoded=encoded,
                eta_maps=maps, source0=source0, summary=summary)


def target_geometry(bundle, source, provider, points):
    """Return per-policy arrays for a saved bank using the exact role mapping."""
    descriptor, encoded, coordinates = _coordinates(bundle, provider, points)
    lo, hi = source['coordinates'][1].min(axis=0), source['coordinates'][1].max(axis=0)
    excess = np.maximum(np.maximum(lo - coordinates[1], coordinates[1] - hi), 0.)
    outside = np.any(excess > 1e-10, axis=1)
    maps, map_summary = [], []
    for item in source['eta_maps']:
        design = _query_design(bundle, item['identity'], item['eta_head'],
                               coordinates, item['law'])
        norms = np.linalg.norm(design, axis=1)
        source_max = float(item['training_design_norm'].max())
        ratio = norms / source_max
        kind, support, eta_head = item['key']
        info = dict(kind=kind, support=support,
            heads=['eta_f', 'eta_g'] if support == 'joint' else
                  ['eta_f' if eta_head == 3 else 'eta_g'],
            source_identity_mass=item['source_identity_mass'],
            selected_features=item['selected_features'],
            training_design_norm_max=source_max)
        maps.append(dict(info, design_norm=norms, design_norm_to_source_max=ratio))
        map_summary.append(dict(info, maximum_target_norm_ratio=float(ratio.max()),
                                count_above_source_max=int(np.sum(ratio > 1. + 1e-10))))
    source0 = _source0(bundle, coordinates)
    return dict(encoder_values=encoded, standardized_encoder_range_excess=excess,
        encoder_outside_source=outside,
        standardized_encoder_norm=np.linalg.norm(coordinates[1], axis=1),
        descriptor_second_bin=descriptor[:, 13], eta_maps=maps, source0=source0,
        summary=dict(policy_count=len(encoded), encoder_outside_source_count=int(outside.sum()),
            encoder_maximum_standardized_range_excess=float(excess.max()),
            eta_maps=map_summary,
            source0={name: {key: [float(source0[name][key].min()), float(source0[name][key].max())]
                           for key in ('mean_eta', 'variance_eta', 'noise')}
                     for name in ('f', 'g')}))
