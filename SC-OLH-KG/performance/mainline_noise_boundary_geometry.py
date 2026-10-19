"""Deterministic source-policy geometry for the frozen eta extrapolation audit."""
import numpy as np

from representation.legacy_function_prior import legacy_public_coordinates
from representation.legacy_observable_prior import legacy_exposure_descriptors


def _rows(values, training, focus):
    values = np.asarray(values)
    norms = np.linalg.norm(values, axis=1)
    return dict(training_min=values[training].min(axis=0).tolist(),
                training_max=values[training].max(axis=0).tolist(),
                training_norm_range=[float(norms[training].min()), float(norms[training].max())],
                focus=values[focus].tolist(), focus_norm=norms[focus].tolist())


def geometry(bundle, task, provider, training_ids, focus_ids):
    """Read X and fitted maps only; no outcomes, fitting, or simulator calls.

    Columns 13 and 66 are the ordered descriptor's second occupancy bin and
    its compressed square in the encoder input library. Their contributions
    below sum exactly with the remaining terms in each normalized latent axis;
    they are an algebraic attribution, not a modified-policy experiment.
    """
    training = np.asarray(training_ids, dtype=int)
    focus = np.asarray(focus_ids, dtype=int)
    descriptor = legacy_exposure_descriptors(provider, task.X_integer, 50)
    encoder = bundle['encoder']
    encoded = encoder.transform(descriptor)
    public = (legacy_public_coordinates(task.X)-bundle['mean_center'])/bundle['mean_scale']
    risk = (encoded-bundle['risk_center'])/bundle['risk_scale']
    bin_values = descriptor[:, 13]
    result = dict(family=task.name, training_policy_ids=training.tolist(),
        focus_policy_ids=focus.tolist(), focus_policy_mean=task.X_integer[focus].mean(axis=1).tolist(),
        descriptor=_rows(descriptor, training, focus), encoder=_rows(encoded, training, focus),
        standardized_public=_rows(public, training, focus),
        standardized_risk=_rows(risk, training, focus),
        occupancy_second_bin=dict(interval='[0.125, 0.25)', descriptor_column=13,
            compressed_square_library_column=66,
            training_mean=float(bin_values[training].mean()),
            training_sd=float(bin_values[training].std()),
            training_nonzero=[[int(i), float(bin_values[i])] for i in training if bin_values[i] > 0],
            focus=bin_values[focus].tolist()), encoder_axes=[], eta_maps=[])

    for name, model in [('f_mean', encoder.f_mean), ('g_mean', encoder.g_mean),
                        ('eta_g', encoder.g_variance.coordinate),
                        ('eta_f', encoder.f_variance.coordinate)]:
        assert model.latent_transform == 'identity' and model.feature_mode == 'linear'
        library = np.vstack([model._input_library(row) for row in descriptor])
        z = (library-model.feature_mean)/model.feature_scale
        loading = model.alignment_components.T/model.latent_scale
        latent = z@loading-model.latent_mean/model.latent_scale
        compressed = z[focus, 13, None]*loading[13]
        squared = z[focus, 66, None]*loading[66]
        exported = np.vstack([model.features_profile(descriptor[i]) for i in focus])
        error = float(np.max(np.abs(exported-latent[focus])))
        assert error < 1e-10
        result['encoder_axes'].append(dict(name=name, normalized_latent=_rows(latent, training, focus),
            focus_compressed_bin_z=z[focus, 13].tolist(),
            focus_squared_bin_z=z[focus, 66].tolist(),
            compressed_bin_latent_contribution=compressed.tolist(),
            squared_bin_latent_contribution=squared.tolist(),
            remaining_latent_contribution=(latent[focus]-compressed-squared).tolist(),
            maximum_reconstruction_error=error))

    seen = set()
    for identity in bundle['identities']:
        kind, support, _ = identity.split('/')
        for head in (2, 3):
            key = (kind, support, head if support == 'head_specific' else -1)
            if key in seen:
                continue
            seen.add(key)
            basis = bundle['maps'][key]
            inputs = np.column_stack([public, risk]) if support == 'joint' else risk
            selected = basis._standardized_library(inputs)[:, basis.selected_idx_]
            features = basis.transform(inputs)
            reconstructed = selected@basis.whitening_
            error = float(np.max(np.abs(reconstructed-features)))
            assert error < 1e-10
            names = ([basis.library_names_[int(i)] for i in basis.selected_idx_]
                     if kind == 'spectral' else basis.function_names_)
            singular = np.linalg.svd(basis.whitening_, compute_uv=False)
            mass = sum(float(np.exp(w)) for ident, w in zip(bundle['identities'], bundle['log_weights'])
                       if ident.startswith(f'{kind}/{support}/'))
            result['eta_maps'].append(dict(kind=kind, support=support,
                heads=['eta_g', 'eta_f'] if support == 'joint' else ['eta_g' if head == 2 else 'eta_f'],
                source_identity_mass=mass, selected_features=names,
                selected_before_whitening=_rows(selected, training, focus),
                design=_rows(np.column_stack([np.ones(len(inputs)), features]), training, focus),
                whitening_singular_values=singular.tolist(),
                maximum_reconstruction_error=error))
    return result
