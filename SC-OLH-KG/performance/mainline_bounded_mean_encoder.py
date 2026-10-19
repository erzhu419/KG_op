"""One source-only candidate: bound the observable f/g mean latent axes.

The full current source likelihood is refitted. Only the two mean-coordinate
constructors change; the eta-specific coordinates keep their original recipe.
"""

from copy import deepcopy
from functools import partial
from unittest.mock import patch

from performance.mainline_legacy_source_models import fit_source_models
from representation import legacy_observable_prior
from representation.boundary_coordinate import SourceAlignedBoundaryCoordinate


def fit_candidate_source_models(training_archive, source_problems, *, settings=None):
    """Fit the unchanged source bundle with training-selected bounded mean axes.

    The existing source_tanh selector reuses the alignment and standardization
    learned from every supplied training row inside its domain-wise coefficient
    holdouts. Its scores are temperature-selection diagnostics, not a nested
    estimate of unseen-domain accuracy. Outer held-policy rows remain excluded.
    """
    constructor = partial(SourceAlignedBoundaryCoordinate, latent_transform="source_tanh")
    # variance_coordinate has its own imported class reference, so this local
    # replacement reaches f_mean/g_mean only and is restored after the fit.
    with patch.object(legacy_observable_prior, "SourceAlignedBoundaryCoordinate", constructor):
        bundle = fit_source_models(training_archive, source_problems, settings=settings)

    encoder = bundle["encoder"]
    heads = {}
    for name in ("f_mean", "g_mean"):
        heads[name] = deepcopy(getattr(encoder, name).latent_transform_diagnostics)
        heads[name]["selection_target"] = (
            "source_objective_mean" if name == "f_mean" else "source_centered_constraint_mean")
    bundle["diagnostics"]["bounded_mean_encoder"] = {
        "mean_heads": heads,
        "source_record_count": int(encoder.source_count),
        "eta_specific_encoder_transforms": {
            name: getattr(encoder, name).coordinate.latent_transform
            for name in ("f_variance", "g_variance")},
        "selection_scope": "supplied outer-fold training rows only",
        "inner_domain_holdout_reuses_supervised_embedding": True,
        "inner_scores_are_nested_generalization_estimates": False,
        "full_source_likelihood_refitted": True,
    }
    return bundle
