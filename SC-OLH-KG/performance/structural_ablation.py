"""Auditable structural-prior and HVD ablation profiles."""

from __future__ import annotations


PRIOR_COMPONENTS = (
    "low_frequency",
    "orthogonality",
    "sparsity",
    "additivity",
)


ORDERED_DEFAULTS = {
    "frequency_penalty": 0.10,
    "basis_mode": "diagonal_quadratic",
}


def _profile(*enabled):
    active = set(enabled)
    unknown = active.difference(PRIOR_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown structural-prior components: {sorted(unknown)}")
    low_frequency = "low_frequency" in active
    orthogonality = "orthogonality" in active
    sparsity = "sparsity" in active
    additivity = "additivity" in active
    return {
        "meta_spectral_low_frequency_prior": "low_frequency" in active,
        "meta_spectral_frequency_adaptation": "low_frequency" in active,
        "meta_spectral_orthogonalization": (
            "symmetric" if "orthogonality" in active else "none"),
        "meta_spectral_coefficient_shrinkage": "sparsity" in active,
        # Spike-and-slab target gating is a stronger challenger, not the
        # definition of the source-learned sparse coefficient prior.
        "meta_spectral_adaptive_sparsity": False,
        "meta_spectral_additive_adaptation": "additivity" in active,
        # The ordered cumulative-risk implementation realizes the same four
        # assumptions through a different basis.  These switches must move
        # with their spectral counterparts or a nominal ``none`` row still
        # contains the assumptions being ablated.
        "meta_ordered_exposure_frequency_penalty": (
            ORDERED_DEFAULTS["frequency_penalty"] if low_frequency else 0.0
        ),
        "meta_ordered_exposure_orthogonal_coordinates": orthogonality,
        "meta_ordered_exposure_adaptive_sparsity": sparsity,
        "meta_ordered_exposure_latent_structure_selection": sparsity,
        "meta_ordered_exposure_group_shared_shrinkage": False,
        "meta_ordered_exposure_group_ridge_learning": sparsity,
        # Diagonal quadratic blocks are the ordered additive model.  The
        # control keeps the same A,N coordinate but restores all pairwise
        # interactions instead of silently retaining additivity.
        "meta_ordered_exposure_basis_mode": (
            ORDERED_DEFAULTS["basis_mode"] if additivity else "full_quadratic"
        ),
        "structural_prior_active_components": sorted(active),
    }


STRUCTURAL_PRIOR_PROFILES = {
    "none": _profile(),
    "low_frequency_only": _profile("low_frequency"),
    "orthogonality_only": _profile("orthogonality"),
    "sparsity_only": _profile("sparsity"),
    "additivity_only": _profile("additivity"),
    "full": _profile(*PRIOR_COMPONENTS),
    "leave_out_low_frequency": _profile(
        "orthogonality", "sparsity", "additivity"),
    "leave_out_orthogonality": _profile(
        "low_frequency", "sparsity", "additivity"),
    "leave_out_sparsity": _profile(
        "low_frequency", "orthogonality", "additivity"),
    "leave_out_additivity": _profile(
        "low_frequency", "orthogonality", "sparsity"),
}


HVD_PROFILES = {
    "pooled": {
        "variance_mode": "pooled",
        "hvd_use_cumulative_provider": False,
    },
    "class": {
        "variance_mode": "class",
        "hvd_use_cumulative_provider": False,
    },
    "orthogonal_pointwise": {
        "variance_mode": "orthogonal",
        "hvd_use_cumulative_provider": False,
    },
    "factor_pointwise": {
        "variance_mode": "factor",
        "hvd_use_cumulative_provider": False,
    },
    "factor_cumulative": {
        "variance_mode": "factor",
        "hvd_use_cumulative_provider": True,
    },
}


def apply_structural_prior_profile(config, name):
    profile = str(name or "inherit")
    if profile == "inherit":
        return config
    if profile not in STRUCTURAL_PRIOR_PROFILES:
        raise ValueError(f"unknown structural-prior profile {profile!r}")
    config.update(STRUCTURAL_PRIOR_PROFILES[profile])
    config["structural_prior_profile"] = profile
    return config


def apply_hvd_profile(config, name):
    profile = str(name or "inherit")
    if profile == "inherit":
        return config
    if profile not in HVD_PROFILES:
        raise ValueError(f"unknown HVD profile {profile!r}")
    config.update(HVD_PROFILES[profile])
    config["hvd_ablation_profile"] = profile
    return config
