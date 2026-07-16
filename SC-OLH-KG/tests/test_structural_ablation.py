from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.structural_ablation import (  # noqa: E402
    PRIOR_COMPONENTS,
    STRUCTURAL_PRIOR_PROFILES,
    apply_hvd_profile,
    apply_structural_prior_profile,
)
from representation.transferable_spectral import (  # noqa: E402
    SourceDomainBatch,
    TransferableSpectralBasis,
)
from variance.orthogonal_hvd import OrthogonalHVD  # noqa: E402


def _enabled_components(profile):
    return {
        "low_frequency": bool(profile["meta_spectral_low_frequency_prior"]),
        "orthogonality": profile["meta_spectral_orthogonalization"] != "none",
        "sparsity": bool(profile["meta_spectral_coefficient_shrinkage"]),
        "additivity": bool(profile["meta_spectral_additive_adaptation"]),
    }


def test_only_and_leave_one_out_profiles_change_exactly_declared_priors():
    for component in PRIOR_COMPONENTS:
        only = _enabled_components(
            STRUCTURAL_PRIOR_PROFILES[f"{component}_only"])
        assert {name for name, enabled in only.items() if enabled} == {component}
        leave_out = _enabled_components(
            STRUCTURAL_PRIOR_PROFILES[f"leave_out_{component}"])
        assert {name for name, enabled in leave_out.items() if enabled} == (
            set(PRIOR_COMPONENTS) - {component})


def test_profiles_are_applied_without_mutating_unrelated_budget_fields():
    config = {"N": 20, "n0": 10, "variance_mode": "class"}
    apply_structural_prior_profile(config, "full")
    apply_hvd_profile(config, "factor_cumulative")
    assert config["N"] == 20
    assert config["n0"] == 10
    assert config["structural_prior_profile"] == "full"
    assert config["variance_mode"] == "factor"
    assert config["hvd_use_cumulative_provider"] is True


def test_low_frequency_prior_is_a_real_spectral_selection_switch():
    rng = np.random.default_rng(11)
    batches = []
    for domain in ("a", "b"):
        psi = rng.normal(size=(12, 3))
        signals = np.column_stack([
            psi[:, 0],
            np.sin(psi[:, 1]),
        ])
        batches.append(SourceDomainBatch(domain, psi, signals))
    common = dict(active_dim=3, max_library_size=20, low_frequency_components=4)
    low = TransferableSpectralBasis(
        **common, use_low_frequency_score=True).fit(batches)
    no_low = TransferableSpectralBasis(
        **common, use_low_frequency_score=False).fit(batches)
    assert low.diagnostics()["low_frequency_prior"] is True
    assert no_low.diagnostics()["low_frequency_prior"] is False
    assert low.fingerprint() != no_low.fingerprint()


def test_pointwise_factor_profile_cannot_consume_provider_features():
    class Provider:
        @staticmethod
        def cumulative_risk_features(x, output_index=0):
            del x, output_index
            return np.array([1.0, 2.0, 3.0])

    disabled = OrthogonalHVD(
        mode="factor", use_cumulative_provider=False)
    enabled = OrthogonalHVD(
        mode="factor", use_cumulative_provider=True)
    assert disabled._cumulative_features((1, 2), Provider(), 1) is None
    np.testing.assert_allclose(
        enabled._cumulative_features((1, 2), Provider(), 1),
        [1.0, 2.0, 3.0],
    )
