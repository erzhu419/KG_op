import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.proposal_coverage import (  # noqa: E402
    ProposalCoverageInputs,
    deterministic_atlas_coverage_audit,
    effective_dimension_transfer_radius,
    feasible_mass_lower_bound,
    geometric_atlas_coverage_audit,
    iid_hit_probability_lower_bound,
    proposal_coverage_audit,
    normalized_ranks,
    rank_aligned_atlas_coverage_audit,
    source_only_rank_alignment_calibration,
)


def test_effective_dimension_radius_matches_registered_formula():
    assert effective_dimension_transfer_radius(2, 0.5, 1.0, 10) == 0.2


def test_feasible_mass_and_n0_hit_lower_bound():
    inputs = ProposalCoverageInputs(
        source_miss=0.1,
        domain_shift=0.05,
        effective_dim=2.0,
        log_library=0.5,
        inverse_confidence_log=1.0,
        source_samples=10,
        n0=10,
    )
    assert feasible_mass_lower_bound(inputs) == pytest.approx(0.65)
    assert iid_hit_probability_lower_bound(0.65, 10) == pytest.approx(
        1.0 - 0.35**10)
    audit = proposal_coverage_audit(inputs)
    assert audit["theory_contract_id"] == (
        "source_target_proposal_coverage_v1")
    assert audit["n0_at_least_one_hit_probability_lower"] > 0.999


def test_vacuous_transfer_bound_is_reported_as_zero_not_hidden():
    inputs = ProposalCoverageInputs(
        source_miss=0.8,
        domain_shift=0.3,
        effective_dim=2.0,
        log_library=1.0,
        inverse_confidence_log=1.0,
        source_samples=10,
        n0=10,
    )
    assert feasible_mass_lower_bound(inputs) == 0.0
    assert proposal_coverage_audit(inputs)[
        "n0_at_least_one_hit_probability_lower"] == 0.0


def test_deterministic_atlas_uses_support_coverage_not_iid_formula():
    inputs = ProposalCoverageInputs(
        source_miss=0.1,
        domain_shift=0.05,
        effective_dim=2.0,
        log_library=0.5,
        inverse_confidence_log=1.0,
        source_samples=10,
        n0=10,
    )
    audit = deterministic_atlas_coverage_audit(
        inputs,
        atlas_size=10,
        unique_design_fingerprints=1,
    )
    assert audit["independent_draws_assumed"] is False
    assert audit["finite_atlas_feasible_mass_lower"] == pytest.approx(0.65)
    assert audit["positive_mass_certifies_feasible_support_member"] is True
    assert "n0_at_least_one_hit_probability_lower" not in audit

    with pytest.raises(ValueError):
        deterministic_atlas_coverage_audit(
            inputs,
            atlas_size=11,
            unique_design_fingerprints=1,
        )


def test_rank_aligned_atlas_coverage_uses_source_calibration_and_safe_depth():
    source_a = normalized_ranks([0.0, 1.0, 2.0, 3.0])
    source_b = normalized_ranks([0.0, 1.1, 2.2, 3.3])
    calibration = source_only_rank_alignment_calibration(
        [source_a, source_b], delta=0.05)
    assert calibration["target_labels_used"] is False

    audit = rank_aligned_atlas_coverage_audit(
        source_rank=[0.0, 0.2, 0.8, 1.0],
        target_rank=[0.0, 0.2, 0.8, 1.0],
        target_feasible=[True, True, False, False],
        atlas_indices=[0, 2],
        source_only_alignment_bound=0.0,
    )
    assert audit["one_sided_source_rank_atlas_cover_error"] == 0.0
    assert audit["theorem_conditions_hold"] is True
    assert audit["observed_atlas_contains_feasible"] is True


def test_geometric_atlas_coverage_uses_cover_shift_and_safe_radius():
    audit = geometric_atlas_coverage_audit(
        atlas_coordinate=[[0.0], [1.0]],
        source_support_coordinate=[[0.0], [0.9]],
        target_library_coordinate=[[0.0], [0.2], [0.9], [1.0]],
        target_feasible=[True, True, False, False],
        atlas_target_indices=[0, 3],
        target_margin=[-1.0, -0.5, 0.2, 0.3],
    )
    assert audit["source_support_atlas_cover_radius"] == pytest.approx(0.1)
    assert audit["finite_library_theorem_conditions_hold"] is True
    assert audit["observed_atlas_contains_feasible"] is True
    assert audit["lipschitz_audit"] is not None


@pytest.mark.parametrize(
    "inputs",
    [
        ProposalCoverageInputs(-0.1, 0, 1, 1, 1, 10, 10),
        ProposalCoverageInputs(0.1, -0.1, 1, 1, 1, 10, 10),
        ProposalCoverageInputs(0.1, 0.1, -1, 1, 1, 10, 10),
        ProposalCoverageInputs(0.1, 0.1, 1, 1, 1, 0, 10),
        ProposalCoverageInputs(0.1, 0.1, 1, 1, 1, 10, 0),
    ],
)
def test_invalid_theorem_inputs_are_rejected(inputs):
    with pytest.raises(ValueError):
        proposal_coverage_audit(inputs)
