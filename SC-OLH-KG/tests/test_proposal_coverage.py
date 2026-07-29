import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.proposal_coverage import (  # noqa: E402
    ProposalCoverageInputs,
    effective_dimension_transfer_radius,
    feasible_mass_lower_bound,
    iid_hit_probability_lower_bound,
    proposal_coverage_audit,
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
