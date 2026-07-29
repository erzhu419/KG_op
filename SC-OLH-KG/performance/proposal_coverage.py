"""Numerical bridge for the source-to-target proposal coverage theorem."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ProposalCoverageInputs:
    """Auditable finite-task quantities used by the Lean theorem."""

    source_miss: float
    domain_shift: float
    effective_dim: float
    log_library: float
    inverse_confidence_log: float
    source_samples: int
    n0: int

    def validate(self):
        values = asdict(self)
        for name, value in values.items():
            if name in {"source_samples", "n0"}:
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(self.source_miss) <= 1.0:
            raise ValueError("source_miss must lie in [0, 1]")
        if float(self.domain_shift) < 0.0:
            raise ValueError("domain_shift must be nonnegative")
        if float(self.effective_dim) < 0.0:
            raise ValueError("effective_dim must be nonnegative")
        if float(self.log_library) < 0.0:
            raise ValueError("log_library must be nonnegative")
        if float(self.inverse_confidence_log) < 0.0:
            raise ValueError("inverse_confidence_log must be nonnegative")
        if int(self.source_samples) <= 0:
            raise ValueError("source_samples must be positive")
        if int(self.n0) <= 0:
            raise ValueError("n0 must be positive")
        return self


def effective_dimension_transfer_radius(
    effective_dim,
    log_library,
    inverse_confidence_log,
    source_samples,
):
    """Match ``effectiveDimensionTransferRadius`` in Lean exactly."""

    source_samples = int(source_samples)
    if source_samples <= 0:
        raise ValueError("source_samples must be positive")
    return (
        float(effective_dim) * float(log_library)
        + float(inverse_confidence_log)
    ) / source_samples


def feasible_mass_lower_bound(inputs):
    """Return the conservative one-draw heldout feasible-mass lower bound."""

    inputs = inputs.validate()
    radius = effective_dimension_transfer_radius(
        inputs.effective_dim,
        inputs.log_library,
        inputs.inverse_confidence_log,
        inputs.source_samples,
    )
    return max(
        0.0,
        1.0 - (
            float(inputs.source_miss)
            + float(inputs.domain_shift)
            + radius
        ),
    )


def iid_hit_probability_lower_bound(feasible_mass_lower, n0):
    """Return ``1 - (1 - p_lower)^n0`` with probability-domain checks."""

    feasible_mass_lower = float(feasible_mass_lower)
    n0 = int(n0)
    if not 0.0 <= feasible_mass_lower <= 1.0:
        raise ValueError("feasible_mass_lower must lie in [0, 1]")
    if n0 <= 0:
        raise ValueError("n0 must be positive")
    return 1.0 - (1.0 - feasible_mass_lower) ** n0


def proposal_coverage_audit(inputs):
    """Serialize every theorem input and both resulting lower bounds."""

    inputs = inputs.validate()
    radius = effective_dimension_transfer_radius(
        inputs.effective_dim,
        inputs.log_library,
        inputs.inverse_confidence_log,
        inputs.source_samples,
    )
    feasible_mass_lower = feasible_mass_lower_bound(inputs)
    hit_lower = iid_hit_probability_lower_bound(feasible_mass_lower, inputs.n0)
    return {
        "theory_contract_id": "source_target_proposal_coverage_v1",
        "inputs": asdict(inputs),
        "effective_dimension_transfer_radius": radius,
        "single_draw_feasible_mass_lower": feasible_mass_lower,
        "n0_at_least_one_hit_probability_lower": hit_lower,
        "formula": "1-(1-p_lower)^n0",
        "target_outcomes_used_to_fit_proposal": False,
        "requires_source_only_domain_shift_calibration": True,
    }
