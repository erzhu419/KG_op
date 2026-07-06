"""Admissibility audit helpers for structural-prior experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AdmissibilityAudit:
    """Machine-readable information-use declaration for a run variant."""

    variant: str
    uses_true_objective: bool = False
    uses_true_constraint: bool = False
    uses_true_sigma: bool = False
    uses_true_optimum: bool = False
    uses_true_boundary: bool = False
    uses_hidden_axis: bool = False
    uses_problem_specific_formula: bool = False
    uses_problem_initial_samples: bool = False
    uses_problem_state_anchors: bool = False
    uses_problem_refinement: bool = False
    uses_target_eval_data: bool = True
    uses_source_data: bool = False
    uses_frozen_meta_prior: bool = False
    notes: str = ""

    @property
    def admissible_mainline(self):
        """Whether this variant is admissible as a non-oracle main result."""

        forbidden = (
            self.uses_true_objective
            or self.uses_true_constraint
            or self.uses_true_sigma
            or self.uses_true_optimum
            or self.uses_true_boundary
            or self.uses_hidden_axis
            or self.uses_problem_specific_formula
            or self.uses_problem_initial_samples
            or self.uses_problem_state_anchors
            or self.uses_problem_refinement
        )
        return bool(not forbidden)

    def to_dict(self):
        out = asdict(self)
        out["admissible_mainline"] = self.admissible_mainline
        return out


def strict_universal_audit():
    return AdmissibilityAudit(
        variant="strict_universal",
        notes="No source data and target-specific structural hooks hidden.",
    )


def lodo_meta_prior_audit():
    return AdmissibilityAudit(
        variant="lodo_meta_prior",
        uses_source_data=True,
        uses_frozen_meta_prior=True,
        notes=(
            "Frozen source-trained meta-prior; target hand-coded structural "
            "hooks are hidden from candidate generation and HVD."
        ),
    )


def domain_tuned_audit():
    return AdmissibilityAudit(
        variant="domain_tuned_upper_bound",
        uses_problem_specific_formula=True,
        uses_problem_initial_samples=True,
        uses_problem_state_anchors=True,
        uses_problem_refinement=True,
        notes="Problem-defined anchors/refinement/risk coordinates are enabled.",
    )
