"""Problem registry for RZDT variance-study experiments.

The registry is the single place where synthetic benchmark names are mapped
to classes, metadata, and recommended variance-partition features.  Runners
should use this module instead of hard-coding problem classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from gpr_kg import (  # noqa: E402
    RCZDT_Curve2D,
    RCZDT_MisalignedV,
    RCZDT_StepV,
    RZDT1,
    RZDT1_VC,
    RZDT2,
    RZDT2_VC,
    RZDT5_RR,
    RZDT5_RR_VC,
)


@dataclass(frozen=True)
class ProblemSpec:
    """Metadata needed for reproducible problem construction."""

    name: str
    factory: Callable
    suite: str
    reference_point: tuple[float, float] = (1.5, 1.5)
    tau: float = 0.0
    variance_features: tuple[int, ...] = (0,)
    recommended_partition_features: tuple[int, ...] = (0,)
    description: str = ""


ORIGINAL_RZDT_PROBLEMS = ("RZDT1", "RZDT2", "RZDT5_RR")
VARIANCE_CRITICAL_PROBLEMS = ("RZDT1_VC", "RZDT2_VC", "RZDT5_RR_VC")
RCZDT_PROBLEMS = ("RCZDT-Curve2D", "RCZDT-MisalignedV", "RCZDT-StepV")
ALL_RZDT_PROBLEMS = (
    ORIGINAL_RZDT_PROBLEMS
    + VARIANCE_CRITICAL_PROBLEMS
    + RCZDT_PROBLEMS)


PROBLEM_REGISTRY: dict[str, ProblemSpec] = {
    "RZDT1": ProblemSpec(
        name="RZDT1",
        factory=RZDT1,
        suite="original_rzdt",
        description="Convex RZDT1 with monotone sqrt heteroscedasticity.",
    ),
    "RZDT2": ProblemSpec(
        name="RZDT2",
        factory=RZDT2,
        suite="original_rzdt",
        description="Concave RZDT2 with bell-shaped heteroscedasticity.",
    ),
    "RZDT5_RR": ProblemSpec(
        name="RZDT5_RR",
        factory=RZDT5_RR,
        suite="original_rzdt",
        description="Hyperbolic enlarged-grid RZDT5_RR.",
    ),
    "RZDT1_VC": ProblemSpec(
        name="RZDT1_VC",
        factory=RZDT1_VC,
        suite="variance_critical",
        description="Variance-critical RZDT1 constraint calibration.",
    ),
    "RZDT2_VC": ProblemSpec(
        name="RZDT2_VC",
        factory=RZDT2_VC,
        suite="variance_critical",
        description="Variance-critical RZDT2 constraint calibration.",
    ),
    "RZDT5_RR_VC": ProblemSpec(
        name="RZDT5_RR_VC",
        factory=RZDT5_RR_VC,
        suite="variance_critical",
        description="Variance-critical RZDT5_RR constraint calibration.",
    ),
    "RCZDT-Curve2D": ProblemSpec(
        name="RCZDT-Curve2D",
        factory=RCZDT_Curve2D,
        suite="rczdt",
        variance_features=(0, 1),
        recommended_partition_features=(0, 1),
        description=(
            "Two-coordinate curved Pareto set with center-peaked "
            "heteroscedasticity."),
    ),
    "RCZDT-MisalignedV": ProblemSpec(
        name="RCZDT-MisalignedV",
        factory=RCZDT_MisalignedV,
        suite="rczdt",
        variance_features=(2,),
        recommended_partition_features=(2,),
        description=(
            "Three-coordinate Pareto set where variance is mainly governed "
            "by x3 rather than the primary Pareto coordinate."),
    ),
    "RCZDT-StepV": ProblemSpec(
        name="RCZDT-StepV",
        factory=RCZDT_StepV,
        suite="rczdt",
        variance_features=(2,),
        recommended_partition_features=(2,),
        description=(
            "Piecewise Pareto manifold with region-type step "
            "heteroscedasticity."),
    ),
}


def problem_names_for_suite(suite: str) -> tuple[str, ...]:
    """Return problem names for a named benchmark suite."""
    if suite == "original_rzdt":
        return ORIGINAL_RZDT_PROBLEMS
    if suite == "variance_critical":
        return VARIANCE_CRITICAL_PROBLEMS
    if suite == "rczdt":
        return RCZDT_PROBLEMS
    if suite == "all":
        return ALL_RZDT_PROBLEMS
    raise ValueError(f"Unknown problem suite: {suite}")


def get_problem_spec(name: str) -> ProblemSpec:
    """Return registry metadata for one problem."""
    try:
        return PROBLEM_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(PROBLEM_REGISTRY))
        raise ValueError(f"Unknown problem {name!r}. Valid names: {valid}") from exc


def make_problem(name: str, d: int, sigma: float, alpha: float):
    """Instantiate a registered RZDT problem with paper-default settings."""
    spec = get_problem_spec(name)
    problem = spec.factory(
        d=d,
        L=100,
        sigma=sigma,
        heteroscedastic=True,
        alpha=alpha,
    )
    problem.tau = float(spec.tau)
    problem.ref_point = np.array(spec.reference_point, dtype=float)
    problem.problem_name = spec.name
    problem.problem_suite = spec.suite
    problem.variance_features = tuple(spec.variance_features)
    problem.recommended_partition_features = tuple(
        spec.recommended_partition_features)
    return problem
