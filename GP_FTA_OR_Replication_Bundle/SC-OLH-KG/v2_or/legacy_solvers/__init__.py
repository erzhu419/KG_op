"""Real legacy simulation-optimization solvers and trajectory archive I/O."""

from v2_or.legacy_solvers.solvers import (
    LEGACY_SOLVER_REGISTRY,
    TrajectoryArchiveRecorder,
    generate_legacy_archive,
    load_legacy_archive_json,
    run_constrained_cmaes,
    run_constrained_nsga2,
    run_heuristic_powell,
    run_penalty_spsa,
    save_legacy_archive_json,
)

__all__ = [
    "LEGACY_SOLVER_REGISTRY",
    "TrajectoryArchiveRecorder",
    "generate_legacy_archive",
    "load_legacy_archive_json",
    "run_constrained_cmaes",
    "run_constrained_nsga2",
    "run_heuristic_powell",
    "run_penalty_spsa",
    "save_legacy_archive_json",
]
