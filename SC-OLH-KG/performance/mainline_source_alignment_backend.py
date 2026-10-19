"""Backend-only source intervention; the target frontend keeps its real prior.

This module constructs a projection context from already supplied source
objects. It does not permute archives, fit models, or start target observations.
"""
from copy import copy

from algorithms.single_olhkg import SingleOLHKGAlgorithm
from core.gpr import ParametricGPR
from representation.meta_prior import ObservableConstraintMeanBasis


SOURCE_PERMUTED_ROLE_OPERATORS = {
    'source_permuted_role_after_init_point_only': 'after_init_point_only',
    'source_permuted_role_point_only': 'point_only',
}
SOURCE_PERMUTED_ROLE_ARMS = frozenset(SOURCE_PERMUTED_ROLE_OPERATORS)


def build_role_projection_context(frontend, source_prior):
    """Use the actual target geometry/config with a separate native risk law.

    The two source-law methods need only problem, config and gpr. Skipping the
    optimizer constructor avoids target initialization and any optimizer RNG
    consumption. The new basis/model never replace the frontend's proposal GPR
    or its MetaPriorProblemAdapter. No target observations enter this context.
    """
    context = object.__new__(SingleOLHKGAlgorithm)
    context.problem = frontend.problem
    context.config = copy(frontend.config)
    context.source_prior = source_prior
    context.gpr = [None, ParametricGPR(
        d=frontend.problem.d,
        basis_map=ObservableConstraintMeanBasis(source_prior, frontend.problem))]
    return context
