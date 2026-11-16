"""Verification-Aware Knowledge Gradient (`V-KG`) and sequential optimization loop."""

from v2_or.acquisition.verification_kg import (
    SequentialRunResult,
    generate_dynamic_candidate_pool,
    run_tls_olh_kg,
    score_constrained_ei,
    score_constrained_ts,
    score_verification_kg,
)

__all__ = [
    "SequentialRunResult",
    "generate_dynamic_candidate_pool",
    "run_tls_olh_kg",
    "score_constrained_ei",
    "score_constrained_ts",
    "score_verification_kg",
]
