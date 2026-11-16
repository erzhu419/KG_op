"""Distribution-adaptive exact gated terminal verification."""

from v2_or.verification.gated_verifier_v2 import (
    GateCertificate,
    TerminalDeploymentReceipt,
    build_feasible_aware_shortlist,
    exact_binomial_gate,
    gaussian_noncentral_t_gate,
    select_feasible_aware_initial_incumbent,
    verify_and_deploy_shortlist,
)

__all__ = [
    "GateCertificate",
    "TerminalDeploymentReceipt",
    "build_feasible_aware_shortlist",
    "exact_binomial_gate",
    "gaussian_noncentral_t_gate",
    "select_feasible_aware_initial_incumbent",
    "verify_and_deploy_shortlist",
]
