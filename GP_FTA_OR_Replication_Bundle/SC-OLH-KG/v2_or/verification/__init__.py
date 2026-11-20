"""Distribution-adaptive exact gated terminal verification & gate-power primitives."""

from v2_or.verification.gated_verifier_v2 import (
    GateCertificate,
    TerminalDeploymentReceipt,
    build_feasible_aware_shortlist,
    exact_binomial_critical_k,
    exact_binomial_gate,
    exact_binomial_pass_power,
    gaussian_noncentral_t_gate,
    posterior_binomial_gate_power,
    select_empirical_feasible_incumbent,
    select_feasible_aware_initial_incumbent,
    verify_and_deploy_shortlist,
)

__all__ = [
    "GateCertificate",
    "TerminalDeploymentReceipt",
    "build_feasible_aware_shortlist",
    "exact_binomial_critical_k",
    "exact_binomial_gate",
    "exact_binomial_pass_power",
    "gaussian_noncentral_t_gate",
    "posterior_binomial_gate_power",
    "select_empirical_feasible_incumbent",
    "select_feasible_aware_initial_incumbent",
    "verify_and_deploy_shortlist",
]
