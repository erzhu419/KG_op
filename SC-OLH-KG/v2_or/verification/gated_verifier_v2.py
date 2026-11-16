"""Feasible-Aware 3-Slot Shortlist & Distribution-Adaptive Exact Gated Verifier.

Implements Theorem 3(a) of the Path-B Operations Research reconstruction:
1. Fixes the legacy Slot-2 shortlist bug: `select_feasible_aware_initial_incumbent`
   filters initial design points by empirical chance-safety margin before ranking by
   objective, never wasting Slot-2 verification calls on blatantly infeasible points.
2. Supports two exact finite-sample safety gates with strict familywise error rate
   `P(deploy any infeasible policy) <= delta_1 + delta_2 + delta_3 = delta`:
   - `"exact_binomial"`: Clopper-Pearson / exact Binomial tail test on
     `V = sum I{Y_g > 0} ~ Binomial(K, p_viol)` requiring `P_{Bin(K, alpha)}(V <= v_obs) <= delta_i`.
     Valid for ANY simulation noise distribution (skewed, heavy-tailed, discrete, multimodal).
   - `"gaussian_noncentral_t"`: Exact noncentral Student-t upper tolerance bound on
     `mu_g + z_{1-alpha} * sigma_g <= 0` when constraint noise is Gaussian.
3. Paired/independent objective guard between Slot 1 (challenger) and Slot 2 (initial
   safe incumbent) prevents online search regressions while preserving exact FWER `<= delta`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.stats import binom, nct, norm

from v2_or.contracts import BaseDynamicORProblem, SufficientStatsRecord


@dataclass(frozen=True)
class GateCertificate:
    """Result of testing a single frozen candidate policy on fresh verification samples."""

    slot_index: int
    role: str
    certified: bool
    method: str
    n_samples: int
    delta_allocated: float
    p_value_or_margin: float
    observed_violation_rate: float
    sample_mean_obj: float
    sample_mean_con: float


@dataclass(frozen=True)
class TerminalDeploymentReceipt:
    """Complete auditable record of shortlist verification and final policy deployment."""

    deployed_policy: np.ndarray
    certified: bool
    selected_slot: int | None
    selected_role: str
    verification_calls_used: int
    familywise_delta: float
    certificates: tuple[GateCertificate, ...]
    shortlist_roles: tuple[str, ...]


def exact_binomial_gate(
    con_samples: np.ndarray,
    *,
    alpha: float,
    delta_slot: float,
) -> tuple[bool, float, float]:
    """Test `H_0: P(Y_g > 0) > alpha` at exact level `delta_slot` with zero distributional assumptions."""
    vals = np.asarray(con_samples, dtype=float).reshape(-1)
    k_tot = int(vals.size)
    if k_tot < 1:
        return False, 1.0, 1.0
    v_obs = int(np.sum(vals > 0.0))
    # Exact upper tail probability P_{p = alpha}(Binomial(K, alpha) <= v_obs)
    p_val = float(binom.cdf(v_obs, k_tot, float(alpha)))
    return bool(p_val <= float(delta_slot)), p_val, float(v_obs / k_tot)


def gaussian_noncentral_t_gate(
    con_samples: np.ndarray,
    *,
    alpha: float,
    delta_slot: float,
) -> tuple[bool, float, float]:
    """Exact noncentral-t upper confidence bound on `mu_g + z_{1-alpha} * sigma_g <= 0`."""
    vals = np.asarray(con_samples, dtype=float).reshape(-1)
    n = int(vals.size)
    if n < 2:
        return False, float("inf"), 1.0
    mean_g = float(np.mean(vals))
    std_g = float(np.std(vals, ddof=1))
    z_a = float(norm.ppf(1.0 - float(alpha)))
    nc = z_a * np.sqrt(float(n))
    q_nct = float(nct.ppf(1.0 - float(delta_slot), n - 1, nc))
    upper_margin = mean_g + (q_nct / np.sqrt(float(n))) * std_g
    viol_rate = float(np.mean(vals > 0.0))
    return bool(upper_margin <= 0.0), float(upper_margin), viol_rate


def select_feasible_aware_initial_incumbent(
    initial_records: Sequence[SufficientStatsRecord],
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    """Select the lowest-objective empirically margin-safe policy from initial evaluations."""
    if not initial_records:
        raise ValueError("initial_records cannot be empty")
    z_a = float(norm.ppf(1.0 - float(alpha)))
    safe_candidates: list[tuple[float, float, np.ndarray]] = []
    all_candidates: list[tuple[float, float, np.ndarray]] = []

    for rec in initial_records:
        std_g = float(np.sqrt(max(rec.ssq_con / max(rec.n_reps - 1, 1), 1e-8))) if rec.n_reps > 1 else 0.05
        emp_margin = float(rec.mean_con + 0.85 * z_a * std_g)
        all_candidates.append((emp_margin, float(rec.mean_obj), rec.x_norm))
        if emp_margin <= 0.0 and rec.violation_count == 0:
            safe_candidates.append((float(rec.mean_obj), emp_margin, rec.x_norm))

    if safe_candidates:
        safe_candidates.sort(key=lambda t: (t[0], t[1]))
        return safe_candidates[0][2].copy()

    # If no initial policy is empirically safe, pick the least-violating initial policy
    all_candidates.sort(key=lambda t: (t[0], t[1]))
    return all_candidates[0][2].copy()


def build_feasible_aware_shortlist(
    *,
    candidate_pool: Sequence[np.ndarray],
    mean_obj: np.ndarray,
    mean_con: np.ndarray,
    std_con: np.ndarray,
    ucb_aleat_var_con: np.ndarray,
    r_perp_con: np.ndarray,
    initial_records: Sequence[SufficientStatsRecord],
    alpha: float = 0.05,
    epistemic_beta: float = 1.28,
) -> list[tuple[str, np.ndarray]]:
    """Construct up to 3 distinct frozen candidates: Primary, Safe-Initial-Incumbent, Safe-Interior."""
    z_a = float(norm.ppf(1.0 - float(alpha)))
    eff_std_con = np.sqrt(std_con ** 2 + 0.25 * r_perp_con)
    cert_margin = mean_con + float(epistemic_beta) * eff_std_con + z_a * np.sqrt(np.maximum(ucb_aleat_var_con, 1e-8))

    # Slot 1: Posterior primary challenger
    safe_idx = np.where(cert_margin <= 0.0)[0]
    if safe_idx.size > 0:
        primary_i = int(safe_idx[np.argmin(mean_obj[safe_idx])])
    else:
        # Relax slightly to posterior mean quantile if conservative UCB set is empty
        nom_margin = mean_con + z_a * np.sqrt(np.maximum(ucb_aleat_var_con, 1e-8))
        semi_safe = np.where(nom_margin <= 0.0)[0]
        if semi_safe.size > 0:
            primary_i = int(semi_safe[np.argmin(mean_obj[semi_safe])])
        else:
            primary_i = int(np.argmin(cert_margin))
    x_primary = np.asarray(candidate_pool[primary_i], dtype=float).copy()

    # Slot 2: Feasible-aware initial incumbent guard
    x_init_inc = select_feasible_aware_initial_incumbent(initial_records, alpha=alpha)

    # Slot 3: Deepest posterior safe-interior policy (maximizing safety buffer, tie-breaking by objective)
    interior_scores = cert_margin + 0.05 * (mean_obj - float(np.min(mean_obj))) / max(float(np.ptp(mean_obj)), 1e-4)
    interior_order = np.argsort(interior_scores)

    shortlist: list[tuple[str, np.ndarray]] = [("posterior_primary", x_primary)]
    seen = {np.round(x_primary, 5).tobytes()}

    init_key = np.round(x_init_inc, 5).tobytes()
    if init_key not in seen:
        shortlist.append(("safe_initial_incumbent", x_init_inc))
        seen.add(init_key)

    for idx in interior_order:
        if len(shortlist) >= 3:
            break
        cand = np.asarray(candidate_pool[int(idx)], dtype=float).copy()
        ck = np.round(cand, 5).tobytes()
        if ck not in seen:
            shortlist.append(("posterior_safe_interior", cand))
            seen.add(ck)

    return shortlist


def verify_and_deploy_shortlist(
    problem: BaseDynamicORProblem,
    shortlist: Sequence[tuple[str, np.ndarray]],
    *,
    seed: int,
    familywise_delta: float = 0.05,
    slot_budgets: tuple[int, int, int] = (112, 128, 140),
    gate_mode: str = "auto",
) -> TerminalDeploymentReceipt:
    """Execute independent terminal safety certification with exact finite-sample FWER `<= delta`."""
    if not shortlist:
        raise ValueError("Shortlist cannot be empty")
    rng = np.random.default_rng(seed)

    if gate_mode == "auto":
        use_mode = "gaussian_noncentral_t" if problem.is_gaussian_constraint_noise else "exact_binomial"
    else:
        use_mode = str(gate_mode)

    # Asymmetric FWER allocation summing to `familywise_delta` (favoring Slot 1 power)
    n_slots = min(len(shortlist), 3)
    if n_slots == 1:
        deltas = (familywise_delta,)
    elif n_slots == 2:
        deltas = (0.625 * familywise_delta, 0.375 * familywise_delta)
    else:
        deltas = (0.50 * familywise_delta, 0.30 * familywise_delta, 0.20 * familywise_delta)

    certs: list[GateCertificate] = []
    calls_used = 0

    def _run_slot(idx: int) -> GateCertificate:
        nonlocal calls_used
        role, x_pol = shortlist[idx]
        k_bud = int(slot_budgets[min(idx, len(slot_budgets) - 1)])
        d_alloc = float(deltas[idx])
        samples = problem.simulate_replications(x_pol, k_bud, rng, split="verification")
        calls_used += k_bud
        if use_mode == "exact_binomial":
            ok, stat_val, viol_r = exact_binomial_gate(samples[:, 1], alpha=problem.alpha, delta_slot=d_alloc)
        elif use_mode == "gaussian_noncentral_t":
            ok, stat_val, viol_r = gaussian_noncentral_t_gate(samples[:, 1], alpha=problem.alpha, delta_slot=d_alloc)
        else:
            raise ValueError(f"Unknown gate_mode {use_mode!r}")
        cert = GateCertificate(
            slot_index=idx,
            role=role,
            certified=ok,
            method=use_mode,
            n_samples=k_bud,
            delta_allocated=d_alloc,
            p_value_or_margin=stat_val,
            observed_violation_rate=viol_r,
            sample_mean_obj=float(np.mean(samples[:, 0])),
            sample_mean_con=float(np.mean(samples[:, 1])),
        )
        certs.append(cert)
        return cert

    # Check if Slot 2 is the initial incumbent guard
    has_incumbent_guard = n_slots >= 2 and shortlist[1][0] == "safe_initial_incumbent"

    c0 = _run_slot(0)
    if c0.certified and not has_incumbent_guard:
        return TerminalDeploymentReceipt(
            deployed_policy=shortlist[0][1].copy(),
            certified=True,
            selected_slot=0,
            selected_role=shortlist[0][0],
            verification_calls_used=calls_used,
            familywise_delta=familywise_delta,
            certificates=tuple(certs),
            shortlist_roles=tuple(r for r, _ in shortlist[:n_slots]),
        )

    if has_incumbent_guard:
        # Only pay Slot-2 budget if Slot 1 failed safety OR if we guard against objective regression
        c1 = _run_slot(1)
        if c0.certified and c1.certified:
            # Both certified safe at FWER <= delta_1 + delta_2! Deploy whichever has better verified mean objective!
            # (Zero extra FWER cost because BOTH policies already passed their independent safety gates!)
            if c0.sample_mean_obj <= c1.sample_mean_obj:
                chosen_idx = 0
            else:
                chosen_idx = 1
            return TerminalDeploymentReceipt(
                deployed_policy=shortlist[chosen_idx][1].copy(),
                certified=True,
                selected_slot=chosen_idx,
                selected_role=shortlist[chosen_idx][0],
                verification_calls_used=calls_used,
                familywise_delta=familywise_delta,
                certificates=tuple(certs),
                shortlist_roles=tuple(r for r, _ in shortlist[:n_slots]),
            )
        if c0.certified:
            return TerminalDeploymentReceipt(
                deployed_policy=shortlist[0][1].copy(),
                certified=True,
                selected_slot=0,
                selected_role=shortlist[0][0],
                verification_calls_used=calls_used,
                familywise_delta=familywise_delta,
                certificates=tuple(certs),
                shortlist_roles=tuple(r for r, _ in shortlist[:n_slots]),
            )
        if c1.certified:
            return TerminalDeploymentReceipt(
                deployed_policy=shortlist[1][1].copy(),
                certified=True,
                selected_slot=1,
                selected_role=shortlist[1][0],
                verification_calls_used=calls_used,
                familywise_delta=familywise_delta,
                certificates=tuple(certs),
                shortlist_roles=tuple(r for r, _ in shortlist[:n_slots]),
            )
        start_rem = 2
    else:
        start_rem = 1

    for idx in range(start_rem, n_slots):
        ci = _run_slot(idx)
        if ci.certified:
            return TerminalDeploymentReceipt(
                deployed_policy=shortlist[idx][1].copy(),
                certified=True,
                selected_slot=idx,
                selected_role=shortlist[idx][0],
                verification_calls_used=calls_used,
                familywise_delta=familywise_delta,
                certificates=tuple(certs),
                shortlist_roles=tuple(r for r, _ in shortlist[:n_slots]),
            )

    # Abstain safely onto Operational Anchor if no shortlist candidate passed certification
    best_fallback = min(certs, key=lambda c: (c.observed_violation_rate, c.sample_mean_obj))
    return TerminalDeploymentReceipt(
        deployed_policy=shortlist[best_fallback.slot_index][1].copy(),
        certified=False,
        selected_slot=None,
        selected_role="abstained_uncertified",
        verification_calls_used=calls_used,
        familywise_delta=familywise_delta,
        certificates=tuple(certs),
        shortlist_roles=tuple(r for r, _ in shortlist[:n_slots]),
    )
