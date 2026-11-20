"""Feasible-Aware 3-Slot Shortlist, Exact Binomial Gate Power & Strict Deployment Contract.

Implements a transparent, finite-sample safety certification contract:
1. Freezes up to 3 evaluated candidate policies BEFORE running independent verification:
   - Slot 0 (`"posterior_primary"`): Maximizes expected net certified deployment utility
     using exact terminal binomial gate power `C_{K_0}(p)` (`K_0 = 112, delta_0 = 0.025`).
   - Slot 1 (`"empirical_feasible_incumbent"`): Lowest-objective evaluated policy that
     empirically satisfies strict sample feasibility (`K_1 = 128, delta_1 = 0.015`).
   - Slot 2 (`"conservative_posterior_interior"`): Evaluated policy maximizing posterior
     terminal gate pass probability (`K_2 = 140, delta_2 = 0.010`).
2. Under independent identically distributed (IID) verification replications on the
   verification law `P_ver`, guarantees exact finite-sample familywise false-certification
   control:
       Pr_{P_ver}(certified AND P_ver(Y_g(x_deploy) > 0) > alpha) <= delta_0 + delta_1 + delta_2 = delta.
3. Enforces a strict abstention contract: when no shortlist slot passes its gate,
   `certified = False`, `decision_state = "abstention"`, and `deployed_policy = None`
   (never silently returning an uncertified policy as a deployed fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr
from scipy.stats import binom, nct, norm

from v2_or.contracts import BaseDynamicORProblem, SufficientStatsRecord


def _gh_1d(order: int = 9) -> tuple[np.ndarray, np.ndarray]:
    pts, wts = hermgauss(int(order))
    return pts * np.sqrt(2.0), wts / np.sqrt(np.pi)


_GH9_Z, _GH9_W = _gh_1d(9)


@dataclass(frozen=True)
class GateCertificate:
    """Result of testing a single frozen candidate policy on fresh verification samples."""

    slot_index: int
    role: str
    certified: bool
    method: str
    n_samples: int
    delta_allocated: float
    critical_k: int | None
    observed_violations: int
    p_value_or_margin: float
    observed_violation_rate: float
    sample_mean_obj: float
    sample_mean_con: float


@dataclass(frozen=True)
class TerminalDeploymentReceipt:
    """Complete auditable record of shortlist verification and final decision state."""

    decision_state: str  # "certified_deployment" or "abstention"
    deployed_policy: np.ndarray | None
    selected_uncertified_candidate: np.ndarray | None
    primary_policy: np.ndarray
    shortlist_policies: tuple[np.ndarray, ...]
    certified: bool
    selected_slot: int | None
    selected_role: str
    verification_calls_used: int
    familywise_delta: float
    slot_budgets_configured: tuple[int, ...]
    slot_deltas_configured: tuple[float, ...]
    certificates: tuple[GateCertificate, ...]
    shortlist_roles: tuple[str, ...]


@lru_cache(maxsize=128)
def exact_binomial_critical_k(n_samples: int, alpha: float, delta_slot: float) -> int:
    """Largest integer `k*` such that `BinomCDF(k*; K, alpha) <= delta_slot` (`-1` if none)."""
    k_tot = int(n_samples)
    if k_tot < 1:
        return -1
    ks = np.arange(0, k_tot + 1, dtype=int)
    cdfs = binom.cdf(ks, k_tot, float(alpha))
    valid = np.where(cdfs <= float(delta_slot))[0]
    return int(valid[-1]) if valid.size > 0 else -1


def exact_binomial_pass_power(
    p_viol: np.ndarray | float,
    n_samples: int,
    k_crit: int,
) -> np.ndarray:
    """Exact pass probability `C_K(p) = BinomCDF(k_crit; K, p)` as a function of `p`."""
    p_arr = np.clip(np.asarray(p_viol, dtype=float), 0.0, 1.0)
    k = int(k_crit)
    n = int(n_samples)
    if k < 0 or n < 1:
        return np.zeros_like(p_arr, dtype=float)
    q = 1.0 - p_arr
    if k == 0:
        return np.asarray(q ** n, dtype=float)
    if k == 1 and n >= 1:
        return np.asarray((q ** (n - 1)) * (1.0 + (n - 1) * p_arr), dtype=float)
    if k == 2 and n >= 2:
        poly = q * q + n * p_arr * q + (0.5 * n * (n - 1)) * (p_arr * p_arr)
        return np.asarray((q ** (n - 2)) * poly, dtype=float)
    return np.asarray(binom.cdf(k, n, p_arr), dtype=float)


def posterior_binomial_gate_power(
    mean_con: np.ndarray,
    std_con: np.ndarray,
    aleat_var_con: np.ndarray,
    *,
    n_samples: int = 112,
    alpha: float = 0.05,
    delta_slot: float = 0.025,
    r_perp_con: np.ndarray | None = None,
) -> np.ndarray:
    """Posterior expectation of exact terminal binomial gate power `E_n[C_K(Phi(g / sigma_g))]`.

    Unlike a naive Gaussian `z_{1-alpha}` margin (which assigns `0.50` at the boundary
    `p = alpha = 0.05` whereas `C_{112}(0.05) = 0.022058`), this integrates the exact
    terminal gate power curve `BinomCDF(k*; K, p)` over epistemic uncertainty in `g(x)`.
    """
    mu = np.asarray(mean_con, dtype=float).reshape(-1)
    sd = np.asarray(std_con, dtype=float).reshape(-1)
    av = np.maximum(np.asarray(aleat_var_con, dtype=float).reshape(-1), 1e-8)
    if r_perp_con is not None:
        rp = np.maximum(np.asarray(r_perp_con, dtype=float).reshape(-1), 0.0)
        eff_sd = np.sqrt(sd ** 2 + 0.25 * rp)
    else:
        eff_sd = np.maximum(sd, 0.0)

    k_crit = exact_binomial_critical_k(int(n_samples), float(alpha), float(delta_slot))
    if k_crit < 0:
        return np.zeros_like(mu, dtype=float)

    # Shape (m_cand, 9): latent constraint mean draws g_q = mu + eff_sd * z_q
    g_draws = mu[:, None] + eff_sd[:, None] * _GH9_Z[None, :]
    p_draws = ndtr(g_draws / np.sqrt(av)[:, None])
    power_draws = exact_binomial_pass_power(p_draws, int(n_samples), k_crit)
    return np.clip(power_draws @ _GH9_W, 0.0, 1.0)


def exact_binomial_gate(
    con_samples: np.ndarray,
    *,
    alpha: float,
    delta_slot: float,
) -> tuple[bool, float, float, int, int]:
    """Test `H_0: P(Y_g > 0) > alpha` at exact level `delta_slot` with zero distributional assumptions."""
    vals = np.asarray(con_samples, dtype=float).reshape(-1)
    k_tot = int(vals.size)
    k_crit = exact_binomial_critical_k(k_tot, alpha, delta_slot)
    if k_tot < 1:
        return False, 1.0, 1.0, 0, k_crit
    v_obs = int(np.sum(vals > 0.0))
    p_val = float(binom.cdf(v_obs, k_tot, float(alpha)))
    return bool(p_val <= float(delta_slot)), p_val, float(v_obs / k_tot), v_obs, k_crit


def gaussian_noncentral_t_gate(
    con_samples: np.ndarray,
    *,
    alpha: float,
    delta_slot: float,
) -> tuple[bool, float, float, int]:
    """Exact noncentral-t upper confidence bound on `mu_g + z_{1-alpha} * sigma_g <= 0`."""
    vals = np.asarray(con_samples, dtype=float).reshape(-1)
    n = int(vals.size)
    v_obs = int(np.sum(vals > 0.0))
    if n < 2:
        return False, float("inf"), 1.0, v_obs
    mean_g = float(np.mean(vals))
    std_g = float(np.std(vals, ddof=1))
    z_a = float(norm.ppf(1.0 - float(alpha)))
    nc = z_a * np.sqrt(float(n))
    q_nct = float(nct.ppf(1.0 - float(delta_slot), n - 1, nc))
    upper_margin = mean_g + (q_nct / np.sqrt(float(n))) * std_g
    viol_rate = float(v_obs / n)
    return bool(upper_margin <= 0.0), float(upper_margin), viol_rate, v_obs


def _select_empirical_feasible_incumbent_with_idx(
    evaluated_records: Sequence[SufficientStatsRecord],
    *,
    alpha: float = 0.05,
    slot_budget: int = 128,
    delta_slot: float = 0.015,
    pooled_aleat_var_con: np.ndarray | None = None,
    posterior_gate_power: np.ndarray | None = None,
    posterior_mean_obj: np.ndarray | None = None,
    fallback_loss: float | None = None,
) -> tuple[np.ndarray, int]:
    """Select the empirically safe evaluated policy maximizing cascade utility with pooled-variance guard."""
    if not evaluated_records:
        raise ValueError("evaluated_records cannot be empty")
    n_eval = len(evaluated_records)
    k_crit = max(exact_binomial_critical_k(slot_budget, alpha, delta_slot), 0)
    # Target violation probability achieving >= 75% power on Slot 1 (~0.007 for K=128, k*=1)
    p_target = max(0.006, (k_crit + 0.7) / float(max(slot_budget, 1)))
    z_pow = float(norm.ppf(1.0 - min(p_target, float(alpha))))

    use_pool = pooled_aleat_var_con is not None and len(pooled_aleat_var_con) == n_eval
    use_p_gate = posterior_gate_power is not None and len(posterior_gate_power) == n_eval
    use_mf = posterior_mean_obj is not None and len(posterior_mean_obj) == n_eval

    # Fallback global scale across multi-rep records when pooled_aleat_var_con is omitted
    multi_vars = [
        rec.ssq_con / float(rec.n_reps - 1)
        for rec in evaluated_records
        if rec.n_reps > 1 and rec.ssq_con > 0.0
    ]
    med_emp_var = float(np.median(multi_vars)) if multi_vars else 2.5e-3

    strict_safe: list[tuple[float, float, int, np.ndarray]] = []
    zero_viol: list[tuple[float, float, int, np.ndarray]] = []
    all_cands: list[tuple[float, float, int, np.ndarray]] = []

    log_mult = float(np.log(max(n_eval, 2)))
    for idx, rec in enumerate(evaluated_records):
        dof = max(int(rec.n_reps) - 1, 0)
        s2_emp = float(rec.ssq_con / float(dof)) if dof >= 1 else med_emp_var
        v_pool = float(max(pooled_aleat_var_con[idx], 1e-10)) if use_pool else med_emp_var
        # Empirical-Bayes / chi-square shrinkage eliminating 1-dof (`batch_reps=2`) winner's curse
        v_shrunk = max(0.65 * v_pool, (float(dof) * s2_emp + 2.0 * v_pool) / float(dof + 2.0))
        std_g = float(np.sqrt(max(v_shrunk, 1e-10)))
        epist_mult = 0.08 * float(np.sqrt(log_mult / float(max(rec.n_reps, 1))))
        pow_margin = float(rec.mean_con + (z_pow + epist_mult) * std_g)
        obj_est = 0.5 * float(rec.mean_obj) + 0.5 * float(posterior_mean_obj[idx]) if use_mf else float(rec.mean_obj)
        p_g1 = float(posterior_gate_power[idx]) if use_p_gate else 1.0

        all_cands.append((pow_margin, obj_est, idx, rec.x_norm))
        if rec.violation_count == 0:
            zero_viol.append((pow_margin, obj_est, idx, rec.x_norm))
            if pow_margin <= 0.0 and p_g1 >= 0.55:
                strict_safe.append((obj_est, pow_margin, idx, rec.x_norm))

    if strict_safe:
        strict_safe.sort(key=lambda t: (t[0], t[1]))
        return strict_safe[0][3].copy(), int(strict_safe[0][2])
    if zero_viol:
        zero_viol.sort(key=lambda t: (t[0], t[1]))
        return zero_viol[0][3].copy(), int(zero_viol[0][2])
    all_cands.sort(key=lambda t: (t[0], t[1]))
    return all_cands[0][3].copy(), int(all_cands[0][2])


def select_empirical_feasible_incumbent(
    evaluated_records: Sequence[SufficientStatsRecord],
    *,
    alpha: float = 0.05,
    slot_budget: int = 128,
    delta_slot: float = 0.015,
    pooled_aleat_var_con: np.ndarray | None = None,
) -> np.ndarray:
    """Select the empirically safe evaluated policy meeting gate-power safety with pooled-variance guard."""
    pol, _ = _select_empirical_feasible_incumbent_with_idx(
        evaluated_records,
        alpha=alpha,
        slot_budget=slot_budget,
        delta_slot=delta_slot,
        pooled_aleat_var_con=pooled_aleat_var_con,
    )
    return pol


select_feasible_aware_initial_incumbent = select_empirical_feasible_incumbent


DEFAULT_SLOT_BUDGETS = (112, 128, 140)


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
    nomination_mode: str = "binomial_gate_power",
    reference_obj: float | None = None,
    slot_budgets: tuple[int, int, int] = DEFAULT_SLOT_BUDGETS,
    familywise_delta: float = 0.05,
) -> list[tuple[str, np.ndarray]]:
    """Construct up to 3 frozen evaluated candidates via Bellman cascade gate-power optimality."""
    mf = np.asarray(mean_obj, dtype=float).reshape(-1)
    mg = np.asarray(mean_con, dtype=float).reshape(-1)
    sg = np.asarray(std_con, dtype=float).reshape(-1)
    uv = np.maximum(np.asarray(ucb_aleat_var_con, dtype=float).reshape(-1), 1e-8)
    rp = np.maximum(np.asarray(r_perp_con, dtype=float).reshape(-1), 0.0)
    spread_f = max(float(np.std(mf)), 1e-8)
    ptp_f = max(float(np.ptp(mf)), spread_f)
    eps_f = 1e-4 * spread_f

    d0 = 0.50 * float(familywise_delta)
    d1 = 0.30 * float(familywise_delta)
    d2 = 0.20 * float(familywise_delta)

    p_gate1_rec = None
    j_rem2: float | None = None

    if nomination_mode in {"binomial_gate_power", "cascade_bellman_3slot"}:
        p_gate0 = posterior_binomial_gate_power(
            mg,
            sg,
            uv,
            n_samples=int(slot_budgets[0]),
            alpha=alpha,
            delta_slot=d0,
            r_perp_con=rp,
        )
        p_gate1 = posterior_binomial_gate_power(
            mg,
            sg,
            uv,
            n_samples=int(slot_budgets[1]),
            alpha=alpha,
            delta_slot=d1,
            r_perp_con=rp,
        )
        p_gate2 = posterior_binomial_gate_power(
            mg,
            sg,
            uv,
            n_samples=int(slot_budgets[2]),
            alpha=alpha,
            delta_slot=d2,
            r_perp_con=rp,
        )
        if len(p_gate1) == len(initial_records):
            p_gate1_rec = p_gate1

        j_ref = (
            max(float(reference_obj), float(np.min(mf)) + 0.20 * spread_f)
            if (reference_obj is not None and np.isfinite(reference_obj))
            else float(np.quantile(mf, 0.85) + 0.25 * spread_f)
        )

        # Backward induction across the 3-slot verification cascade:
        # 1. Slot 2 (conservative interior safety net)
        interior_scores = -p_gate2 + 0.02 * (mf - float(np.min(mf))) / ptp_f
        interior_order = np.argsort(interior_scores)
        i2 = int(interior_order[0])
        j_rem2 = float(p_gate2[i2] * min(float(mf[i2]), j_ref) + (1.0 - p_gate2[i2]) * j_ref)

        # 2. Slot 1 (multiplicity-guarded pooled-variance empirical incumbent)
        x_emp_inc, inc_idx = _select_empirical_feasible_incumbent_with_idx(
            initial_records,
            alpha=alpha,
            slot_budget=int(slot_budgets[1]),
            delta_slot=d1,
            pooled_aleat_var_con=uv if len(uv) == len(initial_records) else None,
            posterior_gate_power=p_gate1_rec,
            posterior_mean_obj=mf if len(mf) == len(initial_records) else None,
            fallback_loss=j_rem2,
        )
        if len(mf) == len(initial_records) and 0 <= inc_idx < len(mf):
            j_rem1 = float(
                p_gate1[inc_idx] * min(float(mf[inc_idx]), j_rem2) + (1.0 - p_gate1[inc_idx]) * j_rem2
            )
        else:
            j_rem1 = j_rem2

        # 3. Slot 0 (Bellman-optimal primary contender against cascade continuation loss `j_rem1`)
        cand_gain_vs_rem1 = np.maximum(j_rem1 - mf, 0.0) * p_gate0
        viable_cascade = np.where((p_gate0 >= 0.55) & (mf < j_rem1 - eps_f))[0]
        if viable_cascade.size > 0:
            primary_i = int(viable_cascade[np.argmax(cand_gain_vs_rem1[viable_cascade])])
        else:
            high_pow = np.where(p_gate0 >= 0.55)[0]
            if high_pow.size > 0:
                pos_saving = np.where(mf[high_pow] < j_ref - eps_f)[0]
                if pos_saving.size > 0:
                    sub_hp = high_pow[pos_saving]
                    net_util = (j_ref - mf[sub_hp]) * p_gate0[sub_hp]
                    primary_i = int(sub_hp[np.argmax(net_util)])
                else:
                    # Never invert to argmax(eps_f * p_gate0); select lowest-cost certifiable policy!
                    primary_i = int(high_pow[np.argmin(mf[high_pow])])
            elif float(np.max(p_gate0)) >= 0.10:
                j_hi = max(j_ref, float(np.quantile(mf, 0.85)) + 0.25 * spread_f)
                net_util = np.maximum(j_hi - mf, 0.0) * p_gate0
                primary_i = int(np.argmax(net_util)) if float(np.max(net_util)) > 0.0 else int(np.argmax(p_gate0))
            else:
                primary_i = int(np.argmax(p_gate0 - 0.02 * (mf - float(np.min(mf))) / ptp_f))
    else:
        z_a = float(norm.ppf(1.0 - float(alpha)))
        eff_std_con = np.sqrt(sg ** 2 + 0.25 * rp)
        cert_margin = mg + float(epistemic_beta) * eff_std_con + z_a * np.sqrt(uv)
        safe_idx = np.where(cert_margin <= 0.0)[0]
        if safe_idx.size > 0:
            primary_i = int(safe_idx[np.argmin(mf[safe_idx])])
        else:
            nom_margin = mg + z_a * np.sqrt(uv)
            semi_safe = np.where(nom_margin <= 0.0)[0]
            primary_i = int(semi_safe[np.argmin(mf[semi_safe])]) if semi_safe.size > 0 else int(np.argmin(cert_margin))
        interior_scores = cert_margin + 0.05 * (mf - float(np.min(mf))) / ptp_f
        interior_order = np.argsort(interior_scores)
        x_emp_inc, _ = _select_empirical_feasible_incumbent_with_idx(
            initial_records,
            alpha=alpha,
            slot_budget=int(slot_budgets[1]),
            delta_slot=d1,
            pooled_aleat_var_con=uv if len(uv) == len(initial_records) else None,
        )

    x_primary = np.asarray(candidate_pool[primary_i], dtype=float).copy()

    shortlist: list[tuple[str, np.ndarray]] = [("posterior_primary", x_primary)]
    seen = {np.round(x_primary, 5).tobytes()}

    inc_key = np.round(x_emp_inc, 5).tobytes()
    if inc_key not in seen:
        shortlist.append(("empirical_feasible_incumbent", x_emp_inc))
        seen.add(inc_key)

    for idx in interior_order:
        if len(shortlist) >= 3:
            break
        cand = np.asarray(candidate_pool[int(idx)], dtype=float).copy()
        ck = np.round(cand, 5).tobytes()
        if ck not in seen:
            shortlist.append(("conservative_posterior_interior", cand))
            seen.add(ck)

    return shortlist


def verify_and_deploy_shortlist(
    problem: BaseDynamicORProblem,
    shortlist: Sequence[tuple[str, np.ndarray]],
    *,
    seed: int,
    familywise_delta: float = 0.05,
    slot_budgets: tuple[int, int, int] = DEFAULT_SLOT_BUDGETS,
    gate_mode: str = "auto",
) -> TerminalDeploymentReceipt:
    """Execute independent terminal safety certification with strict abstention semantics."""
    if not shortlist:
        raise ValueError("Shortlist cannot be empty")
    rng = np.random.default_rng(seed)

    if gate_mode == "auto":
        use_mode = "gaussian_noncentral_t" if problem.is_gaussian_constraint_noise else "exact_binomial"
    else:
        use_mode = str(gate_mode)

    n_slots = min(len(shortlist), 3)
    if n_slots == 1:
        deltas = (float(familywise_delta),)
    elif n_slots == 2:
        deltas = (0.625 * float(familywise_delta), 0.375 * float(familywise_delta))
    else:
        deltas = (0.50 * float(familywise_delta), 0.30 * float(familywise_delta), 0.20 * float(familywise_delta))

    used_budgets = tuple(int(slot_budgets[min(i, len(slot_budgets) - 1)]) for i in range(n_slots))
    primary_x = np.asarray(shortlist[0][1], dtype=float).copy()
    shortlist_pols = tuple(np.asarray(x, dtype=float).copy() for _, x in shortlist[:n_slots])
    shortlist_rls = tuple(str(r) for r, _ in shortlist[:n_slots])

    certs: list[GateCertificate] = []
    calls_used = 0

    def _run_slot(idx: int) -> GateCertificate:
        nonlocal calls_used
        role, x_pol = shortlist[idx]
        k_bud = used_budgets[idx]
        d_alloc = float(deltas[idx])
        samples = problem.simulate_replications(x_pol, k_bud, rng, split="verification")
        calls_used += k_bud
        if use_mode == "exact_binomial":
            ok, stat_val, viol_r, v_obs, k_crit = exact_binomial_gate(
                samples[:, 1], alpha=problem.alpha, delta_slot=d_alloc
            )
        elif use_mode == "gaussian_noncentral_t":
            ok, stat_val, viol_r, v_obs = gaussian_noncentral_t_gate(
                samples[:, 1], alpha=problem.alpha, delta_slot=d_alloc
            )
            k_crit = None
        else:
            raise ValueError(f"Unknown gate_mode {use_mode!r}")
        cert = GateCertificate(
            slot_index=idx,
            role=role,
            certified=ok,
            method=use_mode,
            n_samples=k_bud,
            delta_allocated=d_alloc,
            critical_k=k_crit,
            observed_violations=v_obs,
            p_value_or_margin=stat_val,
            observed_violation_rate=viol_r,
            sample_mean_obj=float(np.mean(samples[:, 0])),
            sample_mean_con=float(np.mean(samples[:, 1])),
        )
        certs.append(cert)
        return cert

    def _make_certified_receipt(chosen_idx: int) -> TerminalDeploymentReceipt:
        return TerminalDeploymentReceipt(
            decision_state="certified_deployment",
            deployed_policy=shortlist_pols[chosen_idx].copy(),
            selected_uncertified_candidate=None,
            primary_policy=primary_x,
            shortlist_policies=shortlist_pols,
            certified=True,
            selected_slot=chosen_idx,
            selected_role=shortlist_rls[chosen_idx],
            verification_calls_used=calls_used,
            familywise_delta=float(familywise_delta),
            slot_budgets_configured=used_budgets,
            slot_deltas_configured=deltas,
            certificates=tuple(certs),
            shortlist_roles=shortlist_rls,
        )

    has_incumbent_guard = n_slots >= 2 and shortlist_rls[1] in {
        "empirical_feasible_incumbent",
        "safe_initial_incumbent",
    }

    c0 = _run_slot(0)
    if c0.certified and not has_incumbent_guard:
        return _make_certified_receipt(0)

    if has_incumbent_guard:
        c1 = _run_slot(1)
        if c0.certified and c1.certified:
            chosen_idx = 0 if c0.sample_mean_obj <= c1.sample_mean_obj else 1
            return _make_certified_receipt(chosen_idx)
        if c0.certified:
            return _make_certified_receipt(0)
        if c1.certified:
            return _make_certified_receipt(1)
        start_rem = 2
    else:
        start_rem = 1

    for idx in range(start_rem, n_slots):
        ci = _run_slot(idx)
        if ci.certified:
            return _make_certified_receipt(idx)

    # Strict abstention: NO uncertified candidate is ever placed into `deployed_policy`!
    best_uncert = min(certs, key=lambda c: (c.observed_violation_rate, c.sample_mean_obj))
    return TerminalDeploymentReceipt(
        decision_state="abstention",
        deployed_policy=None,
        selected_uncertified_candidate=shortlist_pols[best_uncert.slot_index].copy(),
        primary_policy=primary_x,
        shortlist_policies=shortlist_pols,
        certified=False,
        selected_slot=None,
        selected_role="abstained_uncertified",
        verification_calls_used=calls_used,
        familywise_delta=float(familywise_delta),
        slot_budgets_configured=used_budgets,
        slot_deltas_configured=deltas,
        certificates=tuple(certs),
        shortlist_roles=shortlist_rls,
    )
