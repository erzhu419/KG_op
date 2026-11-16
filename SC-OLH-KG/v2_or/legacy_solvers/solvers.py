"""Authentic legacy simulation-optimization solvers for source archive generation & target baselines.

Implements four canonical operational / simulation-optimization algorithms:
1. `PenaltySPSA`: Spall (1992) Simultaneous Perturbation Stochastic Approximation with
   augmented-Lagrangian / adaptive exterior penalty for chance constraints.
2. `ConstrainedCMAES`: Hansen (2006) Covariance Matrix Adaptation Evolution Strategy
   with Deb's stochastic feasibility-first ranking.
3. `ConstrainedNSGA2`: Deb et al. (2002) elitist evolutionary tournament optimizer
   with probabilistic margin feasibility domination (mirroring `nsga2_direct.py`).
4. `HeuristicPowellSearch`: Domain operational rule anchor followed by coordinate /
   Powell pattern refinement with adaptive boundary replication.

Every evaluation logs complete `LegacyTrajectoryRecord` entries (preserving the genuine
non-uniform, optimization-biased trajectory distribution and heterogeneous replication
counts `n_reps >= 2`).
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import norm

from v2_or.contracts import (
    BaseDynamicORProblem,
    LegacyTrajectoryRecord,
    SufficientStatsRecord,
)


def _empirical_chance_margin(rec: SufficientStatsRecord, alpha: float) -> float:
    """Estimate `mean_con + z_{1-alpha} * std_con` from `rec`."""
    z = float(norm.ppf(1.0 - alpha))
    if rec.n_reps > 1:
        std = float(np.sqrt(max(rec.ssq_con / (rec.n_reps - 1), 1e-12)))
    else:
        std = 0.05
    return float(rec.mean_con + z * std)


def _deb_feasible_key(rec: SufficientStatsRecord, alpha: float) -> tuple[int, float, float]:
    """Deb's feasibility-first lexicographic ordering (smaller tuple is better)."""
    margin = _empirical_chance_margin(rec, alpha)
    if margin <= 0.0:
        return (0, float(rec.mean_obj), margin)
    return (1, margin, float(rec.mean_obj))


class TrajectoryArchiveRecorder:
    """Meters and records every policy evaluation during a legacy solver trajectory."""

    def __init__(self, problem: BaseDynamicORProblem, solver_name: str) -> None:
        self.problem = problem
        self.solver_name = str(solver_name)
        self.records: list[SufficientStatsRecord] = []
        self.trajectory_log: list[LegacyTrajectoryRecord] = []

    @property
    def total_calls(self) -> int:
        return int(sum(r.n_reps for r in self.records))

    def evaluate(
        self,
        x_norm: np.ndarray,
        reps: int,
        rng: np.random.Generator,
    ) -> SufficientStatsRecord:
        x_clean = self.problem.clip_policy(x_norm)
        exposure = self.problem.nominal_exposure(x_clean)
        samples = self.problem.simulate_replications(x_clean, int(reps), rng, split="search")
        rec = SufficientStatsRecord.from_samples(x_clean, samples, exposure)
        step_idx = len(self.trajectory_log)
        self.records.append(rec)
        self.trajectory_log.append(
            LegacyTrajectoryRecord(
                task_id=self.problem.task_id,
                solver_name=self.solver_name,
                step_index=step_idx,
                policy_dim=int(self.problem.d),
                x_norm=tuple(float(v) for v in x_clean),
                n_reps=rec.n_reps,
                mean_obj=rec.mean_obj,
                ssq_obj=rec.ssq_obj,
                mean_con=rec.mean_con,
                ssq_con=rec.ssq_con,
                violation_count=rec.violation_count,
                psi=tuple(float(v) for v in exposure.psi),
                A_local=tuple(float(v) for v in exposure.A_local),
                N_shared=tuple(float(v) for v in exposure.N_shared),
            )
        )
        return rec

    def best_incumbent(self) -> SufficientStatsRecord:
        if not self.records:
            raise RuntimeError("No evaluations recorded")
        return min(self.records, key=lambda r: _deb_feasible_key(r, self.problem.alpha))


def run_penalty_spsa(
    problem: BaseDynamicORProblem,
    *,
    budget_calls: int,
    seed: int,
    reps_per_eval: int = 3,
) -> TrajectoryArchiveRecorder:
    """Spall (1992) SPSA with adaptive chance-constraint exterior penalty."""
    rng = np.random.default_rng(seed)
    rec = TrajectoryArchiveRecorder(problem, "penalty_spsa")
    d = problem.d
    # Start from a jittered operational anchor or mid-box point
    x = np.clip(problem.domain_heuristic_anchor() + rng.normal(0.0, 0.12, size=d), 0.05, 0.95)
    rec.evaluate(x, min(reps_per_eval, budget_calls), rng)

    penalty_weight = 6.0
    k = 0
    while rec.total_calls + 2 * reps_per_eval <= budget_calls:
        k += 1
        ck = min(0.14 / ((k + 1.0) ** 0.101), 0.15)
        ak = 0.08 / ((k + 8.0) ** 0.602)
        delta = rng.choice([-1.0, 1.0], size=d)
        x_plus = np.clip(x + ck * delta, 0.0, 1.0)
        x_minus = np.clip(x - ck * delta, 0.0, 1.0)

        r_plus = rec.evaluate(x_plus, reps_per_eval, rng)
        r_minus = rec.evaluate(x_minus, reps_per_eval, rng)

        m_plus = max(_empirical_chance_margin(r_plus, problem.alpha), 0.0)
        m_minus = max(_empirical_chance_margin(r_minus, problem.alpha), 0.0)
        L_plus = r_plus.mean_obj + penalty_weight * (m_plus + 2.0 * m_plus ** 2)
        L_minus = r_minus.mean_obj + penalty_weight * (m_minus + 2.0 * m_minus ** 2)

        grad_est = ((L_plus - L_minus) / (2.0 * ck)) / delta
        step = np.clip(ak * grad_est, -0.12, 0.12)
        x = np.clip(x - step, 0.0, 1.0)
        if m_plus > 0.02 and m_minus > 0.02:
            penalty_weight = min(penalty_weight * 1.20, 60.0)

        # Periodically evaluate current center with slightly higher replication count
        if k % 3 == 0 and rec.total_calls + (reps_per_eval + 2) <= budget_calls:
            rec.evaluate(x, reps_per_eval + 2, rng)

    rem = budget_calls - rec.total_calls
    if rem >= 1:
        rec.evaluate(x, rem, rng)
    return rec


def run_constrained_cmaes(
    problem: BaseDynamicORProblem,
    *,
    budget_calls: int,
    seed: int,
    reps_per_eval: int = 3,
    pop_size: int = 8,
) -> TrajectoryArchiveRecorder:
    """Diagonal/low-ranksep-CMA-ES with Deb's stochastic feasibility ranking."""
    rng = np.random.default_rng(seed)
    rec = TrajectoryArchiveRecorder(problem, "constrained_cmaes")
    d = problem.d
    mu = max(pop_size // 2, 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mu_eff = 1.0 / float(np.sum(weights ** 2))

    mean = np.clip(0.5 * problem.domain_heuristic_anchor() + 0.5 * rng.uniform(0.2, 0.8, size=d), 0.05, 0.95)
    sigma = 0.22
    diag_std = np.ones(d, dtype=float)
    p_sigma = np.zeros(d, dtype=float)
    p_c = np.zeros(d, dtype=float)
    c_sigma = (mu_eff + 2.0) / (d + mu_eff + 5.0)
    c_c = 4.0 / (d + 4.0)
    c_cov = 2.0 / ((d + 1.4) ** 2)

    while rec.total_calls + reps_per_eval <= budget_calls:
        gen_evals: list[tuple[np.ndarray, np.ndarray, SufficientStatsRecord]] = []
        for _ in range(pop_size):
            if rec.total_calls + reps_per_eval > budget_calls:
                break
            z = rng.normal(0.0, 1.0, size=d)
            cand = np.clip(mean + sigma * diag_std * z, 0.0, 1.0)
            ev = rec.evaluate(cand, reps_per_eval, rng)
            gen_evals.append((cand, z, ev))
        if len(gen_evals) < mu:
            break

        gen_evals.sort(key=lambda item: _deb_feasible_key(item[2], problem.alpha))
        elites = gen_evals[:mu]
        old_mean = mean.copy()
        mean = np.sum([w * item[0] for w, item in zip(weights, elites)], axis=0)
        z_mean = np.sum([w * item[1] for w, item in zip(weights, elites)], axis=0)

        p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * z_mean
        p_c = (1.0 - c_c) * p_c + np.sqrt(c_c * (2.0 - c_c) * mu_eff) * ((mean - old_mean) / max(sigma, 1e-8))
        diag_var = (1.0 - c_cov) * (diag_std ** 2) + c_cov * (p_c ** 2)
        diag_std = np.clip(np.sqrt(diag_var), 0.25, 2.5)
        sigma = float(np.clip(sigma * np.exp((c_sigma / 2.0) * (np.linalg.norm(p_sigma) / np.sqrt(d) - 1.0)), 0.03, 0.40))

        # Replicate generation winner for sharper boundary certification
        if rec.total_calls + 2 <= budget_calls:
            rec.evaluate(elites[0][0], min(2, budget_calls - rec.total_calls), rng)

    return rec


def run_constrained_nsga2(
    problem: BaseDynamicORProblem,
    *,
    budget_calls: int,
    seed: int,
    reps_per_eval: int = 3,
    pop_size: int = 10,
) -> TrajectoryArchiveRecorder:
    """SBX + polynomial-mutation evolutionary optimizer with feasibility tournament."""
    rng = np.random.default_rng(seed)
    rec = TrajectoryArchiveRecorder(problem, "constrained_nsga2")
    d = problem.d

    pop: list[SufficientStatsRecord] = []
    # Seed population with Sobol-like stratified points + jittered domain heuristic
    init_pts = [problem.domain_heuristic_anchor()]
    for i in range(pop_size - 1):
        base = (i + rng.uniform(0.0, 1.0, size=d)) / float(max(pop_size - 1, 1))
        rng.shuffle(base)
        init_pts.append(np.clip(base, 0.0, 1.0))

    for x0 in init_pts:
        if rec.total_calls + reps_per_eval > budget_calls:
            break
        pop.append(rec.evaluate(x0, reps_per_eval, rng))

    while len(pop) >= 4 and rec.total_calls + reps_per_eval <= budget_calls:
        offspring: list[SufficientStatsRecord] = []
        for _ in range(pop_size // 2):
            if rec.total_calls + reps_per_eval > budget_calls:
                break
            idx_a, idx_b = rng.choice(len(pop), size=2, replace=False)
            p1 = pop[idx_a] if _deb_feasible_key(pop[idx_a], problem.alpha) <= _deb_feasible_key(pop[idx_b], problem.alpha) else pop[idx_b]
            idx_c, idx_d = rng.choice(len(pop), size=2, replace=False)
            p2 = pop[idx_c] if _deb_feasible_key(pop[idx_c], problem.alpha) <= _deb_feasible_key(pop[idx_d], problem.alpha) else pop[idx_d]

            alpha_blend = rng.uniform(-0.15, 1.15, size=d)
            child = np.clip(alpha_blend * p1.x_norm + (1.0 - alpha_blend) * p2.x_norm, 0.0, 1.0)
            mut_mask = rng.uniform(0.0, 1.0, size=d) < max(2.0 / d, 0.08)
            if np.any(mut_mask):
                child[mut_mask] = np.clip(child[mut_mask] + rng.normal(0.0, 0.12, size=int(np.sum(mut_mask))), 0.0, 1.0)
            offspring.append(rec.evaluate(child, reps_per_eval, rng))

        combined = pop + offspring
        combined.sort(key=lambda r: _deb_feasible_key(r, problem.alpha))
        pop = combined[:pop_size]

    return rec


def run_heuristic_powell(
    problem: BaseDynamicORProblem,
    *,
    budget_calls: int,
    seed: int,
    reps_per_eval: int = 4,
) -> TrajectoryArchiveRecorder:
    """Practitioner operational heuristic + low-frequency block pattern search."""
    rng = np.random.default_rng(seed)
    rec = TrajectoryArchiveRecorder(problem, "heuristic_powell")
    d = problem.d
    anchor = problem.domain_heuristic_anchor()
    best_rec = rec.evaluate(anchor, reps_per_eval, rng)
    best_x = anchor.copy()

    # Construct practitioner low-dimensional adjustment directions (level, tilt, peak boost, smoothing)
    u = np.linspace(-1.0, 1.0, d)
    directions = [
        np.ones(d, dtype=float),
        u,
        np.cos(np.pi * u),
        np.sin(2.0 * np.pi * u),
    ]
    if d >= 4:
        even_dir = np.zeros(d, dtype=float)
        even_dir[0::2] = 1.0
        odd_dir = np.zeros(d, dtype=float)
        odd_dir[1::2] = 1.0
        directions.extend([even_dir, odd_dir])

    step_size = 0.14
    dir_idx = 0
    while rec.total_calls + reps_per_eval <= budget_calls:
        vec = directions[dir_idx % len(directions)]
        vec = vec / max(np.linalg.norm(vec) / np.sqrt(d), 1e-8)
        improved = False
        for sign in (1.0, -1.0):
            if rec.total_calls + reps_per_eval > budget_calls:
                break
            cand = np.clip(best_x + sign * step_size * vec + rng.normal(0.0, 0.015, size=d), 0.0, 1.0)
            ev = rec.evaluate(cand, reps_per_eval, rng)
            if _deb_feasible_key(ev, problem.alpha) < _deb_feasible_key(best_rec, problem.alpha):
                best_rec = ev
                best_x = cand
                improved = True
                break
        dir_idx += 1
        if not improved and dir_idx % len(directions) == 0:
            step_size = max(step_size * 0.60, 0.025)

    return rec


LEGACY_SOLVER_REGISTRY = {
    "penalty_spsa": run_penalty_spsa,
    "constrained_cmaes": run_constrained_cmaes,
    "constrained_nsga2": run_constrained_nsga2,
    "heuristic_powell": run_heuristic_powell,
}


def generate_legacy_archive(
    source_tasks: Sequence[BaseDynamicORProblem],
    *,
    solvers: Sequence[str] = (
        "penalty_spsa",
        "constrained_cmaes",
        "constrained_nsga2",
        "heuristic_powell",
    ),
    per_solver_budget: int = 54,
    reps_per_eval: int = 3,
    seed: int = 20261101,
) -> tuple[list[LegacyTrajectoryRecord], dict]:
    """Run real legacy optimizers across `source_tasks` and return full archive + ledger."""
    rng = np.random.default_rng(seed)
    all_records: list[LegacyTrajectoryRecord] = []
    per_task_summary = []

    for task in source_tasks:
        task_calls_before = task.search_calls
        solver_counts = {}
        for solver_name in solvers:
            fn = LEGACY_SOLVER_REGISTRY[solver_name]
            sub_seed = int(rng.integers(1, 2**31 - 1))
            recorder = fn(
                task,
                budget_calls=int(per_solver_budget),
                seed=sub_seed,
                reps_per_eval=int(reps_per_eval),
            )
            all_records.extend(recorder.trajectory_log)
            solver_counts[solver_name] = int(recorder.total_calls)
        per_task_summary.append(
            {
                "task_id": task.task_id,
                "policy_dim": int(task.d),
                "total_calls": int(task.search_calls - task_calls_before),
                "solver_calls": solver_counts,
            }
        )

    metadata = {
        "n_source_tasks": len(source_tasks),
        "solvers": list(solvers),
        "per_solver_budget": int(per_solver_budget),
        "reps_per_eval": int(reps_per_eval),
        "total_logged_evaluations": len(all_records),
        "total_simulation_calls": int(sum(r.n_reps for r in all_records)),
        "per_task_summary": per_task_summary,
    }
    return all_records, metadata


def save_legacy_archive_json(
    path: Path | str,
    records: Sequence[LegacyTrajectoryRecord],
    metadata: dict,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "records": [asdict(r) for r in records],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_legacy_archive_json(
    path: Path | str,
) -> tuple[list[LegacyTrajectoryRecord], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = [LegacyTrajectoryRecord(**row) for row in payload["records"]]
    return records, payload["metadata"]
