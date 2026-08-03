#!/usr/bin/env python3
"""Stochastic five-seed OPSD development gate with exact verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.designs import (  # noqa: E402
    common_sobol_integer_design,
    integer_design_fingerprint,
    load_frozen_source_informed_design,
    next_sobol_integer_candidate,
)
from performance.benchmark_quality import json_safe  # noqa: E402
from performance.materialize_external_energy_design import target_task_id  # noqa: E402
from problems.energy_reliability import OPSDStorageReliabilityProblem  # noqa: E402


GATE_CONTRACT_ID = "opsd_energy_stochastic_online_development_v1"
VERIFIER_CONTRACT_ID = "opsd_energy_familywise_exact_binomial_shortlist_v1"


def _binomial_lower(successes, trials, delta):
    successes = int(successes)
    trials = int(trials)
    if successes <= 0:
        return 0.0
    return float(beta.ppf(
        float(delta), successes, trials - successes + 1))


def _policy_summary(problem, point):
    z = problem.normalize(point)
    return {
        "fingerprint": integer_design_fingerprint([point]),
        "mean": float(np.mean(z)),
        "std": float(np.std(z)),
        "mean_absolute_ramp": float(np.mean(np.abs(np.diff(z)))),
        "minimum": float(np.min(z)),
        "maximum": float(np.max(z)),
    }


def _select_shortlist(problem, records, size=3):
    """Objective, safe-interior, and Bayes-risk empirical roles."""

    if not records:
        raise ValueError("cannot select an empty energy shortlist")
    feasible = [row for row in records if row["observation"][1] <= problem.tau]
    roles = []
    if feasible:
        roles.append((
            "observed_feasible_objective",
            min(feasible, key=lambda row: row["observation"][0]),
        ))
        roles.append((
            "observed_safe_interior",
            min(feasible, key=lambda row: row["observation"][1]),
        ))
    penalty = 2.0
    roles.append((
        "empirical_bayes_risk",
        min(records, key=lambda row: (
            row["observation"][0]
            + penalty * max(row["observation"][1] - problem.tau, 0.0)
        )),
    ))
    roles.extend((
        "objective_fallback",
        row,
    ) for row in sorted(records, key=lambda row: row["observation"][0]))

    selected = []
    seen = set()
    for role, row in roles:
        point = tuple(row["point"])
        if point in seen:
            continue
        seen.add(point)
        selected.append({"role": role, **row})
        if len(selected) == min(int(size), len(records)):
            break
    return selected


def _verify_shortlist(
    problem,
    shortlist,
    *,
    seed,
    budgets=(80, 128, 128),
    familywise_delta=0.05,
):
    budgets = tuple(map(int, budgets))
    if len(budgets) != len(shortlist):
        raise ValueError("verification budgets must match the shortlist")
    rng = np.random.default_rng(int(seed) + 7_771_003)
    per_candidate_delta = float(familywise_delta) / len(shortlist)
    target_probability = 1.0 - float(problem.alpha)
    attempts = []
    deployed = None
    for rank, (record, budget) in enumerate(zip(shortlist, budgets), start=1):
        samples = np.vstack([
            problem.simulate_from_split(record["point"], "verification", rng)
            for _ in range(int(budget))
        ])
        successes = int(np.sum(samples[:, 1] <= problem.tau))
        lower = _binomial_lower(successes, budget, per_candidate_delta)
        certified = bool(lower >= target_probability)
        attempt = {
            "rank": int(rank),
            "role": record["role"],
            "policy": _policy_summary(problem, record["point"]),
            "budget": int(budget),
            "successes": successes,
            "empirical_feasible_probability": float(successes / budget),
            "familywise_exact_lower": lower,
            "certified": certified,
            "objective_sample_mean": float(np.mean(samples[:, 0])),
            "samples_logged": False,
        }
        attempts.append(attempt)
        if certified:
            deployed = record
            break
    verification_calls = int(sum(row["budget"] for row in attempts))
    return deployed, {
        "contract_id": VERIFIER_CONTRACT_ID,
        "status": "certified" if deployed is not None else "abstained",
        "certified": bool(deployed is not None),
        "familywise_delta": float(familywise_delta),
        "per_candidate_delta": per_candidate_delta,
        "target_probability": target_probability,
        "candidate_budgets": list(budgets),
        "verification_calls": verification_calls,
        "attempts": attempts,
        "verification_split": "next_full_calendar_year",
        "search_samples_reused": False,
        "posterior_updated_from_verification": False,
        "target_oracle_used": False,
    }


def run_gate(
    *,
    data_path,
    seed,
    arm,
    market="DK_2",
    year=2018,
    dimension=1000,
    n0=10,
    N=13,
    design_path=None,
    verification_budgets=(80, 128, 128),
):
    arm = str(arm)
    if arm not in {"frozen_proposal", "common_sobol"}:
        raise ValueError("energy gate arm must be frozen_proposal or common_sobol")
    if int(N) < int(n0):
        raise ValueError("N must be at least n0")
    problem = OPSDStorageReliabilityProblem(
        data_path, market=market, year=year, d=dimension)
    task = target_task_id(market, year)
    design_contract = None
    source_calls = 0
    if arm == "frozen_proposal":
        if not design_path:
            raise ValueError("frozen proposal arm requires a design artifact")
        points, design_contract = load_frozen_source_informed_design(
            design_path,
            heldout=task,
            seed=int(seed),
            n0=int(n0),
            dimension=int(dimension),
        )
        payload = json.loads(Path(design_path).read_text(encoding="utf-8"))
        source_calls = int(payload["source_archive_simulator_calls"])
    else:
        points = tuple(common_sobol_integer_design(
            problem, int(n0), int(seed)))

    rng = np.random.default_rng(int(seed) + 1_909_117)
    records = []
    observed = []
    for index, point in enumerate(points):
        observation = problem.simulate(point, rng)
        observed.append(tuple(point))
        records.append({
            "point": tuple(point),
            "observation": observation,
            "source": "initial_design",
            "evaluation_index": int(index),
        })
    while len(records) < int(N):
        point = next_sobol_integer_candidate(
            problem,
            int(seed),
            observed=observed,
            seed_offset=330_107,
        )
        observation = problem.simulate(point, rng)
        observed.append(tuple(point))
        records.append({
            "point": tuple(point),
            "observation": observation,
            "source": "neutral_sobol_continuation",
            "evaluation_index": int(len(records)),
        })

    shortlist = _select_shortlist(problem, records, size=3)
    deployed, verification = _verify_shortlist(
        problem,
        shortlist,
        seed=int(seed),
        budgets=verification_budgets,
    )
    truth = None
    if deployed is not None:
        population = problem.split_population(deployed["point"], "verification")
        true_probability = float(np.mean(population[:, 1] <= problem.tau))
        truth = {
            "policy": _policy_summary(problem, deployed["point"]),
            "verification_population_size": int(len(population)),
            "true_feasible_probability": true_probability,
            "truly_chance_feasible": bool(
                true_probability >= 1.0 - problem.alpha),
            "true_objective_mean": float(np.mean(population[:, 0])),
            "used_for_search_or_selection": False,
        }
    compact_search = [{
        "evaluation_index": row["evaluation_index"],
        "source": row["source"],
        "policy": _policy_summary(problem, row["point"]),
        "observation": np.asarray(row["observation"], dtype=float).tolist(),
    } for row in records]
    return {
        "schema_version": 1,
        "contract_id": GATE_CONTRACT_ID,
        "status": "ok",
        "development_only": True,
        "confirmatory_target_opened": False,
        "arm": arm,
        "market": market,
        "year": int(year),
        "dimension": int(dimension),
        "seed": int(seed),
        "n0": int(n0),
        "target_search_calls": int(N),
        "source_calls": source_calls,
        "verification_calls": int(verification["verification_calls"]),
        "total_target_calls": int(N + verification["verification_calls"]),
        "design_contract": design_contract,
        "information_contract": problem.information_contract(),
        "search_records": compact_search,
        "shortlist": [{
            "rank": index + 1,
            "role": row["role"],
            "policy": _policy_summary(problem, row["point"]),
        } for index, row in enumerate(shortlist)],
        "verification": verification,
        "deployment_truth_audit": truth,
        "independently_certified": bool(verification["certified"]),
        "false_certificate": bool(
            verification["certified"]
            and truth is not None
            and not truth["truly_chance_feasible"]
        ),
        "target_oracle_used_during_search": False,
        "target_outcomes_used_to_fit_proposal": False,
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--arm", choices=("frozen_proposal", "common_sobol"), required=True)
    parser.add_argument("--design")
    parser.add_argument("--market", default="DK_2")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--verification-budgets", default="80,128,128")
    args = parser.parse_args()
    result = run_gate(
        data_path=args.data,
        seed=args.seed,
        arm=args.arm,
        market=args.market,
        year=args.year,
        dimension=args.d,
        n0=args.n0,
        N=args.N,
        design_path=args.design,
        verification_budgets=tuple(
            int(value) for value in args.verification_budgets.split(",")),
    )
    _atomic_json(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "out": str(args.out),
        "arm": result["arm"],
        "seed": result["seed"],
        "independently_certified": result["independently_certified"],
        "false_certificate": result["false_certificate"],
        "verification_calls": result["verification_calls"],
    }, indent=2, sort_keys=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
