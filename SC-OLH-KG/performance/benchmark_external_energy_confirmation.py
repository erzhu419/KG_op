#!/usr/bin/env python3
"""Untouched GB_GBN confirmation for the frozen OPSD front end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_external_energy_gate import (  # noqa: E402
    _atomic_json,
    run_gate,
)


CONFIRMATORY_CONTRACT_ID = "opsd_energy_gb_gbn_confirmatory_v1"
CONFIRMATORY_MARKET = "GB_GBN"
CONFIRMATORY_YEAR = 2018
CONFIRMATORY_SEEDS = tuple(range(100, 120))


def run_confirmation(
    *,
    data_path,
    seed,
    arm,
    dimension=1000,
    n0=10,
    N=13,
    design_path=None,
    verification_budgets=(80, 128, 128),
):
    if int(seed) not in CONFIRMATORY_SEEDS:
        raise ValueError("seed is outside the frozen GB_GBN confirmatory set")
    result = run_gate(
        data_path=data_path,
        seed=seed,
        arm=arm,
        market=CONFIRMATORY_MARKET,
        year=CONFIRMATORY_YEAR,
        dimension=dimension,
        n0=n0,
        N=N,
        design_path=design_path,
        verification_budgets=verification_budgets,
    )
    result.update({
        "contract_id": CONFIRMATORY_CONTRACT_ID,
        "evidence_phase": "confirmatory_holdout",
        "development_only": False,
        "confirmatory_target_opened": True,
        "method_frozen_before_target_outcomes": True,
        "development_contract_id": (
            "opsd_energy_stochastic_online_development_gate_v1"),
        "development_gate_status": "pass",
        "target_market_was_unopened_during_method_selection": True,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--arm", choices=("frozen_proposal", "common_sobol"), required=True)
    parser.add_argument("--design")
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--verification-budgets", default="80,128,128")
    args = parser.parse_args()
    result = run_confirmation(
        data_path=args.data,
        seed=args.seed,
        arm=args.arm,
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
