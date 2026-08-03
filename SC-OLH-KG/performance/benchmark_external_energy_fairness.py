#!/usr/bin/env python3
"""Post-confirmatory OPSD structured and total-cost controls.

These controls were registered only after the GB_GBN confirmatory target had
been opened.  They cannot promote or tune the frozen method.  Their sole role
is to test whether the original common-Sobol comparison was too weak.
"""

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


FAIRNESS_CONTRACT_ID = "opsd_energy_postconfirmatory_fairness_v1"
MARKET = "GB_GBN"
YEAR = 2018
SEEDS = tuple(range(100, 120))
ARMS = {
    "low_frequency_grid_n13": {
        "runner_arm": "low_frequency_grid",
        "N": 13,
        "source_calls": 0,
        "role": "natural_target_label_free_structured_control",
    },
    "common_sobol_n397": {
        "runner_arm": "common_sobol",
        "N": 397,
        "source_calls": 0,
        "role": "source_plus_search_cost_matched_target_only_control",
    },
}


def run_fairness(
    *, data_path, seed, arm, dimension=1000, n0=10,
    verification_budgets=(80, 128, 128),
):
    if int(seed) not in SEEDS:
        raise ValueError("seed is outside the registered fairness set")
    if arm not in ARMS:
        raise ValueError(f"unknown fairness arm {arm!r}")
    contract = ARMS[arm]
    result = run_gate(
        data_path=data_path,
        seed=int(seed),
        arm=contract["runner_arm"],
        market=MARKET,
        year=YEAR,
        dimension=int(dimension),
        n0=int(n0),
        N=int(contract["N"]),
        verification_budgets=verification_budgets,
    )
    result.update({
        "contract_id": FAIRNESS_CONTRACT_ID,
        "fairness_arm": str(arm),
        "fairness_role": contract["role"],
        "evidence_phase": "post_confirmatory_fairness_audit",
        "confirmatory_claim_eligible": False,
        "method_or_hyperparameter_tuning_permitted": False,
        "gb_gbn_outcomes_were_open_before_registration": True,
        "frozen_method_changed": False,
        "source_calls": int(contract["source_calls"]),
        "source_plus_target_search_calls": int(
            contract["source_calls"] + contract["N"]),
        "comparison_reference": {
            "frozen_source_calls": 384,
            "frozen_target_search_calls": 13,
            "frozen_source_plus_target_search_calls": 397,
        },
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--verification-budgets", default="80,128,128")
    args = parser.parse_args()
    result = run_fairness(
        data_path=args.data,
        seed=args.seed,
        arm=args.arm,
        dimension=args.d,
        n0=args.n0,
        verification_budgets=tuple(
            int(value) for value in args.verification_budgets.split(",")),
    )
    _atomic_json(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "out": str(args.out),
        "arm": args.arm,
        "seed": int(args.seed),
        "independently_certified": result["independently_certified"],
        "false_certificate": result["false_certificate"],
        "verification_calls": result["verification_calls"],
    }, indent=2, sort_keys=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
