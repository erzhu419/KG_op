#!/usr/bin/env python3
"""Analyze the paired pooled/provider-cumulative HVD CPU gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODES = ("pooled", "provider_cumulative_factor")
DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_csv(value):
    return tuple(
        item.strip() for item in str(value).split(",") if item.strip())


def finite_regret(result):
    value = result.get("feasible_regret")
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def load_row(path):
    payload = read_json(path)
    if payload.get("status") != "ok":
        return None
    result = payload["result"]
    mode = str(result.get(
        "aleatoric_head_mode",
        payload.get("information_contract", {}).get("aleatoric_head_mode"),
    ))
    if mode not in MODES:
        return None
    audit = result.get("post_run_aleatoric_audit", {})
    verification = result.get("terminal_verification", {})
    return {
        "path": str(path),
        "heldout": str(payload["heldout"]),
        "seed": int(payload["seed"]),
        "mode": mode,
        "method": str(payload.get("method", result.get("method"))),
        "initial_points_fingerprint": str(
            payload["initial_points_fingerprint"]),
        "source_archive_fingerprint": str(
            payload["source_archive_fingerprint"]),
        "n_search": int(result["n_search_simulations"]),
        "n_verification": int(result.get(
            "n_verification_simulations", 0)),
        "n_total_target": int(result.get(
            "n_target_simulations_total",
            result["n_search_simulations"],
        )),
        "true_feasible": bool(result["true_feasible"]),
        "feasible_regret": finite_regret(result),
        "independently_certified": bool(
            verification.get("certified", False)),
        "false_certification": bool(
            verification.get("certified", False)
            and not bool(result["true_feasible"])),
        "log_variance_rmse": float(audit["log_variance_rmse"]),
        "upper_coverage": float(audit["upper_coverage"]),
        "variance_shape_correlation": float(
            audit["variance_shape_correlation"]),
        "aleatoric_calibration_multiplier": float(
            result["aleatoric_head"]["calibration_multiplier"]),
    }


def summarize(rows):
    regrets = [
        row["feasible_regret"] for row in rows
        if row["feasible_regret"] is not None
    ]
    return {
        "row_count": int(len(rows)),
        "true_feasible_count": int(sum(
            row["true_feasible"] for row in rows)),
        "independently_certified_count": int(sum(
            row["independently_certified"] for row in rows)),
        "false_certification_count": int(sum(
            row["false_certification"] for row in rows)),
        "median_feasible_regret": (
            None if not regrets else float(np.median(regrets))),
        "mean_verification_calls": float(np.mean([
            row["n_verification"] for row in rows
        ])) if rows else None,
        "median_verification_calls": float(np.median([
            row["n_verification"] for row in rows
        ])) if rows else None,
        "median_log_variance_rmse": float(np.median([
            row["log_variance_rmse"] for row in rows
        ])) if rows else None,
        "median_variance_shape_correlation": float(np.median([
            row["variance_shape_correlation"] for row in rows
        ])) if rows else None,
        "median_upper_coverage": float(np.median([
            row["upper_coverage"] for row in rows
        ])) if rows else None,
        "median_calibration_multiplier": float(np.median([
            row["aleatoric_calibration_multiplier"] for row in rows
        ])) if rows else None,
    }


def regret_not_harmed(pooled, provider):
    left = pooled["median_feasible_regret"]
    right = provider["median_feasible_regret"]
    if left is None or right is None:
        return False
    return bool(right <= left + 1e-12)


def analyze(root, *, domains, seeds):
    root = Path(root)
    rows = [
        row
        for path in root.rglob("result.json")
        if (row := load_row(path)) is not None
    ]
    by_key = {}
    for row in rows:
        key = (row["heldout"], row["seed"])
        if row["mode"] in by_key.setdefault(key, {}):
            raise ValueError(f"duplicate row for {key}/{row['mode']}")
        by_key[key][row["mode"]] = row
    expected = {
        (str(domain), int(seed))
        for domain in domains for seed in seeds
    }
    complete = {
        key: modes for key, modes in by_key.items()
        if set(modes) == set(MODES)
    }
    for key, modes in complete.items():
        pooled = modes["pooled"]
        provider = modes["provider_cumulative_factor"]
        for field in (
            "initial_points_fingerprint",
            "source_archive_fingerprint",
            "n_search",
            "method",
        ):
            if pooled[field] != provider[field]:
                raise ValueError(f"{key} changed paired field {field}")

    summaries_by_domain = {}
    domain_gates = {}
    for domain in domains:
        domain_rows = [row for row in rows if row["heldout"] == domain]
        pooled = summarize([
            row for row in domain_rows if row["mode"] == "pooled"])
        provider = summarize([
            row for row in domain_rows
            if row["mode"] == "provider_cumulative_factor"
        ])
        summaries_by_domain[domain] = {
            "pooled": pooled,
            "provider_cumulative_factor": provider,
        }
        checks = {
            "all_seed_pairs_present": bool(all(
                (domain, seed) in complete for seed in seeds)),
            "calibration_improved": bool(
                provider["median_log_variance_rmse"] is not None
                and pooled["median_log_variance_rmse"] is not None
                and provider["median_log_variance_rmse"]
                < pooled["median_log_variance_rmse"]),
            "shape_correlation_improved": bool(
                provider["median_variance_shape_correlation"] is not None
                and pooled["median_variance_shape_correlation"] is not None
                and provider["median_variance_shape_correlation"]
                > pooled["median_variance_shape_correlation"] + 0.25),
            "feasibility_not_harmed": bool(
                provider["true_feasible_count"]
                >= pooled["true_feasible_count"]),
            "false_certification_not_harmed": bool(
                provider["false_certification_count"]
                <= pooled["false_certification_count"]),
            "regret_not_harmed": regret_not_harmed(pooled, provider),
            "verification_cost_not_harmed": bool(
                provider["mean_verification_calls"] is not None
                and pooled["mean_verification_calls"] is not None
                and provider["mean_verification_calls"]
                <= pooled["mean_verification_calls"] + 1e-12),
        }
        operational_gain = bool(
            provider["true_feasible_count"] > pooled["true_feasible_count"]
            or (
                provider["median_feasible_regret"] is not None
                and pooled["median_feasible_regret"] is not None
                and provider["median_feasible_regret"]
                < pooled["median_feasible_regret"] - 1e-12
            )
            or (
                provider["mean_verification_calls"] is not None
                and pooled["mean_verification_calls"] is not None
                and provider["mean_verification_calls"]
                < pooled["mean_verification_calls"] - 1e-12
            )
        )
        domain_gates[domain] = {
            "checks": checks,
            "pass_calibration_and_noninferiority": bool(all(checks.values())),
            "operational_gain_present": operational_gain,
        }

    all_expected = bool(set(complete) == expected)
    calibration_and_noninferiority = bool(
        all_expected
        and all(row["pass_calibration_and_noninferiority"]
                for row in domain_gates.values())
    )
    operational_gain_domains = [
        domain for domain, row in domain_gates.items()
        if row["operational_gain_present"]
    ]
    return {
        "schema_version": 1,
        "gate_id": "provider_cumulative_hvd_cpu_sequential_gate_v1",
        "status": "complete",
        "root": str(root),
        "row_count": int(len(rows)),
        "complete_pair_count": int(len(complete)),
        "all_expected_pairs_present": all_expected,
        "summaries_by_domain": summaries_by_domain,
        "domain_gates": domain_gates,
        "gate": {
            "advance_from_5_to_20_seeds": calibration_and_noninferiority,
            "retain_hvd_as_calibration_component": (
                calibration_and_noninferiority),
            "promote_hvd_as_core_contribution": bool(
                calibration_and_noninferiority
                and operational_gain_domains),
            "operational_gain_domains": operational_gain_domains,
            "decision_rule": (
                "Provider HVD must improve variance calibration/shape in "
                "every domain without worsening feasibility, false "
                "certification, regret, or verification cost. Core status "
                "additionally requires an operational gain."
            ),
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    domains = parse_csv(args.domains)
    seeds = tuple(range(
        int(args.seed_start),
        int(args.seed_start) + int(args.n_seeds),
    ))
    payload = analyze(args.root, domains=domains, seeds=seeds)
    atomic_json(args.out, payload)
    print(json.dumps({
        "out": str(args.out),
        "row_count": payload["row_count"],
        "gate": payload["gate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
