#!/usr/bin/env python3
"""Audit the same-proposal, same-SAAS, same-verifier HVD causal gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODES = ("pooled", "cumulative_factor")
DEFAULT_DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_regret(result):
    value = result.get("feasible_regret")
    if value is None or not np.isfinite(float(value)):
        return float("inf")
    return float(value)


def _row(path):
    payload = _read_json(path)
    if payload.get("status") != "ok":
        return None
    result = payload["result"]
    audit = result.get("post_run_aleatoric_audit", {})
    verification = result.get("terminal_verification", {})
    contract = payload["information_contract"]
    mode = str(contract["aleatoric_head_mode"])
    return {
        "path": str(path),
        "mode": mode,
        "heldout": str(payload["heldout"]),
        "seed": int(payload["seed"]),
        "initial_points_fingerprint": str(
            payload["initial_points_fingerprint"]),
        "source_archive_fingerprint": str(
            payload["source_archive_fingerprint"]),
        "aleatoric_head_contract": str(
            contract["aleatoric_head_contract"]),
        "n_search": int(result.get(
            "n_search_simulations", result["n_simulations"])),
        "n_verification": int(result.get(
            "n_verification_simulations", 0)),
        "true_feasible": bool(result["true_feasible"]),
        "feasible_regret": _safe_regret(result),
        "independently_certified": bool(
            verification.get("certified", False)),
        "false_certification": bool(
            verification.get("certified", False)
            and not bool(result["true_feasible"])),
        "log_variance_rmse": float(audit["log_variance_rmse"]),
        "upper_coverage": float(audit["upper_coverage"]),
        "variance_shape_correlation": float(
            audit["variance_shape_correlation"]),
    }


def _summary(rows):
    finite_regret = [
        row["feasible_regret"] for row in rows
        if np.isfinite(row["feasible_regret"])
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
            None if not finite_regret
            else float(np.median(finite_regret))
        ),
        "median_log_variance_rmse": (
            None if not rows else float(np.median([
                row["log_variance_rmse"] for row in rows]))
        ),
        "median_upper_coverage": (
            None if not rows else float(np.median([
                row["upper_coverage"] for row in rows]))
        ),
        "median_variance_shape_correlation": (
            None if not rows else float(np.median([
                row["variance_shape_correlation"] for row in rows]))
        ),
    }


def _noninferiority_gate(pooled, factor):
    return {
        "calibration_improved": bool(
            factor["median_log_variance_rmse"] is not None
            and pooled["median_log_variance_rmse"] is not None
            and factor["median_log_variance_rmse"]
            < pooled["median_log_variance_rmse"]
        ),
        "shape_correlation_improved": bool(
            factor["median_variance_shape_correlation"] is not None
            and pooled["median_variance_shape_correlation"] is not None
            and factor["median_variance_shape_correlation"]
            > pooled["median_variance_shape_correlation"]
        ),
        "feasibility_not_harmed": bool(
            factor["true_feasible_count"]
            >= pooled["true_feasible_count"]
        ),
        "false_certification_not_harmed": bool(
            factor["false_certification_count"]
            <= pooled["false_certification_count"]
        ),
        "regret_not_harmed": bool(
            factor["median_feasible_regret"] is not None
            and pooled["median_feasible_regret"] is not None
            and factor["median_feasible_regret"]
            <= pooled["median_feasible_regret"] + 1e-12
        ),
    }


def analyze(root, *, expected_domains=None, expected_seeds=None):
    root = Path(root)
    rows = [
        row
        for path in root.rglob("result.json")
        if (row := _row(path)) is not None
    ]
    by_key = {}
    for row in rows:
        key = (row["heldout"], row["seed"])
        by_key.setdefault(key, {})[row["mode"]] = row
    complete = {
        key: modes for key, modes in by_key.items()
        if set(modes) == set(MODES)
    }
    pair_audits = []
    for (heldout, seed), modes in sorted(complete.items()):
        pooled = modes["pooled"]
        factor = modes["cumulative_factor"]
        if pooled["initial_points_fingerprint"] != (
            factor["initial_points_fingerprint"]
        ):
            raise ValueError(f"{heldout}/{seed} changed the initial proposal")
        if pooled["source_archive_fingerprint"] != (
            factor["source_archive_fingerprint"]
        ):
            raise ValueError(f"{heldout}/{seed} changed the source archive")
        if pooled["n_search"] != factor["n_search"]:
            raise ValueError(f"{heldout}/{seed} changed the search budget")
        regret_gain = None
        if (
            np.isfinite(pooled["feasible_regret"])
            and np.isfinite(factor["feasible_regret"])
        ):
            regret_gain = float(
                pooled["feasible_regret"]
                - factor["feasible_regret"])
        pair_audits.append({
            "heldout": heldout,
            "seed": seed,
            "factor_minus_pooled_log_variance_rmse": float(
                factor["log_variance_rmse"]
                - pooled["log_variance_rmse"]),
            "factor_minus_pooled_upper_coverage": float(
                factor["upper_coverage"] - pooled["upper_coverage"]),
            "factor_minus_pooled_shape_correlation": float(
                factor["variance_shape_correlation"]
                - pooled["variance_shape_correlation"]),
            "factor_feasible_gain": int(
                factor["true_feasible"]) - int(pooled["true_feasible"]),
            "factor_false_certification_gain": int(
                factor["false_certification"]
            ) - int(pooled["false_certification"]),
            "factor_regret_gain": regret_gain,
        })
    summaries = {
        mode: _summary([
            row for row in rows if row["mode"] == mode
        ])
        for mode in MODES
    }
    observed_domains = sorted({row["heldout"] for row in rows})
    summaries_by_domain = {}
    domain_gates = {}
    for domain in observed_domains:
        summaries_by_domain[domain] = {
            mode: _summary([
                row for row in rows
                if row["mode"] == mode and row["heldout"] == domain
            ])
            for mode in MODES
        }
        domain_gate = _noninferiority_gate(
            summaries_by_domain[domain]["pooled"],
            summaries_by_domain[domain]["cumulative_factor"],
        )
        domain_gate["complete_pair_count"] = int(sum(
            heldout == domain for heldout, _seed in complete
        ))
        domain_gate["promote_in_domain"] = bool(all(
            domain_gate[key]
            for key in (
                "calibration_improved",
                "shape_correlation_improved",
                "feasibility_not_harmed",
                "false_certification_not_harmed",
                "regret_not_harmed",
            )
        ))
        domain_gates[domain] = domain_gate
    factor = summaries["cumulative_factor"]
    pooled = summaries["pooled"]
    complete_count = int(len(complete))
    gate = _noninferiority_gate(pooled, factor)
    gate.update({
        "complete_pair_count": complete_count,
        "all_rows_paired": bool(
            complete_count > 0 and 2 * complete_count == len(rows)),
    })
    expected_keys = None
    if expected_domains is not None and expected_seeds is not None:
        expected_keys = {
            (str(domain), int(seed))
            for domain in expected_domains
            for seed in expected_seeds
        }
    gate["all_expected_pairs_present"] = bool(
        expected_keys is None or set(complete) == expected_keys)
    expected_domain_set = (
        set(observed_domains)
        if expected_domains is None
        else set(map(str, expected_domains))
    )
    gate["all_expected_domains_observed"] = bool(
        set(observed_domains) == expected_domain_set)
    gate["promote_hvd_as_core"] = bool(
        gate["all_rows_paired"]
        and gate["all_expected_pairs_present"]
        and gate["all_expected_domains_observed"]
        and all(
            domain_gates.get(domain, {}).get(
                "promote_in_domain", False)
            for domain in expected_domain_set
        )
    )
    promoted_domains = sorted(
        domain for domain, value in domain_gates.items()
        if value["promote_in_domain"]
    )
    gate["promoted_domains"] = promoted_domains
    gate["retain_as_domain_conditional_calibration_component"] = bool(
        promoted_domains and not gate["promote_hvd_as_core"])
    return {
        "schema_version": 1,
        "root": str(root),
        "row_count": int(len(rows)),
        "summaries": summaries,
        "summaries_by_domain": summaries_by_domain,
        "domain_gates": domain_gates,
        "gate": gate,
        "pair_audits": pair_audits,
        "interpretation": (
            "HVD may be a core optimization contribution only when shape "
            "calibration improves without worsening feasibility, regret, or "
            "independent false certification under the paired SAAS contract."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--expected-domains",
        default=",".join(DEFAULT_DOMAINS),
    )
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    report = analyze(
        args.root,
        expected_domains=[
            item.strip()
            for item in args.expected_domains.split(",")
            if item.strip()
        ],
        expected_seeds=range(
            args.seed_start, args.seed_start + args.n_seeds),
    )
    _atomic_json(args.out, report)
    print(json.dumps(report["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
