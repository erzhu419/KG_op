#!/usr/bin/env python3
"""Offline causal gate for the observable-provider cumulative HVD repair.

The audit freezes one replicated source archive, fits every variance head
before drawing held-out audit points, and uses target truth only after fitting
to score calibration.  It runs no optimizer and consumes no target simulator
budget.  Passing this gate permits, but does not replace, a sequential CPU
backend comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive  # noqa: E402
from core.designs import common_sobol_integer_design  # noqa: E402
from performance.benchmark_lodo_meta_prior import (  # noqa: E402
    build_scalarized_problem,
)
from performance.benchmark_quality import json_safe  # noqa: E402
from variance.source_archive_hvd import (  # noqa: E402
    FrozenSourceArchiveAleatoricHead,
    SourceArchiveHVDConfig,
)


DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
MODES = (
    "pooled",
    "cumulative_factor",
    "provider_cumulative_factor",
)


def parse_csv(value):
    return tuple(
        item.strip() for item in str(value).split(",") if item.strip())


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def archive_path(root, heldout):
    return (
        Path(root)
        / heldout
        / f"heldout_{heldout}.json"
    )


def audit_points(problem, head, *, seed, audit_size):
    points = common_sobol_integer_design(
        problem,
        max(8, int(audit_size)),
        int(seed) + 9041081,
    )
    predicted = np.asarray([
        head.predict_variance(point) for point in points
    ], dtype=float)
    upper = np.asarray([
        head.predict_certification_variance(point) for point in points
    ], dtype=float)
    truth = np.asarray([
        max(float(problem.true_sigma(point)[1]) ** 2, 1e-14)
        for point in points
    ], dtype=float)
    log_error = np.log(np.maximum(predicted, 1e-14)) - np.log(truth)
    centered_prediction = predicted - float(np.mean(predicted))
    centered_truth = truth - float(np.mean(truth))
    denominator = float(np.sqrt(
        np.sum(centered_prediction ** 2)
        * np.sum(centered_truth ** 2)
    ))
    return {
        "seed": int(seed),
        "audit_size": int(len(points)),
        "log_variance_rmse": float(np.sqrt(np.mean(log_error ** 2))),
        "variance_rmse": float(np.sqrt(np.mean(
            (predicted - truth) ** 2))),
        "variance_shape_correlation": (
            0.0
            if denominator <= 1e-14
            else float(np.sum(
                centered_prediction * centered_truth) / denominator)
        ),
        "upper_coverage": float(np.mean(truth <= upper)),
        "mean_predicted_to_true_ratio": float(
            np.mean(predicted) / max(float(np.mean(truth)), 1e-14)),
        "mean_upper_to_true_ratio": float(
            np.mean(upper) / max(float(np.mean(truth)), 1e-14)),
        "target_truth_used_post_fit_only": True,
        "target_truth_used_for_selection": False,
    }


def summarize(rows):
    metrics = (
        "log_variance_rmse",
        "variance_rmse",
        "variance_shape_correlation",
        "upper_coverage",
        "mean_predicted_to_true_ratio",
        "mean_upper_to_true_ratio",
    )
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "median": float(np.median([row[metric] for row in rows])),
            "minimum": float(np.min([row[metric] for row in rows])),
            "maximum": float(np.max([row[metric] for row in rows])),
        }
        for metric in metrics
    }


def run(args):
    domains = parse_csv(args.domains)
    modes = parse_csv(args.modes)
    seeds = tuple(range(
        int(args.seed_start),
        int(args.seed_start) + int(args.n_seeds),
    ))
    cells = {}
    rows = []
    for heldout in domains:
        archive = FrozenTransferArchive.load(
            archive_path(args.archive_root, heldout))
        problem = build_scalarized_problem(
            heldout,
            int(args.d),
            int(args.L),
            float(args.sigma),
            float(args.alpha),
            (0.5, 0.5),
        )
        cells[heldout] = {}
        for mode in modes:
            started = time.perf_counter()
            head = FrozenSourceArchiveAleatoricHead(
                archive=archive,
                target_problem=problem,
                config=SourceArchiveHVDConfig(
                    mode=mode,
                    calibration_delta=float(args.calibration_delta),
                    calibration_quantile=float(args.calibration_quantile),
                    provider_ridge_per_source_row=float(
                        args.provider_ridge_per_source_row),
                ),
            )
            diagnostics = head.diagnostics()
            mode_rows = [
                audit_points(
                    problem,
                    head,
                    seed=seed,
                    audit_size=int(args.audit_size),
                )
                for seed in seeds
            ]
            row = {
                "heldout": heldout,
                "mode": mode,
                "source_archive_fingerprint": archive.fingerprint,
                "source_simulator_calls": int(archive.simulator_calls),
                "target_audit_calls": 0,
                "target_outcomes_used_to_fit": False,
                "target_oracle_used_to_fit": False,
                "coordinate_id": diagnostics["coordinate_id"],
                "calibration_id": diagnostics["calibration_id"],
                "calibration_multiplier": float(
                    diagnostics["calibration_multiplier"]),
                "fit_wall_time_sec": float(time.perf_counter() - started),
                "summary": summarize(mode_rows),
                "seed_rows": mode_rows,
                "head_diagnostics": diagnostics,
            }
            rows.append(row)
            cells[heldout][mode] = row

    decisions = {}
    for heldout in domains:
        pooled = cells[heldout]["pooled"]["summary"]
        provider = cells[heldout][
            "provider_cumulative_factor"]["summary"]
        checks = {
            "log_rmse_strictly_better_than_pooled": bool(
                provider["log_variance_rmse"]["median"]
                < pooled["log_variance_rmse"]["median"]),
            "shape_correlation_at_least_0_5": bool(
                provider["variance_shape_correlation"]["median"] >= 0.5),
            "shape_correlation_gain_at_least_0_25": bool(
                provider["variance_shape_correlation"]["median"]
                - pooled["variance_shape_correlation"]["median"]
                >= 0.25),
            "upper_coverage_at_least_0_95": bool(
                provider["upper_coverage"]["mean"] >= 0.95),
            "finite_calibration_multiplier": bool(np.isfinite(
                cells[heldout]["provider_cumulative_factor"][
                    "calibration_multiplier"])),
        }
        decisions[heldout] = {
            "checks": checks,
            "pass": bool(all(checks.values())),
            "paired_median_log_rmse_improvement": float(
                pooled["log_variance_rmse"]["median"]
                - provider["log_variance_rmse"]["median"]),
            "paired_median_shape_correlation_gain": float(
                provider["variance_shape_correlation"]["median"]
                - pooled["variance_shape_correlation"]["median"]),
        }
    all_pass = bool(decisions and all(
        row["pass"] for row in decisions.values()))
    return {
        "schema_version": 1,
        "gate_id": "provider_cumulative_hvd_offline_causal_gate_v1",
        "status": "complete",
        "information_contract": {
            "source_archive_frozen_before_target_audit": True,
            "source_oracle_labels_used": False,
            "target_outcomes_used_to_fit": False,
            "target_oracle_used_to_fit": False,
            "target_truth_timing": "post_fit_audit_only",
            "target_audit_simulator_calls": 0,
            "provider_track": "descriptor_conditional_observable_A_N_schema",
            "final_safety_role": "independent_terminal_verifier",
        },
        "config": {
            "domains": list(domains),
            "modes": list(modes),
            "d": int(args.d),
            "audit_size": int(args.audit_size),
            "seeds": list(seeds),
            "provider_ridge_per_source_row": float(
                args.provider_ridge_per_source_row),
            "calibration_delta": float(args.calibration_delta),
        },
        "decisions": decisions,
        "all_domains_pass": all_pass,
        "advance_to_cpu_sequential_gate": all_pass,
        "rows": rows,
    }


def compact_payload(payload):
    return {
        key: value
        for key, value in payload.items()
        if key != "rows"
    } | {
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"seed_rows", "head_diagnostics"}
            }
            for row in payload["rows"]
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compact-out", default="")
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--audit-size", type=int, default=256)
    parser.add_argument("--calibration-delta", type=float, default=0.05)
    parser.add_argument("--calibration-quantile", type=float, default=0.95)
    parser.add_argument(
        "--provider-ridge-per-source-row", type=float, default=1.0)
    args = parser.parse_args()
    payload = run(args)
    atomic_json(args.out, payload)
    if args.compact_out:
        atomic_json(args.compact_out, compact_payload(payload))
    print(json.dumps({
        "out": str(args.out),
        "all_domains_pass": payload["all_domains_pass"],
        "decisions": payload["decisions"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
