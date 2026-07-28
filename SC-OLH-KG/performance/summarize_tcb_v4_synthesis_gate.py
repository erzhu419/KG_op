"""Aggregate the strict nested-LODO TCB-V4 synthesis gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_quality import json_safe  # noqa: E402
from performance import summarize_tcb_v2_source_gate as shared  # noqa: E402


SYNTHESIS_FIELDS = (
    "coefficient_ridge",
    "coefficient_prior_strength",
    "coefficient_floor",
    "synthesis_mode",
)
CONFIG_FIELDS = tuple(
    field for field in shared.CONFIG_FIELDS if field != "pilot_policy"
) + SYNTHESIS_FIELDS + ("pilot_policy",)
MODEL_CONFIG_FIELDS = tuple(
    field for field in CONFIG_FIELDS if field != "pilot_policy")


def _synthesis_audit(rows):
    diagnostics = [row.get("adapter_diagnostics", {}) for row in rows]
    coefficient_checks = []
    covariance_checks = []
    dimension_checks = []
    for row, diagnostic in zip(rows, diagnostics):
        coefficients = np.asarray(
            diagnostic.get("coefficients", []), dtype=float)
        eigenvalues = np.asarray(
            diagnostic.get("parameter_covariance_eigenvalues", []),
            dtype=float,
        )
        coefficient_checks.append(bool(
            len(coefficients) == int(row["family_count"])
            and np.all(np.isfinite(coefficients))
            and np.all(coefficients >= -1e-12)
        ))
        covariance_checks.append(bool(
            len(eigenvalues) == int(row["adapter_effective_dimension"])
            and np.all(np.isfinite(eigenvalues))
            and np.min(eigenvalues) >= -1e-10
        ))
        dimension_checks.append(bool(
            int(row["adapter_effective_dimension"])
            == int(row["family_count"]) + 1
        ))
    checks = {
        "family_coefficients_are_nonnegative": bool(
            all(coefficient_checks)),
        "coefficient_covariance_is_psd": bool(all(covariance_checks)),
        "adapter_dimension_matches_dictionary": bool(all(dimension_checks)),
        "source_dictionary_frozen": bool(all(
            bool(item.get("source_dictionary_frozen", False))
            for item in diagnostics
        )),
        "target_domain_label_unused": bool(all(
            not bool(item.get("target_label_used", True))
            for item in diagnostics
        )),
        "target_oracle_unused": bool(all(
            not bool(item.get("target_oracle_used", True))
            for item in diagnostics
        )),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "mean_effective_family_count": float(np.mean([
            item.get("effective_family_count", np.nan)
            for item in diagnostics
        ])),
        "mean_parameter_covariance_trace": float(np.mean([
            item.get("parameter_covariance_trace", np.nan)
            for item in diagnostics
        ])),
    }


def summarize(rows, args):
    old_config = shared.CONFIG_FIELDS
    old_model = shared.MODEL_CONFIG_FIELDS
    shared.CONFIG_FIELDS = CONFIG_FIELDS
    shared.MODEL_CONFIG_FIELDS = MODEL_CONFIG_FIELDS
    try:
        result = shared.summarize(rows, args)
    finally:
        shared.CONFIG_FIELDS = old_config
        shared.MODEL_CONFIG_FIELDS = old_model
    audit = _synthesis_audit(rows)
    result["schema_version"] = 4
    result["gate_model"] = "tcb_v4_boundary_family_synthesis"
    result["synthesis_posterior_audit"] = audit
    result["gate_pass_before_synthesis_audit"] = bool(result["gate_pass"])
    result["gate_pass"] = bool(result["gate_pass"] and audit["passed"])
    if not result["gate_pass"]:
        result["promoted_candidate"] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-domains", type=int, default=5)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--false-safe-tolerance", type=float, default=0.01)
    parser.add_argument("--minimum-spearman", type=float, default=0.35)
    parser.add_argument("--minimum-nonvacuous-rate", type=float, default=0.50)
    parser.add_argument("--maximum-adapter-dimension", type=int, default=8)
    args = parser.parse_args()
    rows, invalid = shared.load_rows(args.input_root)
    result = summarize(rows, args)
    result["invalid_files"] = invalid
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({
        "gate_pass": result["gate_pass"],
        "n_input_rows": result["n_input_rows"],
        "n_configurations": result["n_configurations"],
        "n_passing": result["n_passing"],
        "synthesis_audit": result["synthesis_posterior_audit"]["passed"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
