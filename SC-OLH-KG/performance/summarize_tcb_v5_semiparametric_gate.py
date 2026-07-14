"""Aggregate the strict nested-LODO TCB-V5 semiparametric gate."""

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


EXTRA_FIELDS = (
    "coefficient_ridge",
    "coefficient_prior_strength",
    "coefficient_floor",
    "synthesis_mode",
    "semiparametric_residual_features",
    "semiparametric_residual_ridge",
    "semiparametric_lengthscale_multiplier",
)
CONFIG_FIELDS = tuple(
    field for field in shared.CONFIG_FIELDS if field != "pilot_policy"
) + EXTRA_FIELDS + ("pilot_policy",)
MODEL_CONFIG_FIELDS = tuple(
    field for field in CONFIG_FIELDS if field != "pilot_policy")


def _semiparametric_audit(rows):
    checks_by_row = []
    effective_family_count = []
    covariance_trace = []
    for row in rows:
        diagnostic = row.get("adapter_diagnostics", {})
        coefficients = np.asarray(
            diagnostic.get("coefficients", []), dtype=float)
        residual = np.asarray(
            diagnostic.get("residual_coefficients", []), dtype=float)
        eigenvalues = np.asarray(
            diagnostic.get("parameter_covariance_eigenvalues", []),
            dtype=float,
        )
        expected_dimension = (
            int(row["family_count"])
            + int(row["semiparametric_residual_features"])
            + 1
        )
        checks_by_row.append({
            "family_coefficients_nonnegative": bool(
                len(coefficients) == int(row["family_count"])
                and np.all(np.isfinite(coefficients))
                and np.all(coefficients >= -1e-12)
            ),
            "residual_dimension_matches": bool(
                len(residual)
                == int(row["semiparametric_residual_features"])
            ),
            "covariance_psd": bool(
                len(eigenvalues) == expected_dimension
                and np.all(np.isfinite(eigenvalues))
                and np.min(eigenvalues) >= -1e-10
            ),
            "effective_dimension_matches": bool(
                int(row["adapter_effective_dimension"])
                == expected_dimension
            ),
            "source_dictionary_frozen": bool(
                diagnostic.get("source_dictionary_frozen", False)
            ),
            "residual_dictionary_frozen": bool(
                diagnostic.get(
                    "orthogonal_residual_dictionary_frozen", False)
            ),
            "source_projection_orthogonal": bool(
                float(row["semiparametric_orthogonality_relative"])
                <= 1e-10
            ),
            "projection_target_label_free": bool(
                not row["residual_dictionary_target_labels_used"]
            ),
            "target_label_unused": bool(
                not diagnostic.get("target_label_used", True)
            ),
            "target_oracle_unused": bool(
                not diagnostic.get("target_oracle_used", True)
            ),
        })
        effective_family_count.append(
            diagnostic.get("effective_family_count", np.nan))
        covariance_trace.append(
            diagnostic.get("parameter_covariance_trace", np.nan))
    check_names = tuple(checks_by_row[0]) if checks_by_row else ()
    checks = {
        name: bool(all(item[name] for item in checks_by_row))
        for name in check_names
    }
    return {
        "checks": checks,
        "passed": bool(checks and all(checks.values())),
        "mean_effective_family_count": float(np.mean(
            effective_family_count)),
        "mean_parameter_covariance_trace": float(np.mean(
            covariance_trace)),
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
    audit = _semiparametric_audit(rows)
    result["schema_version"] = 5
    result["gate_model"] = "tcb_v5_orthogonal_semiparametric_boundary"
    result["semiparametric_posterior_audit"] = audit
    result["gate_pass_before_semiparametric_audit"] = bool(
        result["gate_pass"])
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
        "semiparametric_audit": result[
            "semiparametric_posterior_audit"]["passed"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
