"""Aggregate the strict nested-LODO TCB-V3 boundary-family gate."""

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


FAMILY_FIELDS = (
    "family_delta",
    "evidence_temperature",
    "family_guard_scale",
    "family_strategy",
)
CONFIG_FIELDS = tuple(
    field for field in shared.CONFIG_FIELDS if field != "pilot_policy"
) + FAMILY_FIELDS + ("pilot_policy",)
MODEL_CONFIG_FIELDS = tuple(
    field for field in CONFIG_FIELDS if field != "pilot_policy")


def _family_audit(rows):
    diagnostics = [row.get("adapter_diagnostics", {}) for row in rows]
    finite_weights = []
    mass_checks = []
    for row, diagnostic in zip(rows, diagnostics):
        weights = np.asarray(
            diagnostic.get("posterior_weights", []), dtype=float)
        finite_weights.append(bool(
            len(weights) > 0
            and np.all(np.isfinite(weights))
            and np.all(weights >= 0.0)
            and np.isclose(float(np.sum(weights)), 1.0)
        ))
        required = 1.0 - float(row["family_delta"])
        mass_checks.append(
            float(diagnostic.get("credible_family_mass", -np.inf))
            + 1e-12 >= required
        )
    checks = {
        "posterior_weights_are_simplex": bool(all(finite_weights)),
        "credible_family_mass_reaches_declared_level": bool(
            all(mass_checks)),
        "target_domain_label_unused": bool(all(
            not bool(item.get("target_label_used", True))
            for item in diagnostics
        )),
        "target_oracle_unused": bool(all(
            not bool(item.get("target_oracle_used", True))
            for item in diagnostics
        )),
        "family_parameters_frozen": bool(all(
            bool(item.get("family_parameters_frozen", False))
            for item in diagnostics
        )),
        "loo_predictive_evidence": bool(all(
            item.get("evidence_protocol")
            == "leave_one_pilot_out_generalized_bayes"
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
        "mean_credible_family_count": float(np.mean([
            item.get("credible_family_count", np.nan)
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
    audit = _family_audit(rows)
    result["schema_version"] = 3
    result["gate_model"] = "tcb_v3_boundary_family_mixture"
    result["family_posterior_audit"] = audit
    result["gate_pass_before_family_audit"] = bool(result["gate_pass"])
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
        "family_audit": result["family_posterior_audit"]["passed"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
