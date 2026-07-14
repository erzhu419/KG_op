"""Strict source-only LODO gate for the TCB-V3 boundary-family posterior.

The data, pilot policies, evaluation truth boundary, and leakage audit are the
same as TCB-V2.  Only the source model changes: a frozen finite family library
is updated by leave-one-pilot-out generalized Bayes evidence, while its
credible-family envelope supplies the certificate.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance import benchmark_tcb_v2_source_gate as shared  # noqa: E402
from performance.benchmark_quality import json_safe  # noqa: E402
from representation.transferable_boundary import (  # noqa: E402
    BoundaryFamilyMixturePosterior,
)


def _fit_family_model_from_rows(args, rows):
    raw_descriptors = np.vstack([row["descriptor"] for row in rows])
    provider_descriptors = [
        descriptor
        for row in rows
        for descriptor in row["provider_risk_descriptor"]
    ]
    provider_coordinates = [
        coordinate
        for row in rows
        for coordinate in row["provider_risk_coordinate"]
    ]
    projector = shared._BoundaryDescriptorProjector(args).fit(raw_descriptors)
    descriptors = projector.transform(
        raw_descriptors, provider_descriptors, provider_coordinates)
    margins = np.concatenate([row["margin"] for row in rows])
    variances = np.concatenate([row["variance"] for row in rows])
    replicate_count = np.concatenate([
        row["replicate_count"] for row in rows])
    domains = np.concatenate([row["domain"] for row in rows])
    model = BoundaryFamilyMixturePosterior(
        base_model_kwargs={
            "coordinate": args.coordinate,
            "geometry": args.geometry,
            "rank": args.rank,
            "ridge": args.ridge,
            "domain_penalty": args.domain_penalty,
            "boundary_temperature": args.boundary_temperature,
            "adaptation_ridge": args.adaptation_ridge,
            "upper_alpha": args.upper_alpha,
            "calibration_prior_df": args.calibration_prior_df,
            "hierarchy_iterations": args.hierarchy_iterations,
            "effect_ridge": args.effect_ridge,
            "rotation_mode": args.rotation_mode,
            "rotation_ridge": args.rotation_ridge,
            "target_residual_rank": args.target_residual_rank,
            "residual_ridge": args.residual_ridge,
        },
        family_delta=args.family_delta,
        evidence_temperature=args.evidence_temperature,
        family_guard_scale=args.family_guard_scale,
        family_strategy=args.family_strategy,
    ).fit(
        descriptors,
        margins,
        domains,
        margin_variance=variances,
        replicate_count=replicate_count,
    )
    model.boundary_descriptor_projector_ = projector
    model.gate_row_metadata_ = {
        "family_delta": float(args.family_delta),
        "evidence_temperature": float(args.evidence_temperature),
        "family_guard_scale": float(args.family_guard_scale),
        "family_strategy": str(model.diagnostics()["family_strategy"]),
        "family_count": int(model.diagnostics()["family_count"]),
    }
    model.diagnostics_.update({
        "descriptor_mode": projector.mode,
        "descriptor_dimension": int(descriptors.shape[1]),
        "provider_structural_input": bool("provider_" in projector.mode),
    })
    return model


def run_gate(args, heldout, target_seed):
    result = shared.run_gate(
        args,
        heldout,
        target_seed,
        model_builder=_fit_family_model_from_rows,
    )
    result["schema_version"] = 3
    result["gate_model"] = "tcb_v3_boundary_family_mixture"
    result["leakage_contract"] = {
        "outer_target_excluded_from_source_library": True,
        "target_domain_label_used": False,
        "target_pilot_outcomes_used_for_family_posterior": True,
        "target_evaluation_truth_used_after_fit_only": True,
        "family_parameters_frozen_on_target": True,
    }
    return result


def build_parser():
    parser = shared.build_parser()
    parser.description = (
        "Strict source-only TCB-V3 boundary-family mixture gate")
    parser.add_argument("--family-delta", type=float, default=0.10)
    parser.add_argument("--evidence-temperature", type=float, default=0.50)
    parser.add_argument("--family-guard-scale", type=float, default=0.0)
    parser.add_argument(
        "--family-strategy",
        choices=(
            "pooled_plus_leave_one_source_domain_out",
            "source_domain_atoms",
            "pooled_plus_source_domain_atoms",
        ),
        default="pooled_plus_leave_one_source_domain_out",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.domains = tuple(shared.parse_csv(args.domains))
    args.pilot_policies = tuple(shared.parse_csv(args.pilot_policies))
    if args.heldout not in args.domains:
        raise ValueError("heldout domain must be present in --domains")
    result = run_gate(args, args.heldout, args.target_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(
        f"DONE tcb_v3_family_gate rows={len(result['rows'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
