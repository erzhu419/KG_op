"""Strict source-only gate for orthogonal semiparametric TCB-V5."""

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
    BoundaryFamilySemiparametricPosterior,
)


def _fit_semiparametric_model_from_rows(args, rows):
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
    model = BoundaryFamilySemiparametricPosterior(
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
            "rotation_mode": "none",
            "rotation_ridge": args.rotation_ridge,
            "target_residual_rank": 0,
            "residual_ridge": args.residual_ridge,
        },
        coefficient_ridge=args.coefficient_ridge,
        coefficient_prior_strength=args.coefficient_prior_strength,
        coefficient_floor=args.coefficient_floor,
        residual_feature_count=args.semiparametric_residual_features,
        residual_ridge=args.semiparametric_residual_ridge,
        residual_lengthscale_multiplier=(
            args.semiparametric_lengthscale_multiplier),
    ).fit(
        descriptors,
        margins,
        domains,
        margin_variance=variances,
        replicate_count=replicate_count,
    )
    model.boundary_descriptor_projector_ = projector
    model.gate_row_metadata_ = {
        "coefficient_ridge": float(args.coefficient_ridge),
        "coefficient_prior_strength": float(
            args.coefficient_prior_strength),
        "coefficient_floor": float(args.coefficient_floor),
        "family_count": int(model.diagnostics()["family_count"]),
        "synthesis_mode": "nonnegative_source_atom_dictionary",
        "semiparametric_residual_features": int(
            model.diagnostics()["residual_feature_count"]),
        "semiparametric_residual_ridge": float(
            args.semiparametric_residual_ridge),
        "semiparametric_lengthscale_multiplier": float(
            args.semiparametric_lengthscale_multiplier),
        "semiparametric_orthogonality_relative": float(
            model.diagnostics()["orthogonality_relative"]),
        "residual_dictionary_target_labels_used": bool(
            model.diagnostics()[
                "target_labels_used_for_residual_dictionary"]),
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
        model_builder=_fit_semiparametric_model_from_rows,
    )
    result["schema_version"] = 5
    result["gate_model"] = "tcb_v5_orthogonal_semiparametric_boundary"
    result["leakage_contract"] = {
        "outer_target_excluded_from_source_dictionary": True,
        "target_domain_label_used": False,
        "target_pilot_outcomes_update_coefficients": True,
        "target_evaluation_truth_used_after_fit_only": True,
        "source_atoms_frozen_on_target": True,
        "orthogonal_residual_dictionary_frozen_on_target": True,
        "residual_projection_uses_source_descriptors_without_target_labels": True,
    }
    return result


def build_parser():
    parser = shared.build_parser()
    parser.description = (
        "Strict source-only TCB-V5 orthogonal semiparametric gate")
    parser.add_argument("--coefficient-ridge", type=float, default=0.1)
    parser.add_argument(
        "--coefficient-prior-strength", type=float, default=0.5)
    parser.add_argument("--coefficient-floor", type=float, default=0.0)
    parser.add_argument(
        "--semiparametric-residual-features", type=int, default=2)
    parser.add_argument(
        "--semiparametric-residual-ridge", type=float, default=10.0)
    parser.add_argument(
        "--semiparametric-lengthscale-multiplier", type=float, default=1.0)
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
        f"DONE tcb_v5_semiparametric_gate rows={len(result['rows'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
