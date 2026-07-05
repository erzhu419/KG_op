"""Compare deterministic, manifold, self-supervised, and transformer encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_quality  # noqa: E402
from benchmark_quality import json_safe, parse_csv, write_csv  # noqa: E402


def _encoder_args(args, encoder_kind):
    fields = vars(args).copy()
    fields["encoder_kind"] = encoder_kind
    fields["modes"] = args.modes
    fields["sc_modes"] = args.sc_modes
    fields["out_prefix"] = (
        f"{args.out_prefix}_{encoder_kind}"
        if args.out_prefix
        else f"encoder_suite_{encoder_kind}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    return SimpleNamespace(**fields)


def run_suite(args):
    rows = []
    summary_rows = []
    encoder_results = {}
    for encoder_kind in parse_csv(args.encoder_kinds):
        print(f"[encoder-suite] encoder={encoder_kind}", flush=True)
        enc_args = _encoder_args(args, encoder_kind)
        result = benchmark_quality.run_benchmark(enc_args)
        paths = benchmark_quality.write_outputs(enc_args, result)
        encoder_results[encoder_kind] = {
            "paths": paths,
            "summary": result["summary"],
        }
        for row in result["rows"]:
            row = dict(row)
            row["encoder_kind"] = encoder_kind
            rows.append(row)
        for summary in result["summary"].values():
            flat = benchmark_quality.flatten_summary(summary)
            flat["encoder_kind"] = encoder_kind
            summary_rows.append(flat)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": vars(args),
        "encoder_results": encoder_results,
        "rows": rows,
        "summary_rows": summary_rows,
    }


def write_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"encoder_suite_{time.strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{prefix}.json"
    rows_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(rows_path, result["rows"])
    write_csv(summary_path, result["summary_rows"])
    return {
        "json": str(json_path),
        "rows_csv": str(rows_path),
        "summary_csv": str(summary_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="StatePolicyRZDT1")
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--K1", type=int, default=20)
    parser.add_argument("--K2", type=int, default=1)
    parser.add_argument("--posterior_pool_size", type=int, default=220)
    parser.add_argument("--posterior_keep", type=int, default=12)
    parser.add_argument("--axis_candidate_count", type=int, default=-1)
    parser.add_argument("--structured_candidate_count", type=int, default=0)
    parser.add_argument("--state_candidate_count", type=int, default=-1)
    parser.add_argument("--state_inverse_pool_size", type=int, default=500)
    parser.add_argument("--state_inverse_neighbors", type=int, default=2)
    parser.add_argument("--n_thr", type=int, default=5)
    parser.add_argument("--eval_pool_size", type=int, default=300)
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_mean", type=float, default=0.10)
    parser.add_argument("--lambda_coupling", type=float, default=0.05)
    parser.add_argument("--beta_g", type=float, default=2.0)
    parser.add_argument("--certification_mode", default="theory",
                        choices=["theory", "legacy"])
    parser.add_argument("--coupling_safety_z", type=float, default=0.5)
    parser.add_argument("--coupling_gate_temperature", type=float, default=0.25)
    parser.add_argument("--recommendation_safety_z", type=float, default=0.5)
    parser.add_argument("--recommendation_noise_floor_scale", type=float, default=1.0)
    parser.add_argument("--recommendation_infeasible_penalty", type=float, default=5.0)
    parser.add_argument("--disable_recommendation_calibration", action="store_true")
    parser.add_argument("--recommendation_calibration_ridge", type=float, default=1e-6)
    parser.add_argument("--disable_recommendation_axis_oracle", action="store_true")
    parser.add_argument("--use_state_basis", action="store_true")
    parser.add_argument(
        "--state_basis_mode",
        default="raw+state",
        choices=["raw", "state", "raw+state", "manifold", "raw+manifold"],
    )
    parser.add_argument(
        "--encoder_kinds",
        default=(
            "synthetic,pca_manifold,kernel_manifold,ssl_masked,"
            "ssl_contrastive,ssl_next_risk,ssl_transformer"
        ),
    )
    parser.add_argument("--encoder_latent_dim", type=int, default=8)
    parser.add_argument("--encoder_fit_pool_size", type=int, default=512)
    parser.add_argument("--exact_kg_mc_samples", type=int, default=0)
    parser.add_argument("--exact_kg_use_score", action="store_true")
    parser.add_argument("--exact_kg_blend", type=float, default=0.0)
    parser.add_argument("--acquisition_modes", default="additive")
    parser.add_argument("--modes", default="")
    parser.add_argument("--sc_modes", default="factor")
    parser.add_argument("--baseline_variant", default="factor+sc")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run_suite(args)
    paths = write_outputs(args, result)
    print(json.dumps(json_safe({
        "paths": paths,
        "summary_rows": result["summary_rows"],
    }), indent=2))


if __name__ == "__main__":
    main()
