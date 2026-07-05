"""Benchmark optional exact posterior-update KG against the additive proxy."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_quality  # noqa: E402
from benchmark_quality import (  # noqa: E402
    compare_to_baseline,
    flatten_summary,
    json_safe,
    parse_csv,
    printable_summary,
    summarize_variant,
    write_csv,
)


def _safe_name(value):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))


def _method_args(args, method):
    values = dict(vars(args))
    values["exact_method"] = method
    values["acquisition_mode"] = "additive"
    values["exact_kg_mc_samples"] = 0
    values["exact_kg_use_score"] = False
    values["exact_kg_blend"] = 0.0
    if method == "additive":
        pass
    elif method == "exact":
        values["acquisition_mode"] = "exact_mc"
        values["exact_kg_mc_samples"] = int(args.exact_mc_samples)
        values["exact_kg_use_score"] = True
    elif method.startswith("blend"):
        blend = float(method.removeprefix("blend"))
        values["acquisition_mode"] = "blend"
        values["exact_kg_mc_samples"] = int(args.exact_mc_samples)
        values["exact_kg_blend"] = blend
    else:
        raise ValueError(f"Unknown exact-KG method: {method}")
    return SimpleNamespace(**values)


def _run_one(payload):
    args_dict, method, seed = payload
    args = SimpleNamespace(**args_dict)
    run_args = _method_args(args, method)
    row = benchmark_quality.run_variant_once(
        run_args,
        args.variance_mode,
        seed,
        args.use_state_coupling,
        run_args.acquisition_mode,
    )
    suffix = "+sc" if args.use_state_coupling else ""
    row["variant"] = f"{method}{suffix}"
    row["exact_method"] = method
    return row


def run_benchmark(args):
    seeds = parse_csv(args.seeds, int)
    if not seeds:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    methods = parse_csv(args.methods)
    payloads = [
        (vars(args), method, seed)
        for method in methods
        for seed in seeds
    ]
    rows = []
    if int(args.jobs) > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as pool:
            futures = {pool.submit(_run_one, payload): payload for payload in payloads}
            for future in as_completed(futures):
                _, method, seed = futures[future]
                print(f"[exact-kg] done method={method} seed={seed}", flush=True)
                rows.append(future.result())
    else:
        for payload in payloads:
            _, method, seed = payload
            print(f"[exact-kg] method={method} seed={seed}", flush=True)
            rows.append(_run_one(payload))

    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    summaries = {
        variant: summarize_variant(variant_rows)
        for variant, variant_rows in grouped.items()
    }
    compare_to_baseline(summaries, args.baseline_variant)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            key: json_safe(value)
            for key, value in vars(args).items()
        },
        "rows": rows,
        "summary": summaries,
    }


def write_outputs(args, result):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix
    if not prefix:
        prefix = (
            f"exact_kg_{_safe_name(args.problem)}_"
            f"{_safe_name(args.variance_mode)}_n{int(args.N)}_s{int(args.n_seeds)}"
        )
    json_path = out_dir / f"{prefix}.json"
    rows_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    write_csv(rows_path, result["rows"])
    write_csv(summary_path, [
        flatten_summary(summary)
        for summary in result["summary"].values()
    ])
    return {
        "json": str(json_path),
        "rows_csv": str(rows_path),
        "summary_csv": str(summary_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="RegimeRZDT1")
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=24)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--K1", type=int, default=18)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--posterior_pool_size", type=int, default=220)
    parser.add_argument("--posterior_keep", type=int, default=12)
    parser.add_argument("--axis_candidate_count", type=int, default=-1)
    parser.add_argument("--structured_candidate_count", type=int, default=0)
    parser.add_argument("--state_candidate_count", type=int, default=-1)
    parser.add_argument("--state_inverse_pool_size", type=int, default=400)
    parser.add_argument("--state_inverse_neighbors", type=int, default=2)
    parser.add_argument("--n_thr", type=int, default=5)
    parser.add_argument("--eval_pool_size", type=int, default=300)
    parser.add_argument("--variance_mode", default="orthogonal")
    parser.add_argument("--lambda_feas", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.25)
    parser.add_argument("--lambda_mean", type=float, default=0.10)
    parser.add_argument("--lambda_coupling", type=float, default=0.05)
    parser.add_argument("--coupling_safety_z", type=float, default=0.5)
    parser.add_argument("--coupling_gate_temperature", type=float, default=0.25)
    parser.add_argument("--recommendation_safety_z", type=float, default=0.5)
    parser.add_argument("--recommendation_noise_floor_scale", type=float, default=1.0)
    parser.add_argument("--recommendation_infeasible_penalty", type=float, default=5.0)
    parser.add_argument("--disable_recommendation_calibration", action="store_true")
    parser.add_argument("--recommendation_calibration_ridge", type=float, default=1e-6)
    parser.add_argument("--disable_recommendation_axis_oracle", action="store_true")
    parser.add_argument("--use_state_coupling", action="store_true")
    parser.add_argument("--use_state_basis", action="store_true")
    parser.add_argument(
        "--state_basis_mode",
        default="raw+state",
        choices=["raw", "state", "raw+state", "manifold", "raw+manifold"],
    )
    parser.add_argument(
        "--encoder_kind",
        default="synthetic",
        choices=[
            "synthetic",
            "self_supervised",
            "masked",
            "contrastive",
            "transformer",
            "pca_manifold",
            "kernel_manifold",
            "ssl_masked",
            "ssl_contrastive",
            "ssl_next_risk",
            "ssl_transformer",
        ],
    )
    parser.add_argument("--encoder_latent_dim", type=int, default=8)
    parser.add_argument("--encoder_fit_pool_size", type=int, default=512)
    parser.add_argument("--methods", default="additive,blend0.25,blend0.5,exact")
    parser.add_argument("--exact_mc_samples", type=int, default=4)
    parser.add_argument("--baseline_variant", default="additive")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out_dir", default=str(ROOT / "profiles"))
    parser.add_argument("--out_prefix", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.N <= args.n0:
        raise ValueError("--N must be larger than --n0")
    if int(args.exact_mc_samples) <= 0:
        raise ValueError("--exact_mc_samples must be positive")

    result = run_benchmark(args)
    paths = write_outputs(args, result)
    print(json.dumps(json_safe({
        "paths": paths,
        "summary": printable_summary(result),
    }), indent=2))


if __name__ == "__main__":
    main()
