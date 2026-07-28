"""Profile one read-only BoTorch candidate step from an existing checkpoint."""

from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pickle
import resource
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.botorch_adapters import (  # noqa: E402
    BoTorchBaseline,
    BoTorchBaselineConfig,
    botorch_runtime_fingerprint,
)
from performance.benchmark_lodo_meta_prior import (  # noqa: E402
    build_scalarized_problem,
)
from performance.benchmark_quality import json_safe, parse_weights  # noqa: E402


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_size(path):
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return len(payload.get("history", [])), int(payload.get("schema_version", 0))


def run(args):
    history_size, schema_version = _checkpoint_size(args.checkpoint)
    if history_size < args.n0:
        raise ValueError(
            f"checkpoint has {history_size} rows, fewer than n0={args.n0}"
        )
    problem = build_scalarized_problem(
        args.heldout,
        args.d,
        args.L,
        args.sigma,
        args.alpha,
        parse_weights(args.weights),
    )
    config = BoTorchBaselineConfig(
        N=history_size + 1,
        n0=args.n0,
        seed=args.seed,
        method="botorch_saasbo",
        raw_samples=args.raw_samples,
        num_restarts=args.num_restarts,
        maxiter=args.maxiter,
        timeout_sec=args.timeout_sec,
        saas_warmup_steps=args.saas_warmup_steps,
        saas_num_samples=args.saas_num_samples,
        saas_thinning=args.saas_thinning,
        saas_max_tree_depth=args.saas_max_tree_depth,
        saas_mc_samples=args.saas_mc_samples,
        strict_failures=True,
        checkpoint_path=str(args.checkpoint),
        checkpoint_resume=True,
        torch_device=args.torch_device,
        saas_parallel_models=bool(args.saas_parallel_models),
        saas_parallel_min_total_steps=int(
            args.saas_parallel_min_total_steps),
        saas_parallel_threads_per_model=int(
            args.saas_parallel_threads_per_model),
    )
    baseline = BoTorchBaseline(problem, config)
    started = time.perf_counter()
    if args.profile_out:
        profiler = cProfile.Profile()
        profiler.enable()
        candidate = baseline._next_candidate()
        profiler.disable()
    else:
        candidate = baseline._next_candidate()
    elapsed = time.perf_counter() - started
    if args.profile_out:
        profile_path = Path(args.profile_out)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(profile_path)
    return {
        "status": "ok",
        "heldout": args.heldout,
        "seed": int(args.seed),
        "checkpoint": str(args.checkpoint),
        "checkpoint_schema_version": int(schema_version),
        "history_size": int(history_size),
        "candidate": list(map(int, candidate)),
        "candidate_time_sec": float(elapsed),
        "max_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
        "runtime_fingerprint": botorch_runtime_fingerprint(baseline._torch_device),
        "stochastic_stage_seed": int(
            baseline._stage_seed("saas_nuts:objective")
        ),
        "saas_parallel_models": bool(baseline._use_parallel_saas_models()),
        "saas_parallel_fit_count": int(baseline._saas_parallel_fit_count),
        "saas_parallel_failures": int(baseline._saas_parallel_failures),
        "saas_parallel_last_error": str(
            baseline._saas_parallel_last_error),
        "saas_parallel_threads_per_model": int(
            baseline._saas_parallel_threads()),
        "profile_out": str(args.profile_out or ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile-out", default="")
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--timeout-sec", type=float, default=7200.0)
    parser.add_argument("--saas-warmup-steps", type=int, default=256)
    parser.add_argument("--saas-num-samples", type=int, default=128)
    parser.add_argument("--saas-thinning", type=int, default=16)
    parser.add_argument("--saas-max-tree-depth", type=int, default=6)
    parser.add_argument("--saas-mc-samples", type=int, default=256)
    parser.add_argument(
        "--saas-parallel-models",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--saas-parallel-min-total-steps", type=int, default=64)
    parser.add_argument("--saas-parallel-threads-per-model", type=int, default=0)
    args = parser.parse_args()
    try:
        payload = run(args)
    except Exception as exc:
        payload = {
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(args.out, payload)
        raise
    _atomic_json(args.out, payload)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
