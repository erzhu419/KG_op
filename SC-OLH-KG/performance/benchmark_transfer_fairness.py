#!/usr/bin/env python3
"""Run one transfer-CBO row under an immutable source/target contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import FrozenTransferArchive  # noqa: E402
from baselines.transfer_bo_adapters import (  # noqa: E402
    TRANSFER_METHODS,
    TransferBOConfig,
    TransferConstrainedBO,
)
from core.designs import (  # noqa: E402
    load_frozen_source_informed_design,
)
from performance.benchmark_lodo_meta_prior import build_scalarized_problem  # noqa: E402
from performance.benchmark_quality import json_safe, parse_weights  # noqa: E402


FORMAL_SOURCE_STEPS = {
    "safe_fpacoh_cbo": 10_000,
    "fsbo_cbo": 50_000,
    "malibo_cbo": 2_048,
    "metabo_cbo": 10_000,
    "hyperbo_cbo": 10_000,
    "rgpe_cbo": 1,
    "stacked_transfer_gp_cbo": 1,
    "mtgp_cbo": 1,
}


def load_source_initial_design(path, *, archive, heldout, seed, n0, dimension):
    if not path:
        raise ValueError(
            "source_informed comparison requires --initial-design-file")
    points, contract = load_frozen_source_informed_design(
        path,
        heldout=heldout,
        seed=seed,
        n0=n0,
        dimension=dimension,
    )
    if contract["source_archive_fingerprint"] != archive.fingerprint:
        raise ValueError("source-informed design archive fingerprint mismatch")
    return points, contract["fingerprint"]


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_one(args):
    archive = FrozenTransferArchive.load(args.archive)
    problem = build_scalarized_problem(
        args.heldout,
        args.d,
        args.L,
        args.sigma,
        args.alpha,
        parse_weights(args.weights),
    )
    source_domains = [
        name for name in args.domains.split(",")
        if name.strip() and name.strip() != args.heldout
    ]
    archive.validate(
        expected_domains=map(str.strip, source_domains),
        expected_dimension=args.d,
    )
    if args.require_source_domains > 0 and len(archive.tasks) != int(
        args.require_source_domains
    ):
        raise ValueError("formal LODO row has the wrong number of source domains")
    initial_points = None
    initial_design_fingerprint = None
    if args.initial_design == "source_informed":
        initial_points, initial_design_fingerprint = load_source_initial_design(
            args.initial_design_file,
            archive=archive,
            heldout=args.heldout,
            seed=args.seed,
            n0=args.n0,
            dimension=problem.d,
        )
    source_steps = int(args.source_train_steps)
    if source_steps <= 0:
        source_steps = FORMAL_SOURCE_STEPS[args.method]
    checkpoint = (
        Path(args.checkpoint_dir)
        / f"seed{int(args.seed):04d}.pkl"
    )
    config = TransferBOConfig(
        method=args.method,
        N=args.N,
        n0=args.n0,
        seed=args.seed,
        candidate_pool_size=args.candidate_pool_size,
        beta_g=args.beta_g,
        beta_risk=args.beta_risk,
        initial_design=args.initial_design,
        initial_points=initial_points,
        implementation=args.implementation,
        source_train_steps=source_steps,
        target_finetune_steps=args.target_finetune_steps,
        checkpoint_path=str(checkpoint),
        checkpoint_resume=True,
        progress_logging=True,
        progress_label=(
            f"transfer:{args.implementation}:{args.heldout}:"
            f"{args.method}:seed={int(args.seed)}"
        ),
    )
    started = time.time()
    payload = {
        "schema_version": 1,
        "status": "running",
        "method": args.method,
        "implementation": args.implementation,
        "heldout_target_domain": args.heldout,
        "source_domains": list(archive.source_domains),
        "seed": int(args.seed),
        "comparison_contract": {
            "split": "leave_one_domain_out",
            "n_source_domains": int(len(archive.tasks)),
            "source_archive_fingerprint": archive.fingerprint,
            "source_profiles_per_domain": archive.profiles_per_domain,
            "source_simulator_calls": int(archive.simulator_calls),
            "target_dimension": int(args.d),
            "target_initial_calls_n0": int(args.n0),
            "target_total_calls_N": int(args.N),
            "total_source_plus_target_calls": int(
                archive.simulator_calls + args.N),
            "source_archive_identical_across_methods": True,
            "source_archive_identical_across_target_seeds": True,
            "target_initial_design": args.initial_design,
            "common_target_initial_design": (
                "frozen_source_informed_rank_spanning"
                if args.initial_design == "source_informed"
                else "scrambled_sobol"
            ),
            "source_informed_initial_proposal": bool(
                args.initial_design == "source_informed"),
            "source_informed_initial_fingerprint": (
                initial_design_fingerprint),
            "source_oracle_aided": False,
            "target_oracle_used_for_selection": False,
            "source_training_schedule": int(source_steps),
            "target_finetune_steps_per_refit": int(
                args.target_finetune_steps),
        },
    }
    try:
        result = TransferConstrainedBO(problem, archive, config).run()
        payload["comparison_contract"][
            "target_initial_design_fingerprint"
        ] = result["target_information_contract"][
            "initial_design_fingerprint"
        ]
        if (
            initial_design_fingerprint is not None
            and payload["comparison_contract"][
                "target_initial_design_fingerprint"
            ] != initial_design_fingerprint
        ):
            raise RuntimeError(
                "optimizer did not consume the frozen initial design exactly")
        payload.update({
            "status": "ok",
            "result": result,
            "wall_time_sec": float(time.time() - started),
        })
    except Exception as exc:
        payload.update({
            "status": (
                "failed_official_runtime"
                if args.implementation == "official" else "failed"
            ),
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "wall_time_sec": float(time.time() - started),
            "checkpoint_path": str(checkpoint),
        })
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=TRANSFER_METHODS, required=True)
    parser.add_argument(
        "--implementation", choices=("official", "paper_core"),
        default="official",
    )
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--domains", default=(
        "FactorShockStatePolicyRZDT1,InventorySupplyChain,QueueResourceControl"
    ))
    parser.add_argument("--archive", required=True)
    parser.add_argument(
        "--initial-design",
        choices=("common_sobol", "source_informed"),
        default="common_sobol",
    )
    parser.add_argument("--initial-design-file", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--d", type=int, default=50)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--require-source-domains", type=int, default=2)
    parser.add_argument("--candidate-pool-size", type=int, default=1024)
    parser.add_argument("--beta-g", type=float, default=2.0)
    parser.add_argument("--beta-risk", type=float, default=2.0)
    parser.add_argument("--source-train-steps", type=int, default=0)
    parser.add_argument("--target-finetune-steps", type=int, default=100)
    args = parser.parse_args()
    payload = run_one(args)
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "method": args.method,
        "implementation": args.implementation,
        "heldout": args.heldout,
        "seed": args.seed,
        "out": args.out,
    }, indent=2))
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
