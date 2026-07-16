#!/usr/bin/env python3
"""Freeze source-only target warm starts for paired transfer comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import (  # noqa: E402
    FrozenTransferArchive,
    frozen_archive_from_meta_prior,
)
from core.designs import integer_design_fingerprint  # noqa: E402
from performance.benchmark_lodo_meta_prior import (  # noqa: E402
    build_scalarized_problem,
    train_meta_prior,
)
from performance.benchmark_quality import parse_weights  # noqa: E402
from performance.benchmark_sota_fairness import (  # noqa: E402
    oracle_free_lodo_config,
)
from performance.structural_ablation import (  # noqa: E402
    STRUCTURAL_PRIOR_PROFILES,
    apply_structural_prior_profile,
)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_source_designs(
    manifest,
    heldout,
    archive_path,
    output,
    *,
    dimension=None,
    source_dimension=None,
    n0=10,
    seed_start=0,
    n_seeds=20,
    structural_prior_profile="inherit",
    proposal_mode="rank_spanning",
):
    """Build designs from the same frozen, oracle-free source observations.

    Reconstructing the archive before proposing points is intentional: the
    task fails if the proposal prior and the archive consumed by a transfer
    optimizer differ by even one observed value.
    """

    target_config = oracle_free_lodo_config(manifest)
    if dimension is not None:
        target_config["d"] = int(dimension)
    archive = FrozenTransferArchive.load(archive_path)
    archive_dimension = int(archive.tasks[0].X.shape[1])
    source_dimension = (
        archive_dimension if source_dimension is None else int(source_dimension)
    )
    archive.validate(expected_dimension=source_dimension)
    source_config = dict(target_config)
    source_config["d"] = int(source_dimension)
    source_config["meta_source_dimension"] = int(source_dimension)
    apply_structural_prior_profile(source_config, structural_prior_profile)
    prior = train_meta_prior(source_config, heldout, 0, teacher=False)
    reconstructed = frozen_archive_from_meta_prior(
        prior, source_seed=int(archive.source_seed))
    if reconstructed.fingerprint != archive.fingerprint:
        raise ValueError(
            "source-informed proposal archive does not match frozen archive: "
            f"{reconstructed.fingerprint} != {archive.fingerprint}"
        )

    problem = build_scalarized_problem(
        heldout,
        int(target_config["d"]),
        int(target_config["L"]),
        float(target_config["sigma"]),
        float(target_config["alpha"]),
        parse_weights(target_config["weights"]),
    )
    proposal_mode = str(proposal_mode)
    if proposal_mode not in {"rank_spanning", "risk_coordinate_atlas"}:
        raise ValueError(f"unknown source proposal mode {proposal_mode!r}")
    designs = {}
    for offset in range(int(n_seeds)):
        seed = int(seed_start) + offset
        generator = (
            prior.dimension_equivariant_initial_candidates
            if proposal_mode == "risk_coordinate_atlas"
            else prior.initial_universal_candidates
        )
        points = generator(problem, n=int(n0), rng=np.random.default_rng(seed))
        points = [tuple(map(int, point)) for point in points]
        if len(points) != int(n0) or len(set(points)) != int(n0):
            raise RuntimeError(
                "source-informed proposal did not produce n0 unique points")
        designs[str(seed)] = {
            "points": [list(point) for point in points],
            "fingerprint": integer_design_fingerprint(points),
        }

    payload = {
        "schema_version": 1,
        "design_kind": (
            "frozen_source_informed_risk_coordinate_atlas"
            if proposal_mode == "risk_coordinate_atlas"
            else "frozen_source_informed_rank_spanning"
        ),
        "proposal_mode": proposal_mode,
        "structural_prior_profile": str(structural_prior_profile),
        "structural_prior_active_components": list(source_config.get(
            "structural_prior_active_components", [])),
        "heldout_target_domain": str(heldout),
        "dimension": int(problem.d),
        "source_dimension": int(source_dimension),
        "dimension_holdout": bool(int(source_dimension) != int(problem.d)),
        "n0": int(n0),
        "seed_start": int(seed_start),
        "n_seeds": int(n_seeds),
        "source_archive_fingerprint": archive.fingerprint,
        "source_archive_oracle_aided": False,
        "target_labels_used": False,
        "target_oracle_used": False,
        "proposal_diagnostics": dict(getattr(
            prior, "dimension_equivariant_proposal_diagnostics", {})),
        "designs": designs,
    }
    _atomic_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--source-d", type=int, default=None)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument(
        "--structural-prior-profile",
        choices=("inherit", *STRUCTURAL_PRIOR_PROFILES),
        default="inherit",
    )
    parser.add_argument(
        "--proposal-mode",
        choices=("rank_spanning", "risk_coordinate_atlas"),
        default="rank_spanning",
    )
    args = parser.parse_args()
    payload = materialize_source_designs(
        args.manifest,
        args.heldout,
        args.archive,
        args.out,
        dimension=args.d,
        source_dimension=args.source_d,
        n0=args.n0,
        seed_start=args.seed_start,
        n_seeds=args.n_seeds,
        structural_prior_profile=args.structural_prior_profile,
        proposal_mode=args.proposal_mode,
    )
    print(json.dumps({
        "status": "ok",
        "heldout": args.heldout,
        "n_designs": len(payload["designs"]),
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
