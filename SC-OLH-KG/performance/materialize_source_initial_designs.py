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
    n0=10,
    seed_start=0,
    n_seeds=20,
):
    """Build designs from the same frozen, oracle-free source observations.

    Reconstructing the archive before proposing points is intentional: the
    task fails if the proposal prior and the archive consumed by a transfer
    optimizer differ by even one observed value.
    """

    config = oracle_free_lodo_config(manifest)
    if dimension is not None:
        config["d"] = int(dimension)
    archive = FrozenTransferArchive.load(archive_path)
    archive.validate(expected_dimension=int(config["d"]))
    prior = train_meta_prior(config, heldout, 0, teacher=False)
    reconstructed = frozen_archive_from_meta_prior(
        prior, source_seed=int(archive.source_seed))
    if reconstructed.fingerprint != archive.fingerprint:
        raise ValueError(
            "source-informed proposal archive does not match frozen archive: "
            f"{reconstructed.fingerprint} != {archive.fingerprint}"
        )

    problem = build_scalarized_problem(
        heldout,
        int(config["d"]),
        int(config["L"]),
        float(config["sigma"]),
        float(config["alpha"]),
        parse_weights(config["weights"]),
    )
    designs = {}
    for offset in range(int(n_seeds)):
        seed = int(seed_start) + offset
        points = prior.initial_universal_candidates(
            problem,
            n=int(n0),
            rng=np.random.default_rng(seed),
        )
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
        "design_kind": "frozen_source_informed_rank_spanning",
        "heldout_target_domain": str(heldout),
        "dimension": int(problem.d),
        "n0": int(n0),
        "seed_start": int(seed_start),
        "n_seeds": int(n_seeds),
        "source_archive_fingerprint": archive.fingerprint,
        "source_archive_oracle_aided": False,
        "target_labels_used": False,
        "target_oracle_used": False,
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
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()
    payload = materialize_source_designs(
        args.manifest,
        args.heldout,
        args.archive,
        args.out,
        dimension=args.d,
        n0=args.n0,
        seed_start=args.seed_start,
        n_seeds=args.n_seeds,
    )
    print(json.dumps({
        "status": "ok",
        "heldout": args.heldout,
        "n_designs": len(payload["designs"]),
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
