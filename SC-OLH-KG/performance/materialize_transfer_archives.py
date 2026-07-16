#!/usr/bin/env python3
"""Materialize one immutable source archive for each LODO split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import frozen_archive_from_meta_prior  # noqa: E402
from performance.benchmark_sota_fairness import oracle_free_lodo_config  # noqa: E402
from performance.benchmark_lodo_meta_prior import train_meta_prior  # noqa: E402
from performance.benchmark_quality import parse_csv  # noqa: E402


def materialize_one(
    manifest,
    heldout,
    output,
    *,
    dimension=None,
    overwrite=False,
):
    output = Path(output)
    config = oracle_free_lodo_config(manifest)
    if dimension is not None:
        config["d"] = int(dimension)
    if output.is_file() and not overwrite:
        archive = frozen_archive_from_path(output)
    else:
        prior = train_meta_prior(config, heldout, 0, teacher=False)
        archive = frozen_archive_from_meta_prior(prior, source_seed=0)
        archive.save(output)
    expected = [
        name for name in parse_csv(config["domains"]) if name != heldout
    ]
    archive.validate(
        expected_domains=expected,
        expected_dimension=int(config["d"]),
    )
    return {
        "heldout": heldout,
        "path": str(output),
        **archive.information_contract(),
    }


def frozen_archive_from_path(path):
    from baselines.transfer_archive import FrozenTransferArchive
    return FrozenTransferArchive.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--heldouts",
        default=(
            "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
            "QueueResourceControl"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "archives" / "transfer_fair_v1"),
    )
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = []
    for heldout in parse_csv(args.heldouts):
        rows.append(materialize_one(
            args.manifest,
            heldout,
            Path(args.out_dir) / f"heldout_{heldout}.json",
            dimension=args.d,
            overwrite=args.overwrite,
        ))
    print(json.dumps({"archives": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
