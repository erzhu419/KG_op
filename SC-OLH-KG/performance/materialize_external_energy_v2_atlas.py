#!/usr/bin/env python3
"""Materialize one frozen region-held-out OPSD V2 source atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_external_energy_v2 import (  # noqa: E402
    TARGET_MARKETS,
    materialize_source_atlas,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-market", choices=TARGET_MARKETS, required=True)
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--library-size", type=int, default=64)
    parser.add_argument("--source-replications", type=int, default=3)
    parser.add_argument("--family-seed", type=int, default=20260808)
    args = parser.parse_args()
    payload = materialize_source_atlas(
        data_path=args.data,
        target_market=args.target_market,
        year=args.year,
        dimension=args.d,
        alpha=args.alpha,
        n0=args.n0,
        library_size=args.library_size,
        source_replications=args.source_replications,
        family_seed=args.family_seed,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({
        "status": payload["status"],
        "out": str(args.out),
        "target_market": payload["target_market"],
        "source_markets": payload["source_markets"],
        "source_calls": payload["source_calls"],
        "initial_design_fingerprint": payload["initial_design_fingerprint"],
    }, indent=2, sort_keys=True))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
