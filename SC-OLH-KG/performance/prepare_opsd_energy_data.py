#!/usr/bin/env python3
"""Stream and compact the pinned OPSD data package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.opsd import (  # noqa: E402
    DEFAULT_MARKETS,
    DEFAULT_YEARS,
    OPSD_DATA_URL,
    preprocess_opsd,
)


def _csv(value):
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=OPSD_DATA_URL,
        help="Pinned URL or a local copy of the official CSV.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    parser.add_argument(
        "--years", default=",".join(map(str, DEFAULT_YEARS)))
    parser.add_argument("--interpolation-limit", type=int, default=6)
    args = parser.parse_args()
    metadata = preprocess_opsd(
        args.out,
        source=args.source,
        markets=_csv(args.markets),
        years=tuple(map(int, _csv(args.years))),
        interpolation_limit=args.interpolation_limit,
    )
    print(json.dumps({
        "status": "ok",
        "output": metadata["output"],
        "source": metadata["source"],
        "markets": metadata["markets"],
        "years": metadata["years"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
