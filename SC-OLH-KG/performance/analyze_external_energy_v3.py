#!/usr/bin/env python3
"""Fail-closed market/region analysis for forecast-indexed energy V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_external_energy_v2 import (  # noqa: E402
    _atomic_json,
    _write_csv,
    analyze as analyze_energy,
)
from performance.benchmark_external_energy_v3 import (  # noqa: E402
    ARMS,
    CONTRACT_ID,
)


ANALYSIS_CONTRACT_ID = "opsd_forecast_indexed_region_holdout_analysis_v3"


def analyze(paths):
    controls = tuple(arm for arm in ARMS if arm != "source_atlas")
    return analyze_energy(
        paths,
        accepted_contract_ids={CONTRACT_ID},
        controls=controls,
        analysis_contract_id=ANALYSIS_CONTRACT_ID,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv")
    parser.add_argument("--expected-count", type=int, default=540)
    args = parser.parse_args()
    paths = []
    for value in args.inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(path.rglob("cell*.json"))
        elif path.is_file():
            paths.append(path)
    payload = analyze(paths)
    payload["expected_count"] = int(args.expected_count)
    payload["matrix_complete"] = bool(
        payload["status"] == "complete"
        and payload["row_count"] == int(args.expected_count)
    )
    if not payload["matrix_complete"]:
        payload["status"] = "incomplete"
    _atomic_json(args.out, payload)
    if args.csv:
        _write_csv(args.csv, payload["compact_rows"])
    print(json.dumps({
        "status": payload["status"],
        "row_count": payload["row_count"],
        "expected_count": payload["expected_count"],
        "market_count": payload["market_count"],
        "region_count": payload["region_count"],
    }, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        raise SystemExit(1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
