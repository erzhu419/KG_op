#!/usr/bin/env python3
"""Check bundled OPSD and SUMO inputs through the live adapter entry points."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
SC_ROOT = ROOT / "SC-OLH-KG"
sys.path.insert(0, str(SC_ROOT))

from data.opsd import load_opsd_market
from v2_or.dispatch.campaign_runner import build_family_tasks


def verify_opsd() -> dict:
    archive = SC_ROOT / "data/external/opsd_time_series_extended_v2.npz"
    manifest = json.loads(archive.with_suffix(".npz.manifest.json").read_text(encoding="utf-8"))
    rows = {market: len(load_opsd_market(archive, market).timestamp_hour)
            for market in manifest["markets"]}
    if rows != {market: data["row_count"] for market, data in manifest["market_metadata"].items()}:
        raise RuntimeError("The compact OPSD archive does not match its market manifest")
    return {"markets": len(rows), "observed_hours": sum(rows.values())}


def verify_live_sumo() -> dict:
    path = SC_ROOT / "v2_or/manifests/v2_or_u3_live_sumo_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    family = next(family for family in manifest["families"] if family["kind"] == "traffic_live_sumo")
    task = build_family_tasks(family, "target")[0]
    backend = task._get_live_problem()
    scenario = ROOT / "Final_Submission/GPR_KG_Code/experiments/ingolstadt21/data"
    network = Path(backend._sim.NET_FILE).resolve()
    routes = Path(backend._route_files[0]).resolve()
    if network.parent != scenario or routes.parent != scenario:
        raise RuntimeError("The live SUMO adapter resolves scenario files outside the bundle")
    engine_version = backend._sim.traci.getVersion()[1]
    if (backend._sim._BACKEND, version("libsumo"), engine_version) != ("libsumo", "1.22.0", "SUMO 1.22.0"):
        raise RuntimeError("The live SUMO adapter did not load libsumo 1.22.0")
    n_tls = len(ET.parse(network).findall("tlLogic"))
    n_trips = len(ET.parse(routes).findall("trip"))
    if (task.d, backend.d, n_tls, n_trips) != (44, 44, 21, 4283):
        raise RuntimeError("The live SUMO adapter inputs differ from the archived scenario")
    return {"backend": "libsumo", "engine_version": engine_version,
            "dimensions": backend.d, "signals": n_tls, "trips": n_trips}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-sumo", action="store_true", help="Also import and check the pinned libsumo adapter.")
    args = parser.parse_args()
    inputs = {"opsd": verify_opsd()}
    if args.live_sumo:
        inputs["sumo"] = verify_live_sumo()
    print(json.dumps(inputs, indent=2))
