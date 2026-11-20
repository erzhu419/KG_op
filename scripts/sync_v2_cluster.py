#!/usr/bin/env python3
"""Sync isolated `v2_or` code, OPSD data, and Ingolstadt21 assets to cluster NFS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SC_ROOT = REPO_ROOT / "SC-OLH-KG"
STAGE_DIR = REPO_ROOT.parent / "KG_op_v2_or_stage"
REMOTE_ROOT = "/home/zhengliang01/scheduleurm_work/KG_op_v2_or_cluster"
TRANSPORT = REPO_ROOT.parent / "scheduleurm/skill/transport_exec.py"


def collect_files() -> list[str]:
    rels: set[str] = {
        "AGENTS.md",
        "scripts/scheduler_node_policy.py",
        "scripts/run_v2_lean_cluster.py",
        "SC-OLH-KG/data/__init__.py",
        "SC-OLH-KG/data/opsd.py",
        "SC-OLH-KG/data/external/opsd_time_series_extended_v2.npz",
        "SC-OLH-KG/data/external/opsd_time_series_extended_v2.npz.manifest.json",
        "SC-OLH-KG/data/ingolstadt21/__init__.py",
        "SC-OLH-KG/data/ingolstadt21/config.py",
        "SC-OLH-KG/data/ingolstadt21/ingolstadt21_problem.py",
        "SC-OLH-KG/data/ingolstadt21/sumo_sim.py",
        "SC-OLH-KG/data/ingolstadt21/decision_space.json",
        "SC-OLH-KG/data/ingolstadt21/baseline.json",
        "SC-OLH-KG/data/ingolstadt21/data/ingolstadt21.net.xml",
        "SC-OLH-KG/data/ingolstadt21/data/ingolstadt21.rou.xml",
        "Final_Submission/GPR_KG_Code/results/ingolstadt21/decision_space.json",
        "Final_Submission/GPR_KG_Code/results/ingolstadt21/baseline.json",
    }
    for py_or_json in (SC_ROOT / "v2_or").rglob("*"):
        if py_or_json.is_file() and "__pycache__" not in py_or_json.parts and py_or_json.suffix in {".py", ".json"}:
            rels.add(str(py_or_json.relative_to(REPO_ROOT)))

    for lean_p in (REPO_ROOT / "proof/SCOLHKG/V2OR").rglob("*.lean"):
        rels.add(str(lean_p.relative_to(REPO_ROOT)))
    v2_iface = REPO_ROOT / "proof/SCOLHKG/V2ORProofInterface.lean"
    if v2_iface.is_file():
        rels.add(str(v2_iface.relative_to(REPO_ROOT)))

    ing_dir = REPO_ROOT / "Final_Submission/GPR_KG_Code/experiments/ingolstadt21"
    if ing_dir.is_dir():
        for p in ing_dir.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                rels.add(str(p.relative_to(REPO_ROOT)))
    for init_p in (
        REPO_ROOT / "Final_Submission/GPR_KG_Code/experiments/__init__.py",
    ):
        if init_p.is_file():
            rels.add(str(init_p.relative_to(REPO_ROOT)))

    out = sorted(rels)
    for r in out:
        if not (REPO_ROOT / r).is_file():
            raise FileNotFoundError(REPO_ROOT / r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = collect_files()
    total_bytes = sum((REPO_ROOT / r).stat().st_size for r in files)
    print(json.dumps({
        "files": len(files),
        "bytes": total_bytes,
        "remote_root": REMOTE_ROOT,
        "download_bytes": 0,
    }), flush=True)
    if args.dry_run:
        return

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    listing = "\n".join(files) + "\n"
    subprocess.run(
        ["rsync", "-a", "--files-from=-", f"{REPO_ROOT}/", f"{STAGE_DIR}/"],
        input=listing,
        text=True,
        check=True,
    )
    transport = [sys.executable, str(TRANSPORT)]
    ssh = ["ssh", "-o", "ConnectTimeout=30", "-o", "BatchMode=yes", "-J", "jtl110gpu"]
    remote = "zhengliang01@202.197.46.16"
    subprocess.run(
        transport + ["--", "ssh", "-n"] + ssh[1:] + [remote, shlex.join(["mkdir", "-p", f"{REMOTE_ROOT}/SC-OLH-KG"])],
        check=True,
    )
    subprocess.run(
        transport + ["--bulk", "--", "rsync", "-a", "--files-from=-", "-e", shlex.join(ssh), f"{STAGE_DIR}/", f"{remote}:{REMOTE_ROOT}/"],
        input=listing,
        text=True,
        check=True,
    )
    print("v2_or upload complete; zero historical files modified; zero remote results downloaded.", flush=True)


if __name__ == "__main__":
    main()
