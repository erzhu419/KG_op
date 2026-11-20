#!/usr/bin/env python3
"""Derive an anonymous, redistribution-minimized supplement from a verified export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

from export_clean_replication_bundle import verify_bundle
from replication.verify_review_supplement import record

REPLICATION = Path(__file__).resolve().parent / "replication"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SC = Path("SC-OLH-KG")
MANUSCRIPT = SC / "manuscript_v2_or"
ARTIFACTS = SC / "paper_artifacts"
INGOLSTADT = Path("Final_Submission/GPR_KG_Code/experiments/ingolstadt21/data")
REVIEW_INPUTS = {
    SC / "data/external/opsd_time_series_extended_v2.npz",
    MANUSCRIPT / "informs4.cls",
    MANUSCRIPT / "informs_Logo.pdf",
    INGOLSTADT / "ingolstadt21.net.xml",
    INGOLSTADT / "ingolstadt21.rou.xml",
}
OMIT = REVIEW_INPUTS | {
    MANUSCRIPT / "main_or.pdf",
    MANUSCRIPT / "export_clean_replication_bundle.py",
    SC / "docs/gp_fta_or_remediation_20261120.md",
    SC / "docs/gp_fta_systematic_audit_20261120.md",
    SC / "docs/gp_fta_u3_historical_replay_20261120.md",
    Path("README_REPLICATION.md"),
    Path("LICENSES.md"),
}
CLUSTER_SCRIPTS = {"run_v2_lean_cluster.py", "sync_v2_cluster.py"}
PRIVATE = re.compile(r"/home/|/mnt/[a-z]/Users/|\\Users\\|zhengliang|erzhu|jtl110|202\.197\.46\.16", re.I)
PRIVATE_PATH = re.compile(r"/home/[^\s\"'|<>]+")


def _scrub(value):
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if k not in {"server_path", "attempt_directory", "base_proof", "v2_proof"}}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        return PRIVATE_PATH.sub("[execution-cluster path]", value)
    return value


def _copy_entry(source: Path, target: Path, relative: Path) -> None:
    if relative.name == "campaign_runner.py":
        text = source.read_text(encoding="utf-8")
        for name, value in (("remote-cwd", '"."'), ("remote-python", "sys.executable"), ("extra-pythonpath", '\"\"')):
            text, count = re.subn(rf'("--{name}",\n\s*default=)"/home/[^"]+"', rf"\g<1>{value}", text)
            if count != 1:
                raise RuntimeError(f"Expected one deployment default for {name}")
        text = text.replace('default=Path.home() / "mine_code/scheduleurm/skill/scheduler.py"',
                            'default=Path("scheduler.py")')
        target.write_text(text, encoding="utf-8")
    elif relative == Path("scripts/scheduler_node_policy.py"):
        text = source.read_text(encoding="utf-8")
        text = re.sub(r"FORBIDDEN_CPU_NODES = .*\n", "", text)
        text = text.replace("FORBIDDEN_CPU_NODES.intersection(nodes)", "set(nodes).difference(DEFAULT_CPU_NODES)")
        target.write_text(text, encoding="utf-8")
    elif relative.suffix == ".json" and relative.is_relative_to(ARTIFACTS) and PRIVATE.search(source.read_text(encoding="utf-8")):
        data = _scrub(json.loads(source.read_bytes()))
        if relative.name == "p1_diagnostics_summary.json":
            data["sumo_platform_calibration"]["route_file"] = "ingolstadt21.rou.xml"
        target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        shutil.copy2(source, target)
    if target.suffix in {".py", ".json", ".md", ".tex", ".bib", ".csv"} and PRIVATE.search(target.read_text(encoding="utf-8")):
        raise ValueError(f"Review supplement contains an identifying location: {relative}")


def export(source: Path, target: Path) -> dict:
    exact = verify_bundle(source)
    target.mkdir(parents=True)
    omitted_hashes = {}
    adapted = []
    for item in exact["files"]:
        relative = Path(item["relpath"])
        if relative in REVIEW_INPUTS:
            omitted_hashes[item["relpath"]] = item["sha256"]
        if relative in OMIT or relative.name in CLUSTER_SCRIPTS:
            continue
        out = target / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        original = source / relative
        if relative.is_relative_to(MANUSCRIPT) and relative.suffix in {".tex", ".bib"}:
            original = PROJECT_ROOT / relative
        _copy_entry(original, out, relative)
        if record(target, item["relpath"])["sha256"] != item["sha256"]:
            adapted.append(item["relpath"])

    for name, dest in (
        ("README_REVIEW.md", "README_REPLICATION.md"),
        ("RIGHTS_AND_PROVENANCE.md", "LICENSES.md"),
        ("U3_REVIEW_REPLAY.md", "U3_REPLAY.md"),
        ("restore_review_inputs.py", "restore_review_inputs.py"),
        ("verify_review_supplement.py", "verify_review_supplement.py"),
    ):
        _copy_entry(REPLICATION / name, target / dest, Path(dest))
    (target / "REVIEW_INPUTS.json").write_text(json.dumps({"omitted_input_sha256": omitted_hashes}, indent=2) + "\n")
    records = [record(target, path.relative_to(target).as_posix())
               for path in sorted(target.rglob("*")) if path.is_file()]
    manifest = {
        "bundle_name": "GP-FTA Operations Research anonymous review supplement",
        "experiment_execution_sha256": exact["execution_code_sha256"],
        "report_and_deployment_metadata_adapted_for_review": sorted(adapted),
        "files": records,
    }
    (target / "REPLICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing verified independent GP-FTA export")
    parser.add_argument("--export-dir", type=Path, required=True, help="New anonymous supplementary directory")
    args = parser.parse_args()
    verified = export(args.source.resolve(), args.export_dir.resolve())
    print(json.dumps({"files": len(verified["files"]), "directory": str(args.export_dir)}))


if __name__ == "__main__":
    main()
