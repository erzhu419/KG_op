#!/usr/bin/env python3
"""Derive an anonymous, redistribution-minimized supplement from a verified export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

from export_clean_replication_bundle import verify_bundle
from replication.restore_review_inputs import compute_npz_arrays_sha256
from replication.verify_review_supplement import compute_v2_python_sha256, record

REPLICATION = Path(__file__).resolve().parent / "replication"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SC = Path("SC-OLH-KG")
MANUSCRIPT = SC / "manuscript_v2_or"
ARTIFACTS = SC / "paper_artifacts"
INGOLSTADT = SC / "data/ingolstadt21/data"
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
    MANUSCRIPT / "export_review_supplement.py",
    Path("README_REPLICATION.md"),
    Path("LICENSES.md"),
}
CLUSTER_SCRIPTS = {"run_v2_lean_cluster.py", "sync_v2_cluster.py"}
FROZEN_ROOTS = ("frozen_u3", "frozen_independent", "frozen_referee")
STRICT_LEAK_RE = re.compile(
    r"/home/|/mnt/[a-z]/Users/|\\Users\\|zhengliang|erzhu|jtl110|202\.197\.46\.16|"
    r"scheduleurm|cfcmt|InTAS|GPR[_-]KG|GPT[0-9]|KG_op|[\u4e00-\u9fff]",
    re.I,
)
PRIVATE_PATH = re.compile(r"/home/[^\s\"'|<>]+")


def _scrub(value):
    if isinstance(value, dict):
        return {
            k: _scrub(v)
            for k, v in value.items()
            if k not in {"server_path", "attempt_directory", "base_proof", "v2_proof"}
        }
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        out = PRIVATE_PATH.sub("[execution-cluster path]", value)
        out = out.replace("scheduleurm.run_on", "cluster_scheduler.run_on")
        out = out.replace("cfcmt-sumo122-py310-v1", "sumo122-py310-v1")
        return out
    return value


def _scrub_historical_frozen_python(text: str, filename: str) -> str:
    if filename == "campaign_runner.py":
        for name, value in (
            ("remote-cwd", '"."'),
            ("remote-python", "sys.executable"),
            ("extra-pythonpath", '""'),
        ):
            text, count = re.subn(rf'("--{name}",\n\s*default=)"/home/[^"]+"', rf"\g<1>{value}", text)
            if count != 1:
                raise RuntimeError(f"Expected one deployment default for {name}")
        text = text.replace(
            'default=Path.home() / "mine_code/scheduleurm/skill/scheduler.py"',
            'default=Path("scheduler.py")',
        )
        text = text.replace("KG_op_v2/", "gp_fta_v2/")
        text = text.replace('"KG_op"', '"gp_fta"')
    elif filename == "traffic_signal_des.py":
        text = text.replace(
            "Final_Submission/GPR_KG_Code/experiments/ingolstadt21",
            "SC-OLH-KG/data/ingolstadt21",
        )
        text = text.replace("cfcmt-sumo122-py310-v1", "sumo122-py310-v1")
        text = re.sub(
            r'Path\(__file__\)\.resolve\(\)\.parents\[3\]\s*/\s*"Final_Submission"\s*/\s*"GPR_KG_Code"\s*/\s*"results"\s*/\s*"ingolstadt21"',
            'Path(__file__).resolve().parents[2] / "data" / "ingolstadt21"',
            text,
        )
        text = re.sub(
            r'Path\(__file__\)\.resolve\(\)\.parents\[3\]\s*/\s*"Final_Submission"\s*/\s*"GPR_KG_Code"',
            'Path(__file__).resolve().parents[2] / "data"',
            text,
        )
    return text


def _copy_entry(source: Path, target: Path, relative: Path) -> None:
    is_frozen = bool(relative.parts and relative.parts[0] in FROZEN_ROOTS)
    if is_frozen and relative.name in {"campaign_runner.py", "traffic_signal_des.py"}:
        text = _scrub_historical_frozen_python(source.read_text(encoding="utf-8"), relative.name)
        target.write_text(text, encoding="utf-8")
    elif relative.suffix == ".json" and relative.is_relative_to(ARTIFACTS) and STRICT_LEAK_RE.search(source.read_text(encoding="utf-8")):
        data = _scrub(json.loads(source.read_bytes()))
        if relative.name == "p1_diagnostics_summary.json":
            data["sumo_platform_calibration"]["route_file"] = "ingolstadt21.rou.xml"
        target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        shutil.copy2(source, target)

    if target.suffix in {".py", ".patch", ".json", ".md", ".tex", ".bib", ".csv", ".txt", ".lean", ".sty", ".bst"}:
        content = target.read_text(encoding="utf-8")
        match = STRICT_LEAK_RE.search(content)
        if match:
            raise ValueError(f"Review supplement contains forbidden token {match.group(0)!r} in {relative}")


def export(source: Path, target: Path) -> dict:
    exact = verify_bundle(source)
    target.mkdir(parents=True)
    omitted_hashes = {}
    adapted = []
    for item in exact["files"]:
        relative = Path(item["relpath"])
        if relative in REVIEW_INPUTS:
            omitted_hashes[item["relpath"]] = item["sha256"]
        if relative in OMIT or relative.name in CLUSTER_SCRIPTS or relative.is_relative_to(SC / "docs"):
            continue
        out = target / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        original = source / relative
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

    (target / ".gpfta_portable_review").write_text(
        "GP-FTA portable anonymous review supplement marker\n",
        encoding="utf-8",
    )

    opsd_npz = source / SC / "data/external/opsd_time_series_extended_v2.npz"
    opsd_arrays_sha = compute_npz_arrays_sha256(opsd_npz.read_bytes())
    (target / "REVIEW_INPUTS.json").write_text(
        json.dumps(
            {
                "omitted_input_sha256": omitted_hashes,
                "opsd_uncompressed_arrays_sha256": opsd_arrays_sha,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    active_supp_sha = compute_v2_python_sha256(target / SC)
    records = [
        record(target, path.relative_to(target).as_posix())
        for path in sorted(target.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "bundle_name": "GP-FTA Operations Research anonymous review supplement",
        "experiment_execution_sha256": exact["execution_code_sha256"],
        "review_supplement_python_sha256": {
            "prospective_active": active_supp_sha,
        },
        "report_and_deployment_metadata_adapted_for_review": sorted(adapted),
        "files": records,
    }
    (target / "REPLICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing verified GP-FTA export")
    parser.add_argument("--export-dir", type=Path, required=True, help="New anonymous supplementary directory")
    args = parser.parse_args()
    verified = export(args.source.resolve(), args.export_dir.resolve())
    print(json.dumps({
        "files": len(verified["files"]),
        "directory": str(args.export_dir),
        "active_supplement_python_sha256": verified["review_supplement_python_sha256"]["prospective_active"],
    }))


if __name__ == "__main__":
    main()
