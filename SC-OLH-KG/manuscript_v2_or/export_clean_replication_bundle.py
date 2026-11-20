#!/usr/bin/env python3
"""Export a standalone GP-FTA review bundle or verify an existing export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

MANUSCRIPT_DIR = Path(__file__).resolve().parent
SC_ROOT = MANUSCRIPT_DIR.parent
REPO_ROOT = SC_ROOT.parent
REPLICATION_DIR = MANUSCRIPT_DIR / "replication"
MANIFEST = "REPLICATION_MANIFEST.json"
ARCHIVE = Path("frozen_u3/SC-OLH-KG")
FROZEN_TREE = REPO_ROOT / "GP_FTA_OR_Replication_Bundle/SC-OLH-KG"
U3_SHA256 = "5965482dba305513d62307c362558e304c2c5943930648738f5cc71f6e10b89a"
U3_COHORTS = ("v2_or_u3_confirm_v1", "v2_or_u3_live_sumo_v1", "v2_or_u3_messy_archive_v1")
FRESH_COHORT = "v2_or_kg_quadrature_confirm_v1"
FRESH_MANIFEST = Path("v2_or/manifests") / f"{FRESH_COHORT}.json"
LEDGER_COHORTS = (
    "v2_or_p1_gate_check_v1",
    "v2_or_p1_mech_v1",
    "v2_or_p2_confirm_v1",
    "v2_or_p2_live_sumo_v1",
    "v2_or_u1_dev_check_v1",
    "v2_or_u1_dev_check_v2",
    *U3_COHORTS,
    FRESH_COHORT,
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _execution_sha256(sc_root: Path) -> str:
    """Use the same v2_or execution fingerprint as campaign_runner."""
    digest = hashlib.sha256()
    for path in sorted((sc_root / "v2_or").rglob("*.py")):
        digest.update(path.relative_to(sc_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _check_fresh_binding(sc_root: Path, summary: dict, code_sha: str) -> None:
    if summary["execution_code_sha256"] != code_sha:
        raise RuntimeError("Independent panel summary differs from the bundled Python")
    original_path = sc_root / FRESH_MANIFEST
    if _sha256(original_path) != summary["manifest_sha256"]:
        raise RuntimeError("Independent panel summary differs from the preregistered manifest")
    sealed_path = sc_root / "paper_artifacts" / FRESH_COHORT / "sealed_manifest.json"
    sealed = json.loads(sealed_path.read_bytes())
    if sealed.pop("sealed_complete") is not True or sealed.pop("sealed_execution_code_sha256") != code_sha:
        raise RuntimeError("Independent panel sealed manifest differs from the bundled Python")
    sealed.pop("sealed_utc")
    if sealed != json.loads(original_path.read_bytes()):
        raise RuntimeError("Independent panel sealed and original manifests disagree")


def _check_archived_bindings(root: Path, entries: list[dict]) -> str:
    if _execution_sha256(root / ARCHIVE) != U3_SHA256:
        raise RuntimeError("Frozen U3 Python does not match the sealed execution fingerprint")

    corrected_sha = _execution_sha256(root / SC_ROOT.name)
    artifacts = Path(SC_ROOT.name) / "paper_artifacts"
    packaged = {entry["relpath"]: entry["sha256"] for entry in entries}
    for cohort in LEDGER_COHORTS:
        directory = artifacts / cohort
        summary = json.loads((root / directory / "compact_summary.json").read_text(encoding="utf-8"))
        if cohort in U3_COHORTS and summary["execution_code_sha256"] != U3_SHA256:
            raise RuntimeError(f"U3 compact summary disagrees with the bundled Python: {cohort}")
        if cohort == FRESH_COHORT:
            _check_fresh_binding(root / SC_ROOT.name, summary, corrected_sha)
        for name in ("per_run_ledger", "per_slot_certificates"):
            csv = directory / f"{name}.csv"
            recorded = summary[f"{name}_csv_sha256"]
            if packaged[csv.as_posix()] == recorded:
                continue
            # The campaign CSV writer uses CRLF. Some local checkouts store LF.
            local = (root / csv).read_bytes().replace(b"\r\n", b"\n")
            if hashlib.sha256(local.replace(b"\n", b"\r\n")).hexdigest() != recorded:
                raise RuntimeError(f"Archived ledger disagrees with its compact summary: {csv}")
    return corrected_sha


def collect_clean_replication_files() -> dict[Path, Path]:
    files: dict[Path, Path] = {}

    def add(source: Path, destination: Path | None = None) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"Missing replication input: {source}")
        files[destination or source.relative_to(REPO_ROOT)] = source

    for path in (SC_ROOT / "v2_or").rglob("*"):
        if path.suffix in {".py", ".json", ".md"}:
            add(path)
    for path in (FROZEN_TREE / "v2_or").rglob("*"):
        if path.suffix in {".py", ".json", ".md"}:
            add(path, ARCHIVE / path.relative_to(FROZEN_TREE))

    for rel in (
        "scripts/scheduler_node_policy.py",
        "SC-OLH-KG/data/__init__.py",
        "SC-OLH-KG/data/opsd.py",
        "SC-OLH-KG/data/external/opsd_time_series_extended_v2.npz",
        "SC-OLH-KG/data/external/opsd_time_series_extended_v2.npz.manifest.json",
        "SC-OLH-KG/performance/prepare_opsd_energy_data.py",
        "proof/lean-toolchain",
        "proof/lake-manifest.json",
        "proof/SCOLHKG/V2ORProofInterface.lean",
        "SC-OLH-KG/docs/gp_fta_systematic_audit_20261120.md",
        "SC-OLH-KG/docs/gp_fta_or_remediation_20261120.md",
        "SC-OLH-KG/docs/gp_fta_u3_historical_replay_20261120.md",
        "SC-OLH-KG/docs/gp_fta_corrected_kg_confirmation_preregistration_20261120.md",
        "SC-OLH-KG/docs/gp_fta_corrected_kg_confirmation_results_20261120.md",
    ):
        add(REPO_ROOT / rel)
    for path in (REPO_ROOT / "proof/SCOLHKG/V2OR").glob("*.lean"):
        add(path)

    legacy = REPO_ROOT / "Final_Submission/GPR_KG_Code"
    for rel in (
        "experiments/__init__.py",
        "experiments/ingolstadt21/__init__.py",
        "experiments/ingolstadt21/config.py",
        "experiments/ingolstadt21/ingolstadt21_problem.py",
        "experiments/ingolstadt21/sumo_sim.py",
        "experiments/ingolstadt21/data/ingolstadt21.net.xml",
        "experiments/ingolstadt21/data/ingolstadt21.rou.xml",
        "results/ingolstadt21/decision_space.json",
        "results/ingolstadt21/baseline.json",
    ):
        add(legacy / rel)

    artifacts = SC_ROOT / "paper_artifacts"
    for cohort in LEDGER_COHORTS:
        for name in ("compact_summary.json", "per_run_ledger.csv", "per_slot_certificates.csv"):
            add(artifacts / cohort / name)
    for path in (
        artifacts / "v2_or_p0_reaudit_v1/p0_reaudit_index.json",
        artifacts / "v2_or_p0_reaudit_v1/v2_or_live_sumo_v1/per_run_ledger.csv",
        artifacts / "v2_or_p1_diagnostics_v1/p1_diagnostics_summary.json",
        artifacts / "v2_or_u3_messy_archive_v1/stages/search_and_verify/jobs_reconstructed.json",
        artifacts / "v2_or_u3_historical_replay_receipt.json",
        artifacts / FRESH_COHORT / "independent_panel_readout.json",
        artifacts / FRESH_COHORT / "sealed_manifest.json",
    ):
        add(path)
    for directory in artifacts.glob("v2_or_*"):
        if directory.is_dir():
            for path in directory.iterdir():
                if path.is_file() and path.suffix in {".json", ".csv", ".md"}:
                    add(path)
            for path in directory.glob("stages/*/jobs.json"):
                add(path)
            for path in directory.glob("stages/*/completion.json"):
                add(path)
    for path in (artifacts / "v2_or_p0_reaudit_v1").glob("*/*"):
        if path.is_file() and path.suffix in {".json", ".csv"}:
            add(path)
    for path in sorted(artifacts.glob("v2_or_*.json")):
        if path.name != "v2_or_clean_replication_manifest.json":
            add(path)

    for path in MANUSCRIPT_DIR.rglob("*"):
        if REPLICATION_DIR in path.parents:
            continue
        if path.suffix in {".tex", ".bib", ".bst", ".sty", ".cls", ".pdf", ".png", ".py", ".md"}:
            add(path)
    add(REPLICATION_DIR / "lakefile.lean", Path("proof/lakefile.lean"))
    for name in ("README_REPLICATION.md", "LICENSES.md", "requirements.txt", "requirements-live-sumo.txt",
                 "RESCO_INGOLSTADT21_LICENSE.txt", "verify_inputs.py", "read_u3_replay.py"):
        add(REPLICATION_DIR / name, Path(name))
    return files


def _record(root: Path, relpath: Path) -> dict[str, str | int]:
    path = root / relpath
    return {"relpath": relpath.as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def export_bundle(root: Path) -> dict:
    files = collect_clean_replication_files()
    root.mkdir(parents=True)
    for relative, source in sorted(files.items()):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    entries = [_record(root, relative) for relative in sorted(files)]
    corrected_sha = _check_archived_bindings(root, entries)
    manifest = {
        "bundle_name": "GP-FTA_Operations_Research_Review_Bundle",
        "files": entries,
        "execution_code_sha256": {
            "frozen_u3": U3_SHA256,
            "corrected_source": corrected_sha,
        },
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_bundle(root: Path) -> dict:
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if _record(root, Path(entry["relpath"])) != entry:
            raise RuntimeError(f"Exported file differs from the manifest: {entry['relpath']}")
    corrected_sha = _check_archived_bindings(root, manifest["files"])
    if corrected_sha != manifest["execution_code_sha256"]["corrected_source"]:
        raise RuntimeError("Corrected source differs from the exported execution fingerprint")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--export-dir", type=Path, help="New standalone directory")
    action.add_argument("--verify-dir", type=Path, help="Verify a standalone directory from its manifest")
    args = parser.parse_args()
    root = (args.export_dir or args.verify_dir).resolve()
    manifest = export_bundle(root) if args.export_dir else verify_bundle(root)
    print(json.dumps({"directory": str(root), "files_checked": len(manifest["files"]),
                      "execution_code_sha256": manifest["execution_code_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
