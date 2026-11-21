#!/usr/bin/env python3
"""Export a standalone GP-FTA review bundle or verify an existing export."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory

MANUSCRIPT_DIR = Path(__file__).resolve().parent
SC_ROOT = MANUSCRIPT_DIR.parent
REPO_ROOT = SC_ROOT.parent
REPLICATION_DIR = MANUSCRIPT_DIR / "replication"
MANIFEST = "REPLICATION_MANIFEST.json"

ARCHIVE = Path("frozen_u3/SC-OLH-KG")
FROZEN_TREE = REPO_ROOT / "GP_FTA_OR_Replication_Bundle/SC-OLH-KG"
U3_SHA256 = "5965482dba305513d62307c362558e304c2c5943930648738f5cc71f6e10b89a"

INDEPENDENT_ARCHIVE = Path("frozen_independent/SC-OLH-KG")
INDEPENDENT_TREE = REPO_ROOT / "GP_FTA_OR_Replication_Bundle_Independent_20261120/SC-OLH-KG"
INDEPENDENT_COMMIT = "0b5ba915ae22208f61757e82b3597b6f1f7d19ba"
INDEPENDENT_SHA256 = "c76f162df1f878398927b5afacf105081a985491a25b66780fbecc385c1bf75f"

REFEREE_ARCHIVE = Path("frozen_referee/SC-OLH-KG")
REFEREE_COMMIT = "fa22a8f970b81ce54c58cafcbfc7f812aff130f5"
REFEREE_SHA256 = "905eca832b0973cba9710693d17e09050e94b03471cb387308d0052d947a4f3d"

PROSPECTIVE_COMMIT = "5454c63a56bd8310730c4d425b5a7eaefd1f2f96"
PROSPECTIVE_SHA256 = "fe8df9098424907147128161afda75d752cc0a8ab47e4357251f05da6206462b"

U3_COHORTS = ("v2_or_u3_confirm_v1", "v2_or_u3_live_sumo_v1", "v2_or_u3_messy_archive_v1")
FRESH_COHORT = "v2_or_kg_quadrature_confirm_v1"
REFEREE_COHORT = "v2_or_gpfta_referee_ablation_v1"
REVIEW3_DIAGNOSTIC = "v2_or_gpfta_review3_v1"
PROSPECTIVE_COHORT = "v2_or_eckg_sure_confirm_v1"
METAPRIOR_OPSD_COHORT = "v2_or_metaprior_opsd_confirm_v1"

LEDGER_COHORTS = (
    "v2_or_p1_gate_check_v1",
    "v2_or_p1_mech_v1",
    "v2_or_p2_confirm_v1",
    "v2_or_p2_live_sumo_v1",
    "v2_or_u1_dev_check_v1",
    "v2_or_u1_dev_check_v2",
    *U3_COHORTS,
    FRESH_COHORT,
    REFEREE_COHORT,
    PROSPECTIVE_COHORT,
    METAPRIOR_OPSD_COHORT,
)

ROOT_REPLICATION_FILES = (
    "README_REPLICATION.md",
    "LICENSES.md",
    "PREREGISTRATION_METAPRIOR_OPSD_20261120.json",
    "PREREGISTRATION_REV5_PROPAGATION_CONFIRM_20261121.json",
    "PREREGISTRATION_REV7_OR_PANEL_20261121.json",
    "requirements.txt",
    "requirements-live-sumo.txt",
    "RESCO_INGOLSTADT21_LICENSE.txt",
    "verify_inputs.py",
    "summarize_prospective_panel.py",
    "summarize_metaprior_opsd_panel.py",
    "summarize_inventory_covariance.py",
    "summarize_cascade_prefix.py",
    "audit_rev2_cascade_scope.py",
    "summarize_eligible_prefix.py",
    "test_cascade_prefix.py",
    "test_eligible_prefix.py",
)

CLUSTER_ARCHIVE_TOOLS = (
    "read_u3_replay.py",
    "summarize_operating_policy.py",
    "summarize_terminal_replay.py",
    "diagnose_source_loo.py",
    "restore_cascade_prefix.py",
    "score_cascade_prefix.py",
    "refine_cascade_prefix.py",
    "restore_eligible_late_prefix.py",
    "score_eligible_prefix.py",
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


def _check_panel_binding(sc_root: Path, runtime_sc: Path, cohort: str, summary: dict, code_sha: str) -> None:
    if summary["execution_code_sha256"] != code_sha:
        raise RuntimeError(f"{cohort} summary differs from its bundled Python")
    original_path = runtime_sc / "v2_or/manifests" / f"{cohort}.json"
    if _sha256(original_path) != summary["manifest_sha256"]:
        raise RuntimeError(f"{cohort} summary differs from its preregistered manifest")
    sealed_path = sc_root / "paper_artifacts" / cohort / "sealed_manifest.json"
    sealed = json.loads(sealed_path.read_bytes())
    if sealed.pop("sealed_complete") is not True or sealed.pop("sealed_execution_code_sha256") != code_sha:
        raise RuntimeError(f"{cohort} sealed manifest differs from its bundled Python")
    sealed.pop("sealed_utc")
    if sealed != json.loads(original_path.read_bytes()):
        raise RuntimeError(f"{cohort} sealed and preregistered manifests disagree")


def _check_archived_bindings(root: Path, entries: list[dict]) -> dict[str, str]:
    if _execution_sha256(root / ARCHIVE) != U3_SHA256:
        raise RuntimeError("Frozen U3 Python does not match the sealed execution fingerprint")
    if _execution_sha256(root / INDEPENDENT_ARCHIVE) != INDEPENDENT_SHA256:
        raise RuntimeError("Frozen independent Python does not match the sealed execution fingerprint")
    if _execution_sha256(root / REFEREE_ARCHIVE) != REFEREE_SHA256:
        raise RuntimeError("Frozen referee Python does not match the sealed execution fingerprint")

    sc_root = root / SC_ROOT.name
    active_sha = _execution_sha256(sc_root)
    if active_sha != PROSPECTIVE_SHA256:
        raise RuntimeError("Active prospective Python does not match the sealed execution fingerprint")

    artifacts = Path(SC_ROOT.name) / "paper_artifacts"
    packaged = {entry["relpath"]: entry["sha256"] for entry in entries}
    for cohort in LEDGER_COHORTS:
        directory = artifacts / cohort
        summary = json.loads((root / directory / "compact_summary.json").read_text(encoding="utf-8"))
        if cohort in U3_COHORTS and summary["execution_code_sha256"] != U3_SHA256:
            raise RuntimeError(f"U3 compact summary disagrees with the bundled Python: {cohort}")
        if cohort == FRESH_COHORT:
            _check_panel_binding(sc_root, root / INDEPENDENT_ARCHIVE, cohort, summary, INDEPENDENT_SHA256)
        elif cohort == REFEREE_COHORT:
            _check_panel_binding(sc_root, root / REFEREE_ARCHIVE, cohort, summary, REFEREE_SHA256)
        elif cohort in {PROSPECTIVE_COHORT, METAPRIOR_OPSD_COHORT}:
            _check_panel_binding(sc_root, sc_root, cohort, summary, PROSPECTIVE_SHA256)
        for name in ("per_run_ledger", "per_slot_certificates"):
            csv = directory / f"{name}.csv"
            recorded = summary[f"{name}_csv_sha256"]
            if packaged[csv.as_posix()] == recorded:
                continue
            local = (root / csv).read_bytes().replace(b"\r\n", b"\n")
            if hashlib.sha256(local.replace(b"\n", b"\r\n")).hexdigest() != recorded:
                raise RuntimeError(f"Archived ledger disagrees with its compact summary: {csv}")
    return {
        "frozen_u3": U3_SHA256,
        "frozen_independent": INDEPENDENT_SHA256,
        "frozen_referee": REFEREE_SHA256,
        "prospective_active": active_sha,
    }


def _extract_git_v2(stage: Path, subdir: str, commit: str) -> Path:
    dest_root = stage / subdir
    saved = subprocess.check_output(["git", "archive", commit, "SC-OLH-KG/v2_or"], cwd=REPO_ROOT)
    with tarfile.open(fileobj=io.BytesIO(saved), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isfile():
                path = dest_root / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    return dest_root / "SC-OLH-KG"


def collect_clean_replication_files(independent_tree: Path, referee_tree: Path, prospective_tree: Path) -> dict[Path, Path]:
    files: dict[Path, Path] = {}

    def add(source: Path, destination: Path | None = None) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"Missing replication input: {source}")
        files[destination or source.relative_to(REPO_ROOT)] = source

    for path in (SC_ROOT / "v2_or").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md"}:
            add(path)
    for path in (prospective_tree / "v2_or").rglob("*.py"):
        add(path, Path(SC_ROOT.name) / path.relative_to(prospective_tree))
    # The sealed execution predates moving these launchers to scripts/.
    for launcher in ("run_v2_lean_cluster.py", "sync_v2_cluster.py"):
        relative = Path("v2_or/dispatch") / launcher
        add(referee_tree / relative, Path(SC_ROOT.name) / relative)
    grid_simulator = Path("v2_or/simulators/opsd_storage.py")
    add(SC_ROOT / grid_simulator, Path("maintenance/SC-OLH-KG") / grid_simulator)
    grid_test = Path("tests/test_opsd_audit_accounting.py")
    add(SC_ROOT / grid_test, Path("maintenance/SC-OLH-KG") / grid_test)

    for path in (FROZEN_TREE / "v2_or").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".md"}:
            add(path, ARCHIVE / path.relative_to(FROZEN_TREE))
    for path in (independent_tree / "v2_or").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".md"}:
            add(path, INDEPENDENT_ARCHIVE / path.relative_to(independent_tree))
    for path in (referee_tree / "v2_or").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".md"}:
            add(path, REFEREE_ARCHIVE / path.relative_to(referee_tree))

    for rel in (
        "scripts/scheduler_node_policy.py",
        "scripts/run_metaprior_propagation_audit.py",
        "scripts/run_direct_propagation_study.py",
        "scripts/run_eckg_reference_validation.py",
        "scripts/run_eckg_multiscramble_cei_audit.py",
        "scripts/eckg_reference_diagnostic.patch",
        "scripts/summarize_eckg_reference_correction.py",
        "scripts/run_rev7_closedloop_study.py",
        "scripts/run_rev7_prior_and_simopt_study.py",
        "SC-OLH-KG/paper_artifacts/v2_or_rev7_or_panel_v1/simopt_prior_default.pkl",
        "SC-OLH-KG/paper_artifacts/v2_or_u3_confirm_v1/source_bundles/Inventory_CrossHorizon/prior_default.pkl",
        "SC-OLH-KG/paper_artifacts/v2_or_u3_confirm_v1/source_bundles/Queueing_CrossHorizon/prior_default.pkl",
        "SC-OLH-KG/paper_artifacts/v2_or_u3_confirm_v1/source_bundles/OPSD_EuroGrid/prior_default.pkl",
        "SC-OLH-KG/paper_artifacts/v2_or_u3_confirm_v1/source_bundles/OPSD_EuroGrid_Unclamped/prior_default.pkl",
        "SC-OLH-KG/paper_artifacts/v2_or_u3_confirm_v1/source_bundles/Traffic_Ingolstadt/prior_default.pkl",
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
        "SC-OLH-KG/performance/prepare_opsd_energy_data.py",
        "proof/lean-toolchain",
        "proof/lake-manifest.json",
        "proof/SCOLHKG/V2ORProofInterface.lean",
    ):
        add(REPO_ROOT / rel)
    for path in (REPO_ROOT / "proof/SCOLHKG/V2OR").glob("*.lean"):
        add(path)

    artifacts = SC_ROOT / "paper_artifacts"
    for cohort, diagnostic in (
        (PROSPECTIVE_COHORT, "eckg_ref_val"),
        (PROSPECTIVE_COHORT, "eckg_multiscramble_cei"),
        (METAPRIOR_OPSD_COHORT, "propagation_audit"),
        (METAPRIOR_OPSD_COHORT, "direct_propagation_exploratory_86"),
        (METAPRIOR_OPSD_COHORT, "direct_propagation_confirm_fresh_72"),
    ):
        diagnostic_dir = artifacts / cohort / diagnostic
        descriptor = diagnostic_dir / "jobs.json"
        add(descriptor)
        for job in json.loads(descriptor.read_bytes())["jobs"]:
            add(diagnostic_dir / "items" / f"{job['job_id']}.json")
    for cohort in LEDGER_COHORTS:
        for name in ("compact_summary.json", "per_run_ledger.csv", "per_slot_certificates.csv"):
            add(artifacts / cohort / name)
    for path in (
        artifacts / "v2_or_p0_reaudit_v1/p0_reaudit_index.json",
        artifacts / "v2_or_p0_reaudit_v1/v2_or_live_sumo_v1/per_run_ledger.csv",
        artifacts / "v2_or_p1_diagnostics_v1/p1_diagnostics_summary.json",
        artifacts / "v2_or_u3_messy_archive_v1/stages/search_and_verify/jobs_reconstructed.json",
        artifacts / "v2_or_u3_historical_replay_receipt.json",
        artifacts / "v2_or_eckg_lean_receipt.json",
        artifacts / FRESH_COHORT / "independent_panel_readout.json",
        artifacts / FRESH_COHORT / "sealed_manifest.json",
        artifacts / FRESH_COHORT / "operating_policy_stage_summary.json",
        artifacts / FRESH_COHORT / "operating_policy_readout.json",
        artifacts / REFEREE_COHORT / "referee_ablation_readout.json",
        artifacts / REFEREE_COHORT / "operating_policy_diagnostics.json",
        artifacts / REFEREE_COHORT / "sealed_manifest.json",
        artifacts / REFEREE_COHORT / "terminal_replay_readout.json",
        artifacts / REFEREE_COHORT / "cascade_query_v1/receipts.json",
        artifacts / REFEREE_COHORT / "cascade_query_v1/readout.json",
        artifacts / REFEREE_COHORT / "eligible_query_v1/late_prefix_receipts.json",
        artifacts / "v2_or_gpfta_review2_v1/source_loo_readout.json",
        artifacts / REVIEW3_DIAGNOSTIC / "rev2_scope_receipts.json",
        artifacts / REVIEW3_DIAGNOSTIC / "rev2_scope_readout.json",
        artifacts / REVIEW3_DIAGNOSTIC / "eligible_query_receipts.json",
        artifacts / REVIEW3_DIAGNOSTIC / "eligible_query_readout.json",
        artifacts / PROSPECTIVE_COHORT / "sealed_manifest.json",
        artifacts / PROSPECTIVE_COHORT / "prospective_panel_readout.json",
        artifacts / PROSPECTIVE_COHORT / "eckg_prefix_audit_readout.json",
        artifacts / PROSPECTIVE_COHORT / "eckg_reference_validation_readout.json",
        artifacts / PROSPECTIVE_COHORT / "eckg_reference_validation_corrected.json",
        artifacts / PROSPECTIVE_COHORT / "eckg_reference_states.json",
        artifacts / PROSPECTIVE_COHORT / "eckg_reference_correction_execution.json",
        artifacts / PROSPECTIVE_COHORT / "eckg_multiscramble_cei_readout.json",
        artifacts / METAPRIOR_OPSD_COHORT / "sealed_manifest.json",
        artifacts / METAPRIOR_OPSD_COHORT / "metaprior_opsd_panel_readout.json",
        artifacts / METAPRIOR_OPSD_COHORT / "as_reported/metaprior_opsd_panel_readout.json",
        artifacts / METAPRIOR_OPSD_COHORT / "metaprior_propagation_readout.json",
        artifacts / METAPRIOR_OPSD_COHORT / "direct_propagation_exploratory_86_readout.json",
        artifacts / METAPRIOR_OPSD_COHORT / "direct_propagation_confirm_fresh_72_readout.json",
        artifacts / METAPRIOR_OPSD_COHORT / "inventory_covariance_readout.json",
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
        if REPLICATION_DIR in path.parents or "__pycache__" in path.parts:
            continue
        if path.is_file() and path.suffix in {".tex", ".bib", ".bst", ".sty", ".cls", ".pdf", ".png", ".py", ".md"}:
            add(path)
    add(REPLICATION_DIR / "lakefile.lean", Path("proof/lakefile.lean"))
    for name in ROOT_REPLICATION_FILES:
        add(REPLICATION_DIR / name, Path(name))
    for name in CLUSTER_ARCHIVE_TOOLS:
        add(REPLICATION_DIR / name, Path("cluster_archive_tools") / name)
    return files


def _record(root: Path, relpath: Path) -> dict[str, str | int]:
    path = root / relpath
    return {"relpath": relpath.as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def export_bundle(root: Path) -> dict:
    with TemporaryDirectory(prefix="gpfta-frozen-") as stage:
        stage_path = Path(stage)
        indep = INDEPENDENT_TREE if INDEPENDENT_TREE.is_dir() else _extract_git_v2(stage_path, "indep", INDEPENDENT_COMMIT)
        referee = _extract_git_v2(stage_path, "referee", REFEREE_COMMIT)
        prospective = _extract_git_v2(stage_path, "prospective", PROSPECTIVE_COMMIT)
        files = collect_clean_replication_files(indep, referee, prospective)
        root.mkdir(parents=True)
        for relative, source in sorted(files.items()):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    entries = [_record(root, relative) for relative in sorted(files)]
    manifest = {
        "bundle_name": "GP-FTA_Operations_Research_Review_Bundle",
        "files": entries,
        "execution_code_sha256": _check_archived_bindings(root, entries),
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_bundle(root: Path) -> dict:
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if _record(root, Path(entry["relpath"])) != entry:
            raise RuntimeError(f"Exported file differs from the manifest: {entry['relpath']}")
    if _check_archived_bindings(root, manifest["files"]) != manifest["execution_code_sha256"]:
        raise RuntimeError("Bundled source differs from the exported execution fingerprints")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--export-dir", type=Path, help="New standalone directory")
    action.add_argument("--verify-dir", type=Path, help="Verify a standalone directory from its manifest")
    args = parser.parse_args()
    root = (args.export_dir or args.verify_dir).resolve()
    manifest = export_bundle(root) if args.export_dir else verify_bundle(root)
    print(json.dumps({
        "directory": str(root),
        "files_checked": len(manifest["files"]),
        "execution_code_sha256": manifest["execution_code_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
