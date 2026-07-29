#!/usr/bin/env python3
"""Materialize an immutable sparse code snapshot for the SUMO holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


METHOD_CONTRACT_ID = "or_transfer_frontend_saas_v1"
THEORY_CONTRACT_ID = "source_target_geometric_atlas_coverage_v1"
MARKER = ".scolhkg_execution_snapshot.json"
TRAFFIC_EXPERIMENT = (
    "Final_Submission/GPR_KG_Code/experiments/ingolstadt21")
TRAFFIC_DECISION_SPACE = (
    "Final_Submission/GPR_KG_Code/results/ingolstadt21/"
    "decision_space.json")
TRAFFIC_BASELINE = (
    "Final_Submission/GPR_KG_Code/results/ingolstadt21/baseline.json")
TRACKED_PATHS = (
    "SC-OLH-KG",
    "proof",
    "scripts",
    "Final_Submission/GPR_KG_Code/experiments/__init__.py",
    TRAFFIC_EXPERIMENT,
    TRAFFIC_DECISION_SPACE,
    TRAFFIC_BASELINE,
)


def _git(root, *args):
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def snapshot_contract(repository_root, commit="HEAD"):
    repository_root = Path(repository_root).resolve()
    full_commit = _git(repository_root, "rev-parse", str(commit))
    return {
        "schema_version": 1,
        "status": "frozen",
        "snapshot_kind": "traffic_sparse",
        "repository_commit": full_commit,
        "scolhkg_tree": _git(
            repository_root, "rev-parse", f"{full_commit}:SC-OLH-KG"),
        "proof_tree": _git(
            repository_root, "rev-parse", f"{full_commit}:proof"),
        "scripts_tree": _git(
            repository_root, "rev-parse", f"{full_commit}:scripts"),
        "legacy_traffic_tree": _git(
            repository_root,
            "rev-parse",
            f"{full_commit}:{TRAFFIC_EXPERIMENT}",
        ),
        "traffic_decision_space_blob": _git(
            repository_root,
            "rev-parse",
            f"{full_commit}:{TRAFFIC_DECISION_SPACE}",
        ),
        "traffic_baseline_blob": _git(
            repository_root,
            "rev-parse",
            f"{full_commit}:{TRAFFIC_BASELINE}",
        ),
        "method_contract_id": METHOD_CONTRACT_ID,
        "theory_contract_id": THEORY_CONTRACT_ID,
        "tracked_paths": list(TRACKED_PATHS),
        "runtime_results_included": False,
        "runtime_checkpoints_or_model_weights_included": False,
        "target_outcomes_used_to_select_snapshot": False,
    }


def _validate_existing(path, expected):
    marker = path / MARKER
    if not marker.is_file():
        raise RuntimeError(f"traffic snapshot has no marker: {path}")
    observed = json.loads(marker.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError(f"traffic snapshot contract mismatch: {path}")
    return observed


def materialize(repository_root, output_root, *, commit="HEAD"):
    repository_root = Path(repository_root).resolve()
    output_root = Path(output_root).resolve()
    dirty = _git(
        repository_root,
        "status",
        "--short",
        "--untracked-files=no",
    )
    if dirty:
        raise RuntimeError(
            "tracked worktree must be clean before freezing a snapshot")
    contract = snapshot_contract(repository_root, commit)
    target = output_root / contract["repository_commit"]
    contract = {**contract, "snapshot_root": str(target)}
    if target.exists():
        return _validate_existing(target, contract)

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{contract['repository_commit']}.",
        dir=output_root,
    ))
    archive = temporary / "tracked.tar"
    extracted = temporary / "content"
    extracted.mkdir()
    try:
        with archive.open("wb") as handle:
            subprocess.run(
                [
                    "git",
                    "archive",
                    "--format=tar",
                    contract["repository_commit"],
                    *TRACKED_PATHS,
                ],
                cwd=repository_root,
                stdout=handle,
                check=True,
            )
        with tarfile.open(archive, "r") as handle:
            handle.extractall(extracted)
        archive.unlink()
        (extracted / MARKER).write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        extracted.replace(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _validate_existing(target, contract)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--output-root",
        default=(
            "/home/erzhu419/mine_code/"
            "KG_op_traffic_code_snapshots"),
    )
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args()
    contract = materialize(
        args.repository_root,
        args.output_root,
        commit=args.commit,
    )
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
