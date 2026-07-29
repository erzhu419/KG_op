import importlib.util
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/materialize_scolhkg_traffic_snapshot.py"
SPEC = importlib.util.spec_from_file_location(
    "materialize_traffic_snapshot", SCRIPT)
SNAPSHOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAPSHOT)


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_sparse_traffic_snapshot_contains_code_and_static_assets(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "unit@example.invalid")
    _git(repository, "config", "user.name", "Unit Test")
    for root in ("SC-OLH-KG", "proof", "scripts"):
        _write(repository / root / "tracked.txt", root)
    _write(
        repository
        / "Final_Submission/GPR_KG_Code/experiments/__init__.py",
        "",
    )
    _write(
        repository
        / "Final_Submission/GPR_KG_Code/experiments/ingolstadt21"
        / "config.py",
        "VALUE = 1\n",
    )
    _write(
        repository / SNAPSHOT.TRAFFIC_DECISION_SPACE,
        '{"d": 1}\n',
    )
    _write(repository / SNAPSHOT.TRAFFIC_BASELINE, '{"T0": 1}\n')
    _git(repository, "add", *SNAPSHOT.TRACKED_PATHS)
    _git(repository, "commit", "-qm", "traffic snapshot")

    contract = SNAPSHOT.materialize(
        repository,
        tmp_path / "snapshots",
    )
    target = Path(contract["snapshot_root"])
    assert contract["snapshot_kind"] == "traffic_sparse"
    assert (target / SNAPSHOT.MARKER).is_file()
    assert (target / SNAPSHOT.TRAFFIC_DECISION_SPACE).is_file()
    assert (target / SNAPSHOT.TRAFFIC_BASELINE).is_file()
    assert not (target / "checkpoints").exists()

    _write(repository / "SC-OLH-KG/tracked.txt", "dirty")
    with pytest.raises(RuntimeError, match="tracked worktree"):
        SNAPSHOT.materialize(repository, tmp_path / "snapshots")
