import importlib.util
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/materialize_scolhkg_git_snapshot.py"
SPEC = importlib.util.spec_from_file_location(
    "materialize_git_snapshot", SCRIPT)
SNAPSHOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAPSHOT)


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True)


def test_materialize_snapshot_is_content_addressed_and_fail_closed(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "unit@example.invalid")
    _git(repository, "config", "user.name", "Unit Test")
    for root in SNAPSHOT.TRACKED_ROOTS:
        path = repository / root
        path.mkdir()
        (path / "tracked.txt").write_text(root, encoding="utf-8")
    _git(repository, "add", *SNAPSHOT.TRACKED_ROOTS)
    _git(repository, "commit", "-qm", "snapshot")

    output = tmp_path / "snapshots"
    first = SNAPSHOT.materialize(repository, output)
    second = SNAPSHOT.materialize(repository, output)

    assert first == second
    target = Path(first["snapshot_root"])
    assert target.name == first["repository_commit"]
    assert (target / SNAPSHOT.MARKER).is_file()
    assert (target / "SC-OLH-KG/tracked.txt").is_file()

    (repository / "SC-OLH-KG/tracked.txt").write_text(
        "dirty",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tracked worktree"):
        SNAPSHOT.materialize(repository, output)
