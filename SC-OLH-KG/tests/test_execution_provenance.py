import pytest

from performance.execution_provenance import (
    attach_execution_provenance,
    execution_provenance_from_env,
)


VARIABLES = (
    "SCOLHKG_EXECUTION_COMMIT",
    "SCOLHKG_SCOLHKG_TREE",
    "SCOLHKG_PROOF_TREE",
    "SCOLHKG_SCRIPTS_TREE",
    "SCOLHKG_METHOD_CONTRACT_ID",
    "SCOLHKG_THEORY_CONTRACT_ID",
    "SCOLHKG_CODE_SNAPSHOT_ROOT",
    "SCOLHKG_EXECUTION_PROVENANCE_REQUIRED",
    "SCOLHKG_LEGACY_TRAFFIC_TREE",
    "SCOLHKG_TRAFFIC_DECISION_SPACE_BLOB",
    "SCOLHKG_TRAFFIC_BASELINE_BLOB",
)


def _clear(monkeypatch):
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_execution_provenance_is_optional_outside_frozen_runs(monkeypatch):
    _clear(monkeypatch)
    payload = attach_execution_provenance({"status": "ok"})
    assert payload["execution_provenance"] == {
        "status": "unregistered",
        "required": False,
    }


def test_execution_provenance_is_fail_closed_for_frozen_runs(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SCOLHKG_EXECUTION_PROVENANCE_REQUIRED", "1")
    with pytest.raises(RuntimeError, match="incomplete"):
        execution_provenance_from_env()

    values = {
        "SCOLHKG_EXECUTION_COMMIT": "a" * 40,
        "SCOLHKG_SCOLHKG_TREE": "b" * 40,
        "SCOLHKG_PROOF_TREE": "c" * 40,
        "SCOLHKG_SCRIPTS_TREE": "d" * 40,
        "SCOLHKG_METHOD_CONTRACT_ID": "method-v1",
        "SCOLHKG_THEORY_CONTRACT_ID": "theory-v1",
        "SCOLHKG_CODE_SNAPSHOT_ROOT": "/immutable/a",
    }
    for variable, value in values.items():
        monkeypatch.setenv(variable, value)
    provenance = execution_provenance_from_env()
    assert provenance["status"] == "frozen"
    assert provenance["repository_commit"] == "a" * 40
    assert provenance["required"] is True


def test_execution_provenance_requires_complete_traffic_bundle(monkeypatch):
    _clear(monkeypatch)
    values = {
        "SCOLHKG_EXECUTION_COMMIT": "a" * 40,
        "SCOLHKG_SCOLHKG_TREE": "b" * 40,
        "SCOLHKG_PROOF_TREE": "c" * 40,
        "SCOLHKG_SCRIPTS_TREE": "d" * 40,
        "SCOLHKG_METHOD_CONTRACT_ID": "method-v1",
        "SCOLHKG_THEORY_CONTRACT_ID": "theory-v1",
        "SCOLHKG_CODE_SNAPSHOT_ROOT": "/immutable/a",
        "SCOLHKG_LEGACY_TRAFFIC_TREE": "e" * 40,
    }
    for variable, value in values.items():
        monkeypatch.setenv(variable, value)
    with pytest.raises(RuntimeError, match="traffic provenance"):
        execution_provenance_from_env()

    monkeypatch.setenv(
        "SCOLHKG_TRAFFIC_DECISION_SPACE_BLOB", "f" * 40)
    monkeypatch.setenv("SCOLHKG_TRAFFIC_BASELINE_BLOB", "1" * 40)
    provenance = execution_provenance_from_env()
    assert provenance["legacy_traffic_tree"] == "e" * 40
    assert provenance["traffic_decision_space_blob"] == "f" * 40
    assert provenance["traffic_baseline_blob"] == "1" * 40
