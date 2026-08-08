import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.audit_frozen_evidence import (  # noqa: E402
    _sha256,
    audit_spec,
    load_specification,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _spec(expected_count=2):
    return {
        "contract_id": "unit_test_frozen_matrix_v1",
        "matrices": [{
            "name": "primary",
            "relative_glob": "primary/cell*.json",
            "expected_count": expected_count,
            "contract_ids": ["cell_contract_v1"],
            "required_values": {
                "status": "ok",
                "execution_commit": "abc123",
                "information.target_oracle_used": False,
            },
            "unique_key_fields": ["regime", "seed", "arm"],
        }],
    }


def _row(seed):
    return {
        "contract_id": "cell_contract_v1",
        "status": "ok",
        "execution_commit": "abc123",
        "regime": "test",
        "seed": seed,
        "arm": "method",
        "information": {"target_oracle_used": False},
    }


def test_frozen_evidence_audit_accepts_exact_consistent_matrix(tmp_path):
    _write(tmp_path / "primary" / "cell0.json", _row(0))
    _write(tmp_path / "primary" / "cell1.json", _row(1))
    result = audit_spec(tmp_path, _spec())
    assert result["status"] == "complete"
    assert result["publication_ready"] is True
    assert result["failure_count"] == 0
    assert len(result["matrices"][0]["receipts"]) == 2


def test_frozen_evidence_audit_fails_on_partial_or_mixed_matrix(tmp_path):
    first = _row(0)
    second = _row(0)
    second["execution_commit"] = "wrong"
    _write(tmp_path / "primary" / "cell0.json", first)
    _write(tmp_path / "primary" / "cell1.json", second)
    result = audit_spec(tmp_path, _spec(expected_count=3))
    kinds = {
        failure["kind"]
        for failure in result["matrices"][0]["failures"]
    }
    assert result["publication_ready"] is False
    assert "cell_count_mismatch" in kinds
    assert "required_value_mismatch" in kinds
    assert "duplicate_matrix_key" in kinds


def test_frozen_evidence_audit_fails_when_information_field_is_missing(tmp_path):
    row = _row(0)
    del row["information"]
    _write(tmp_path / "primary" / "cell0.json", row)
    result = audit_spec(tmp_path, _spec(expected_count=1))
    failure = next(
        item for item in result["matrices"][0]["failures"]
        if item["kind"] == "required_value_mismatch"
    )
    assert failure["field"] == "information.target_oracle_used"
    assert failure["field_missing"] is True


def test_frozen_evidence_audit_counts_declared_algorithmic_failure(tmp_path):
    specification = _spec()
    matrix = specification["matrices"][0]
    matrix["algorithmic_failure_contract_ids"] = ["cell_error_v1"]
    matrix["algorithmic_failure_required_values"] = {
        "status": "error",
        "execution_commit": "abc123",
    }
    matrix["algorithmic_failure_unique_key_fields"] = [
        "cell.regime", "cell.seed", "cell.arm",
    ]
    _write(tmp_path / "primary" / "cell0.json", _row(0))
    _write(tmp_path / "primary" / "cell1.json", {
        "contract_id": "cell_error_v1",
        "status": "error",
        "execution_commit": "abc123",
        "error_type": "RuntimeError",
        "error_message": "duplicate candidate",
        "cell": {"regime": "test", "seed": 1, "arm": "method"},
    })

    result = audit_spec(tmp_path, specification)

    assert result["status"] == "complete_with_algorithmic_failures"
    assert result["publication_ready"] is True
    assert result["failure_count"] == 0
    assert result["algorithmic_failure_count"] == 1
    matrix_result = result["matrices"][0]
    assert matrix_result["successful_cell_count"] == 1
    assert matrix_result["algorithmic_failure_cell_count"] == 1
    assert matrix_result["accepted_cell_count"] == 2


def test_frozen_evidence_overlay_verifies_parent_digest(tmp_path):
    parent_path = tmp_path / "parent.json"
    _write(parent_path, _spec(expected_count=1))
    overlay_path = tmp_path / "overlay.json"
    _write(overlay_path, {
        "contract_id": "overlay_v1",
        "parent_specification": {
            "path": "parent.json",
            "sha256": _sha256(parent_path),
        },
        "matrix_overrides": {
            "primary": {"expected_count": 3},
        },
        "additional_matrices": [{
            "name": "secondary",
            "relative_glob": "secondary/cell*.json",
            "expected_count": 2,
            "required_values": {"status": "ok"},
            "unique_key_fields": ["seed"],
        }],
    })

    specification = load_specification(overlay_path)

    matrices = {matrix["name"]: matrix for matrix in specification["matrices"]}
    assert specification["contract_id"] == "overlay_v1"
    assert matrices["primary"]["expected_count"] == 3
    assert matrices["secondary"]["expected_count"] == 2
    assert specification["resolved_parent_specification"]["sha256"] == _sha256(
        parent_path)
