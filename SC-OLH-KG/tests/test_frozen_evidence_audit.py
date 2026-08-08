import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.audit_frozen_evidence import audit_spec  # noqa: E402


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
