import json

from performance.build_or_review_evidence_registry import (
    _receipt_root,
    build_registry,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_receipt_root_is_order_invariant():
    first = [
        {"path": "b.json", "sha256": "b" * 64},
        {"path": "a.json", "sha256": "a" * 64},
    ]
    assert _receipt_root(first) == _receipt_root(list(reversed(first)))


def test_registry_is_fail_closed_and_counts_algorithm_failure(tmp_path):
    audit_path = _write(tmp_path / "audit.json", {
        "contract_id": "audit",
        "specification_id": "spec-v1",
        "status": "complete_with_algorithmic_failures",
        "publication_ready": True,
        "matrix_count": 1,
        "failure_count": 0,
        "algorithmic_failure_count": 1,
        "matrices": [{
            "name": "matrix",
            "status": "complete_with_algorithmic_failures",
            "relative_glob": "matrix/*.json",
            "expected_cell_count": 2,
            "observed_cell_count": 2,
            "successful_cell_count": 1,
            "algorithmic_failure_cell_count": 1,
            "failure_count": 0,
            "algorithmic_failures": [{"path": "matrix/failure.json"}],
            "receipts": [
                {"path": "matrix/a.json", "sha256": "a" * 64},
                {"path": "matrix/failure.json", "sha256": "b" * 64},
            ],
        }],
    })
    specification = _write(tmp_path / "spec.json", {"contract_id": "spec"})
    method = _write(tmp_path / "method.json", {"contract_id": "method"})
    analysis = _write(tmp_path / "analysis.json", {
        "contract_id": "analysis",
        "status": "complete_with_algorithmic_failures",
        "failures": [],
        "algorithmic_failure_count": 1,
        "source_analysis": {
            "contract_id": "full-analysis",
            "sha256": "c" * 64,
        },
    })
    payload = build_registry(
        json.loads(audit_path.read_text()),
        audit_path=audit_path,
        specification_path=specification,
        method_specification_path=method,
        analyses=(("analysis", analysis),),
        repository_commit="abc123",
    )
    assert payload["publication_ready"] is True
    assert payload["status"] == "complete_with_algorithmic_failures"
    assert payload["matrices"][0]["receipt_count"] == 2
    assert payload["analyses"][0]["algorithmic_failure_count"] == 1
    assert payload["analyses"][0]["source_analysis_contract_id"] == (
        "full-analysis"
    )
    assert payload["analyses"][0]["source_analysis_sha256"] == "c" * 64
    assert payload["frozen_evidence_audit"]["path"] == "audit.json"
    assert payload["frozen_evidence_specification"]["path"] == "spec.json"


def test_registry_rejects_incomplete_analysis(tmp_path):
    audit_path = _write(tmp_path / "audit.json", {
        "contract_id": "audit",
        "specification_id": "spec-v1",
        "status": "complete",
        "publication_ready": True,
        "matrix_count": 0,
        "failure_count": 0,
        "algorithmic_failure_count": 0,
        "matrices": [],
    })
    specification = _write(tmp_path / "spec.json", {})
    method = _write(tmp_path / "method.json", {})
    analysis = _write(tmp_path / "analysis.json", {
        "contract_id": "analysis",
        "status": "incomplete",
        "failures": ["missing cells"],
    })
    payload = build_registry(
        json.loads(audit_path.read_text()),
        audit_path=audit_path,
        specification_path=specification,
        method_specification_path=method,
        analyses=(("analysis", analysis),),
        repository_commit="abc123",
    )
    assert payload["publication_ready"] is False
    assert payload["status"] == "incomplete"
    assert len(payload["integrity_failures"]) == 2
