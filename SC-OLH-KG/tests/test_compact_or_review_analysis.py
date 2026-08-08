import hashlib

from performance.compact_or_review_analysis import compact_analysis


def test_compact_analysis_omits_only_registered_row_fields(tmp_path):
    source = tmp_path / "full.json"
    source.write_text("full-analysis", encoding="utf-8")
    payload = {
        "schema_version": 3,
        "contract_id": "full-analysis-v3",
        "status": "complete",
        "failures": [],
        "compact_rows": [{"cell": 1}, {"cell": 2}],
        "result_receipts": [{"path": "a", "sha256": "a" * 64}],
        "summaries": [{"rate": 0.75}],
        "paired_comparisons": [{"difference": 0.2}],
    }

    compact = compact_analysis(payload, source_path=source)

    assert compact["status"] == "complete"
    assert compact["source_analysis"]["contract_id"] == "full-analysis-v3"
    assert compact["source_analysis"]["sha256"] == hashlib.sha256(
        b"full-analysis"
    ).hexdigest()
    assert compact["source_analysis"]["omitted_row_fields"] == {
        "compact_rows": {"container_type": "list", "entry_count": 2},
        "result_receipts": {"container_type": "list", "entry_count": 1},
    }
    assert "compact_rows" not in compact["aggregate_analysis"]
    assert "result_receipts" not in compact["aggregate_analysis"]
    assert compact["aggregate_analysis"]["summaries"] == [{"rate": 0.75}]
    assert compact["aggregate_analysis"]["paired_comparisons"] == [
        {"difference": 0.2}
    ]


def test_compact_analysis_preserves_algorithm_failure_outcome(tmp_path):
    source = tmp_path / "failure.json"
    source.write_text("failure-analysis", encoding="utf-8")
    payload = {
        "contract_id": "analysis",
        "status": "complete_with_algorithmic_failures",
        "failures": [],
        "algorithmic_failure_count": 1,
        "algorithmic_failures": [{"cell": "seed-19"}],
        "result_receipts": [{"path": "failure.json", "sha256": "f" * 64}],
    }

    compact = compact_analysis(payload, source_path=source)

    assert compact["status"] == "complete_with_algorithmic_failures"
    assert compact["algorithmic_failure_count"] == 1
    assert compact["aggregate_analysis"]["algorithmic_failures"] == [
        {"cell": "seed-19"}
    ]
