import json

from performance.materialize_uniform_verification_manifest import materialize


def test_uniform_manifest_keeps_only_preverification_shortlist(tmp_path):
    result_path = tmp_path / "result.json"
    payload = {
        "status": "ok",
        "result": {
            "terminal_shortlist_frozen_before_truth_metrics": True,
            "frozen_terminal_shortlist": [
                {
                    "point": [1, 2],
                    "shortlist_role": "primary",
                    "target_oracle_used": False,
                    "verification_samples_used": False,
                },
                {
                    "point": [3, 4],
                    "shortlist_role": "safe",
                    "target_oracle_used": False,
                    "verification_samples_used": False,
                },
                {
                    "point": [5, 6],
                    "shortlist_role": "third",
                    "target_oracle_used": False,
                    "verification_samples_used": False,
                },
            ],
            "history": [[999]],
        },
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    audit = {
        "records": [{
            "track_id": "track",
            "method_identity": "method",
            "status": "ok",
            "path": str(result_path),
            "domain": "FactorShockStatePolicyRZDT1",
            "target_dimension": 2,
            "seed": 80,
            "source_calls": 384,
            "target_search_calls": 13,
            "optimization_calls_excluding_verification": 397,
            "source_archive_fingerprint": "archive",
            "initial_design_fingerprint": "design",
        }],
    }
    frozen = materialize(
        audit,
        selections=["track::method"],
        candidate_budget=8,
    )
    assert frozen["row_count"] == 1
    assert frozen["candidate_budgets"] == [8, 8]
    assert [item["point"] for item in frozen["rows"][0]["shortlist"]] == [
        [1, 2], [3, 4],
    ]
    serialized = json.dumps(frozen)
    assert "history" not in serialized
    assert "999" not in serialized


def test_uniform_manifest_rejects_truth_tainted_shortlist(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({
        "status": "ok",
        "result": {
            "terminal_shortlist_frozen_before_truth_metrics": True,
            "frozen_terminal_shortlist": [
                {
                    "point": [1],
                    "target_oracle_used": True,
                },
                {"point": [2]},
            ],
        },
    }), encoding="utf-8")
    audit = {
        "records": [{
            "track_id": "track",
            "method_identity": "method",
            "status": "ok",
            "path": str(result_path),
            "domain": "FactorShockStatePolicyRZDT1",
            "target_dimension": 1,
            "seed": 80,
            "source_calls": 384,
            "target_search_calls": 13,
            "optimization_calls_excluding_verification": 397,
            "source_archive_fingerprint": "archive",
            "initial_design_fingerprint": "design",
        }],
    }
    try:
        materialize(audit, selections=["track::method"])
    except ValueError as error:
        assert "forbidden information" in str(error)
    else:
        raise AssertionError("truth-tainted shortlist was accepted")
