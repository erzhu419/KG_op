import pytest

from performance.analyze_traffic_universal_library_diagnostic import analyze


def _payload(indices, *, successes=200, trials=200):
    return {
        "execution_provenance": {"repository_commit": "a" * 40},
        "candidates": [{
            "source_index": index,
            "x": [index, index],
            "validation": {
                "R": trials,
                "seeds": list(range(1000, 1000 + trials)),
                "feasible_count": successes,
                "feasible_probability": successes / trials,
                "mean": [1.0, 2.0, 3.0],
            },
        } for index in indices],
    }


def test_posthoc_library_audit_is_redacted_and_not_confirmatory():
    result = analyze(
        [_payload([0, 2]), _payload([1, 3])],
        expected_library_size=4,
    )
    assert result["status"] == "complete"
    assert result["familywise_certified_candidate_count"] == 4
    assert result["admissible_for_method_selection"] is False
    assert result["admissible_for_confirmatory_claim"] is False
    assert result["policy_vectors_exported"] is False
    assert all("x" not in row for row in result["rows"])


def test_posthoc_library_audit_rejects_missing_indices():
    with pytest.raises(ValueError, match="fixed library"):
        analyze([_payload([0, 2])], expected_library_size=3)
