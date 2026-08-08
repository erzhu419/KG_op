import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_external_energy_temporal_audits import (  # noqa: E402
    analyze,
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_audit(path, source, *, seed, certified, block=1.0, disjoint=1.0):
    payload = {
        "schema_version": 1,
        "contract_id": "opsd_postdecision_temporal_block_audit_v1",
        "status": "complete" if certified else "not_certified",
        "postdecision_only": True,
        "used_to_modify_method_or_certificate": False,
        "independently_certified": certified,
        "target_market": "M1",
        "target_region": "R1",
        "target_seed": seed,
        "arm": "source_atlas",
        "source_result_path": str(source),
        "source_result_sha256": _hash(source),
    }
    if certified:
        payload["temporal_audit"] = {
            "inferential_certificate_claimed": False,
            "minimum_chronological_block_feasibility_probability": block,
            "nonoverlapping_summary": {
                "feasibility_probability": disjoint,
                "window_count": 52,
            },
            "sampled_distribution_summary": {
                "feasibility_probability": 0.99,
            },
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_temporal_analysis_is_fail_closed_and_separates_descriptive_stability(
    tmp_path,
):
    sources = []
    audits = []
    for seed in range(3):
        source = tmp_path / f"source_{seed}.json"
        source.write_text(json.dumps({"seed": seed}), encoding="utf-8")
        audit = tmp_path / f"audit_{seed}.json"
        _write_audit(
            audit,
            source,
            seed=seed,
            certified=seed < 2,
            block=1.0 if seed == 0 else 0.94,
            disjoint=0.98,
        )
        sources.append(source)
        audits.append(audit)

    payload = analyze(
        {"v3": audits}, expected_counts={"v3": 3},
        required_probability=0.95)
    assert payload["status"] == "complete"
    assert payload["inferential_certificate_claimed"] is False
    summary = payload["matrix_summaries"][0]
    assert summary["originally_certified_count"] == 2
    assert summary["chronological_block_stable_count"] == 1
    assert summary["nonoverlap_stable_count"] == 2
    assert summary["joint_descriptive_stability_count"] == 1

    sources[0].write_text(json.dumps({"seed": "changed"}), encoding="utf-8")
    broken = analyze({"v3": audits}, expected_counts={"v3": 3})
    assert broken["status"] == "incomplete"
    assert any(
        row["kind"] == "source_result_hash_mismatch"
        for row in broken["failures"]
    )
