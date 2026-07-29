import csv
import json
from pathlib import Path

import pytest

from performance.export_paper_audit_render_inputs import export
from performance.render_paper_artifacts import render


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _record(seed, *, status, feasible, certified):
    return {
        "track_id": "final",
        "method_identity": "canonical_saasbo_every_iteration",
        "implementation": "official",
        "domain": "QueueResourceControl",
        "seed": seed,
        "target_dimension": 1000,
        "source_calls": 384,
        "target_search_calls": 13,
        "target_verification_calls": 80,
        "target_total_calls": 93,
        "source_plus_target_total_calls": 477,
        "status": status,
        "true_feasible": feasible,
        "feasible_regret": 0.1 if feasible else None,
        "constraint_violation": 0.0 if feasible else None,
        "terminal_certified": certified,
        "false_certificate": bool(certified and not feasible),
        "aleatoric_log_variance_rmse": 0.2,
        "aleatoric_variance_rmse": 0.3,
        "aleatoric_upper_coverage": 0.95,
        "aleatoric_variance_shape_correlation": 0.7,
        "target_oracle_used_for_selection": False,
        "source_oracle_aided": False,
        "terminal_verification_updates_optimizer": False,
        "result_sha256": str(seed) * 64,
    }


def test_export_retains_failures_and_locks_renderer_to_audit(tmp_path):
    audit_path = _write_json(tmp_path / "audit.json", {
        "registry_id": "registry",
        "status": "pass",
        "track_audits": [{
            "track_id": "final",
            "status": "pass",
        }],
        "records": [
            _record(1, status="ok", feasible=True, certified=True),
            _record(
                2, status="failed_runtime",
                feasible=None, certified=None),
        ],
    })
    registry_path = _write_json(tmp_path / "registry.json", {
        "registry_id": "registry",
        "paper_render_tracks": ["final"],
    })
    rows = tmp_path / "rows.csv"
    summary = tmp_path / "summary.csv"
    input_manifest = tmp_path / "render_inputs.json"
    manifest = export(
        json.loads(audit_path.read_text()),
        json.loads(registry_path.read_text()),
        audit_path=audit_path,
        registry_path=registry_path,
        rows_path=rows,
        summary_path=summary,
        manifest_path=input_manifest,
    )
    assert manifest["row_count"] == 2
    assert manifest["failure_or_timeout_count"] == 1
    assert manifest["contracts"]["unregistered_tracks_excluded"] is True

    traces = tmp_path / "traces.csv"
    _write_csv(traces, [{
        "target_oracle_used_for_decision": False,
    }])
    paired_statistics = _write_json(
        tmp_path / "paired_statistics.json",
        {
            "status": "complete",
            "inference_families": [{
                "family_id": "primary",
                "hypothesis_count": 1,
            }],
            "rows": [],
        },
    )
    rendered = render(
        rows,
        summary,
        traces,
        tmp_path / "rendered",
        "canonical_saasbo_every_iteration",
        no_plots=True,
        input_manifest_path=input_manifest,
        paired_statistics_path=paired_statistics,
    )
    assert rendered["contracts"][
        "rows_from_passed_registered_paper_audit"] is True
    assert rendered["inputs"]["audit_export_manifest"][
        "contract_id"] == "audited_compact_render_input_v1"
    assert rendered["contracts"]["paired_statistics_preregistered"] is True

    rows.write_text(rows.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="render input manifest"):
        render(
            rows,
            summary,
            traces,
            tmp_path / "tampered",
            "canonical_saasbo_every_iteration",
            no_plots=True,
            input_manifest_path=input_manifest,
            paired_statistics_path=paired_statistics,
        )
