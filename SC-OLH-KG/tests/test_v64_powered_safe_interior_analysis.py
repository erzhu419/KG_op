from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load(
    "v64_analysis",
    REPO
    / "SC-OLH-KG/performance/analyze_v64_powered_safe_interior_gate.py",
)
FIXTURE = _load(
    "v64_analysis_fixture",
    REPO / "SC-OLH-KG/tests/test_v63_safe_interior_analysis.py",
)
FIXTURE.MODULE = MODULE


def test_v64_gate_accepts_powered_safe_interior_fresh_matrix(tmp_path):
    control_root = tmp_path / "control"
    challenger_root = tmp_path / "challenger"
    FIXTURE._write_matrix(
        control_root,
        MODULE.CONTROL,
        seed_start=60,
    )
    FIXTURE._write_matrix(
        challenger_root,
        MODULE.CHALLENGER,
        seed_start=60,
    )
    report = MODULE.analyze(
        control_root,
        challenger_root,
        seed_start=60,
        expected_seeds=3,
    )
    assert report["scope"] == MODULE.REPORT_SCOPE
    assert report["contract_valid"] == {
        MODULE.CONTROL: True,
        MODULE.CHALLENGER: True,
    }
    assert report["terminal_certified_count"] == 9
    assert report["terminal_false_certificate_count"] == 0
    assert report["formal_gate_passed"] is True


def test_v64_gate_rejects_primary_budget_drift(tmp_path):
    control_root = tmp_path / "control"
    challenger_root = tmp_path / "challenger"
    FIXTURE._write_matrix(
        control_root,
        MODULE.CONTROL,
        seed_start=60,
    )
    FIXTURE._write_matrix(
        challenger_root,
        MODULE.CHALLENGER,
        seed_start=60,
    )
    path = (
        challenger_root
        / MODULE.CHALLENGER
        / MODULE.V56.DOMAINS[0]
        / "seed60"
        / "result.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["terminal_verification"][
        "candidate_verification_budgets"
    ][0] = 79
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = MODULE.analyze(
        control_root,
        challenger_root,
        seed_start=60,
        expected_seeds=3,
    )
    assert report["contract_valid"][MODULE.CHALLENGER] is False
    assert report["formal_gate_passed"] is False
