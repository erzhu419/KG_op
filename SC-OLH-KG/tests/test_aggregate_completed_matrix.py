from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "performance"
    / "aggregate_completed_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("aggregate_completed_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_aggregates_sc_result_and_excludes_checkpoint_tree(tmp_path):
    root = tmp_path / "structural_run"
    _write(
        root / "priors" / "full" / "Domain" / "seed0" / "result.json",
        {
            "config": {"d": 50, "N": 20, "n0": 10, "initial_design": "source_informed"},
            "experiment_variant": "structural_backend/priors/full",
            "rows": [{
                "heldout": "Domain",
                "seed": 0,
                "true_feasible": True,
                "feasible_simple_regret": 0.1,
                "initial_has_true_feasible": False,
                "adaptive_rescue": True,
                "posterior_certificate_vacuous": False,
                "false_certificate_count": 0,
                "audit": {"source_simulator_calls": 384},
                "source_target_adaptation_contract": {
                    "target_initial_design_fingerprint": "design-0",
                    "source_archive_fingerprint": "archive-0",
                },
            }],
        },
    )
    _write(root / "checkpoints" / "result.json", {"rows": [{"seed": 99}]})
    (root / "runtime.pkl").write_bytes(b"not read")

    rows, errors = MODULE.load_rows([root])

    assert errors == []
    assert len(rows) == 1
    assert rows[0]["method"] == "full"
    assert rows[0]["adaptive_rescue"] is True
    assert rows[0]["total_calls"] == 404
    assert rows[0]["initial_design_fingerprint"] == "design-0"
    assert rows[0]["source_archive_fingerprint"] == "archive-0"


def test_records_official_runtime_failure_as_a_result_row(tmp_path):
    root = tmp_path / "ratio_run"
    _write(
        root / "official" / "Domain" / "safe_fpacoh_cbo" / "seed0000" / "result.json",
        {
            "method": "safe_fpacoh_cbo",
            "implementation": "official",
            "heldout_target_domain": "Domain",
            "seed": 0,
            "status": "failed_official_runtime",
            "failure_type": "ValueError",
            "comparison_contract": {
                "target_dimension": 1000,
                "target_total_calls_N": 20,
                "target_initial_calls_n0": 10,
                "source_simulator_calls": 384,
            },
        },
    )

    rows, errors = MODULE.load_rows([root])

    assert errors == []
    assert len(rows) == 1
    assert rows[0]["status"] == "failed_official_runtime"
    assert rows[0]["true_feasible"] is None
    assert rows[0]["d_over_target_calls"] == 50.0


def test_aggregates_hvd_identifiability_metrics(tmp_path):
    root = tmp_path / "hvd_run"
    _write(root / "factor" / "result.json", {
        "status": "ok",
        "experiment": "hvd_identifiability",
        "mode": "factor_cumulative",
        "seed": 2,
        "d": 50,
        "n_train_policies": 32,
        "replicates_per_policy": 4,
        "simulator_calls": 128,
        "shared_shock_scale": 2.0,
        "certification_tau": 0.25,
        "log_variance_rmse": 0.3,
        "variance_spearman": 0.8,
        "shared_risk_spearman": 0.7,
        "variance_upper_coverage": 0.95,
        "true_feasible_rate": 0.4,
        "posterior_feasible_rate": 0.3,
        "false_feasible_count": 1,
        "false_feasible_rate": 0.01,
        "false_feasible_fraction_of_certified": 0.1,
        "missed_feasible_rate": 0.2,
        "missed_feasible_fraction_of_true": 0.5,
        "certificate_precision": 0.9,
        "certificate_recall": 0.5,
        "median_predicted_true_ratio": 1.1,
        "median_certified_true_ratio": 1.3,
        "posterior_feasible_count": 4,
        "certificate_nonvacuous": True,
    })

    rows, errors = MODULE.load_rows([root])
    summary = MODULE.summarize_rows(rows)

    assert errors == []
    assert len(rows) == 1
    assert rows[0]["track"] == "hvd_identifiability"
    assert rows[0]["shared_shock_scale"] == 2.0
    assert rows[0]["certification_tau"] == 0.25
    assert rows[0]["replicates_per_policy"] == 4
    assert summary[0]["median_log_variance_rmse"] == 0.3
    assert summary[0]["median_variance_upper_coverage"] == 0.95
    assert summary[0]["median_certificate_recall"] == 0.5
    assert summary[0]["median_missed_feasible_fraction_of_true"] == 0.5
