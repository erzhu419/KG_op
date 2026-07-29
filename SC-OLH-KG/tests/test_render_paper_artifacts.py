from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "performance"
    / "render_paper_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("render_paper_artifacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_renders_compact_tables_and_audit_without_runtime_state(tmp_path):
    rows = []
    for method, regrets in {
        "observed_terminal_closure": [0.1, 0.2],
        "sobol_new": [0.3, None],
    }.items():
        for seed, regret in enumerate(regrets):
            rows.append({
                "run_id": "paper",
                "variant": method,
                "method": method,
                "implementation": "sc_olh",
                "initial_design": "source_informed",
                "domain": "InventorySupplyChain",
                "seed": seed,
                "d": 1000,
                "N": 20,
                "n0": 10,
                "true_feasible": regret is not None,
                "feasible_regret": "" if regret is None else regret,
                "posterior_certified_count": 1,
                "evaluated_point_count": 10,
                "false_certificate_count": 0,
                "wall_time_sec": 12.0,
            })
    traces = [{
        "run_id": "paper",
        "variant": "observed_terminal_closure",
        "method": "observed_terminal_closure",
        "implementation": "sc_olh",
        "initial_design": "source_informed",
        "domain": "InventorySupplyChain",
        "seed": 0,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "target_call": 10,
        "incumbent_feasible_regret_post_run": 0.2,
        "target_oracle_used_for_decision": False,
    }]
    rows_path = tmp_path / "rows.csv"
    summary_path = tmp_path / "grouped_summary.csv"
    traces_path = tmp_path / "traces.csv"
    _write_csv(rows_path, rows)
    _write_csv(summary_path, [{"n": 4}])
    _write_csv(traces_path, traces)

    out = tmp_path / "paper"
    manifest = MODULE.render(
        rows_path,
        summary_path,
        traces_path,
        out,
        "observed_terminal_closure",
        no_plots=True,
    )

    assert (out / "table_main.tex").exists()
    assert (out / "table_frontier.tex").exists()
    assert (out / "table_adaptation.tex").exists()
    assert "SC-OLH (promoted)" in (out / "table_main.tex").read_text()
    statistics = json.loads((out / "paired_statistics.json").read_text())
    assert statistics[0]["paired_n"] == 2
    assert statistics[0]["lexicographic_wins"] == 2
    assert manifest["contracts"]["reads_checkpoints"] is False
    assert manifest["contracts"]["post_run_truth_not_used_for_decisions"] is True


def test_bootstrap_interval_is_deterministic():
    first = MODULE.bootstrap_interval([1.0, 2.0, 3.0], samples=100, seed=7)
    second = MODULE.bootstrap_interval([1.0, 2.0, 3.0], samples=100, seed=7)
    assert first == second


def test_registered_paper_variants_have_stable_labels_and_colors():
    for method in (
        "frozen_proposal",
        "proposal_sobol",
        "promoted_joint_voi",
        "new_point_only",
        "pooled_variance",
        "frozen_source_discrepancy",
        "frozen_crossdim_proposal_only",
        "stacked_transfer_gp_cbo:official_transfergpbo_code",
        "canonical_saasbo_every_iteration",
        "scolh:v69_feasible_first_verified_initial_incumbent",
    ):
        assert MODULE._label(method) != method.replace("_", " ")
        assert method in MODULE.METHOD_COLORS
    assert MODULE._method({
        "method_identity": "canonical_saasbo_every_iteration",
    }) == "canonical_saasbo_every_iteration"


def test_renders_hvd_and_evaluate_or_replicate_diagnostics(tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    MODULE._plot_style()
    hvd_rows = []
    for method, base in (("pooled", 0.8), ("factor_cumulative", 0.3)):
        for scale in (0.5, 2.0):
            hvd_rows.append({
                "track": "hvd_identifiability",
                "method": method,
                "shared_shock_scale": scale,
                "replicates_per_policy": 4,
                "log_variance_rmse": base + 0.1 * scale,
                "variance_upper_coverage": 0.94,
            })
    action_rows = [{
        "domain": "QueueResourceControl",
        "method": "promoted_joint_voi",
        "N": budget,
        "adaptive_new_point_count": 6,
        "adaptive_replication_count": budget - 6,
    } for budget in (20, 40)]

    hvd_stem = tmp_path / "hvd"
    action_stem = tmp_path / "actions"
    assert MODULE.plot_hvd_identifiability(hvd_rows, hvd_stem) is True
    assert MODULE.plot_action_allocation(action_rows, action_stem) is True
    assert hvd_stem.with_suffix(".pdf").exists()
    assert hvd_stem.with_suffix(".svg").exists()
    assert hvd_stem.with_suffix(".tiff").exists()
    assert action_stem.with_suffix(".png").exists()
