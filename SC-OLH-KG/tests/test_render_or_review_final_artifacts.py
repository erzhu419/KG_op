import json
from pathlib import Path

import numpy as np

import performance.render_or_review_final_artifacts as renderer
from performance.render_or_review_final_artifacts import render


ROOT = Path(__file__).resolve().parents[1]


def test_review_renderer_uses_frozen_compact_evidence(tmp_path):
    manuscript = tmp_path / "manuscript"
    manifest = render(
        ROOT / "paper_artifacts/or_review",
        manuscript,
        skip_figures=True,
    )

    assert manifest["status"] == "complete"
    assert manifest["contracts"][
        "reads_compact_audited_artifacts_only"
    ] is True
    assert len(manifest["inputs"]) == 11
    assert len(manifest["outputs"]) == 11
    primary = (manuscript / "tables/review_stress_primary.tex").read_text()
    assert "Source-scored atlas" in primary
    assert "47.3" in primary
    energy = (manuscript / "tables/review_energy_v3.tex").read_text()
    assert "60/90" in energy
    assert "70/90" in energy
    equal_cost = (manuscript / "tables/review_equal_cost.tex").read_text()
    assert "Target-only functional SCBO" in equal_cost
    assert "1 &" in equal_cost
    sensitivity = (manuscript / "tables/review_sensitivity.tex").read_text()
    assert "Chance level $\\alpha$" in sensitivity
    assert "Frequency penalty $\\kappa$" in sensitivity
    outcome_cost = (
        manuscript / "tables/review_outcome_adjusted_cost.tex"
    ).read_text()
    assert "Calls/success" in outcome_cost
    assert "Equal preverification" in outcome_cost
    assert "Source-scored atlas" in outcome_cost
    strata = (manuscript / "tables/review_task_seed_strata.tex").read_text()
    assert "Aligned low frequency" in strata
    assert "Overall & 41 & 32 & 40 & 25 & 22 & 160" in strata


def _revision_fixture(tmp_path):
    revision = tmp_path / "revision"
    revision.mkdir()
    primary = json.loads((ROOT / "paper_artifacts/submission_revision_20260907/analysis.json").read_text())
    # Supplement reference rows are deliberately altered: they must never
    # replace the already published source and legacy summaries.
    for row in primary["summaries"]:
        if row["arm"] not in renderer.SUPPLEMENTAL_ARMS:
            row["certified_true_feasible_deployment_count"] = 0
    (revision / "analysis.json").write_text(json.dumps(primary))
    historical_energy = json.loads((ROOT / "paper_artifacts/or_review/energy_v3.json").read_text())["aggregate_analysis"]
    energy_rows = [
        {**row, "arm": arm, "certified_safe_count": 0,
         "false_certificate_count": 0, "median_objective_if_certified": None}
        for row in historical_energy["market_summaries"] if row["arm"] == "source_atlas"
        for arm in sorted(renderer.SUPPLEMENTAL_ARMS)
    ]
    energy_rows.extend({**row, "certified_safe_count": 0}
                       for row in historical_energy["market_summaries"]
                       if row["arm"] == "source_atlas")
    (revision / "energy_analysis.json").write_text(json.dumps({
        "status": "complete", "study_role": "post-review reused target seeds",
        "market_summaries": energy_rows,
    }))
    return revision


def test_supplement_tables_preserve_references_and_charge_both_source_arms(tmp_path):
    manuscript = tmp_path / "manuscript"
    manifest = render(ROOT / "paper_artifacts/or_review", manuscript,
                      skip_figures=True, revision=_revision_fixture(tmp_path))
    assert len(manifest["inputs"]) == 13
    assert len(manifest["outputs"]) == 12
    assert "previously used task seeds" in manifest["supplement_study_role"]["synthetic"]
    primary = (manuscript / "tables/review_stress_primary.tex").read_text()
    source = next(line for line in primary.splitlines() if line.startswith("Source-scored atlas"))
    assert " & 480 & 91.7 & 47.3 & 0 & " in source
    assert "Standardized DCT maximin & 480" in primary
    assert "Source-greedy portfolio & 480" in primary
    assert "Unstandardized DCT maximin & 480" in primary
    paired = (manuscript / "tables/review_matched_comparisons.tex").read_text()
    assert "1000 & Standardized DCT maximin & +15.62 & [11.25, 20.00] & 28 & 3 & 129" in paired
    assert paired.count("Standardized DCT maximin") == 3
    assert paired.count("Source-greedy portfolio") == 3
    energy = (manuscript / "tables/review_energy_v3.tex").read_text()
    assert "Source-scored atlas & 60/90" in energy
    assert "Unstandardized DCT maximin & 70/90" in energy
    assert "Standardized DCT maximin & 0/90 & 0 & -- &" in energy
    assert "Source-greedy portfolio & 0/90 & 0 & -- &" in energy
    outcome = (manuscript / "tables/review_outcome_adjusted_cost.tex").read_text()
    assert "$N=10$ primary & Source-scored atlas & 47.3 & 1182.7 & 411.3 & --" in outcome
    assert "$N=10$ primary & Source-greedy portfolio & 37.1 & 1596.4 & 612.7 & 1" in outcome
    assert "Equal preverification & Raw Sobol" in outcome
    assert "Equal preverification & Source-greedy portfolio" not in outcome


def test_source_greedy_portfolio_is_not_a_source_free_regime_control(tmp_path):
    primary = json.loads((ROOT / "paper_artifacts/or_review/randomized_profile_primary.json").read_text())["aggregate_analysis"]
    portfolio = [{**row, "arm": "source_greedy_portfolio",
                  "certified_true_feasible_deployment_count": 20}
                 for row in primary["summaries"] if row["arm"] == "source_atlas"]
    output = tmp_path / "regimes.tex"
    renderer._write_regime_table({"summaries": primary["summaries"] + portfolio}, output)
    assert "Source-greedy portfolio" not in output.read_text()


def test_cost_figure_uses_only_primary_n10_and_amortizes_source_portfolio(monkeypatch):
    primary = json.loads((ROOT / "paper_artifacts/or_review/randomized_profile_primary.json").read_text())["aggregate_analysis"]
    revision = json.loads((ROOT / "paper_artifacts/submission_revision_20260907/analysis.json").read_text())
    primary["summaries"].extend(row for row in revision["summaries"]
                                if row["arm"] in renderer.SUPPLEMENTAL_ARMS)
    captured = []
    monkeypatch.setattr(renderer, "_save", lambda fig, stem: captured.append(fig))
    renderer._plot_cost(primary, "unused")
    axes = captured[0].axes
    assert len(axes) == 2
    for ax, deployments in zip(axes, (1, 20)):
        archives, target_search, _ = ax.containers
        np.testing.assert_allclose([bar.get_height() for bar in archives],
                                   [384 / deployments, 0, 384 / deployments, 0], atol=1e-10)
        np.testing.assert_allclose([bar.get_height() for bar in target_search], [10] * 4)


def test_regime_figure_uses_both_matched_controls_and_keeps_negative_differences(monkeypatch):
    revision = json.loads((ROOT / "paper_artifacts/submission_revision_20260907/analysis.json").read_text())
    captured = []
    monkeypatch.setattr(renderer, "_save", lambda fig, stem: captured.append(fig))
    renderer._plot_regime(revision, "unused")
    axes = captured[0].axes[:2]
    assert len(axes) == 2
    for ax, control in zip(axes, ("standardized_generic_dct_maximin", "source_greedy_portfolio")):
        values = ax.images[0].get_array()
        assert values.shape == (8, 3)
        assert renderer.ARM_LABELS[control] in ax.get_title(loc="left")
        expected = {(row["regime"], row["nominal_dimension"]):
                    100 * row["endpoints"]["certified_true_feasible"]["source_minus_control_mean"]
                    for row in revision["paired_comparisons"] if row["control_arm"] == control}
        np.testing.assert_allclose(values, [[expected[regime, dimension]
                                            for dimension in (200, 1000, 10000)]
                                           for regime in renderer.REGIME_LABELS], atol=1e-10)
    assert axes[0].images[0].get_array()[2, 0] < 0
