from pathlib import Path

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
