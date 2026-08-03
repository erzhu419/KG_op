import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "performance/render_or_manuscript_artifacts.py"
SPEC = importlib.util.spec_from_file_location("render_or_artifacts", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_render_or_manuscript_tables_from_compact_audit(tmp_path):
    manifest = MODULE.render(
        audit_path=MODULE.DEFAULT_AUDIT,
        summary_path=MODULE.DEFAULT_SUMMARY,
        dimension_path=MODULE.DEFAULT_DIMENSION,
        coverage_path=MODULE.DEFAULT_COVERAGE,
        energy_path=MODULE.DEFAULT_ENERGY,
        convergence_path=MODULE.DEFAULT_CONVERGENCE,
        output_dir=tmp_path,
        no_plots=True,
    )

    assert manifest["status"] == "complete"
    assert manifest["contracts"]["reads_checkpoints"] is False
    assert manifest["contracts"]["hvd_rendered_as_secondary_diagnostic"] is True
    frontend = (tmp_path / "tables/frontend_backend.tex").read_text()
    assert "20/20" in frontend
    assert "Common Sobol" in frontend
    components = (tmp_path / "tables/frontend_components.tex").read_text()
    assert "60/60" in components
    energy = (tmp_path / "tables/external_energy.tex").read_text()
    assert "Natural low-frequency grid" in energy
    dimension = (tmp_path / "tables/dimension_budget.tex").read_text()
    assert "10000" in dimension
    assert "0.0058" in dimension
