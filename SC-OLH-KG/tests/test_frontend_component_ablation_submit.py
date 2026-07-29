import importlib.util
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "scripts/submit_scolhkg_frontend_component_ablation_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location(
    "frontend_component_submit", SCRIPT)
SUBMIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMIT)


def _args(tmp_path):
    return SimpleNamespace(
        deploy=tmp_path,
        run_id="frontend_components",
        archive_run_id="source_archive",
        heldouts=",".join(SUBMIT.DOMAINS),
        components=",".join(SUBMIT.COMPONENTS),
        seed_start=80,
        n_seeds=2,
        source_d=50,
        d=1000,
        n0=10,
        offline_source_calls=384,
        design_cpu=12,
        design_ram_mb=16384,
        run_ram_mb=4096,
    )


def test_frontend_component_matrix_is_causal_and_checkpoint_free(tmp_path):
    specs = SUBMIT.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 3 * 3
    assert len({spec["signature"] for spec in specs}) == len(specs)
    assert all(spec["allowed_nodes"] == list(SUBMIT.CPU_NODES)
               for spec in specs)
    assert all("checkpoints" in spec["stage_excludes"] for spec in specs)
    assert all(spec["vram"] == 0 for spec in specs)

    design_specs = [
        spec for spec in specs if "/design/" in spec["signature"]]
    result_specs = [
        spec for spec in specs if "/result/" in spec["signature"]]
    assert len(design_specs) == 9
    assert len(result_specs) == 18
    assert all("--proposal-mode risk_objective_atlas" in spec["cmd"]
               for spec in design_specs)
    for component in SUBMIT.COMPONENTS:
        assert sum(
            f"--proposal-component-mode {component}" in spec["cmd"]
            for spec in design_specs
        ) == 3


def test_frontend_components_share_v69_but_account_source_cost(tmp_path):
    specs = SUBMIT.build_specs(_args(tmp_path))
    runs = [spec for spec in specs if "/result/" in spec["signature"]]
    universal = [
        spec for spec in runs if "/universal_only/" in spec["signature"]]
    source = [
        spec
        for spec in runs
        if "/source_templates_only/" in spec["signature"]
    ]
    combined = [
        spec for spec in runs if "/combined/" in spec["signature"]]

    assert all("--offline-source-calls 0" in spec["cmd"]
               for spec in universal)
    assert all("--offline-source-calls 384" in spec["cmd"]
               for spec in source + combined)
    for spec in runs:
        assert "--terminal-verification-candidate-budgets 80,128,128" in (
            spec["cmd"])
        assert "--terminal-objective-incumbent-guard" in spec["cmd"]
        assert "--terminal-objective-comparison-budget 8" in spec["cmd"]
        assert spec["cpu"] == 1
