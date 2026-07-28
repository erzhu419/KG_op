import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = _load(
    "promoted_v51_certification_budget_submit",
    REPO
    / "scripts/submit_scolhkg_promoted_v51_certification_budget_gate_scheduler.py",
)


def _args(tmp_path):
    defaults = submit.closure._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "deploy": deploy,
        "python": defaults.REMOTE_PYTHON,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": submit.DEFAULT_SOURCE_RUN_ID,
        "remote_design_only": True,
        "run_id": "cert-budget",
        "rank": 4,
        "budgets": "20,40,80",
        "replication_caps": "5,10,20",
        "source_d": 50,
        "source_records_per_domain": 64,
        "d": 1000,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "cpu": 12,
        "ram_mb": 8192,
    })()


def test_certification_budget_screen_is_fully_sharded_and_matched(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 4 * 3 * 5
    assert sum("/new_only/" in spec["signature"] for spec in specs) == 45
    for cap in (5, 10, 20):
        selected = [
            spec for spec in specs
            if f"/joint_cap{cap}/" in spec["signature"]
        ]
        assert len(selected) == 45
        assert all(f"--replication-max-per-solution {cap}" in spec["cmd"]
                   for spec in selected)
        assert all("--decision-backend sobol_exact_joint_voi" in spec["cmd"]
                   for spec in selected)
    assert all("--decision-recommend-observed-only" in spec["cmd"]
               for spec in specs)
    assert all(spec["cpu"] == 12 and spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
