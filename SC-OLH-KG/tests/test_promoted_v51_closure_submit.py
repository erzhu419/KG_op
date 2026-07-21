import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = _load(
    "promoted_v51_closure_submit",
    REPO / "scripts/submit_scolhkg_promoted_v51_closure_gate_scheduler.py",
)


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "deploy": deploy,
        "python": defaults.REMOTE_PYTHON,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": submit.DEFAULT_SOURCE_RUN_ID,
        "remote_design_only": True,
        "run_id": "promoted-v51-closure",
        "rank": 4,
        "source_d": 50,
        "source_records_per_domain": 64,
        "theory_contract_id": "v51_statistical_closure_v2",
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 20,
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


def test_closure_gate_replays_the_promoted_profile_on_all_seeds(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 20
    assert all("/promoted_v51_closure_sequential/" in spec["signature"]
               for spec in specs)
    assert all("/observed_terminal_closure/" in spec["signature"]
               for spec in specs)
    assert all("--decision-backend sobol_exact_joint_voi" in spec["cmd"]
               for spec in specs)
    assert all("--decision-recommend-observed-only" in spec["cmd"]
               for spec in specs)
    assert all("--finalist-replication-budget 0" in spec["cmd"]
               for spec in specs)
    assert all("--certification-recheck-top-k 0" in spec["cmd"]
               for spec in specs)
    assert all(
        "--implementation-contract-id "
        "promoted_v51_observed_terminal_closure" in spec["cmd"]
        for spec in specs
    )
    assert all(
        "--theory-contract-id v51_statistical_closure_v2" in spec["cmd"]
        for spec in specs
    )
    assert all(spec["cpu"] == 12 and spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
