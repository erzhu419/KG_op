import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = _load(
    "promoted_v51_voi_fidelity_submit",
    REPO / "scripts/submit_scolhkg_promoted_v51_voi_fidelity_gate_scheduler.py",
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
        "run_id": "voi-fidelity",
        "rank": 4,
        "mc_samples": "2,8,32",
        "shortlist_sizes": "1,4,8,32",
        "source_d": 50,
        "source_records_per_domain": 64,
        "d": 1000,
        "N": 11,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "pool_size": 512,
        "variance_audit_size": 128,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "cpu": 12,
        "ram_mb": 8192,
    })()


def test_voi_fidelity_gate_uses_one_common_state_per_variant(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 4 * 3 * 5
    assert all("--N 11 --n0 10" in spec["cmd"] for spec in specs)
    assert all("--exact-sampling-mode antithetic_nested" in spec["cmd"]
               for spec in specs)
    assert all("--decision-recommend-observed-only" in spec["cmd"]
               for spec in specs)
    assert all(spec["cpu"] == 12 and spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    for mc in (2, 8, 32):
        for shortlist in (1, 4, 8, 32):
            selected = [spec for spec in specs if (
                f"/mc{mc}_k{shortlist}/" in spec["signature"])]
            assert len(selected) == 15
            assert all(f"--exact-mc-samples {mc}" in spec["cmd"]
                       for spec in selected)
            assert all(
                f"--evaluate-or-replicate-new-action-count {shortlist}"
                in spec["cmd"] for spec in selected)
