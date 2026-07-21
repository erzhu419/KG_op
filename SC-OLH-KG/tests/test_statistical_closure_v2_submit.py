from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_statistical_closure_v2_audit_scheduler.py"
SPEC = importlib.util.spec_from_file_location("statistical_closure_submit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _registration():
    return {
        "schema_version": 1,
        "status": "frozen",
        "contracts": {
            "implementation_contract_id": MODULE.IMPLEMENTATION_CONTRACT_ID,
            "theory_contract_id": MODULE.THEORY_CONTRACT_ID,
        },
        "profile": {
            "replication_cap": 5,
            "exact_mc_samples": 2,
            "exact_shortlist_size": 4,
            "exact_sampling_mode": "antithetic",
        },
    }


def _args(tmp_path):
    defaults = MODULE.closure._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(MODULE.CPU_NODES),
        "deploy": deploy,
        "python": defaults.REMOTE_PYTHON,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": MODULE.DEFAULT_SOURCE_RUN_ID,
        "remote_design_only": True,
        "run_id": MODULE.DEFAULT_RUN_ID,
        "rank": 4,
        "source_d": 50,
        "source_records_per_domain": 64,
        "d": 1000,
        "N": 20,
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


def test_v2_audit_is_separate_and_preserves_promoted_v51_behavior(tmp_path):
    specs = MODULE.build_specs(_args(tmp_path), _registration())
    assert len(specs) == 3 * 5
    assert all("/statistical_closure_v2_audit_sequential/" in spec["signature"]
               for spec in specs)
    assert all("/statistical_closure_v2/" in spec["signature"]
               for spec in specs)
    assert all("--exact-mc-samples 2" in spec["cmd"] for spec in specs)
    assert all("--exact-sampling-mode antithetic" in spec["cmd"]
               for spec in specs)
    assert all("--evaluate-or-replicate-new-action-count 4" in spec["cmd"]
               for spec in specs)
    assert all("--replication-max-per-solution 5" in spec["cmd"]
               for spec in specs)
    assert all(
        "--implementation-contract-id "
        "promoted_v51_observed_terminal_closure" in spec["cmd"]
        for spec in specs
    )
    assert all("--theory-contract-id v51_statistical_closure_v2" in spec["cmd"]
               for spec in specs)
    assert all(spec["theory_audit_contract"]["exact_mc_samples"] == 2
               for spec in specs)
    assert all(spec["allowed_nodes"] == list(MODULE.CPU_NODES)
               for spec in specs)
    assert all(spec["cpu"] == 12 and spec["vram"] == 0 for spec in specs)
