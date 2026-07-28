from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_v52_safeguarded_closure_gate_scheduler.py"
SPEC = importlib.util.spec_from_file_location("v52_submit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


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
        "run_id": "v52-test",
        "rank": 4,
        "source_d": 50,
        "source_records_per_domain": 64,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 2,
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "variants": ",".join(MODULE.VARIANTS),
        "exact_mc_samples": 2,
        "exact_sampling_mode": "antithetic_nested",
        "action_eta": 0.01,
        "rollout_depth": 2,
        "rollout_max_arms": 4,
        "rollout_mc_samples": 2,
        "rollout_eta": 0.02,
        "cpu": 12,
        "ram_mb": 8192,
    })()


def test_v52_gate_is_paired_and_keeps_literal_v51_control(tmp_path):
    specs = MODULE.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 3 * 2
    by_variant = {
        variant: [spec for spec in specs if f"/{variant}/" in spec["signature"]]
        for variant in MODULE.VARIANTS
    }
    assert all(len(rows) == 6 for rows in by_variant.values())

    control = by_variant[MODULE.CONTROL]
    assert all("--policy-improvement-mode off" in spec["cmd"]
               for spec in control)
    assert all("--evaluate-or-replicate-new-action-count 4" in spec["cmd"]
               for spec in control)
    assert all(
        "--implementation-contract-id "
        "promoted_v51_observed_terminal_closure" in spec["cmd"]
        for spec in control
    )

    superset = by_variant["action_superset"]
    assert all("--policy-improvement-mode action_superset" in spec["cmd"]
               for spec in superset)
    assert all("--evaluate-or-replicate-new-action-count 6" in spec["cmd"]
               for spec in superset)
    assert all(
        "canonical_plus_posterior_risk_certificate_coverage" in spec["cmd"]
        for spec in superset
    )

    rollout = by_variant["guarded_rollout"]
    assert all("--policy-improvement-rollout-depth 2" in spec["cmd"]
               for spec in rollout)
    assert all("--policy-improvement-rollout-mc-error-bound 0.02"
               in spec["cmd"] for spec in rollout)

    challengers = [
        spec for variant in MODULE.CHALLENGERS for spec in by_variant[variant]
    ]
    assert all(
        "--implementation-contract-id v52_safeguarded_policy_improvement"
        in spec["cmd"] for spec in challengers
    )
    assert all("--theory-contract-id v52_safeguarded_closure_v1"
               in spec["cmd"] for spec in challengers)
    assert all(spec["allowed_nodes"] == list(MODULE.CPU_NODES)
               for spec in specs)
    assert all(spec["cpu"] == 12 and spec["vram"] == 0 for spec in specs)
