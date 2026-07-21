from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_paper_main_matrix_scheduler.py"
SPEC = importlib.util.spec_from_file_location("paper_main_submit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _registration(status="frozen"):
    return {
        "schema_version": 1,
        "status": status,
        "contracts": {
            "implementation_contract_id": MODULE.IMPLEMENTATION_CONTRACT_ID,
            "theory_contract_id": MODULE.THEORY_CONTRACT_ID,
        },
        "freeze_evidence": {
            "replication_cap": 5,
            "exact_mc_samples": 8,
            "exact_shortlist_size": 4,
            "exact_sampling_mode": "antithetic_nested",
            "d1000_frontier_passed": True,
        },
        "source_contract": {
            "archive_calls": 384,
            "source_dimension": 50,
            "records_per_source_domain": 64,
            "replicates_per_record": 3,
        },
        "synthetic_domains": [
            "FactorShockStatePolicyRZDT1",
            "InventorySupplyChain",
            "QueueResourceControl",
        ],
        "frontier": [{"d": 200, "budgets": [20], "seeds": 2}],
        "main_variants": list(MODULE.VARIANTS),
    }


def _args(tmp_path, **updates):
    deploy = tmp_path / "deploy"
    values = {
        "scheduler": Path("scheduler.py"),
        "deploy": deploy,
        "python": MODULE.closure._root_module().REMOTE_PYTHON,
        "source_manifest": (
            deploy / "SC-OLH-KG/performance/manifests/source.json"),
        "archive_run_id": MODULE.designs.DEFAULT_ARCHIVE_RUN_ID,
        "run_id": "paper-test",
        "nodes": ",".join(MODULE.CPU_NODES),
        "variants": "promoted_joint_voi,pooled_variance,frozen_source_discrepancy",
        "dims": "",
        "budgets": "",
        "n0": 10,
        "rank": 4,
        "pool_size": 64,
        "variance_audit_size": 64,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "cpu": 12,
        "ram_mb": 8192,
    }
    values.update(updates)
    return type("Args", (), values)()


def test_awaiting_registration_cannot_submit():
    with pytest.raises(RuntimeError, match="not frozen"):
        MODULE.validate_freeze(_registration(status="awaiting_gate_b"))


def test_inspection_counts_frozen_registered_matrix():
    registration = json.loads(MODULE.DEFAULT_REGISTRATION.read_text())
    plan = MODULE.inspect_plan(registration)
    assert plan["status"] == "frozen"
    assert plan["design_tasks"] == 9
    assert plan["run_tasks"] == 2520
    assert plan["total_tasks"] == 2529


def test_frozen_matrix_builds_design_dependencies_and_exact_profiles(tmp_path):
    specs = MODULE.build_specs(_args(tmp_path), _registration())
    assert len(specs) == 3 + (3 * 2 * 3)
    assert len({spec["signature"] for spec in specs}) == len(specs)
    assert all(set(spec["allowed_nodes"]).issubset(set(MODULE.CPU_NODES))
               for spec in specs)
    run_specs = [spec for spec in specs if "/paper_main_v1_" in spec["signature"]]
    promoted = next(spec for spec in run_specs
                    if "/promoted_joint_voi/" in spec["signature"])
    pooled = next(spec for spec in run_specs
                  if "/pooled_variance/" in spec["signature"])
    frozen = next(spec for spec in run_specs
                  if "/frozen_source_discrepancy/" in spec["signature"])
    assert "--exact-mc-samples 8" in promoted["cmd"]
    assert "--exact-sampling-mode antithetic_nested" in promoted["cmd"]
    assert "--evaluate-or-replicate-new-action-count 4" in promoted["cmd"]
    assert "--replication-max-per-solution 5" in promoted["cmd"]
    assert (
        "--implementation-contract-id "
        "promoted_v51_observed_terminal_closure" in promoted["cmd"]
    )
    assert (
        "--theory-contract-id v51_statistical_closure_v2"
        in promoted["cmd"]
    )
    assert "--hvd-profile pooled" in pooled["cmd"]
    assert "--no-source-discrepancy-update" in frozen["cmd"]
    assert promoted["wait_for_files"]
    assert "checkpoints" in promoted["stage_excludes"]
