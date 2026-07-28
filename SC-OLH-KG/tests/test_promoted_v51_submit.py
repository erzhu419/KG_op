import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = _load(
    "promoted_v51_submit",
    REPO / "scripts/submit_scolhkg_promoted_v51_scheduler.py",
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
        "run_id": "promoted-v51",
        "rank": 4,
        "source_d": 50,
        "source_records_per_domain": 64,
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


def test_promoted_v51_is_the_balanced_four_profile(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 20
    assert all("/promoted_v51/" in spec["signature"] for spec in specs)
    assert all("--decision-backend sobol_exact_joint_voi" in spec["cmd"]
               for spec in specs)
    assert all("--evaluate-or-replicate-new-action-count 4" in spec["cmd"]
               for spec in specs)
    assert all(
        "--evaluate-or-replicate-new-action-policy "
        "canonical_plus_posterior_risk" in spec["cmd"]
        for spec in specs
    )
    assert all("--adaptive-replication-voi" in spec["cmd"]
               for spec in specs)
    assert all(spec["wait_for_files"] == [] for spec in specs)


def test_promoted_baseline_record_matches_observed_terminal_closure():
    record = json.loads((
        REPO / "SC-OLH-KG/performance/promoted_baseline.json"
    ).read_text())
    assert record["name"] == "v51_observed_terminal_closure"
    assert record["profile_key"] == "observed_terminal_closure"
    assert record["submission_entrypoint"] == (
        "scripts/submit_scolhkg_promoted_v51_closure_gate_scheduler.py"
    )
    assert record["decision"]["new_action_count"] == 4
    assert record["decision"]["terminal_action_universe"] == (
        "charged_observed_policies"
    )
    assert record["decision"]["current_fantasy_final_contract_shared"] is True
    assert record["evidence"]["true_feasible"] == 60
    assert record["evidence"]["adaptive_losses"] == 0
    assert record["evidence"]["contracts"][
        "observed_terminal_closure"
    ] is True
