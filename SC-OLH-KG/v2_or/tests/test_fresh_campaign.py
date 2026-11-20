from __future__ import annotations

import json

import numpy as np
import pytest

from v2_or.contracts import BaseDynamicORProblem
from v2_or.dispatch import campaign_runner as runner


class SubstituteProblem(BaseDynamicORProblem):
    def __init__(self, *, fail=False):
        super().__init__()
        self.raw_calls = 0
        self.fail = fail

    def nominal_exposure(self, x_norm):
        raise AssertionError("No target search should execute in the wiring test")

    def _simulate_raw(self, x_norm, reps, rng, split="search"):
        self.raw_calls += reps
        if self.fail:
            raise ValueError("substitute failed")
        return np.zeros((reps, 2))


def _inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SC_ROOT", tmp_path)
    path = tmp_path / "manifest.json"
    manifest = {
        "campaign_name": "fresh",
        "output_dir": "paper_artifacts/fresh",
        "reuse_source_campaign": "u3",
        "reuse_source_execution_code_sha256": "u3-code",
        "families": [{
            "name": "inventory", "kind": "inventory", "target_config": {"n_tasks": 1},
            "target_eval_seeds": [12, 13], "online_step_budgets": [60],
            "n0_init": 8, "batch_reps": 2, "target_call_cap": 516, "audit_replications": 8,
            "arms": [{"name": "KG", "kind": "tls_olh_kg"},
                     {"name": "TS", "kind": "tls_olh_kg"},
                     {"name": "PublicPsi", "kind": "target_only_psi"}],
        }],
    }
    runner._write_json(path, manifest)
    return path, manifest


def test_common_budget_and_anchor_barriers(tmp_path, monkeypatch):
    path, manifest = _inputs(tmp_path, monkeypatch)
    historical = tmp_path / "paper_artifacts/u3"
    bundle = historical / "source_bundles/inventory"
    bundle.mkdir(parents=True)
    archive = b"[]"
    (bundle / "archive.json").write_bytes(archive)
    (bundle / "prior_default.pkl").touch()
    runner._write_json(historical / "stages/source_archive/completion.json", {
        "status": "complete", "execution_code_sha256": "u3-code",
        "prior_diagnostics": {"inventory": {
            "archive_sha256": runner._sha256_bytes(archive), "variants": {"default": {}},
        }},
    })

    stages = runner._campaign_dir(manifest) / "stages"
    assert runner.describe_stage(path, "source_archive")["n_jobs"] == 0
    source = runner.finish_stage(stages / "source_archive/jobs.json")
    assert (source["source_simulation_calls_new"], source["source_refits_new"]) == (0, 0)
    desc = runner.describe_stage(path, "search_and_verify")
    assert desc["n_jobs"] == 6
    assert runner._target_call_limits(manifest["families"][0], 60, 8, 2) == {"search": 136, "verification": 380}
    with pytest.raises(RuntimeError, match="search_and_verify"):
        runner.describe_stage(path, "audit_anchors")
    for job in desc["jobs"]:
        runner._write_json(stages / "search_and_verify/items" / f"{job['job_id']}.json", {})
    runner.finish_stage(stages / "search_and_verify/jobs.json")
    anchors = runner.describe_stage(path, "audit_anchors")
    assert anchors["n_jobs"] == 1
    with pytest.raises(RuntimeError, match="audit_anchors"):
        runner.describe_stage(path, "audit_and_summarize")
    problem = SubstituteProblem()
    monkeypatch.setattr(runner, "build_family_tasks", lambda *_: [problem])
    for job in anchors["jobs"]:
        args = path.read_text(), anchors["manifest_sha256"], anchors["execution_code_sha256"], "audit_anchors", job
        assert runner._worker_entry(args)["status"] == "ok"
        assert runner._worker_entry(args)["status"] == "cached"
    anchor_completion = runner.finish_stage(stages / "audit_anchors/jobs.json")
    assert anchor_completion["anchor_audit_calls_used"] == problem.raw_calls == 8
    assert runner.describe_stage(path, "audit_and_summarize")["n_jobs"] == 6

    changed = json.loads(path.read_text())
    changed["families"][0]["target_call_cap"] = 517
    runner._write_json(path, changed)
    with pytest.raises(RuntimeError, match="differs from the frozen stage"):
        runner.describe_stage(path, "search_and_verify")


def test_attempt_fee_retained_with_no_reset(tmp_path, monkeypatch):
    path, manifest = _inputs(tmp_path, monkeypatch)
    stage = "audit_anchors"
    problem = SubstituteProblem(fail=True)
    monkeypatch.setattr(runner, "build_family_tasks", lambda *_: [problem])
    args = path.read_text(), "fresh-manifest", "fresh-code", stage, {
        "job_id": "anchor_inventory_t00", "family": "inventory", "target_task_idx": 0,
    }
    assert runner._worker_entry(args)["status"] == "failed"
    assert "Previous paid" in runner._worker_entry(args)["error"]
    assert problem.raw_calls == 8
    assert problem.audit_calls == 0
    fees = runner._campaign_fees(manifest)[stage]
    assert fees["attempt_status"] == {"failed": 1}
    assert fees["attempted_calls_recorded"] == {"audit": 8}
    with pytest.raises(RuntimeError, match="exceeds audit cap"):
        problem.simulate_replications(np.asarray([0.5]), 1, np.random.default_rng(1), split="audit")
    assert problem.raw_calls == 8
