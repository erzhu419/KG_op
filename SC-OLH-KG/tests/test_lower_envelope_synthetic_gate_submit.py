import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    ROOT / "scripts/submit_scolhkg_lower_envelope_synthetic_gate_scheduler.py")
SPEC = importlib.util.spec_from_file_location(
    "lower_envelope_synthetic_submit", SCRIPT_PATH)
SUBMIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMIT)


def _args(tmp_path):
    deploy = tmp_path / "deploy"
    code_root = tmp_path / "snapshot"
    code_root.mkdir(parents=True)
    marker = {
        "repository_commit": "a" * 40,
        "scolhkg_tree": "b" * 40,
        "proof_tree": "c" * 40,
        "scripts_tree": "d" * 40,
        "method_contract_id": "lower_envelope_synthetic_paired_gate_v1",
        "theory_contract_id": "reserved_sentinel_atlas_coverage_v2",
        "snapshot_root": str(code_root),
        "status": "frozen",
    }
    (code_root / SUBMIT.SNAPSHOT_MARKER).write_text(
        json.dumps(marker), encoding="utf-8")
    archive = (
        deploy / "SC-OLH-KG/archives/archive-run/"
        "FactorShockStatePolicyRZDT1/"
        "heldout_FactorShockStatePolicyRZDT1.json"
    )
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps({
        "fingerprint": "archive-fingerprint",
        "tasks": [{
            "X": [[0.0] * 50],
            "Y_replicates": [[0.0] * 384],
        }],
    }), encoding="utf-8")
    return SimpleNamespace(
        scheduler=tmp_path / "scheduler.py",
        deploy=deploy,
        code_root=code_root,
        require_frozen_snapshot=True,
        run_id="paired-lower-envelope-test",
        archive_run_id="archive-run",
        heldouts="FactorShockStatePolicyRZDT1",
        frontends="v1,lower_envelope_v2",
        backends="proposal_only,botorch_scbo",
        seed_start=80,
        n_seeds=2,
        source_d=50,
        d=1000,
        N=13,
        n0=10,
        offline_source_calls=384,
        raw_samples=64,
        num_restarts=2,
        maxiter=20,
        ts_candidates=128,
        candidate_timeout_sec=60.0,
        cpu=12,
        ram_mb=24576,
        dispatch=False,
        dry_run=True,
    )


def test_lower_envelope_gate_is_paired_cpu_only_and_budget_matched(tmp_path):
    args = _args(tmp_path)
    audit = SUBMIT.validate_archives(args)
    specs = SUBMIT.build_specs(args)

    assert audit["FactorShockStatePolicyRZDT1"]["source_calls"] == 384
    assert len(specs) == 10
    assert all(spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == list(SUBMIT.CPU_NODES)
               for spec in specs)
    assert all("saas" not in spec["cmd"].lower() for spec in specs)
    designs = [spec for spec in specs if "/design/" in spec["signature"]]
    assert len(designs) == 2
    v1 = next(spec for spec in designs if "/v1/" in spec["signature"])
    v2 = next(
        spec for spec in designs
        if "/lower_envelope_v2/" in spec["signature"])
    assert "--protect-lower-envelope-sentinel" not in v1["cmd"]
    assert "--protect-lower-envelope-sentinel" in v2["cmd"]
    assert all(
        "SCOLHKG_METHOD_CONTRACT_ID="
        "lower_envelope_synthetic_paired_gate_v1" in spec["cmd"]
        for spec in specs
    )
    scbo = [spec for spec in specs if "/scbo/" in spec["signature"]]
    assert len(scbo) == 4
    assert all("--target-budget 13" in spec["cmd"] for spec in scbo)
    assert all("--n0 10" in spec["cmd"] for spec in specs if "/design/" not in spec["signature"])
