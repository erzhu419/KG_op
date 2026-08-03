from pathlib import Path
from types import SimpleNamespace
import json

from scripts.submit_scolhkg_traffic_universal_library_diagnostic import (
    CPU_NODES,
    build_specs,
)


def test_posthoc_submitter_is_cpu_only_and_diagnostic(tmp_path):
    code_root = tmp_path / "snapshot"
    code_root.mkdir()
    snapshot = {
        "status": "frozen",
        "repository_commit": "a" * 40,
        "scolhkg_tree": "b" * 40,
        "proof_tree": "c" * 40,
        "scripts_tree": "d" * 40,
        "legacy_traffic_tree": "e" * 40,
        "traffic_decision_space_blob": "f" * 40,
        "traffic_baseline_blob": "1" * 40,
        "method_contract_id": "posthoc",
        "theory_contract_id": "coverage",
        "snapshot_root": str(code_root),
    }
    (code_root / ".scolhkg_execution_snapshot.json").write_text(
        json.dumps(snapshot), encoding="utf-8")
    args = SimpleNamespace(
        code_root=code_root,
        deploy=tmp_path / "deploy",
        archive_run_id="archive",
        source_split_heldout="FactorShockStatePolicyRZDT1",
        run_id="diagnostic",
        source_d=50,
        expected_library_size=111,
        R=200,
        verification_seed_start=1200000,
        num_shards=3,
        cpu=12,
        ram_mb=24576,
    )
    specs, observed = build_specs(args)
    assert observed == snapshot
    assert len(specs) == 5
    assert all(spec["vram"] == 0 for spec in specs)
    assert all(set(spec["allowed_nodes"]) == set(CPU_NODES) for spec in specs)
    assert not any("saas" in spec["cmd"].lower() for spec in specs)
    assert sum("--num-shards 3" in spec["cmd"] for spec in specs) == 3
    assert "--redact-policy-vectors" in specs[-1]["cmd"]
