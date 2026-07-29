import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.submit_scolhkg_source_hvd_saas_gate_scheduler import (  # noqa: E402
    METHOD_CONTRACT_ID,
    build_specs,
)


def test_source_hvd_submitter_pairs_only_the_aleatoric_head(tmp_path):
    args = SimpleNamespace(
        deploy=tmp_path,
        run_id="unit",
        archive_run_id="archive",
        design_run_id="design",
        heldouts="QueueResourceControl",
        modes="pooled,cumulative_factor",
        seed_start=80,
        n_seeds=2,
        source_d=50,
        d=1000,
        N=13,
        n0=10,
        offline_source_calls=384,
        audit_size=64,
        cpu=12,
        ram_mb=24576,
        vram_mb=2048,
        code_root=None,
        require_frozen_snapshot=False,
    )
    specs = build_specs(args)
    assert len(specs) == 4
    assert all(spec["allowed_nodes"] == [
        "node001", "node002", "node003",
        "node004", "node005", "node006",
    ] for spec in specs)
    assert all(spec["cpu"] == 12 for spec in specs)
    assert all(spec["vram"] == 0 for spec in specs)
    assert all("--torch-device cpu" in spec["cmd"] for spec in specs)
    assert all(spec["allow_cpu_training"] is True for spec in specs)
    assert all("--protocol shared_archive_hvd_n13" in spec["cmd"]
               for spec in specs)
    assert sum("--aleatoric-head-mode pooled" in spec["cmd"]
               for spec in specs) == 2
    assert sum("--aleatoric-head-mode cumulative_factor" in spec["cmd"]
               for spec in specs) == 2
    assert all("--target-budget 13" in spec["cmd"] for spec in specs)
    assert all("--terminal-objective-incumbent-guard" in spec["cmd"]
               for spec in specs)


def test_source_hvd_submitter_uses_immutable_code_snapshot(tmp_path):
    deploy = tmp_path / "deploy"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    marker = {
        "status": "frozen",
        "repository_commit": "a" * 40,
        "scolhkg_tree": "b" * 40,
        "proof_tree": "c" * 40,
        "scripts_tree": "d" * 40,
        "method_contract_id": "unused_snapshot_default",
        "theory_contract_id": "source_target_geometric_atlas_coverage_v1",
        "snapshot_root": str(snapshot.resolve()),
    }
    (snapshot / ".scolhkg_execution_snapshot.json").write_text(
        json.dumps(marker),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        deploy=deploy,
        code_root=snapshot,
        require_frozen_snapshot=True,
        run_id="frozen",
        archive_run_id="archive",
        design_run_id="design",
        heldouts="QueueResourceControl",
        modes="pooled,cumulative_factor",
        seed_start=80,
        n_seeds=1,
        source_d=50,
        d=1000,
        N=13,
        n0=10,
        offline_source_calls=384,
        audit_size=64,
        cpu=12,
        ram_mb=24576,
        vram_mb=2048,
    )
    specs = build_specs(args)
    assert len(specs) == 2
    assert all(
        spec["cwd"] == str(snapshot / "SC-OLH-KG")
        for spec in specs
    )
    assert all(
        "SCOLHKG_EXECUTION_PROVENANCE_REQUIRED=1" in spec["cmd"]
        for spec in specs
    )
    assert all(
        f"SCOLHKG_METHOD_CONTRACT_ID={METHOD_CONTRACT_ID}" in spec["cmd"]
        for spec in specs
    )
    assert all(
        str(deploy / "SC-OLH-KG" / "profiles") in spec["cmd"]
        for spec in specs
    )
