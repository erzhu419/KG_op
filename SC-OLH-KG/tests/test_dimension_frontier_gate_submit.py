from pathlib import Path
import json
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.submit_scolhkg_dimension_frontier_gate_scheduler import (  # noqa: E402
    build_specs,
)


def test_frontier_gate_is_sharded_and_uses_correct_node_families(tmp_path):
    args = SimpleNamespace(
        deploy=tmp_path,
        run_id="unit",
        archive_run_id="archive",
        heldouts="QueueResourceControl",
        backends="proposal_only,saasbo",
        dimensions="200,10000",
        seed_start=80,
        n_seeds=2,
        source_d=50,
        N=13,
        budgets="13,20",
        n0=10,
        offline_source_calls=384,
        cpu=12,
        ram_mb=16384,
        gpu_cpu=12,
        gpu_ram_mb=24576,
        vram_mb=2048,
    )
    specs = build_specs(args)
    assert len(specs) == 2 * (1 + 2 * 3)
    designs = [spec for spec in specs if "/design/" in spec["signature"]]
    proposal = [
        spec for spec in specs if "/proposal/" in spec["signature"]]
    saas = [spec for spec in specs if "/saas/" in spec["signature"]]
    assert len(designs) == 2
    assert len(proposal) == 4
    assert len(saas) == 8
    assert all(spec["allowed_nodes"] == [
        "node001", "node002", "node003",
        "node004", "node005", "node006",
    ] for spec in designs + proposal)
    assert all(spec["allowed_nodes"] == [
        "jtl110gpu", "jtl110gpu2", "node007",
    ] for spec in saas)
    assert all("--proposal-mode risk_objective_atlas" in spec["cmd"]
               for spec in designs)
    assert sum("--target-budget 13" in spec["cmd"] for spec in saas) == 4
    assert sum("--target-budget 20" in spec["cmd"] for spec in saas) == 4
    assert all("--terminal-objective-incumbent-guard" in spec["cmd"]
               for spec in proposal + saas)


def test_frontier_gate_binds_frozen_execution_snapshot(tmp_path):
    code_root = tmp_path / "snapshot"
    code_root.mkdir()
    marker = {
        "status": "frozen",
        "repository_commit": "a" * 40,
        "scolhkg_tree": "b" * 40,
        "proof_tree": "c" * 40,
        "scripts_tree": "d" * 40,
        "theory_contract_id": (
            "source_target_geometric_atlas_coverage_v1"),
        "snapshot_root": str(code_root),
    }
    (code_root / ".scolhkg_execution_snapshot.json").write_text(
        json.dumps(marker),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        deploy=tmp_path / "deploy",
        code_root=code_root,
        require_frozen_snapshot=True,
        run_id="frozen",
        archive_run_id="archive",
        heldouts="QueueResourceControl",
        backends="proposal_only,saasbo",
        dimensions="200",
        seed_start=80,
        n_seeds=1,
        source_d=50,
        N=13,
        budgets="13",
        n0=10,
        offline_source_calls=384,
        cpu=12,
        ram_mb=16384,
        gpu_cpu=12,
        gpu_ram_mb=24576,
        vram_mb=2048,
    )

    specs = build_specs(args)

    assert len(specs) == 3
    assert all(
        "SCOLHKG_EXECUTION_PROVENANCE_REQUIRED=1" in spec["cmd"]
        for spec in specs
    )
    assert all(
        str(code_root / "SC-OLH-KG") in spec["cwd"]
        for spec in specs
    )
