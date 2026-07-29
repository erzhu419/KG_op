from pathlib import Path
import json
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.submit_scolhkg_final_paper_bridge_scheduler import (  # noqa: E402
    build_specs,
)


def test_final_paper_bridge_is_final_contract_and_gpu_sharded(tmp_path):
    args = SimpleNamespace(
        deploy=tmp_path,
        run_id="unit",
        archive_run_id="archive",
        source_d=50,
        d=5,
        seed_start=80,
        n_seeds=2,
        N=13,
        n0=10,
        cpu=12,
        ram_mb=16384,
        gpu_cpu=12,
        gpu_ram_mb=24576,
        vram_mb=2048,
    )
    specs = build_specs(args)
    assert len(specs) == 7
    design = specs[0]
    runs = specs[1:]
    assert design["allowed_nodes"] == [
        "node001", "node002", "node003",
        "node004", "node005", "node006",
    ]
    assert all(spec["allowed_nodes"] == [
        "jtl110gpu", "jtl110gpu2", "node007",
    ] for spec in runs)
    assert all("jtl311linux" not in str(spec) for spec in specs)
    assert all(
        "--offline-source-calls-override 384" in spec["cmd"]
        for spec in runs
    )
    assert all("--target-budget 13" in spec["cmd"] for spec in runs)
    assert all("--saas-refit-schedule every_iteration" in spec["cmd"]
               for spec in runs)
    assert all("--terminal-objective-incumbent-guard" in spec["cmd"]
               for spec in runs)
    assert all(
        "--no-terminal-safe-interior-require-provider" in spec["cmd"]
        for spec in runs
    )
    assert all("checkpoints" in spec["stage_excludes"] for spec in specs)


def test_final_paper_bridge_binds_frozen_execution_snapshot(tmp_path):
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
        source_d=50,
        d=5,
        seed_start=80,
        n_seeds=1,
        N=13,
        n0=10,
        cpu=12,
        ram_mb=16384,
        gpu_cpu=12,
        gpu_ram_mb=24576,
        vram_mb=2048,
    )

    specs = build_specs(args)

    assert len(specs) == 4
    assert all(
        "SCOLHKG_EXECUTION_PROVENANCE_REQUIRED=1" in spec["cmd"]
        for spec in specs
    )
    assert all(
        str(code_root / "SC-OLH-KG") in spec["cwd"]
        for spec in specs
    )
