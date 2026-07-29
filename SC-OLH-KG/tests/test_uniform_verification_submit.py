import json
from types import SimpleNamespace

import pytest

from scripts.submit_scolhkg_uniform_verifier_scheduler import (
    build_specs,
    load_or_materialize_manifest,
)


def test_uniform_verifier_submitter_shards_on_cpu_nodes(tmp_path):
    args = SimpleNamespace(
        deploy=tmp_path,
        run_id="unit",
        shard_size=8,
        cpu=8,
        ram_mb=8192,
    )
    manifest = {"row_count": 17}
    specs = build_specs(args, manifest)
    assert len(specs) == 3
    assert all(spec["vram"] == 0 for spec in specs)
    assert all(spec["cpu"] == 8 for spec in specs)
    assert all(spec["allowed_nodes"] == [
        "node001", "node002", "node003",
        "node004", "node005", "node006",
    ] for spec in specs)
    assert all("jtl110cpu" not in str(spec) for spec in specs)
    assert all("jtl311linux" not in str(spec) for spec in specs)
    assert all("checkpoints" in spec["stage_excludes"] for spec in specs)


def test_uniform_recovery_reuses_byte_identical_frozen_manifest(tmp_path):
    manifest = {
        "contract_id": "frozen",
        "row_count": 17,
        "selection_counts": {"method": 17},
    }
    path = (
        tmp_path / "SC-OLH-KG" / "archives" / "unit"
        / "uniform_verification_manifest.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(
        deploy=tmp_path,
        run_id="unit",
        reuse_existing_manifest=True,
        audit="",
        selection=[],
        candidate_budget=128,
        familywise_delta=0.05,
    )

    loaded = load_or_materialize_manifest(args)

    assert loaded == manifest


def test_uniform_submitter_requires_source_audit_outside_recovery(tmp_path):
    args = SimpleNamespace(
        deploy=tmp_path,
        run_id="unit",
        reuse_existing_manifest=False,
        audit="",
        selection=[],
        candidate_budget=128,
        familywise_delta=0.05,
    )
    with pytest.raises(ValueError, match="audit"):
        load_or_materialize_manifest(args)
