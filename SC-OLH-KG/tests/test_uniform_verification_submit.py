from types import SimpleNamespace

from scripts.submit_scolhkg_uniform_verifier_scheduler import build_specs


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
