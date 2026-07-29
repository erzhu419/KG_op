from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.submit_scolhkg_source_hvd_saas_gate_scheduler import (  # noqa: E402
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
    )
    specs = build_specs(args)
    assert len(specs) == 4
    assert all(spec["allowed_nodes"] == [
        "jtl110gpu", "jtl110gpu2", "node007"] for spec in specs)
    assert all(spec["cpu"] == 12 for spec in specs)
    assert all("--protocol shared_archive_hvd_n13" in spec["cmd"]
               for spec in specs)
    assert sum("--aleatoric-head-mode pooled" in spec["cmd"]
               for spec in specs) == 2
    assert sum("--aleatoric-head-mode cumulative_factor" in spec["cmd"]
               for spec in specs) == 2
    assert all("--target-budget 13" in spec["cmd"] for spec in specs)
    assert all("--terminal-objective-incumbent-guard" in spec["cmd"]
               for spec in specs)

