from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.submit_scolhkg_final_transfer_matrix_scheduler import (  # noqa: E402
    METHODS,
    build_specs,
)


def test_final_transfer_matrix_uses_risk_atlas_v69_and_cpu_nodes(tmp_path):
    args = SimpleNamespace(
        deploy=tmp_path,
        run_id="unit",
        archive_run_id="archive",
        design_run_id="design",
        heldouts="QueueResourceControl",
        methods=",".join(METHODS),
        seed_start=80,
        n_seeds=2,
        source_d=50,
        d=1000,
        N=13,
        n0=10,
        offline_source_calls=384,
        source_train_steps=2048,
        target_finetune_steps=100,
        cpu=12,
        ram_mb=16384,
        skip_existing_success=False,
    )
    specs = build_specs(args)
    assert len(specs) == 2 * len(METHODS)
    assert all(spec["allowed_nodes"] == [
        "node001", "node002", "node003",
        "node004", "node005", "node006",
    ] for spec in specs)
    assert all("--initial-design source_informed" in spec["cmd"]
               for spec in specs)
    assert all("--source-dimension-adapter ordered_dct_quadratic"
               in spec["cmd"] for spec in specs)
    assert all(
        "--terminal-verification-candidate-budgets 80,128,128"
        in spec["cmd"] for spec in specs)
    assert all("--terminal-objective-incumbent-guard" in spec["cmd"]
               for spec in specs)

