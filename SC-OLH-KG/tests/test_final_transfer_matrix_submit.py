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


def test_final_transfer_matrix_selects_exact_remote_missing_cells(tmp_path):
    only_cells = tmp_path / "missing.json"
    only_cells.write_text(
        """
        {
          "cells": [
            {
              "method": "stacked_transfer_gp_cbo",
              "heldout": "FactorShockStatePolicyRZDT1",
              "seed": 89
            },
            {
              "method": "hyperbo_cbo",
              "heldout": "InventorySupplyChain",
              "seed": 97
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    args = SimpleNamespace(
        deploy=tmp_path,
        run_id="unit",
        archive_run_id="archive",
        design_run_id="design",
        heldouts="FactorShockStatePolicyRZDT1,InventorySupplyChain",
        methods="stacked_transfer_gp_cbo,hyperbo_cbo",
        seed_start=80,
        n_seeds=20,
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
        only_cells_file=only_cells,
    )

    specs = build_specs(args)

    assert len(specs) == 2
    assert {
        tuple(spec["signature"].rsplit("/", 3)[-3:])
        for spec in specs
    } == {
        (
            "FactorShockStatePolicyRZDT1",
            "stacked_transfer_gp_cbo",
            "seed0089",
        ),
        ("InventorySupplyChain", "hyperbo_cbo", "seed0097"),
    }
