import math
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SC-OLH-KG"))

from performance.analyze_traffic_final_contract import (  # noqa: E402
    audit_oos_payload,
    exact_binomial_lower,
)
from scripts.submit_scolhkg_final_traffic_gate_scheduler import (  # noqa: E402
    build_specs,
)


def _candidate(index, successes, *, seed=80, trials=100):
    return {
        "method": "PaperFinal-SourceProposal-SAAS",
        "partition": f"paper_final_external_v1_seed{seed:04d}",
        "run_seed": seed,
        "source_index": index,
        "x": [index + 1] * 21,
        "validation": {
            "R": trials,
            "seeds": list(range(900000, 900000 + trials)),
            "feasible_count": successes,
            "feasible_probability": successes / trials,
            "mean": [1.0 + index, 2.0 + index, 0.8],
        },
    }


def test_exact_familywise_binomial_certificate_is_not_wilson_shortcut():
    lower = exact_binomial_lower(100, 100, 0.05 / 3.0)
    assert lower >= 0.95
    assert exact_binomial_lower(99, 100, 0.05 / 3.0) < 0.95
    assert math.isclose(lower, (0.05 / 3.0) ** (1.0 / 100.0))


def test_traffic_audit_deploys_first_certified_frozen_rank():
    payload = {
        "candidates": [
            _candidate(0, 99),
            _candidate(1, 100),
            _candidate(2, 100),
        ],
    }
    audit = audit_oos_payload(payload)
    assert audit["deployed_source_index"] == 1
    assert audit["deployed_certified"]
    assert audit["verification_calls"] == 300
    assert audit["verification_samples_used_to_reorder_shortlist"] is False


def test_traffic_submitter_uses_cpu_nodes_and_keeps_checkpoints_remote(tmp_path):
    args = SimpleNamespace(
        deploy=tmp_path,
        run_id="unit",
        archive_run_id="archive",
        source_d=50,
        seed_start=80,
        n_seeds=2,
        N=13,
        n0=10,
        R=100,
        verification_seed_start=900000,
        cpu=12,
        ram_mb=24576,
    )
    specs = build_specs(args)
    assert len(specs) == 6
    expected_nodes = [
        "node001", "node002", "node003",
        "node004", "node005", "node006",
    ]
    assert all(spec["allowed_nodes"] == expected_nodes for spec in specs)
    assert all(spec["vram"] == 0 for spec in specs)
    assert all("jtl311linux" not in str(spec) for spec in specs)
    search = [spec for spec in specs if "/search/" in spec["signature"]]
    oos = [spec for spec in specs if "/oos/" in spec["signature"]]
    assert len(search) == 2
    assert len(oos) == 2
    assert all("--torch-device cpu" in spec["cmd"] for spec in search)
    assert all(
        "--historical-anchor" not in spec["cmd"] for spec in search)
    assert all("--R 100" in spec["cmd"] for spec in oos)
    assert all("--source-indexes 0,1,2" in spec["cmd"] for spec in oos)
    assert all(
        "checkpoints" in spec["stage_excludes"] for spec in specs)
