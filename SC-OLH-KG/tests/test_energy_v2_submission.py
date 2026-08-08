import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/submit_scolhkg_energy_v2_scheduler.py"
SPEC = importlib.util.spec_from_file_location("energy_v2_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


class Args:
    deploy = Path("/deploy")
    python = submit.REMOTE_PYTHON
    design_workers = 18
    design_ram_mb = 16384
    target_shards = 6
    target_workers = 75
    target_ram_mb = 32768


def _registration():
    return {
        "contract_id": "or_review_confirmatory_execution_v1",
        "method_freeze_commit": "abc123",
    }


def test_energy_design_uses_absolute_python_and_region_holdout_runner():
    spec = submit.build_design_spec(Args, _registration())
    assert str(Args.python) in spec["cmd"]
    assert " python3 " not in f" {spec['cmd']} "
    assert "--phase designs" in spec["cmd"]
    assert spec["cpu"] == 18
    assert spec["allowed_nodes"] == list(submit.CPU_NODES)


def test_energy_target_shards_are_complete_and_wait_for_frozen_designs(
    tmp_path,
):
    args = type("TempArgs", (), {
        **{
            name: getattr(Args, name)
            for name in (
                "python", "design_workers", "design_ram_mb",
                "target_shards", "target_workers", "target_ram_mb",
            )
        },
        "deploy": tmp_path,
    })()
    design_dir = tmp_path / submit.DESIGN_RELATIVE
    design_dir.mkdir(parents=True)
    (design_dir / "design_matrix.summary.json").write_text(json.dumps({
        "status": "complete",
        "freeze_commit": "abc123",
        "market_count": 18,
    }), encoding="utf-8")
    for index in range(18):
        (design_dir / f"source_atlas__target-market{index}.json").write_text(
            "{}", encoding="utf-8")
    specs = submit.build_target_specs(args, _registration())
    assert len(specs) == 6
    bounds = sorted(tuple(map(
        int, spec["signature"].rsplit("/", 1)[-1].split("-")
    )) for spec in specs)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == submit.TARGET_CELL_COUNT
    assert all(left[1] == right[0]
               for left, right in zip(bounds, bounds[1:]))
    assert all(spec["wait_for_files"] for spec in specs)
    assert all("--phase targets" in spec["cmd"] for spec in specs)
