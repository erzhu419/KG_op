import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/submit_scolhkg_or_review_confirmatory_scheduler.py"
SPEC = importlib.util.spec_from_file_location("or_review_submit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Args:
    registration = MODULE.DEFAULT_REGISTRATION
    deploy = MODULE.DEFAULT_DEPLOY
    python = MODULE.REMOTE_PYTHON
    matrices = ",".join(MODULE.MATRIX_SPECS)
    shards = 6
    workers = 128
    ram_mb = 65536


def test_partition_is_disjoint_and_complete():
    ranges = MODULE._partition(800, 6)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 800
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))


def test_confirmatory_specs_use_absolute_remote_python_and_full_coverage():
    registration = MODULE.load_registration(Args.registration)
    specs = MODULE.build_specs(Args, registration)
    assert len(specs) == 4 * Args.shards
    assert all(str(Args.python) in spec["cmd"] for spec in specs)
    assert all(" python3 " not in f" {spec['cmd']} " for spec in specs)
    assert all(spec["cpu"] == Args.workers for spec in specs)
    assert all(spec["allowed_nodes"] == list(MODULE.CPU_NODES) for spec in specs)
    assert all(spec["ram_mb"] == Args.ram_mb for spec in specs)
    assert all(spec["cwd"] == str(Args.deploy) for spec in specs)
    assert all(
        spec["result_dir"].startswith(str(Args.deploy)) for spec in specs
    )
    assert all(
        spec["local_result_dir"].startswith(str(ROOT / "SC-OLH-KG"))
        for spec in specs
    )

    for matrix, (total, _, _) in MODULE.MATRIX_SPECS.items():
        selected = [
            spec for spec in specs
            if f"/{matrix}/" in spec["signature"]
        ]
        bounds = []
        for spec in selected:
            fields = spec["signature"].rsplit("/", 1)[-1].split("-")
            bounds.append(tuple(map(int, fields)))
        assert sorted(bounds)[0][0] == 0
        assert sorted(bounds)[-1][1] == total
        assert all(
            left[1] == right[0]
            for left, right in zip(sorted(bounds), sorted(bounds)[1:])
        )


def test_preflight_uses_same_interpreter_and_freeze_contract():
    registration = MODULE.load_registration(Args.registration)
    spec = MODULE.build_preflight_spec(Args, registration)
    assert str(Args.python) in spec["cmd"]
    assert registration["method_freeze_commit"] in spec["cmd"]
    assert "--start 0 --end 1 --workers 1" in spec["cmd"]
