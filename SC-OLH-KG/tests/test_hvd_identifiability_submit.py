import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_hvd_identifiability_scheduler.py"
SPEC = importlib.util.spec_from_file_location("hvd_ident_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def test_hvd_gate_is_one_cell_per_task_and_has_no_checkpoint_sync(tmp_path):
    args = type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "modes": ",".join(submit.MODES),
        "shock_scales": "0,0.25,1,4",
        "replications": "2,4,8,16",
        "seed_start": 0,
        "n_seeds": 5,
        "d": 50,
        "n_train": 32,
        "tau": 0.25,
        "activation_min_records": 16,
        "cpu": 1,
        "ram_mb": 2048,
        "run_id": "hvd-ident",
        "deploy": tmp_path / "deploy",
        "python": submit.REMOTE_PYTHON,
    })()
    specs = submit.build_specs(args)

    assert len(specs) == 5 * 4 * 4 * 5
    assert len({spec["signature"] for spec in specs}) == len(specs)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all("benchmark_hvd_identifiability.py" in spec["cmd"] for spec in specs)
    assert all("--tau 0.25" in spec["cmd"] for spec in specs)
    assert all("ckpt_dir" not in spec for spec in specs)
    assert all("checkpoints" in spec["stage_excludes"] for spec in specs)
