import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "scripts/submit_scolhkg_observable_state_coordinate_offline_gate_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location(
    "observable_state_coordinate_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def _args(tmp_path, **overrides):
    deploy = tmp_path / "deploy"
    values = {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": submit.DEFAULT_SOURCE_RUN_ID,
        "run_id": "observable-state-offline",
        "source_d": 50,
        "d": 1000,
        "N": 10,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "boundary_coordinate_pool_size": 512,
        "python": submit.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_v2_gate_is_140_checkpoint_free_cpu_shards(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 7 * 4 * 5
    assert len({spec["signature"] for spec in specs}) == 140
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("--runtime-checkpoint-dir ''" in spec["cmd"] for spec in specs)
    assert all("--truth-pool-max-candidates 0" in spec["cmd"]
               for spec in specs)
    assert all("ckpt_dir" not in spec for spec in specs)


def test_v2_gate_separates_observable_mainline_and_provider_bound(tmp_path):
    specs = submit.build_specs(_args(tmp_path, n_seeds=1))
    state = next(spec for spec in specs
                 if "/observable_state_phi_r4/" in spec["signature"])
    provider = next(spec for spec in specs
                    if "/provider_exposure_phi_r4/" in spec["signature"])
    assert "--observable-mean-input-mode observable_state_exposure" in (
        state["cmd"])
    assert "oracle_free_v2_challenger" in state["description"]
    assert "--observable-mean-input-mode provider_exposure" in provider["cmd"]
    assert "structure_aware_upper_bound" in provider["description"]

