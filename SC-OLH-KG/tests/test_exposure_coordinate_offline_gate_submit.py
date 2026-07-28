import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "scripts/submit_scolhkg_exposure_coordinate_offline_gate_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location(
    "exposure_coordinate_offline_submit", SCRIPT)
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
        "run_id": "exposure-coordinate-offline",
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


def test_offline_gate_is_160_independent_checkpoint_free_cpu_shards(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 8 * 4 * 5
    assert len({spec["signature"] for spec in specs}) == 160
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES) for spec in specs)
    assert all("SCOLHKG_OFFLINE=1" in spec["cmd"] for spec in specs)
    assert all("--N 10 --n0 10" in spec["cmd"] for spec in specs)
    assert all("--runtime-checkpoint-dir ''" in spec["cmd"] for spec in specs)
    assert all("ckpt_dir" not in spec for spec in specs)
    assert all("--boundary-coordinate-candidate-count 0" in spec["cmd"]
               for spec in specs)


def test_offline_gate_separates_oracle_free_and_provider_upper_bound(tmp_path):
    specs = submit.build_specs(_args(tmp_path, n_seeds=1))
    latent = next(
        spec for spec in specs if "/latent_control/" in spec["signature"])
    learned = next(
        spec for spec in specs
        if "/learned_exposure_phi_r4/" in spec["signature"])
    provider = next(
        spec for spec in specs
        if "/provider_exposure_phi_r4/" in spec["signature"])
    assert "--observable-mean-mode latent" in latent["cmd"]
    assert "--observable-mean-input-mode policy_profile" in latent["cmd"]
    assert "--observable-mean-latent-dim 4" in learned["cmd"]
    assert "--observable-mean-input-mode source_learned_exposure" in (
        learned["cmd"])
    assert "oracle_free_challenger" in learned["description"]
    assert "--observable-mean-input-mode provider_exposure" in provider["cmd"]
    assert "structure_aware_upper_bound" in provider["description"]


def test_offline_gate_rejects_sequential_budget(tmp_path):
    try:
        submit.build_specs(_args(tmp_path, N=20, n0=10))
    except ValueError as error:
        assert "N == n0" in str(error)
    else:
        raise AssertionError("offline gate accepted sequential target calls")
