import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_boundary_coordinate_gate_scheduler.py"
SPEC = importlib.util.spec_from_file_location(
    "boundary_coordinate_gate_submit", SCRIPT)
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
        "run_id": "boundary-coordinate-gate",
        "source_d": 50,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "boundary_coordinate_pool_size": 512,
        "evaluate_interval": 20,
        "python": submit.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_gate_is_sixty_independent_oracle_free_cpu_shards(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 4 * 5
    assert len({spec["signature"] for spec in specs}) == 60
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES) for spec in specs)
    assert all("SCOLHKG_OFFLINE=1" in spec["cmd"] for spec in specs)
    assert all("--initial-design source_informed" in spec["cmd"] for spec in specs)
    assert all("--hvd-profile factor_hierarchical" in spec["cmd"] for spec in specs)
    assert all("--decision-backend sobol_new" in spec["cmd"] for spec in specs)
    assert all("--truth-pool-diagnostics" in spec["cmd"] for spec in specs)
    assert all("--runtime-checkpoint-interval 0" in spec["cmd"] for spec in specs)
    assert all("--N 20 --n0 10" in spec["cmd"] for spec in specs)


def test_gate_changes_only_phi_mean_and_phi_proposal(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        n_seeds=1,
    ))
    latent = next(
        spec for spec in specs if "/latent_control/" in spec["signature"])
    mean = next(
        spec for spec in specs if "/phi_mean_only/" in spec["signature"])
    joint = next(
        spec for spec in specs if "/phi_mean_proposal/" in spec["signature"])
    assert "--observable-mean-mode latent" in latent["cmd"]
    assert "--observable-mean-training-target constraint_mean" in latent["cmd"]
    assert "--boundary-coordinate-candidate-count 0" in latent["cmd"]
    assert "--observable-mean-mode boundary_aligned" in mean["cmd"]
    assert "--observable-mean-training-target chance_margin" in mean["cmd"]
    assert "--boundary-coordinate-candidate-count 0" in mean["cmd"]
    assert "--observable-mean-mode boundary_aligned" in joint["cmd"]
    assert "--boundary-coordinate-candidate-count 12" in joint["cmd"]


def test_factor_shock_is_the_only_domain_with_two_shock_levels(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        variants="phi_mean_proposal",
        n_seeds=1,
    ))
    factor = [
        spec for spec in specs
        if "FactorShockStatePolicyRZDT1" in spec["signature"]
    ]
    inventory = [
        spec for spec in specs if "InventorySupplyChain" in spec["signature"]
    ]
    queue = [
        spec for spec in specs if "QueueResourceControl" in spec["signature"]
    ]
    assert len(factor) == 2
    assert len(inventory) == 1
    assert len(queue) == 1
    assert any("--target-shared-shock-scale 0.0" in spec["cmd"] for spec in factor)
    assert any("--target-shared-shock-scale 4.0" in spec["cmd"] for spec in factor)
