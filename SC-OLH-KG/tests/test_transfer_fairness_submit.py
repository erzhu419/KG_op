import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_transfer_fairness_scheduler.py"
SPEC = importlib.util.spec_from_file_location("transfer_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def _args(**overrides):
    values = {
        "nodes": "node001,node002,node003,node004,node005,node006",
        "methods": ",".join(submit.ALL_METHODS),
        "heldouts": (
            "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
            "QueueResourceControl"
        ),
        "deploy": Path("/deploy"),
        "manifest": Path(
            "/deploy/SC-OLH-KG/performance/manifests/base.json"),
        "run_id": "audit",
        "implementation": "paper_core",
        "allow_unconfigured": False,
        "seed_start": 0,
        "n_seeds": 20,
        "python": "/python",
        "N": 20,
        "n0": 10,
        "d": 50,
        "initial_design": "common_sobol",
        "source_train_steps": 2,
        "target_finetune_steps": 1,
        "cpu": 12,
        "ram_mb": 16384,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_group_submit_has_three_archives_and_480_transfer_rows():
    specs = submit.build_specs(_args())
    assert len(specs) == 3 + 3 * 8 * 20
    archive_specs = [
        spec for spec in specs if "/transfer_archive/" in spec["signature"]
    ]
    run_specs = [
        spec for spec in specs if "/transfer_fairness/" in spec["signature"]
    ]
    assert len(archive_specs) == 3
    assert len(run_specs) == 480
    assert all("require_node" not in spec for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all(spec["cpu"] == 12 for spec in specs)
    assert all(spec["cwd"] == "/deploy/SC-OLH-KG" for spec in specs)
    assert all(spec["wait_for_files"] for spec in run_specs)
    assert all("SCOLHKG_OFFLINE=1" in spec["cmd"] for spec in specs)
    assert all("--N 20 --n0 10" in spec["cmd"] for spec in run_specs)
    assert all("--d 50" in spec["cmd"] for spec in run_specs)
    assert len({spec["ckpt_dir"] for spec in run_specs}) == len(run_specs)


def test_official_submit_configures_every_transfer_method_explicitly():
    specs = submit.build_specs(_args(implementation="official"))
    assert submit.CONFIGURED_OFFICIAL == set(submit.ALL_METHODS)
    assert len(specs) == 3 + 3 * 8 * 20
    assert all(
        "SCOLHKG_EXTERNAL_REPO_ROOT=" in spec["cmd"]
        for spec in specs if "/transfer_fairness/" in spec["signature"]
    )
    assert all(
        str(submit.TRANSFER_TORCH_OVERLAY) in spec["cmd"]
        for spec in specs if "/transfer_fairness/" in spec["signature"]
    )


def test_source_informed_submit_freezes_three_design_files_for_all_methods():
    specs = submit.build_specs(_args(initial_design="source_informed"))
    design_specs = [
        spec for spec in specs
        if "/transfer_initial_design/" in spec["signature"]
    ]
    run_specs = [
        spec for spec in specs if "/transfer_fairness/" in spec["signature"]
    ]
    assert len(specs) == 3 + 3 + 3 * 8 * 20
    assert len(design_specs) == 3
    assert all("materialize_source_initial_designs.py" in spec["cmd"]
               for spec in design_specs)
    assert all("--initial-design source_informed" in spec["cmd"]
               for spec in run_specs)
    assert all("--initial-design-file" in spec["cmd"] for spec in run_specs)
    assert all(len(spec["wait_for_files"]) == 2 for spec in run_specs)
