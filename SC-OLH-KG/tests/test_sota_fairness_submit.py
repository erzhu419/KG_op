import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_sota_fairness_scheduler.py"
SPEC = importlib.util.spec_from_file_location("sota_fair_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def test_fairness_scheduler_builds_three_contracts_by_twenty_seeds():
    parser_args = type("Args", (), {
        "nodes": "node001,node002,node003,node004,node005,node006",
        "protocols": "target_n20,shared_archive_n20,target_n404",
        "methods": "botorch_turbo,botorch_scbo,botorch_saasbo",
        "heldouts": (
            "FactorShockStatePolicyRZDT1,InventorySupplyChain,"
            "QueueResourceControl"
        ),
        "deploy": Path("/deploy"),
        "run_id": "audit",
        "seed_start": 0,
        "n_seeds": 20,
        "python": "/python",
        "manifest": Path("/deploy/SC-OLH-KG/base.json"),
        "candidate_timeout_sec": 3600.0,
        "cpu": 12,
        "ram_mb": 32768,
    })()
    specs = submit.build_specs(parser_args)
    assert len(specs) == 3 * 3 * 3 * 20
    assert all(spec["require_node"] is None for spec in specs)
    assert all(spec["cpu"] == 12 for spec in specs)
    assert all(spec["vram"] == 8192 for spec in specs)
    assert all("--torch-device cuda" in spec["cmd"] for spec in specs)
    assert all("checkpoint" in spec["ckpt_dir"] for spec in specs)
    assert all("SCOLHKG_OFFLINE=1" in spec["cmd"] for spec in specs)
    assert all("botorch_saasbo" not in spec["description"] or
               "--candidate-timeout-sec 3600.0" in spec["cmd"]
               for spec in specs)


def test_fairness_scheduler_can_route_explicit_cpu_ablation():
    parser_args = type("Args", (), {
        "nodes": "node001,node002",
        "gpu_nodes": "node007",
        "gpu_methods": "",
        "protocols": "target_n13",
        "methods": "botorch_turbo",
        "heldouts": "InventorySupplyChain",
        "deploy": Path("/deploy"),
        "run_id": "cpu_audit",
        "seed_start": 0,
        "n_seeds": 1,
        "python": "/python",
        "manifest": Path("/deploy/SC-OLH-KG/base.json"),
        "candidate_timeout_sec": 3600.0,
        "cpu": 12,
        "ram_mb": 8192,
    })()

    [spec] = submit.build_specs(parser_args)

    assert spec["require_node"] is None
    assert spec["allowed_nodes"] == ["node001", "node002"]
    assert spec["vram"] == 0
    assert "--torch-device cpu" in spec["cmd"]


def test_long_saas_periodic_run_is_explicitly_labelled():
    parser_args = type("Args", (), {
        "nodes": "node001,node002",
        "gpu_nodes": "jtl110gpu,jtl110gpu2,node007",
        "gpu_methods": "botorch_saasbo",
        "protocols": "target_cost_matched_n397",
        "methods": "botorch_saasbo",
        "heldouts": "InventorySupplyChain",
        "deploy": Path("/deploy"),
        "run_id": "periodic_saas_v2",
        "seed_start": 80,
        "n_seeds": 1,
        "python": "/python",
        "manifest": Path("/deploy/SC-OLH-KG/base.json"),
        "candidate_timeout_sec": 3600.0,
        "cpu": 12,
        "ram_mb": 8192,
        "saas_refit_schedule": "doubling",
        "saas_refit_interval": 16,
        "saas_refit_growth_factor": 2.0,
        "saas_refit_max_history": 80,
    })()

    [spec] = submit.build_specs(parser_args)

    assert spec["allowed_nodes"] == [
        "jtl110gpu", "jtl110gpu2", "node007"]
    assert "periodic-hyperposterior" in spec["description"]
    assert "--saas-refit-schedule doubling" in spec["cmd"]
    assert "--saas-refit-growth-factor 2.0" in spec["cmd"]
    assert "--saas-refit-max-history 80" in spec["cmd"]
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in spec["cmd"]
    assert spec["vram_resource_family"] == (
        "KG-SYNTH/sota-fairness/periodic_saas_v2/botorch_saasbo/"
        "InventorySupplyChain/seed0080"
    )
    assert "jtl311linux" not in spec["allowed_nodes"]


def test_shared_archive_uses_exact_frozen_design_when_run_id_is_set():
    parser_args = type("Args", (), {
        "nodes": "node001,node002",
        "gpu_nodes": "node007",
        "gpu_methods": "",
        "protocols": "shared_archive_n13",
        "methods": "botorch_turbo",
        "heldouts": "InventorySupplyChain",
        "deploy": Path("/deploy"),
        "source_run_id": "paper_archive",
        "run_id": "shared_design_audit",
        "seed_start": 80,
        "n_seeds": 1,
        "python": "/python",
        "manifest": Path("/deploy/SC-OLH-KG/base.json"),
        "candidate_timeout_sec": 3600.0,
        "cpu": 12,
        "ram_mb": 8192,
    })()

    [spec] = submit.build_specs(parser_args)

    expected = (
        "/deploy/SC-OLH-KG/archives/paper_archive/"
        "InventorySupplyChain/source_initial_designs.json"
    )
    assert spec["wait_for_files"] == [expected]
    assert (
        "--initial-design-file "
        "archives/paper_archive/InventorySupplyChain/"
        "source_initial_designs.json"
    ) in spec["cmd"]


def test_fairness_scheduler_dispatches_only_newly_submitted_tasks():
    command = submit.build_dispatch_command(
        Path("/scheduler.py"), ["t100", "t101"])

    assert command[-4:] == ["--task-id", "t100", "--task-id", "t101"]
