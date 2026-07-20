import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = _load(
    "mean_v51_s20_design_submit",
    REPO / "scripts/submit_scolhkg_mean_alignment_v51_s20_designs_scheduler.py",
)


def test_v51_s20_design_submitter_uses_frozen_archive_without_target_oracle(
    tmp_path,
):
    deploy = tmp_path / "deploy"
    args = type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "heldouts": ",".join(submit.HELDOUTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "python": submit.root_submit.REMOTE_PYTHON,
        "archive_run_id": submit.DEFAULT_ARCHIVE_RUN_ID,
        "output_source_run_id": submit.DEFAULT_OUTPUT_SOURCE_RUN_ID,
        "run_id": "v51-s20-designs",
        "source_d": 50,
        "d": 1000,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 20,
        "cpu": 12,
        "ram_mb": 8192,
    })()
    specs = submit.build_specs(args)
    assert len(specs) == 3
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("--seed-start 0 --n-seeds 20" in spec["cmd"]
               for spec in specs)
    assert all("--proposal-mode risk_objective_atlas" in spec["cmd"]
               for spec in specs)
    assert all("--structural-prior-profile low_frequency_only" in spec["cmd"]
               for spec in specs)
    assert all("heldout_" in spec["cmd"] for spec in specs)
    assert all("&& &&" not in spec["cmd"] for spec in specs)
    assert all("target" not in spec["description"].lower()
               for spec in specs)
