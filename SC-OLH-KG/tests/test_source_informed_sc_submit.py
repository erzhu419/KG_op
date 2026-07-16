import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_source_informed_matched_scheduler.py"
SPEC = importlib.util.spec_from_file_location("source_sc_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def test_matched_source_informed_matrix_has_60_independent_tasks(tmp_path):
    deploy = tmp_path / "deploy"
    for heldout in submit.HELDOUTS:
        path = (
            deploy / "SC-OLH-KG" / "archives" / "source-run" / heldout
            / "source_initial_designs.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    args = type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "heldouts": ",".join(submit.HELDOUTS),
        "deploy": deploy,
        "manifest": (
            deploy / "SC-OLH-KG/performance/manifests/base.json"),
        "source_run_id": "source-run",
        "run_id": "matched-run",
        "seed_start": 0,
        "n_seeds": 20,
        "d": 50,
        "N": 20,
        "n0": 10,
        "initial_design": "source_informed",
        "python": submit.REMOTE_PYTHON,
        "cpu": 12,
        "exact_jobs": 12,
        "ram_mb": 8192,
    })()

    specs = submit.build_specs(args)

    assert len(specs) == 60
    assert len({spec["signature"] for spec in specs}) == 60
    assert all(spec["cpu"] == 12 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("--initial-design source_informed" in spec["cmd"]
               for spec in specs)
    assert all("--initial-design-file" in spec["cmd"] for spec in specs)
    assert all("--exact-jobs 12" in spec["cmd"] for spec in specs)
    assert all("--d 50 --N 20 --n0 10" in spec["cmd"] for spec in specs)
    assert all(len(spec["wait_for_files"]) == 1 for spec in specs)
    assert all("checkpoints" in spec["stage_excludes"] for spec in specs)


def test_common_sobol_matrix_needs_no_source_design(tmp_path):
    deploy = tmp_path / "deploy"
    args = type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "heldouts": ",".join(submit.HELDOUTS),
        "deploy": deploy,
        "manifest": (
            deploy / "SC-OLH-KG/performance/manifests/base.json"),
        "source_run_id": "unused",
        "run_id": "common-run",
        "seed_start": 0,
        "n_seeds": 5,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "initial_design": "common_sobol",
        "python": submit.REMOTE_PYTHON,
        "cpu": 12,
        "exact_jobs": 12,
        "ram_mb": 8192,
    })()

    specs = submit.build_specs(args)

    assert len(specs) == 15
    assert all("--initial-design common_sobol" in spec["cmd"]
               for spec in specs)
    assert all("--initial-design-file" not in spec["cmd"] for spec in specs)
    assert all("--d 1000 --N 20 --n0 10" in spec["cmd"] for spec in specs)
    assert all(spec["wait_for_files"] == [] for spec in specs)
