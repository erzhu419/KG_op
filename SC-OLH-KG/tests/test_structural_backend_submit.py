import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_structural_backend_matrix_scheduler.py"
SPEC = importlib.util.spec_from_file_location("structural_backend_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def _args(tmp_path, tracks=submit.TRACKS, n_seeds=10):
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "heldouts": ",".join(submit.HELDOUTS),
        "tracks": ",".join(tracks),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": "source-run",
        "run_id": "causal-matrix",
        "seed_start": 0,
        "n_seeds": n_seeds,
        "d": 50,
        "N": 20,
        "n0": 10,
        "python": submit.REMOTE_PYTHON,
        "cpu": 12,
        "exact_jobs": 12,
        "ram_mb": 8192,
    })()


def test_full_matrix_is_single_seed_sharded_and_causally_labeled(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(submit.experiment_variants(submit.TRACKS)) == 43
    assert len(specs) == 43 * 3 * 10
    assert len({spec["signature"] for spec in specs}) == len(specs)
    assert all(spec["cpu"] == 12 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES)
               for spec in specs)
    assert all("SCOLHKG_OFFLINE=1" in spec["cmd"] for spec in specs)
    assert all("--d 50 --N 20 --n0 10" in spec["cmd"] for spec in specs)
    assert all("--finalist-replication-budget 0" in spec["cmd"]
               for spec in specs)
    assert all("--finalist-empirical-override off" in spec["cmd"]
               for spec in specs)
    assert all("checkpoints" in spec["stage_excludes"] for spec in specs)


def test_backend_track_matches_initial_design_for_every_backend(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path, tracks=("backends",), n_seeds=2))
    assert len(specs) == 2 * len(submit.BACKENDS) * 3 * 2
    commands = [spec["cmd"] for spec in specs]
    for backend in submit.BACKENDS:
        assert any(
            f"--decision-backend {backend}" in command
            and "--initial-design common_sobol" in command
            for command in commands
        )
        assert any(
            f"--decision-backend {backend}" in command
            and "--initial-design source_informed" in command
            for command in commands
        )
    for spec in specs:
        source_informed = "--initial-design source_informed" in spec["cmd"]
        assert bool(spec["wait_for_files"]) is source_informed
        assert ("--initial-design-file" in spec["cmd"]) is source_informed


def test_matrix_contains_all_prior_hvd_discrepancy_and_recheck_controls(tmp_path):
    commands = [
        spec["cmd"] for spec in submit.build_specs(_args(
            tmp_path,
            tracks=("priors", "hvd", "discrepancy", "recheck"),
            n_seeds=1,
        ))
    ]
    for profile in submit.PRIOR_PROFILES:
        assert any(
            f"--structural-prior-profile {profile}" in command
            for command in commands
        )
    for profile in submit.HVD_PROFILES:
        assert any(f"--hvd-profile {profile}" in command for command in commands)
    assert any("--source-discrepancy-update" in command for command in commands)
    assert any("--no-source-discrepancy-update" in command for command in commands)
    for top_k in (0, 1, 2):
        assert any(
            f"--certification-recheck-top-k {top_k}" in command
            for command in commands
        )


def test_posterior_dominance_supplement_pairs_both_exact_kg_baselines(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        tracks=("replication_dominance",),
        n_seeds=10,
    ))
    assert len(specs) == 2 * 3 * 10
    commands = [spec["cmd"] for spec in specs]
    common_sobol = [
        command for command in commands
        if "replication_dominance/common_sobol_posterior_dominance"
        in command
    ]
    source_informed = [
        command for command in commands
        if "replication_dominance/source_informed_posterior_dominance"
        in command
    ]
    assert len(common_sobol) == len(source_informed) == 30
    assert all("--adaptive-replication-voi" in command
               and "--posterior-dominance-enabled" in command
               and "--exact-terminal-mode bayes_risk_dominance" in command
               and "--replication-candidate-count 10" in command
               for command in commands)
    assert all("--initial-design common_sobol" in command
               and "--initial-design-file" not in command
               for command in common_sobol)
    assert all("--initial-design source_informed" in command
               and "--initial-design-file" in command
               for command in source_informed)
    assert all("--certification-recheck-top-k 0" in command
               and "--finalist-replication-budget 0" in command
               and "--decision-backend exact_kg" in command
               for command in commands)
