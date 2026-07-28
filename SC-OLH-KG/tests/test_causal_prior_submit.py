import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_causal_prior_matrix_scheduler.py"
SPEC = importlib.util.spec_from_file_location("causal_prior_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def _args(tmp_path, **overrides):
    deploy = tmp_path / "deploy"
    values = {
        "nodes": ",".join(submit.CPU_NODES),
        "heldouts": ",".join(submit.HELDOUTS),
        "profiles": ",".join(submit.PROFILES),
        "causal_modes": ",".join(submit.CAUSAL_MODES),
        "proposal_modes": "risk_coordinate_atlas",
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "run_id": "causal-v2",
        "source_d": 50,
        "d": 50,
        "N": 20,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "source_design_mode": "universal_mixture",
        "python": submit.REMOTE_PYTHON,
        "cpu": 12,
        "run_cpu": 1,
        "ram_mb": 8192,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_causal_matrix_retrains_every_proposal_and_separates_three_paths(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    archive = [s for s in specs if "/causal_prior_archive/" in s["signature"]]
    designs = [s for s in specs if "/causal_prior_design/" in s["signature"]]
    runs = [s for s in specs if "/causal_prior_v2/" in s["signature"]]

    assert len(archive) == 3
    assert len(designs) == 3 * 10
    assert len(runs) == 3 * 10 * 3 * 5
    assert len(specs) == 483
    assert all("--proposal-mode risk_coordinate_atlas" in s["cmd"] for s in designs)
    assert all("--decision-backend sobol" in s["cmd"] for s in runs)
    assert all("--hvd-profile pooled" in s["cmd"] for s in runs)
    assert all("--source-discrepancy-update" in s["cmd"] for s in runs)
    assert all(s["cpu"] == 12 for s in archive + designs)
    assert all(s["cpu"] == 1 for s in runs)

    proposal_only = [s for s in runs if "/proposal_only/" in s["signature"]]
    posterior_only = [s for s in runs if "/posterior_only/" in s["signature"]]
    joint = [s for s in runs if "/joint/" in s["signature"]]
    assert all("--structural-prior-profile none" in s["cmd"] for s in proposal_only)
    assert all("--initial-design common_sobol" in s["cmd"] for s in posterior_only)
    assert all("--initial-design-file" not in s["cmd"] for s in posterior_only)
    assert all("--initial-design source_informed" in s["cmd"] for s in joint)
    assert all(len(s["wait_for_files"]) == 1 for s in proposal_only + joint)
    assert all("checkpoints" in s["stage_excludes"] for s in specs)


def test_dimension_holdout_trains_source_and_target_at_declared_dimensions(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        profiles="none,full",
        causal_modes="proposal_only,joint",
        proposal_modes=(
            "rank_spanning,risk_coordinate_atlas,risk_objective_atlas"),
        source_d=50,
        d=1000,
        n_seeds=2,
    ))
    designs = [s for s in specs if "/causal_prior_design/" in s["signature"]]
    runs = [s for s in specs if "/causal_prior_v2/" in s["signature"]]

    assert len(designs) == 3 * 3 * 2
    assert len(runs) == 2 * 3 * 2 * 3 * 2
    assert all("--source-d 50 --d 1000" in s["cmd"] for s in designs)
    assert all("--d 1000 --meta-source-d 50" in s["cmd"] for s in runs)
    assert any(
        "--proposal-mode risk_objective_atlas" in s["cmd"]
        for s in designs)


def test_shared_uniform_control_reaches_archive_design_and_online_fit(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        profiles="none,additivity_only",
        causal_modes="proposal_only,joint",
        proposal_modes="risk_objective_atlas",
        source_design_mode="shared_uniform",
        n_seeds=2,
    ))
    archives = [s for s in specs if "/causal_prior_archive/" in s["signature"]]
    designs = [s for s in specs if "/causal_prior_design/" in s["signature"]]
    runs = [s for s in specs if "/causal_prior_v2/" in s["signature"]]

    assert all("--source-design-mode shared_uniform" in s["cmd"] for s in archives)
    assert all("--source-design-mode shared_uniform" in s["cmd"] for s in designs)
    assert all("--source-design-mode shared_uniform" in s["cmd"] for s in runs)
