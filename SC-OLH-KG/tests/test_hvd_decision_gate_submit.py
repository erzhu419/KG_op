import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/submit_scolhkg_hvd_decision_gate_scheduler.py"
SPEC = importlib.util.spec_from_file_location("hvd_decision_gate_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def _args(tmp_path, **overrides):
    deploy = tmp_path / "deploy"
    values = {
        "nodes": ",".join(submit.CPU_NODES),
        "heldouts": ",".join(submit.HELDOUTS),
        "hvd_profiles": ",".join(submit.HVD_PROFILES),
        "discrepancy_profiles": ",".join(submit.DISCREPANCY_PROFILES),
        "action_profiles": ",".join(submit.DEFAULT_ACTION_PROFILES),
        "mean_profiles": "legacy",
        "observable_mean_mode": "latent",
        "shock_scales": "0,0.25,1,4",
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": submit.DEFAULT_SOURCE_RUN_ID,
        "run_id": "hvd-decision-gate",
        "source_d": 50,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 5,
        "replication_candidate_count": 10,
        "replication_max_per_solution": 8,
        "safe_interior_candidate_count": 12,
        "safe_interior_pool_size": 512,
        "safe_interior_margin": 0.0,
        "python": submit.REMOTE_PYTHON,
        "cpu": 1,
        "ram_mb": 4096,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_gate_has_complete_matched_single_seed_tasks(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == len(submit.HVD_PROFILES) * 2 * 2 * (4 + 1 + 1) * 5
    assert len({spec["signature"] for spec in specs}) == len(specs)
    assert all(spec["cpu"] == 1 for spec in specs)
    assert all(spec["allowed_nodes"] == list(submit.CPU_NODES) for spec in specs)
    assert all("--d 1000 --meta-source-d 50 --N 20 --n0 10" in spec["cmd"]
               for spec in specs)
    assert all("--initial-design source_informed" in spec["cmd"] for spec in specs)
    assert all("--structural-prior-profile none" in spec["cmd"] for spec in specs)
    assert all("--posterior-dominance-enabled" not in spec["cmd"] for spec in specs)
    assert all("--finalist-replication-budget 0" in spec["cmd"] for spec in specs)
    assert all("checkpoints" in spec["stage_excludes"] for spec in specs)
    assert all(not spec["local_result_dir"].endswith("checkpoints") for spec in specs)


def test_new_and_replication_actions_share_sobol_front_end(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        heldouts="InventorySupplyChain",
        hvd_profiles="factor_cumulative",
        discrepancy_profiles="adaptive",
        n_seeds=1,
    ))
    assert len(specs) == 2
    new = next(spec for spec in specs if "/new_only/" in spec["signature"])
    voi = next(spec for spec in specs if "/hvd_voi/" in spec["signature"])
    assert "--decision-backend sobol_new" in new["cmd"]
    assert "--no-adaptive-replication-voi" in new["cmd"]
    assert "--replication-candidate-count 0" in new["cmd"]
    assert "--decision-backend sobol_hvd_voi" in voi["cmd"]
    assert "--adaptive-replication-voi" in voi["cmd"]
    assert "--replication-candidate-count 10" in voi["cmd"]
    assert "--source-discrepancy-update" in new["cmd"]
    assert "--source-discrepancy-update" in voi["cmd"]


def test_joint_margin_voi_is_an_explicit_replication_challenger(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        heldouts="QueueResourceControl",
        hvd_profiles="factor_cumulative",
        discrepancy_profiles="adaptive",
        action_profiles="new_only,joint_voi",
        n_seeds=1,
    ))
    assert len(specs) == 2
    joint = next(spec for spec in specs if "/joint_voi/" in spec["signature"])
    assert "--decision-backend sobol_joint_voi" in joint["cmd"]
    assert "--adaptive-replication-voi" in joint["cmd"]
    assert "--replication-candidate-count 10" in joint["cmd"]


def test_certificate_depth_separates_ranking_from_candidate_search(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        heldouts="QueueResourceControl",
        hvd_profiles="factor_hierarchical",
        discrepancy_profiles="adaptive",
        action_profiles="certificate_depth_new,certificate_depth_search",
        certificate_modes="joint_tangent",
        mean_profiles="eta_source_adaptive",
        hvd_source_task_weight_modes="constraint_mean",
        n_seeds=1,
    ))
    assert len(specs) == 2
    rank_only = next(
        spec for spec in specs if "/certificate_depth_new/" in spec["signature"])
    search = next(
        spec for spec in specs if "/certificate_depth_search/" in spec["signature"])
    assert "--decision-backend certificate_depth_new" in rank_only["cmd"]
    assert "--safe-interior-candidate-count 0" in rank_only["cmd"]
    assert "--decision-backend certificate_depth_new" in search["cmd"]
    assert "--safe-interior-candidate-count 12" in search["cmd"]
    assert "--safe-interior-pool-size 512" in search["cmd"]
    assert "--no-adaptive-replication-voi" in search["cmd"]


def test_joint_certificate_is_paired_without_changing_action_rule(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        heldouts="InventorySupplyChain",
        hvd_profiles="factor_cumulative",
        discrepancy_profiles="adaptive",
        action_profiles="new_only",
        certificate_modes="separable,joint_tangent",
        n_seeds=1,
    ))
    assert len(specs) == 2
    assert all("--decision-backend sobol_new" in spec["cmd"] for spec in specs)
    assert all("--no-adaptive-replication-voi" in spec["cmd"] for spec in specs)
    separable = next(spec for spec in specs if "/separable/" in spec["signature"])
    joint = next(spec for spec in specs if "/joint_tangent/" in spec["signature"])
    assert "--task-posterior-robust-certificate-mode separable" in separable["cmd"]
    assert "--task-posterior-robust-certificate-mode joint_tangent" in joint["cmd"]


def test_shock_sweep_applies_only_to_factor_shock(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        hvd_profiles="pooled",
        discrepancy_profiles="frozen",
        action_profiles="new_only",
        n_seeds=1,
    ))
    factor = [spec for spec in specs if "FactorShockStatePolicyRZDT1" in spec["signature"]]
    inventory = [spec for spec in specs if "InventorySupplyChain" in spec["signature"]]
    queue = [spec for spec in specs if "QueueResourceControl" in spec["signature"]]
    assert len(factor) == 4
    assert len(inventory) == 1
    assert len(queue) == 1
    assert {token for spec in factor for token in spec["cmd"].split()
            if token in {"0.0", "0.25", "1.0", "4.0"}} == {
                "0.0", "0.25", "1.0", "4.0"
            }


def test_source_mean_profiles_are_paired_and_oracle_free(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        heldouts="InventorySupplyChain",
        hvd_profiles="factor_hierarchical",
        discrepancy_profiles="adaptive",
        action_profiles="new_only",
        certificate_modes="joint_tangent",
        mean_profiles=(
            "legacy,eta_empirical,eta_source_prior,eta_source_adaptive,"
            "eta_source_sequential"
        ),
        n_seeds=1,
    ))
    assert len(specs) == 5
    legacy = next(spec for spec in specs if "mean_legacy/" in spec["signature"])
    empirical = next(
        spec for spec in specs if "mean_eta_empirical/" in spec["signature"])
    source = next(
        spec for spec in specs if "mean_eta_source_prior/" in spec["signature"])
    adaptive = next(
        spec for spec in specs
        if "mean_eta_source_adaptive/" in spec["signature"])
    sequential = next(
        spec for spec in specs
        if "mean_eta_source_sequential/" in spec["signature"])
    assert "--no-observable-mean-coordinate" in legacy["cmd"]
    assert "--no-source-constraint-mean-coefficient-prior" in legacy["cmd"]
    assert "--observable-mean-coordinate" in empirical["cmd"]
    assert "--no-source-constraint-mean-coefficient-prior" in empirical["cmd"]
    assert "--observable-mean-coordinate" in source["cmd"]
    assert "--source-constraint-mean-coefficient-prior" in source["cmd"]
    assert "--source-constraint-mean-adaptation-mode frozen" in source["cmd"]
    assert "--observable-mean-coordinate" in adaptive["cmd"]
    assert "--source-constraint-mean-coefficient-prior" in adaptive["cmd"]
    assert (
        "--source-constraint-mean-adaptation-mode evidence_mixture"
        in adaptive["cmd"]
    )
    assert (
        "--source-constraint-mean-adaptation-mode "
        "sequential_evidence_mixture" in sequential["cmd"]
    )
    assert all("SCOLHKG_OFFLINE=1" in spec["cmd"] for spec in specs)


def test_sequential_mean_gate_pairs_collapsed_and_online_bayes(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        heldouts="InventorySupplyChain",
        hvd_profiles="factor_hierarchical",
        discrepancy_profiles="adaptive",
        action_profiles="new_only",
        certificate_modes="joint_tangent",
        mean_profiles="eta_source_adaptive,eta_source_sequential",
        hvd_source_task_weight_modes="constraint_mean",
        n_seeds=1,
    ))
    assert len(specs) == 2
    assert any(
        "--source-constraint-mean-adaptation-mode evidence_mixture"
        in spec["cmd"] for spec in specs)
    assert any(
        "--source-constraint-mean-adaptation-mode "
        "sequential_evidence_mixture" in spec["cmd"]
        for spec in specs)


def test_constraint_mean_task_law_can_weight_hvd_source_shapes(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        heldouts="InventorySupplyChain",
        hvd_profiles="factor_hierarchical",
        discrepancy_profiles="adaptive",
        action_profiles="new_only",
        certificate_modes="joint_tangent",
        mean_profiles="eta_source_adaptive",
        hvd_source_task_weight_modes="independent,constraint_mean",
        n_seeds=1,
    ))
    assert len(specs) == 2
    independent = next(
        spec for spec in specs if "hvd_task_independent/" in spec["signature"])
    coupled = next(
        spec for spec in specs
        if "hvd_task_constraint_mean/" in spec["signature"])
    assert "--hvd-source-task-weight-mode independent" in independent["cmd"]
    assert "--hvd-source-task-weight-mode constraint_mean" in coupled["cmd"]


def test_consensus_mean_coordinate_is_a_three_parameter_affine_family(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        heldouts="QueueResourceControl",
        hvd_profiles="factor_hierarchical",
        discrepancy_profiles="adaptive",
        action_profiles="new_only",
        certificate_modes="joint_tangent",
        mean_profiles="eta_source_adaptive",
        observable_mean_mode="consensus",
        hvd_source_task_weight_modes="constraint_mean",
        n_seeds=1,
    ))
    assert len(specs) == 1
    spec = specs[0]
    assert "mean_coord_consensus/" in spec["signature"]
    assert "--observable-mean-mode consensus" in spec["cmd"]
    assert "--observable-mean-latent-dim 0" in spec["cmd"]
    assert "--source-constraint-mean-adaptation-mode evidence_mixture" in spec["cmd"]


def test_source_affine_coordinate_is_available_for_boundary_shape_gate(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        heldouts="QueueResourceControl",
        hvd_profiles="factor_hierarchical",
        discrepancy_profiles="adaptive",
        action_profiles="certificate_depth_new",
        certificate_modes="joint_tangent",
        mean_profiles="eta_source_adaptive",
        observable_mean_mode="source_affine",
        hvd_source_task_weight_modes="constraint_mean",
        n_seeds=1,
    ))
    assert len(specs) == 1
    spec = specs[0]
    assert "mean_coord_source_affine/" in spec["signature"]
    assert "--observable-mean-mode source_affine" in spec["cmd"]
    assert "--source-constraint-mean-adaptation-mode evidence_mixture" in spec["cmd"]


def test_source_affine_gate_is_an_eighty_task_paired_matrix(tmp_path):
    common = {
        "hvd_profiles": "factor_hierarchical",
        "discrepancy_profiles": "adaptive",
        "action_profiles": "new_only,certificate_depth_new",
        "certificate_modes": "joint_tangent",
        "mean_profiles": "eta_source_adaptive",
        "hvd_source_task_weight_modes": "constraint_mean",
        "shock_scales": "0,4",
        "n_seeds": 5,
        "run_id": "source-affine-boundary-gate",
    }
    latent = submit.build_specs(_args(
        tmp_path, observable_mean_mode="latent", **common))
    affine = submit.build_specs(_args(
        tmp_path, observable_mean_mode="source_affine", **common))
    combined = latent + affine

    assert len(latent) == 40
    assert len(affine) == 40
    assert len(combined) == 80
    assert len({spec["signature"] for spec in combined}) == 80
    assert all(
        spec["allowed_nodes"] == list(submit.CPU_NODES)
        for spec in combined
    )
    assert all(
        spec["ckpt_dir"] not in spec["local_result_dir"]
        for spec in combined
    )
    assert all("checkpoints" in spec["stage_excludes"] for spec in combined)
    assert all(
        "--observable-mean-mode latent" in spec["cmd"]
        for spec in latent
    )
    assert all(
        "--observable-mean-mode source_affine" in spec["cmd"]
        for spec in affine
    )


def test_source_rank_coordinate_is_available_for_neutral_backend_gate(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        heldouts="InventorySupplyChain",
        hvd_profiles="factor_hierarchical",
        discrepancy_profiles="adaptive",
        action_profiles="new_only",
        certificate_modes="joint_tangent",
        mean_profiles="eta_source_adaptive",
        observable_mean_mode="source_rank",
        hvd_source_task_weight_modes="constraint_mean",
        n_seeds=1,
    ))
    assert len(specs) == 1
    spec = specs[0]
    assert "mean_coord_source_rank/" in spec["signature"]
    assert spec["signature"].count("hvd_task_constraint_mean/") == 1
    assert "--observable-mean-mode source_rank" in spec["cmd"]
    assert "--observable-mean-latent-dim 0" in spec["cmd"]
    assert "--decision-backend sobol_new" in spec["cmd"]


def test_prequential_upper_gate_is_a_forty_task_paired_matrix(tmp_path):
    specs = submit.build_specs(_args(
        tmp_path,
        hvd_profiles="factor_hierarchical",
        discrepancy_profiles="adaptive",
        action_profiles="new_only",
        certificate_modes="joint_tangent",
        mean_profiles="eta_source_adaptive",
        observable_mean_mode="latent",
        hvd_source_task_weight_modes="constraint_mean",
        hvd_target_evidence_modes="replication_only,prequential_upper",
        shock_scales="0,4",
        n_seeds=5,
    ))
    assert len(specs) == 40
    assert len({spec["signature"] for spec in specs}) == 40
    controls = [
        spec for spec in specs
        if "/hvd_evidence_replication_only/" in spec["signature"]
    ]
    challengers = [
        spec for spec in specs
        if "/hvd_evidence_prequential_upper/" in spec["signature"]
    ]
    assert len(controls) == 20
    assert len(challengers) == 20
    assert all(
        "--hvd-cumulative-target-evidence-mode replication_only"
        in spec["cmd"] for spec in controls
    )
    assert all(
        "--hvd-cumulative-target-evidence-mode prequential_upper"
        in spec["cmd"] for spec in challengers
    )
    assert all(
        spec["allowed_nodes"] == list(submit.CPU_NODES)
        for spec in specs
    )
