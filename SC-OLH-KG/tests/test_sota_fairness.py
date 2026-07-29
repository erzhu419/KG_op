import json
from pathlib import Path
from types import SimpleNamespace

from core.designs import (
    common_sobol_integer_design,
    integer_design_fingerprint,
)
from performance.benchmark_sota_fairness import (
    oracle_free_lodo_config,
    post_run_aleatoric_audit,
    run_one,
    source_archive_cost,
)
from performance.benchmark_lodo_meta_prior import build_scalarized_problem


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "performance/manifests/v18b_exactkg_mcdiag.json"


def _args(tmp_path, protocol):
    return SimpleNamespace(
        protocol=protocol,
        method="botorch_turbo",
        heldout="QueueResourceControl",
        seed=3,
        manifest=str(MANIFEST),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        target_budget=5,
        d=8,
        L=100,
        sigma=0.04,
        alpha=0.05,
        weights="0.5,0.5",
        n0=4,
        beta_g=2.0,
        ts_candidates=32,
        raw_samples=8,
        num_restarts=2,
        maxiter=10,
        candidate_timeout_sec=30.0,
        saas_warmup_steps=4,
        saas_num_samples=4,
        saas_thinning=1,
        saas_max_tree_depth=2,
        saas_mc_samples=16,
    )


def test_oracle_free_source_archive_cost_is_384():
    config = oracle_free_lodo_config(MANIFEST)
    assert source_archive_cost(config, "QueueResourceControl") == 384
    assert config["meta_source_observation_mode"] == "replicated"
    assert config["meta_source_observation_replicates"] == 3


def test_post_run_aleatoric_audit_is_explicitly_nonadaptive():
    problem = build_scalarized_problem(
        "QueueResourceControl", 8, 100, 0.04, 0.05, (0.5, 0.5))

    class _Head:
        @staticmethod
        def predict_variance(point):
            return float(problem.true_sigma(point)[1] ** 2)

        @staticmethod
        def predict_certification_variance(point):
            return float(problem.true_sigma(point)[1] ** 2)

    audit = post_run_aleatoric_audit(
        problem, _Head(), seed=3, audit_size=16)
    assert audit["status"] == "ok"
    assert audit["used_for_search_or_selection"] is False
    assert audit["target_oracle_used_post_run_only"] is True
    assert audit["log_variance_rmse"] < 1e-12
    assert audit["upper_coverage"] == 1.0


def test_target_only_and_archive_shared_protocols_have_auditable_designs(tmp_path):
    target = run_one(_args(tmp_path, "target_n20"))
    assert target["status"] == "ok"
    assert target["information_contract"]["offline_source_calls"] == 0
    assert target["result"]["initial_design"] == "sobol"

    shared = run_one(_args(tmp_path, "shared_archive_n20"))
    assert shared["status"] == "ok"
    assert shared["information_contract"]["offline_source_calls"] == 384
    assert shared["information_contract"]["source_oracle_aided"] is False
    assert shared["source_archive_fingerprint"]
    assert shared["initial_points_fingerprint"]
    assert shared["result"]["initial_design"] == "shared_external"


def test_online_sota_reports_search_verification_and_total_budgets(tmp_path):
    args = _args(tmp_path, "target_n20")
    args.terminal_verification = True
    args.terminal_verification_primary_budget = 8
    args.terminal_verification_support_budget = 12
    args.terminal_verification_delta = 0.05
    args.terminal_verification_method = "normal_quantile_tolerance"
    args.terminal_safe_interior_probability_slack = 0.05
    args.terminal_safe_interior_require_provider = True
    payload = run_one(args)
    assert payload["status"] == "ok"
    result = payload["result"]
    contract = payload["information_contract"]
    assert result["n_search_simulations"] == 5
    assert result["n_verification_simulations"] in {8, 20}
    assert result["n_simulations"] == (
        result["n_search_simulations"]
        + result["n_verification_simulations"]
    )
    assert contract["target_search_calls"] == 5
    assert contract["target_verification_calls"] == (
        result["n_verification_simulations"])
    assert contract["target_total_calls"] == result["n_simulations"]
    assert result["terminal_shortlist_frozen_before_truth_metrics"] is True
    assert result["terminal_verification"][
        "posterior_updated_from_verification"
    ] is False


def test_shared_archive_loads_the_exact_frozen_initial_design(tmp_path):
    points = [
        tuple([value] * 8)
        for value in (5, 25, 50, 75)
    ]
    design_path = tmp_path / "source_initial_designs.json"
    design_path.write_text(json.dumps({
        "schema_version": 1,
        "design_kind": "frozen_source_informed_risk_objective_atlas",
        "proposal_mode": "risk_objective_atlas",
        "structural_prior_profile": "low_frequency_only",
        "heldout_target_domain": "QueueResourceControl",
        "dimension": 8,
        "source_dimension": 8,
        "n0": 4,
        "source_archive_fingerprint": "shared-archive-fingerprint",
        "source_archive_oracle_aided": False,
        "target_labels_used": False,
        "target_oracle_used": False,
        "designs": {
            "3": {
                "points": [list(point) for point in points],
                "fingerprint": integer_design_fingerprint(points),
            },
        },
    }), encoding="utf-8")
    args = _args(tmp_path, "shared_archive_n20")
    args.initial_design_file = str(design_path)

    payload = run_one(args)

    assert payload["status"] == "ok"
    assert payload["initial_points"] == [list(point) for point in points]
    assert (
        payload["source_archive_fingerprint"]
        == "shared-archive-fingerprint"
    )
    assert payload["information_contract"]["initial_design_contract"] == (
        "byte_identical_frozen_source_informed_n0"
    )


def test_target_only_can_use_byte_identical_common_sobol(tmp_path):
    args = _args(tmp_path, "target_n20")
    args.common_sobol_initial_design = True

    payload = run_one(args)

    problem = build_scalarized_problem(
        args.heldout,
        args.d,
        args.L,
        args.sigma,
        args.alpha,
        (0.5, 0.5),
    )
    expected = common_sobol_integer_design(
        problem,
        args.n0,
        args.seed,
    )
    assert problem.d == args.d
    assert payload["initial_points"] == [
        list(point) for point in expected
    ]
    assert payload["information_contract"]["offline_source_calls"] == 0
    assert payload["information_contract"]["initial_design_contract"] == (
        "byte_identical_common_sobol_n0"
    )
