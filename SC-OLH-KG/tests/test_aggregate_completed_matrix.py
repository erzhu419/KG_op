from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "performance"
    / "aggregate_completed_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("aggregate_completed_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_aggregates_sc_result_and_excludes_checkpoint_tree(tmp_path):
    root = tmp_path / "structural_run"
    _write(
        root / "priors" / "full" / "Domain" / "seed0" / "result.json",
        {
            "config": {"d": 50, "N": 20, "n0": 10, "initial_design": "source_informed"},
            "experiment_variant": "structural_backend/priors/full",
            "rows": [{
                "heldout": "Domain",
                "seed": 0,
                "true_feasible": True,
                "feasible_simple_regret": 0.1,
                "initial_has_true_feasible": False,
                "adaptive_rescue": True,
                "posterior_certificate_vacuous": False,
                "false_certificate_count": 0,
                "audit": {"source_simulator_calls": 384},
                "source_target_adaptation_contract": {
                    "target_initial_design_fingerprint": "design-0",
                    "source_archive_fingerprint": "archive-0",
                },
            }],
        },
    )
    _write(root / "checkpoints" / "result.json", {"rows": [{"seed": 99}]})
    (root / "runtime.pkl").write_bytes(b"not read")

    rows, errors = MODULE.load_rows([root])

    assert errors == []
    assert len(rows) == 1
    assert rows[0]["method"] == "full"
    assert rows[0]["adaptive_rescue"] is True
    assert rows[0]["total_calls"] == 404
    assert rows[0]["initial_design_fingerprint"] == "design-0"
    assert rows[0]["source_archive_fingerprint"] == "archive-0"


def test_records_official_runtime_failure_as_a_result_row(tmp_path):
    root = tmp_path / "ratio_run"
    _write(
        root / "official" / "Domain" / "safe_fpacoh_cbo" / "seed0000" / "result.json",
        {
            "method": "safe_fpacoh_cbo",
            "implementation": "official",
            "heldout_target_domain": "Domain",
            "seed": 0,
            "status": "failed_official_runtime",
            "failure_type": "ValueError",
            "comparison_contract": {
                "target_dimension": 1000,
                "target_total_calls_N": 20,
                "target_initial_calls_n0": 10,
                "source_simulator_calls": 384,
            },
        },
    )

    rows, errors = MODULE.load_rows([root])

    assert errors == []
    assert len(rows) == 1
    assert rows[0]["status"] == "failed_official_runtime"
    assert rows[0]["true_feasible"] is None
    assert rows[0]["d_over_target_calls"] == 50.0


def test_recovers_replicated_source_calls_for_new_archive_origin(tmp_path):
    root = tmp_path / "shared_uniform_run"
    _write(root / "joint" / "Domain" / "seed0" / "result.json", {
        "config": {"d": 50, "N": 20, "n0": 10},
        "experiment_variant": "causal_prior_v2/joint/atlas/none",
        "rows": [{
            "heldout": "Domain",
            "seed": 0,
            "true_feasible": True,
            "feasible_simple_regret": 0.1,
            "audit": {"source_simulator_calls": 0},
            "source_target_adaptation_contract": {
                "source_simulator_calls": 384,
            },
            "meta_prior": {
                "n_records": 128,
                "training": {
                "source_observation_mode": "replicated",
                "source_observation_replicates": 3,
                "record_origins": {"universal_shared_uniform": 128},
            }},
        }],
    })

    rows, errors = MODULE.load_rows([root])

    assert errors == []
    assert len(rows) == 1
    assert rows[0]["source_calls"] == 384
    assert rows[0]["total_calls"] == 404


def test_aggregates_hvd_identifiability_metrics(tmp_path):
    root = tmp_path / "hvd_run"
    _write(root / "factor" / "result.json", {
        "status": "ok",
        "experiment": "hvd_identifiability",
        "mode": "factor_cumulative",
        "seed": 2,
        "d": 50,
        "n_train_policies": 32,
        "replicates_per_policy": 4,
        "simulator_calls": 128,
        "shared_shock_scale": 2.0,
        "certification_tau": 0.25,
        "log_variance_rmse": 0.3,
        "variance_spearman": 0.8,
        "shared_risk_spearman": 0.7,
        "variance_upper_coverage": 0.95,
        "true_feasible_rate": 0.4,
        "posterior_feasible_rate": 0.3,
        "false_feasible_count": 1,
        "false_feasible_rate": 0.01,
        "false_feasible_fraction_of_certified": 0.1,
        "missed_feasible_rate": 0.2,
        "missed_feasible_fraction_of_true": 0.5,
        "certificate_precision": 0.9,
        "certificate_recall": 0.5,
        "median_predicted_true_ratio": 1.1,
        "median_certified_true_ratio": 1.3,
        "posterior_feasible_count": 4,
        "certificate_nonvacuous": True,
    })

    rows, errors = MODULE.load_rows([root])
    summary = MODULE.summarize_rows(rows)

    assert errors == []
    assert len(rows) == 1
    assert rows[0]["track"] == "hvd_identifiability"
    assert rows[0]["shared_shock_scale"] == 2.0
    assert rows[0]["certification_tau"] == 0.25
    assert rows[0]["replicates_per_policy"] == 4
    assert summary[0]["median_log_variance_rmse"] == 0.3
    assert summary[0]["median_variance_upper_coverage"] == 0.95
    assert summary[0]["median_certificate_recall"] == 0.5
    assert summary[0]["median_missed_feasible_fraction_of_true"] == 0.5


def test_extracts_post_run_paper_trace_and_contract_audit(tmp_path):
    root = tmp_path / "closure_run"
    _write(root / "method" / "Domain" / "seed0" / "result.json", {
        "config": {
            "d": 1000,
            "N": 12,
            "n0": 10,
            "exact_kg_mc_samples": 8,
            "exact_kg_sampling_mode": "antithetic_nested",
            "evaluate_or_replicate_new_action_count": 4,
        },
        "experiment_variant": "paper/main/promoted",
        "rows": [{
            "heldout": "Domain",
            "seed": 0,
            "true_feasible": True,
            "feasible_simple_regret": 0.05,
            "initial_has_true_feasible": True,
            "initial_best_feasible_regret": 0.2,
            "audit_admissible_mainline": True,
            "source_target_adaptation_contract": {
                "source_simulator_calls": 384,
                "source_oracle_aided": False,
                "target_oracle_used_for_adaptation": False,
            },
            "decision_backend_contract": {
                "coherent": True,
                "terminal_value_contract": "bayes:observed_actions:v1",
                "terminal_recommendation_observed_only": True,
                "online_updates_use_budgeted_target_observations_only": True,
                "target_oracle_used": False,
            },
            "certificate_outcome_audit": {
                "evaluated_point_count": 11,
                "posterior_certified_count": 2,
                "certified_true_feasible_count": 2,
                "false_certificate_count": 0,
                "minimum_posterior_margin": -0.1,
                "minimum_true_margin": -0.2,
            },
            "online_action_trace": [{
                "iteration": 0,
                "target_call": 11,
                "action_kind": "new",
                "candidate_source": "sobol",
                "x_fingerprint": "x0",
                "decision_bayes_risk": 1.2,
                "decision_theory_margin": 0.3,
                "observed_response": [2.0, 3.0],
                "true_objective_post_run": 1.0,
                "true_chance_margin_post_run": -0.1,
                "true_feasible_post_run": True,
                "feasible_regret_post_run": 0.1,
                "incumbent_feasible_regret_post_run": 0.1,
                "truth_join_timing": "post_run_after_all_decisions_frozen",
                "target_oracle_used_for_decision": False,
            }],
        }],
    })

    rows, errors = MODULE.load_rows([root])
    traces = MODULE.extract_trace_rows(rows)
    summary = MODULE.summarize_rows(rows)

    assert errors == []
    assert rows[0]["decision_contract_coherent"] is True
    assert rows[0]["exact_mc_samples"] == 8
    assert rows[0]["exact_shortlist_size"] == 4
    assert len(traces) == 2
    assert traces[0]["target_call"] == 10
    assert traces[0]["action_kind"] == "initial_design_summary"
    assert traces[1]["target_call"] == 11
    assert traces[1]["incumbent_feasible_regret_post_run"] == 0.1
    assert traces[1]["target_oracle_used_for_decision"] is False
    domain = next(item for item in summary if item["domain"] == "Domain")
    assert domain["certified_point_coverage"] == 2 / 11
    assert domain["decision_contract_coherent_rate"] == 1.0
    assert domain["target_oracle_decision_count"] == 0


def test_failure_aware_regret_keeps_infeasible_runs_in_denominator():
    finite, status = MODULE._failure_aware_median_regret([
        {"true_feasible": True, "feasible_regret": 0.1},
        {"true_feasible": False, "feasible_regret": None},
        {"true_feasible": False, "feasible_regret": None},
    ])

    assert finite is None
    assert status == "infinite_due_to_infeasible_recommendations"


def test_paper_variant_parser_keeps_method_separate_from_shock_scenario():
    assert MODULE._variant_parts(
        "paper_main_v1_sequential/promoted_joint_voi/shock1", "fallback"
    ) == ("paper_main_v1_sequential", "promoted_joint_voi")


def test_certified_matrix_separates_search_verification_and_source_costs(
    tmp_path,
):
    root = tmp_path / "certified_transfer"
    _write(
        root / "official" / "Domain" / "rgpe_cbo" / "seed0000"
        / "result.json",
        {
            "schema_version": 2,
            "status": "ok",
            "method": "rgpe_cbo",
            "implementation": "official",
            "heldout_target_domain": "Domain",
            "seed": 0,
            "comparison_contract": {
                "target_dimension": 1000,
                "target_initial_calls_n0": 10,
                "source_simulator_calls": 384,
                "target_search_calls": 13,
                "target_verification_calls": 176,
                "target_total_calls": 189,
                "total_source_plus_target_search_calls": 397,
                "total_source_plus_target_verification_calls": 573,
            },
            "result": {
                "n_search_simulations": 13,
                "n_verification_simulations": 176,
                "n_simulations": 189,
                "true_feasible": True,
                "feasible_regret": 0.04,
                "optimization_recommendation_truth": {
                    "true_feasible": False,
                    "feasible_regret": None,
                },
                "terminal_verification": {
                    "enabled": True,
                    "certified": True,
                    "selected_shortlist_rank": 2,
                    "attempt_count": 2,
                    "attempts": [
                        {"certified": False},
                        {"certified": True},
                    ],
                },
            },
        },
    )

    rows, errors = MODULE.load_rows([root])
    summaries = MODULE.summarize_rows(rows)

    assert errors == []
    assert len(rows) == 1
    row = rows[0]
    assert row["search_calls"] == 13
    assert row["verification_calls"] == 176
    assert row["target_total_calls"] == 189
    assert row["source_plus_search_calls"] == 397
    assert row["total_calls"] == 573
    assert row["terminal_certified"] is True
    assert row["terminal_rank1_certified"] is False
    assert row["terminal_fallback_used"] is True
    assert row["terminal_false_certificate"] is False
    domain = next(item for item in summaries if item["domain"] == "Domain")
    assert domain["terminal_certified_rate"] == 1.0
    assert domain["terminal_rank1_certified_rate"] == 0.0
    assert domain["median_verification_calls"] == 176
