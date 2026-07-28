from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_hvd_decision_gate import _parse_variant, analyze  # noqa: E402


def test_parse_variant_ignores_registered_model_prefixes():
    parsed = _parse_variant(
        "hvd_decision_gate/mean_coord_latent/"
        "hvd_evidence_replication_only/hvd_task_constraint_mean/"
        "mean_eta_source_adaptive/factor_hierarchical/adaptive/"
        "joint_voi/joint_tangent/shock4"
    )
    assert parsed == {
        "hvd": "factor_hierarchical",
        "discrepancy": "adaptive",
        "action": "joint_voi",
        "certificate": "joint_tangent",
        "mean_profile": "eta_source_adaptive",
        "shock_label": "shock4",
    }


def test_analyzer_pairs_sequential_source_mean_posterior(tmp_path):
    for index, profile in enumerate((
        "eta_source_adaptive", "eta_source_sequential",
    )):
        row = {
            "experiment_variant": (
                f"hvd_decision_gate/mean_{profile}/factor_hierarchical/"
                "adaptive/new_only/joint_tangent/shock1"
            ),
            "heldout": "InventorySupplyChain",
            "seed": 0,
            "true_feasible": True,
            "posterior_feasible": profile == "eta_source_sequential",
            "false_feasible": False,
            "false_certificate_count": 0,
            "posterior_certificate_vacuous": (
                profile != "eta_source_sequential"),
            "feasible_simple_regret": 0.01,
            "initial_best_feasible_regret": 0.02,
            "adaptive_rescue": profile == "eta_source_sequential",
            "adaptive_loss": False,
            "adaptive_improves_initial_best": (
                profile == "eta_source_sequential"),
            "variance_log_rmse": 0.1,
            "variance_upper_coverage": 0.95,
        }
        path = tmp_path / str(index) / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"rows": [row]}), encoding="utf-8")

    result = analyze(tmp_path, expected_count=2)
    assert result["primary_mean_profile"] == "eta_source_sequential"
    assert result["factor_levels"]["mean_profile"] == (
        "eta_source_adaptive", "eta_source_sequential")
    assert result["paired_effects"]["mean_profile"]["pair_count"] == 1
    assert result["gate"]["advance_to_20_seeds"] is True


def test_analyzer_pairs_all_three_causal_factors_and_applies_gate(tmp_path):
    index = 0
    for hvd in ("pooled", "factor_cumulative"):
        for discrepancy in ("frozen", "adaptive"):
            for action in ("new_only", "hvd_voi"):
                primary = (
                    hvd == "factor_cumulative"
                    and discrepancy == "adaptive"
                    and action == "hvd_voi"
                )
                row = {
                    "experiment_variant": (
                        f"hvd_decision_gate/{hvd}/{discrepancy}/{action}/shock1"
                    ),
                    "heldout": "InventorySupplyChain",
                    "seed": 0,
                    "true_feasible": True,
                    "posterior_feasible": bool(primary),
                    "false_feasible": False,
                    "false_certificate_count": 0,
                    "posterior_certificate_vacuous": not primary,
                    "certificate_precision": 1.0 if primary else None,
                    "feasible_simple_regret": 0.01 if primary else 0.02,
                    "initial_best_feasible_regret": 0.03,
                    "adaptive_rescue": False,
                    "adaptive_loss": False,
                    "adaptive_improves_initial_best": bool(primary),
                    "adaptive_regret_change": -0.02 if primary else -0.01,
                    "variance_log_rmse": 0.1 if hvd == "factor_cumulative" else 0.4,
                    "certified_variance_log_rmse": 0.12,
                    "variance_upper_coverage": 0.95 if primary else 0.90,
                    "median_predicted_true_variance_ratio": 1.0,
                    "median_certified_true_variance_ratio": 1.1,
                    "adaptive_replication_selected_count": 1 if action == "hvd_voi" else 0,
                    "adaptive_new_point_selected_count": 9 if action == "hvd_voi" else 10,
                    "decision_backend_diagnostics": {
                        "mean_selected_constraint_epistemic_information_reduction": 0.2,
                        "mean_selected_hvd_margin_information_reduction": 0.3,
                        "mean_selected_joint_information_reduction": 0.5,
                    },
                }
                path = tmp_path / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
                index += 1

    result = analyze(tmp_path, expected_count=8)
    assert result["parsed_count"] == 8
    assert not result["errors"]
    assert result["gate"]["advance_to_20_seeds"] is True
    assert result["paired_effects"]["hvd"]["pair_count"] == 4
    assert result["paired_effects"]["discrepancy"]["pair_count"] == 4
    assert result["paired_effects"]["action"]["pair_count"] == 4
    assert result["primary_cells"]["InventorySupplyChain/shock1"][
        "selected_replication_count"] == 1
    assert result["primary_cells"]["InventorySupplyChain/shock1"][
        "median_selected_joint_information_reduction"] == 0.5
    assert result["paired_effects_by_stratum"][
        "InventorySupplyChain/shock1"]["hvd"]["pair_count"] == 4


def test_analyzer_includes_controlled_replication_scan(tmp_path):
    gate_root = tmp_path / "gate"
    ident_root = tmp_path / "ident"
    index = 0
    for hvd in ("pooled", "factor_cumulative"):
        for discrepancy in ("frozen", "adaptive"):
            for action in ("new_only", "hvd_voi"):
                primary = (
                    hvd == "factor_cumulative"
                    and discrepancy == "adaptive"
                    and action == "hvd_voi"
                )
                row = {
                    "experiment_variant": (
                        f"hvd_decision_gate/{hvd}/{discrepancy}/{action}/shock1"
                    ),
                    "heldout": "InventorySupplyChain",
                    "seed": 0,
                    "true_feasible": True,
                    "posterior_feasible": primary,
                    "false_feasible": False,
                    "false_certificate_count": 0,
                    "posterior_certificate_vacuous": not primary,
                    "certificate_precision": 1.0 if primary else None,
                    "feasible_simple_regret": 0.01,
                    "initial_best_feasible_regret": 0.02,
                    "adaptive_rescue": False,
                    "adaptive_loss": False,
                    "adaptive_improves_initial_best": primary,
                    "adaptive_regret_change": -0.01,
                    "variance_log_rmse": 0.1,
                    "certified_variance_log_rmse": 0.1,
                    "variance_upper_coverage": 0.95,
                    "median_predicted_true_variance_ratio": 1.0,
                    "median_certified_true_variance_ratio": 1.1,
                    "adaptive_replication_selected_count": int(primary),
                    "adaptive_new_point_selected_count": 10 - int(primary),
                }
                path = gate_root / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
                index += 1

    for mode, rmse in (("pooled", 0.4), ("factor_cumulative", 0.2)):
        for replicates in (2, 16):
            payload = {
                "experiment": "hvd_identifiability",
                "mode": mode,
                "seed": 0,
                "shared_shock_scale": 1.0,
                "replicates_per_policy": replicates,
                "log_variance_rmse": rmse - 0.01 * (replicates == 16),
                "variance_spearman": 0.8,
                "shared_risk_spearman": 0.7 if mode == "factor_cumulative" else None,
                "median_predicted_true_ratio": 1.0,
                "median_certified_true_ratio": 1.1,
                "variance_upper_coverage": 0.95,
                "certificate_nonvacuous": True,
                "false_feasible_count": 0,
                "certificate_precision": 1.0,
                "certificate_recall": 1.0,
            }
            path = ident_root / mode / str(replicates) / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

    result = analyze(
        gate_root,
        expected_count=8,
        identifiability_root=ident_root,
        expected_identifiability_count=4,
    )
    ident = result["identifiability"]
    assert ident["complete_expected_matrix"] is True
    assert ident["paired_factor_vs_pooled"]["pair_count"] == 2
    assert ident["paired_factor_vs_pooled"][
        "median_log_variance_rmse_delta"] < 0.0
    assert ident["replication_trends"][
        "factor_cumulative/shock1"]["pair_count"] == 1
    assert result["gate"]["advance_to_20_seeds"] is True


def test_analyzer_pairs_hierarchical_hvd_and_joint_certificate(tmp_path):
    index = 0
    for hvd in ("factor_cumulative", "factor_hierarchical"):
        for action in ("new_only", "joint_voi"):
            for certificate in ("separable", "joint_tangent"):
                primary = (
                    hvd == "factor_hierarchical"
                    and action == "joint_voi"
                    and certificate == "joint_tangent"
                )
                row = {
                    "experiment_variant": (
                        f"hvd_decision_gate/{hvd}/adaptive/{action}/"
                        f"{certificate}/shock1"
                    ),
                    "heldout": "QueueResourceControl",
                    "seed": 0,
                    "true_feasible": True,
                    "posterior_feasible": primary,
                    "false_feasible": False,
                    "false_certificate_count": 0,
                    "posterior_certificate_vacuous": not primary,
                    "certificate_precision": 1.0 if primary else None,
                    "feasible_simple_regret": 0.01,
                    "initial_best_feasible_regret": 0.02,
                    "adaptive_rescue": False,
                    "adaptive_loss": False,
                    "adaptive_improves_initial_best": primary,
                    "adaptive_regret_change": -0.01,
                    "variance_log_rmse": (
                        0.1 if hvd == "factor_hierarchical" else 0.3
                    ),
                    "certified_variance_log_rmse": 0.2,
                    "variance_upper_coverage": 0.95,
                    "median_predicted_true_variance_ratio": 1.0,
                    "median_certified_true_variance_ratio": 1.1,
                    "adaptive_replication_selected_count": int(
                        action == "joint_voi"),
                    "adaptive_new_point_selected_count": 10 - int(
                        action == "joint_voi"),
                }
                path = tmp_path / str(index) / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
                index += 1

    result = analyze(tmp_path, expected_count=8)
    assert not result["errors"]
    assert result["primary_hvd"] == "factor_hierarchical"
    assert result["primary_action"] == "joint_voi"
    assert result["primary_certificate"] == "joint_tangent"
    assert result["factor_levels"]["hvd"] == (
        "factor_cumulative", "factor_hierarchical")
    assert result["factor_levels"]["certificate"] == (
        "separable", "joint_tangent")
    assert result["paired_effects"]["hvd"]["pair_count"] == 4
    assert result["paired_effects"]["action"]["pair_count"] == 4
    assert result["paired_effects"]["certificate"]["pair_count"] == 4
    assert result["gate"]["advance_to_20_seeds"] is True
