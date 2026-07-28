import json

from performance.analyze_certification_budget_curve import analyze


def _write_result(root, budget, action, seed, *, certified, margin):
    path = root / action / f"seed{seed}" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    variant = (
        "hvd_decision_gate/mean_coord_latent/hvd_task_constraint_mean/"
        "mean_eta_source_adaptive/factor_hierarchical/adaptive/"
        f"{action}/joint_tangent/shock0"
    )
    row = {
        "heldout": "FactorShockStatePolicyRZDT1",
        "seed": seed,
        "N": budget,
        "experiment_variant": variant,
        "true_feasible": True,
        "posterior_certificate_vacuous": not certified,
        "false_certificate_count": 0,
        "feasible_simple_regret": 0.1,
        "adaptive_rescue": certified and action == "joint_voi",
        "adaptive_improves_initial_best": False,
        "adaptive_loss": False,
        "variance_log_rmse": 0.5 if action == "joint_voi" else 0.6,
        "variance_upper_coverage": 1.0,
        "certification_margin_decomposition": {
            "minimum_margin": {
                "final_certificate": {"margin": margin},
            },
        },
    }
    path.write_text(json.dumps({"rows": [row]}))


def test_budget_curve_requires_sound_monotone_joint_voi_gain(tmp_path):
    roots = []
    for budget in (40, 80):
        root = tmp_path / f"n{budget}"
        roots.append((budget, root))
        for action in ("new_only", "joint_voi"):
            _write_result(
                root,
                budget,
                action,
                0,
                certified=(budget == 80 and action == "joint_voi"),
                margin=0.2 if budget == 40 else (
                    -0.1 if action == "joint_voi" else 0.1),
            )

    result = analyze(roots, expected_count=4)

    assert result["parsed_count"] == 4
    assert result["action_effects"]["80"][
        "certificate_nonvacuous_net"] == 1
    assert result["budget_effects"]["joint_voi"][
        "certificate_nonvacuous_net"] == 1
    assert result["gate"]["coordinate_information_sufficient_at_primary_budget"]
    assert result["gate"]["promote_joint_voi"]


def test_budget_curve_rejects_vacuous_larger_budget(tmp_path):
    roots = []
    for budget in (40, 80):
        root = tmp_path / f"n{budget}"
        roots.append((budget, root))
        for action in ("new_only", "joint_voi"):
            _write_result(
                root, budget, action, 0, certified=False,
                margin=0.2 if budget == 40 else 0.1,
            )

    result = analyze(roots, expected_count=4)

    assert not result["gate"][
        "coordinate_information_sufficient_at_primary_budget"]
    assert not result["gate"]["promote_joint_voi"]
