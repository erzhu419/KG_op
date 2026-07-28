import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_promoted_v51_certification_budget_gate import (
    DOMAINS,
    analyze,
)


def _write(root, budget, variant, domain, seed, certified):
    path = root / variant / domain / f"seed{seed}" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "evaluated_point_count": budget,
        "posterior_certified_count": int(certified),
        "posterior_certificate_vacuous": not certified,
        "true_feasible_count": 2,
        "certified_true_feasible_count": int(certified),
        "false_certificate_count": 0,
        "minimum_posterior_margin": -0.1 if certified else 0.1,
        "minimum_true_margin": -0.2,
    }
    row = {
        "N": budget,
        "heldout": domain,
        "seed": seed,
        "true_feasible": True,
        "feasible_simple_regret": 0.01,
        "adaptive_loss": False,
        "adaptive_improves_initial_best": variant != "new_only",
        "online_action_trace_target_oracle_used": False,
        "certificate_outcome_audit": audit,
        "task_initial_design": {
            "fingerprint": f"{domain}-{seed}",
            "source_archive_fingerprint": "source",
            "n_unique": 10,
            "target_labels_used": False,
            "target_oracle_used": False,
        },
        "meta_prior": {"training": {
            "source_archive_simulator_calls": 384,
            "target_seed_used_for_source_training": False,
            "source_episode_target_oracle_used": False,
        }},
        "source_target_adaptation_contract": {
            "source_simulator_calls": 384,
            "source_oracle_aided": False,
            "target_oracle_used_for_adaptation": False,
        },
        "decision_backend_contract": {
            "terminal_value_contract": (
                "bayes_risk:posterior_central:posterior_nominal:"
                "positive_part:observed_actions:v1"
            ),
            "terminal_recommendation_observed_only": True,
            "acquisition_and_recommendation_share_terminal_action_universe": True,
            "acquisition_and_recommendation_share_risk_penalty": True,
            "coherent": True,
            "forced_sampling_override_count": 0,
        },
    }
    payload = {
        "experiment_variant": (
            "promoted_v51_certification_budget_sequential/"
            f"{variant}/shock0"
        ),
        "rows": [row],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_budget_gate_promotes_nonvacuous_sound_joint_action(tmp_path):
    roots = []
    for budget in (20, 40, 80):
        root = tmp_path / f"n{budget}"
        roots.append((budget, root))
        for variant in ("new_only", "joint_cap5"):
            for domain in DOMAINS:
                _write(
                    root, budget, variant, domain, 0,
                    certified=(variant == "joint_cap5" and budget == 80),
                )
    result = analyze(roots, expected_count=18)
    assert result["source_contract"]
    assert result["closure_contract"]
    assert result["initial_pairing_contract"]
    assert result["gate"]["survivors"] == ["joint_cap5"]
    assert result["gate"]["passes"]


def test_budget_gate_rejects_domainwise_vacuity(tmp_path):
    roots = []
    for budget in (20, 80):
        root = tmp_path / f"n{budget}"
        roots.append((budget, root))
        for variant in ("new_only", "joint_cap5"):
            for domain in DOMAINS:
                _write(
                    root, budget, variant, domain, 0,
                    certified=(
                        variant == "joint_cap5"
                        and budget == 80
                        and domain != "QueueResourceControl"
                    ),
                )
    result = analyze(roots, expected_count=12)
    assert not result["survivor_checks"]["joint_cap5"][
        "positive_useful_coverage_in_every_domain"]
    assert not result["gate"]["passes"]
