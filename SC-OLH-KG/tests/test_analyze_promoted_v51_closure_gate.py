from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_promoted_v51_closure_gate import (
    CHALLENGER,
    CONTROL,
    DOMAINS,
    summarize,
)


def _row(variant, domain, seed, regret):
    closure = variant == CHALLENGER
    return {
        "gate_variant": variant,
        "heldout": domain,
        "seed": seed,
        "true_feasible": True,
        "feasible_simple_regret": regret,
        "adaptive_improves_initial_best": closure,
        "adaptive_loss": False,
        "adaptive_new_point_selected_count": 5,
        "adaptive_replication_selected_count": 5,
        "online_action_trace_target_oracle_used": False,
        "certificate_outcome_audit": {
            "false_certificate_count": 0,
            "posterior_certified_count": 1 if closure else 0,
            "posterior_certificate_vacuous": not closure,
        },
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
        "decision_backend": "sobol_exact_joint_voi",
        "decision_backend_contract": {
            "terminal_value_contract": (
                "bayes_risk:posterior_central:posterior_nominal:"
                "positive_part:observed_actions:v1"
            ) if closure else "legacy",
            "acquisition_terminal_observed_only": closure,
            "acquisition_and_recommendation_share_terminal_action_universe": (
                closure),
            "acquisition_and_recommendation_share_risk_penalty": closure,
            "coherent": closure,
            "forced_sampling_override_count": 0,
        },
    }


def test_closure_gate_requires_a_strict_paired_gain():
    rows = []
    for domain in DOMAINS:
        for seed in range(2):
            rows.append(_row(CONTROL, domain, seed, 0.02))
            rows.append(_row(CHALLENGER, domain, seed, 0.01))
    result = summarize(rows, seeds=range(2))
    assert result["source_contract"]
    assert result["closure_contract"]
    assert result["initial_pairing_contract"]
    assert result["paired"] == {"wins": 6, "losses": 0, "ties": 0}
    assert result["gate_passes"]


def test_closure_gate_rejects_a_terminal_contract_mismatch():
    rows = []
    for domain in DOMAINS:
        rows.append(_row(CONTROL, domain, 0, 0.02))
        rows.append(_row(CHALLENGER, domain, 0, 0.01))
    rows[-1]["decision_backend_contract"][
        "acquisition_terminal_observed_only"] = False
    result = summarize(rows, seeds=range(1))
    assert not result["closure_contract"]
    assert not result["gate_passes"]
