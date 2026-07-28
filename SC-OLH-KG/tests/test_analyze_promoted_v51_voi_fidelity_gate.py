import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_promoted_v51_voi_fidelity_gate import analyze


def _write(root, variant, selected, names, scores):
    path = root / variant / "FactorShockStatePolicyRZDT1/seed0/result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "heldout": "FactorShockStatePolicyRZDT1",
        "seed": 0,
        "true_feasible": True,
        "online_action_trace_target_oracle_used": False,
        "online_action_trace": [{
            "x_fingerprint": selected,
            "exact_kg_active_action_fingerprints": names,
            "exact_kg_raw_scores_active": scores,
        }],
        "task_initial_design": {
            "fingerprint": "design",
            "source_archive_fingerprint": "archive",
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
            "terminal_value_contract": "bayes_risk:observed_actions:v1",
            "coherent": True,
            "forced_sampling_override_count": 0,
        },
        "exact_kg_diagnostics": {
            "nested_common_random_numbers": True,
            "sampling_mode": "antithetic_nested",
        },
        "certificate_outcome_audit": {"false_certificate_count": 0},
    }
    path.write_text(json.dumps({
        "experiment_variant": (
            f"promoted_v51_voi_fidelity_sequential/{variant}/shock0"),
        "rows": [row],
    }), encoding="utf-8")


def test_fidelity_gate_selects_smallest_stable_configuration(tmp_path):
    root = tmp_path / "gate"
    reference_names = ["a", "b", "c"]
    _write(root, "mc32_k32", "c", reference_names, [0.0, 0.5, 1.0])
    _write(root, "mc32_k4", "c", ["b", "c"], [0.5, 1.0])
    _write(root, "mc8_k4", "c", ["b", "c"], [0.49, 0.99])
    result = analyze(root, expected_count=3)
    assert result["source_contract"]
    assert result["closure_contract"]
    assert result["initial_pairing_contract"]
    assert result["gate"]["passes"]
    assert result["gate"]["selected_configuration"] == "mc8_k4"


def test_fidelity_gate_rejects_unstable_low_mc_configuration(tmp_path):
    root = tmp_path / "gate"
    _write(root, "mc32_k32", "c", ["a", "b", "c"], [0.0, 0.5, 1.0])
    _write(root, "mc2_k4", "a", ["a", "b", "c"], [1.0, 0.5, 0.0])
    result = analyze(root, expected_count=2)
    assert not result["checks"]["mc2_k4"][
        "selected_arm_agreement_at_least_80pct"]
    assert result["gate"]["selected_configuration"] == "mc32_k32"
