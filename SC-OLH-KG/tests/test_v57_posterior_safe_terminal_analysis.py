from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "SC-OLH-KG/performance/analyze_v57_posterior_safe_terminal_gate.py"
)
SPEC = importlib.util.spec_from_file_location("v57_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(variant, domain, seed):
    control = variant == MODULE.CONTROL
    horizon = 3
    row = {
        "heldout": domain,
        "seed": seed,
        "N": 13,
        "n0": 10,
        "implementation_contract_id": (
            "v56_independent_confirmation_guard"
            if control else "v57_posterior_safe_terminal_closure"
        ),
        "theory_contract_id": (
            "v56_independent_confirmation_finite_look_v1"
            if control else "v57_confirmation_dominance_composition_v1"
        ),
        "decision_backend_contract": {
            "policy_improvement_guard_mode": "independent_confirmation",
            "policy_improvement_score_transform": "bounded_current_gain",
            "policy_improvement_confirmation_samples": 4096,
            "terminal_rule": (
                "posterior_bayes_risk"
                if control else "posterior_dominance"
            ),
            "acquisition_and_recommendation_share_terminal_action_universe": (
                True),
            "target_oracle_used": False,
        },
        "online_action_trace_target_oracle_used": False,
        "online_action_trace": [{
            "policy_improvement_pairwise_audit": {"switched": True},
            "policy_improvement_confirmation": {
                "passed": True,
                "sample_count": 512,
                "pilot_stream_independent": True,
                "simulation_stream_independent": True,
                "target_oracle_used": False,
            },
        }],
        "posterior_dominance_terminal_used": not control,
        "posterior_dominance": {
            "enabled": not control,
            "target_oracle_used": False,
            "switch_count": 1 if not control else 0,
            "delta_switch": 0.05 / horizon if not control else 0.05,
            "history": (
                [] if control else [
                    {
                        "incumbent_before": None,
                        "target_oracle_used": False,
                    },
                    *[
                        {
                            "incumbent_before": [0],
                            "delta_switch": 0.05 / horizon,
                            "target_oracle_used": False,
                        }
                        for _ in range(horizon)
                    ],
                ]
            ),
        },
        "true_feasible": True,
        "feasible_simple_regret": 0.1 if control else 0.05,
        "x_recommended": [seed] if control else [seed + 10],
        "adaptive_improves_initial_best": True,
        "adaptive_loss": False,
        "certificate_outcome_audit": {
            "false_certificate_count": 0,
            "posterior_certified_count": 0,
            "posterior_certificate_vacuous": True,
        },
        "initialization_time_sec": 2.0,
        "finalization_time_sec": 3.0,
        "algorithm_time_sec": 10.0,
    }
    return row


def test_complete_v57_gate_passes_formal_and_promotion_contracts(tmp_path):
    for variant in MODULE.KNOWN:
        for domain in MODULE.V56.DOMAINS:
            for seed in range(5):
                path = (
                    tmp_path / variant / domain
                    / f"seed{seed}" / "result.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "experiment_variant": (
                        f"gate/{variant}/{domain}/seed{seed}"),
                    "rows": [_row(variant, domain, seed)],
                }), encoding="utf-8")
    report = MODULE.analyze(tmp_path)
    assert report["complete_5_seed_matrix"] is True
    assert report["paired_keys_match_control"] is True
    assert report["contract_valid"] == {
        variant: True for variant in MODULE.KNOWN
    }
    assert report["formal_gate_passed"] is True
    assert report["promotion_gate_passed"] is True
    assert report["chance_certificate_nonvacuous"] is False
    assert report["summaries"][MODULE.CHALLENGER][
        "QueueResourceControl"]["maximum_switch_horizon_bound"] == 0.05


def test_v57_contract_rejects_unspent_per_stage_switch_delta():
    row = _row(MODULE.CHALLENGER, MODULE.V56.DOMAINS[0], 0)
    row["posterior_dominance"]["delta_switch"] = 0.05
    for item in row["posterior_dominance"]["history"][1:]:
        item["delta_switch"] = 0.05
    assert MODULE._contract(row, MODULE.CHALLENGER) is False
