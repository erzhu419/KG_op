from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "SC-OLH-KG/performance/analyze_v58_guard_decomposition_gate.py"
)
SPEC = importlib.util.spec_from_file_location("v58_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(variant, domain, seed):
    control = variant == MODULE.CONTROL
    mode = ("epistemic", "aleatoric", "interior")[seed % 3]
    prefix = {
        "epistemic": "guard_epistemic_neighbor",
        "aleatoric": "guard_aleatoric_neighbor",
        "interior": "guard_safe_interior_depth",
    }[mode]
    trace = {
        "policy_improvement_pairwise_audit": {"switched": not control},
        "policy_improvement_confirmation": {
            "passed": not control,
            "sample_count": 512 if not control else 0,
            "pilot_stream_independent": True,
            "simulation_stream_independent": True,
            "target_oracle_used": False,
        },
    }
    if not control:
        trace.update({
            "guard_decomposition": {
                "status": "ok",
                "dominant_mode": mode,
                "anchor": [seed],
                "target_oracle_used": False,
            },
            "guard_decomposition_support": {
                "status": "ok",
                "dominant_mode": mode,
                "anchor": [seed],
                "target_oracle_used": False,
            },
            "exact_kg_active_action_labels": [
                "v51_baseline_new",
                "v51_baseline_new",
                "v51_baseline_new",
                "v51_baseline_new",
                prefix,
                "replicate",
            ],
        })
    return {
        "heldout": domain,
        "seed": seed,
        "N": 13,
        "n0": 10,
        "implementation_contract_id": (
            "promoted_v51_observed_terminal_closure"
            if control else "v58_guard_decomposed_action_support"
        ),
        "theory_contract_id": (
            "v51_statistical_closure_v2"
            if control else "v58_guard_decomposed_policy_improvement_v1"
        ),
        "decision_backend_contract": {
            "evaluate_or_replicate_new_action_policy": (
                "canonical_plus_posterior_risk"
                if control
                else "canonical_plus_posterior_guard_decomposition"
            ),
            "evaluate_or_replicate_baseline_new_action_count": 4,
            "policy_improvement_guard_mode": (
                "uniform_score" if control else "independent_confirmation"
            ),
            "policy_improvement_score_transform": (
                "identity" if control else "bounded_current_gain"
            ),
            "policy_improvement_confirmation_samples": 4096,
            "terminal_rule": "posterior_bayes_risk",
        },
        "online_action_trace_target_oracle_used": False,
        "online_action_trace": [trace],
        "posterior_dominance": {
            "enabled": False,
            "target_oracle_used": False,
        },
        "true_feasible": True,
        "feasible_simple_regret": (
            0.1 if control else (0.05 if seed == 0 else 0.1)
        ),
        "x_recommended": [seed] if control else [seed + 10],
        "adaptive_improves_initial_best": not control,
        "adaptive_loss": False,
        "certificate_outcome_audit": {
            "false_certificate_count": 0,
            "posterior_certified_count": 0 if control else 1,
            "posterior_certificate_vacuous": control,
            "minimum_posterior_margin": 0.2 if control else -0.01,
        },
        "initialization_time_sec": 2.0,
        "finalization_time_sec": 3.0,
        "algorithm_time_sec": 10.0,
    }


def test_complete_v58_gate_requires_safe_nonvacuous_certificate(tmp_path):
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
    assert report["contract_valid"] == {
        variant: True for variant in MODULE.KNOWN
    }
    assert report["confirmation_nonvacuous"] is True
    assert report["chance_certificate_nonvacuous"] is True
    assert report["no_false_certificates"] is True
    assert report["formal_gate_passed"] is True
    assert report["promotion_gate_passed"] is True
    assert report["paired_certificate_progress"][
        "median_minimum_margin_delta"] < 0.0


def test_v58_contract_rejects_oracle_tainted_guard():
    row = _row(MODULE.CHALLENGER, MODULE.V56.DOMAINS[0], 0)
    row["online_action_trace"][0][
        "guard_decomposition"]["target_oracle_used"] = True
    assert MODULE._contract(row, MODULE.CHALLENGER) is False
