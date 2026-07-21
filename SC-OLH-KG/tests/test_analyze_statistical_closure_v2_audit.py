from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_statistical_closure_v2_audit import (  # noqa: E402
    DOMAINS,
    IMPLEMENTATION_CONTRACT_ID,
    THEORY_CONTRACT_ID,
    analyze,
)


def _registration():
    return {
        "freeze_evidence": {
            "replication_cap": 5,
            "exact_mc_samples": 32,
            "exact_shortlist_size": 32,
            "exact_sampling_mode": "antithetic_nested",
        },
    }


def _write(root, domain, seed, *, identifiable=True):
    path = root / domain / f"seed{seed}" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "heldout": domain,
        "seed": seed,
        "implementation_contract_id": IMPLEMENTATION_CONTRACT_ID,
        "theory_contract_id": THEORY_CONTRACT_ID,
        "theory_contract_timing": "declared_before_target_evaluation",
        "online_action_trace_target_oracle_used": False,
        "exact_kg_mc_samples": 32,
        "exact_kg_sampling_mode": "antithetic_nested",
        "evaluate_or_replicate_new_action_count": 32,
        "replication_max_per_solution": 5,
        "true_feasible": True,
        "feasible_simple_regret": 0.01,
        "certificate_outcome_audit": {
            "posterior_certified_count": 1,
            "false_certificate_count": 0,
        },
        "variance_diagnostics": {
            "cumulative_statistical_design": {"1": {
                "theory_contract": THEORY_CONTRACT_ID,
                "projection": "source_shape_mixture",
                "effective_replication_dof": 4.0,
                "active_calibration_dimension": 2,
                "active_geometry": {"rank": 2},
                "lean_excitation_kappa": 0.4,
                "target_evidence_solution_count": 10,
                "replicated_solution_count": 2,
                "active_identifiable": identifiable,
            }},
        },
    }
    path.write_text(json.dumps({
        "experiment_variant": (
            "statistical_closure_v2_audit_sequential/"
            "statistical_closure_v2/shock1"
        ),
        "rows": [row],
    }), encoding="utf-8")


def test_v2_audit_requires_contract_and_active_hvd_assumptions(tmp_path):
    for domain in DOMAINS:
        _write(tmp_path, domain, 0)
    result = analyze(tmp_path, _registration(), expected_count=3)
    assert result["contract_complete"]
    assert result["finite_sample_hvd_assumptions_hold_for_all_runs"]
    assert result["publication_eligible"]
    assert result["overall"]["finite_sample_hvd_applicable_count"] == 3


def test_v2_audit_reports_nonidentifiability_instead_of_hiding_it(tmp_path):
    for index, domain in enumerate(DOMAINS):
        _write(tmp_path, domain, 0, identifiable=index != 0)
    result = analyze(tmp_path, _registration(), expected_count=3)
    assert result["contract_complete"]
    assert not result["finite_sample_hvd_assumptions_hold_for_all_runs"]
    assert not result["publication_eligible"]
    assert result["overall"]["active_identifiable_count"] == 2
