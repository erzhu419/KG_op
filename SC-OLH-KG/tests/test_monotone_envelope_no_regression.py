from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.analyze_monotone_envelope_no_regression import (  # noqa: E402
    evaluate_design_equivalence,
)
from performance.paper_method_contract import (  # noqa: E402
    FRONTEND_CONTRACT_ID,
    FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID,
)


def _payload(contract, monotone):
    points = [[seed, seed + 1] for seed in (80, 81)]
    return {
        "heldout_target_domain": "QueueResourceControl",
        "dimension": 2,
        "source_dimension": 2,
        "n0": 1,
        "seed_start": 80,
        "n_seeds": 2,
        "source_archive_fingerprint": "same-archive",
        "offline_source_calls": 384,
        "source_design_mode": "universal_mixture",
        "proposal_mode": "risk_objective_atlas",
        "proposal_component_mode": "combined",
        "structural_prior_profile": "low_frequency_only",
        "paper_frontend_contract_id": contract,
        "source_monotone_envelope": monotone,
        "source_archive_oracle_aided": False,
        "target_labels_used": False,
        "target_oracle_used": False,
        "proposal_diagnostics": {
            "source_monotone_envelope": {
                "status": "rejected",
                "source_only": True,
            },
        },
        "designs": {
            str(seed): {
                "fingerprint": f"fingerprint-{seed}",
                "points": [point],
            }
            for seed, point in zip((80, 81), points)
        },
    }


def _manifest():
    return {
        "baseline_frontend": FRONTEND_CONTRACT_ID,
        "challenger_frontend": FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID,
        "source_only_equivalence_gate": {
            "domain": "QueueResourceControl",
            "source_dimension": 2,
            "target_dimension": 2,
            "source_calls": 384,
            "n0": 1,
            "seed_start": 80,
            "n_seeds": 2,
        },
    }


def test_exact_frozen_design_identity_closes_no_regression_gate():
    baseline = _payload(FRONTEND_CONTRACT_ID, False)
    challenger = _payload(FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID, True)
    result = evaluate_design_equivalence(_manifest(), baseline, challenger)
    assert result["status"] == "pass"
    assert result["ordered_point_sets_identical"] is True
    assert result["target_replay_required"] is False
    assert result["target_simulator_calls_used_for_gate"] == 0


def test_nonidentical_design_requires_preregistered_target_replay():
    baseline = _payload(FRONTEND_CONTRACT_ID, False)
    challenger = _payload(FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID, True)
    challenger["designs"]["81"]["points"] = [[0, 0]]
    result = evaluate_design_equivalence(_manifest(), baseline, challenger)
    assert result["status"] == "target_replay_required"
    assert result["unequal_seeds"] == [81]
