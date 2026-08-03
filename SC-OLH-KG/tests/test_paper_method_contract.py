import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.paper_method_contract import (  # noqa: E402
    FRONTEND_CONTRACT_ID,
    FRONTEND_LOWER_ENVELOPE_CHALLENGER_ID,
    FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID,
    PAPER_METHOD_CONTRACT_ID,
    paper_method_contract,
    validate_final_protocol,
    validate_frozen_proposal_payload,
)


def _proposal_payload():
    points = [[index, 0, 0] for index in range(10)]
    return {
        "proposal_mode": "risk_objective_atlas",
        "structural_prior_profile": "low_frequency_only",
        "source_archive_oracle_aided": False,
        "target_labels_used": False,
        "target_oracle_used": False,
        "source_dimension": 50,
        "dimension": 1000,
        "n0": 10,
        "designs": {
            "80": {"points": points, "fingerprint": "frozen"},
        },
    }


def test_final_contract_names_frontend_as_novel_and_backend_as_replaceable():
    contract = paper_method_contract()
    assert contract["contract_id"] == PAPER_METHOD_CONTRACT_ID
    assert contract["novel_component"]["role"] == (
        "transferable structural front end")
    assert contract["online_backend"]["headline_novelty_claim"] is False
    assert contract["claim_boundary"]["kg_is_main_contribution"] is False
    assert contract["information_contract"]["target_oracle_used"] is False


def test_contract_returns_an_independent_payload():
    first = paper_method_contract()
    second = paper_method_contract()
    first["primary_budget"]["source_archive_calls"] = -1
    assert second["primary_budget"]["source_archive_calls"] == 384


def test_frozen_proposal_contract_accepts_oracle_free_low_frequency_atlas():
    audit = validate_frozen_proposal_payload(_proposal_payload())
    assert audit["validated"] is True
    assert audit["contract_id"] == FRONTEND_CONTRACT_ID
    assert audit["source_archive_calls"] == 384


def test_lower_envelope_challenger_has_a_distinct_contract():
    payload = _proposal_payload()
    payload.update({
        "paper_frontend_contract_id": (
            FRONTEND_LOWER_ENVELOPE_CHALLENGER_ID),
        "universal_lower_envelope_sentinel": True,
    })
    audit = validate_frozen_proposal_payload(payload)
    assert audit["contract_id"] == FRONTEND_LOWER_ENVELOPE_CHALLENGER_ID

    payload["universal_lower_envelope_sentinel"] = False
    with pytest.raises(ValueError, match="requires its universal sentinel"):
        validate_frozen_proposal_payload(payload)


def test_source_monotone_envelope_challenger_has_a_distinct_contract():
    payload = _proposal_payload()
    payload.update({
        "paper_frontend_contract_id": (
            FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID),
        "source_monotone_envelope": True,
    })
    audit = validate_frozen_proposal_payload(payload)
    assert audit["contract_id"] == FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID

    payload["source_monotone_envelope"] = False
    with pytest.raises(ValueError, match="requires its DC envelope"):
        validate_frozen_proposal_payload(payload)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("proposal_mode", "rank_spanning"),
        ("structural_prior_profile", "full"),
        ("source_archive_oracle_aided", True),
        ("target_labels_used", True),
        ("target_oracle_used", True),
    ],
)
def test_frozen_proposal_contract_rejects_claim_drift(key, bad_value):
    payload = copy.deepcopy(_proposal_payload())
    payload[key] = bad_value
    with pytest.raises(ValueError, match="paper proposal contract violation"):
        validate_frozen_proposal_payload(payload)


def test_final_protocol_accepts_only_the_frozen_combination():
    audit = validate_final_protocol(
        initial_design_mode="source_informed",
        backend="saasbo",
        terminal_profile="v69",
        n0=10,
        target_search_calls=13,
        offline_source_calls=384,
    )
    assert audit["contract_id"] == PAPER_METHOD_CONTRACT_ID
    with pytest.raises(ValueError, match="final paper protocol drift"):
        validate_final_protocol(
            initial_design_mode="source_informed",
            backend="scolh_v69",
            terminal_profile="v69",
            n0=10,
            target_search_calls=13,
            offline_source_calls=384,
        )
