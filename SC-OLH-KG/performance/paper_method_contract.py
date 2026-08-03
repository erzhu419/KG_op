"""Executable contract for the final Operations Research method track."""

from __future__ import annotations

from copy import deepcopy


PAPER_METHOD_CONTRACT_ID = "or_transfer_frontend_saas_v1"
FRONTEND_CONTRACT_ID = "lodo_low_frequency_risk_objective_atlas_v1"
FRONTEND_LOWER_ENVELOPE_CHALLENGER_ID = (
    "lodo_low_frequency_risk_objective_atlas_lower_envelope_v2")
FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID = (
    "lodo_low_frequency_risk_objective_atlas_dc_envelope_v3")
BACKEND_CONTRACT_ID = "canonical_botorch_saasbo_every_iteration_v1"
VERIFIER_CONTRACT_ID = "v69_independent_three_policy_objective_guard_v1"

SOURCE_DOMAINS_PER_HELDOUT = 2
SOURCE_PROFILES_PER_DOMAIN = 64
SOURCE_REPLICATIONS_PER_PROFILE = 3
SOURCE_ARCHIVE_CALLS = (
    SOURCE_DOMAINS_PER_HELDOUT
    * SOURCE_PROFILES_PER_DOMAIN
    * SOURCE_REPLICATIONS_PER_PROFILE
)

TARGET_N0 = 10
TARGET_SEARCH_CALLS = 13
VERIFICATION_CANDIDATE_BUDGETS = (80, 128, 128)
VERIFICATION_FAMILYWISE_DELTA = 0.05
OBJECTIVE_COMPARISON_REPLICATIONS_PER_POLICY = 8
OBJECTIVE_COMPARISON_DELTA = VERIFICATION_FAMILYWISE_DELTA / 3.0

ALLOWED_TARGET_DESCRIPTORS = (
    "heldout task-family identifier",
    "policy dimension and integer bounds",
    "policy/state exposure schema without target outcomes",
    "simulator input and output schema",
)

FORBIDDEN_TARGET_INFORMATION = (
    "target objective observations before the charged target stage",
    "target constraint observations before the charged target stage",
    "target oracle objective or optimizer",
    "target oracle chance margin or feasibility labels",
    "terminal verification responses during search or shortlist construction",
)


def paper_method_contract():
    """Return the immutable method, information, and accounting contract."""

    return deepcopy({
        "schema_version": 1,
        "contract_id": PAPER_METHOD_CONTRACT_ID,
        "novel_component": {
            "role": "transferable structural front end",
            "contract_id": FRONTEND_CONTRACT_ID,
            "source_training": "leave-one-domain-out",
            "structural_prior_profile": "low_frequency_only",
            "proposal_mode": "risk_objective_atlas",
            "dimension_equivariant": True,
            "frozen_before_target_observations": True,
        },
        "online_backend": {
            "role": "replaceable optimization backend",
            "contract_id": BACKEND_CONTRACT_ID,
            "implementation": "BoTorch SaasFullyBayesianSingleTaskGP",
            "refit_schedule": "every_iteration",
            "headline_novelty_claim": False,
        },
        "terminal_verifier": {
            "role": "method-independent deployment certificate",
            "contract_id": VERIFIER_CONTRACT_ID,
            "candidate_budgets": list(VERIFICATION_CANDIDATE_BUDGETS),
            "familywise_delta": VERIFICATION_FAMILYWISE_DELTA,
            "objective_incumbent_guard": True,
            "objective_comparison_replications_per_policy": (
                OBJECTIVE_COMPARISON_REPLICATIONS_PER_POLICY
            ),
            "objective_comparison_delta": OBJECTIVE_COMPARISON_DELTA,
            "updates_optimizer": False,
        },
        "primary_budget": {
            "source_archive_calls": SOURCE_ARCHIVE_CALLS,
            "target_initial_design_calls": TARGET_N0,
            "target_search_calls": TARGET_SEARCH_CALLS,
            "verification_calls_reported_separately": True,
        },
        "information_contract": {
            "track": "descriptor_conditional_lodo",
            "allowed_target_descriptors": list(ALLOWED_TARGET_DESCRIPTORS),
            "forbidden_target_information": list(
                FORBIDDEN_TARGET_INFORMATION),
            "target_outcomes_used_to_fit_proposal": False,
            "target_oracle_used": False,
        },
        "registered_stress_tests": {
            "domain_blind": (
                "Remove the heldout task-family identifier while retaining "
                "shared bounds and unlabeled schemas."
            ),
            "dimension_holdout": (
                "Train the frozen source proposal at d_source and deploy it "
                "unchanged at a different d_target."
            ),
        },
        "claim_boundary": {
            "kg_is_main_contribution": False,
            "saasbo_is_main_contribution": False,
            "manifold_or_transformer_is_main_contribution": False,
            "hvd_optimization_gain_status": (
                "calibration_only_not_core_after_provider_20_seed_gate"),
            "supported_primary_claim": (
                "A source-only, target-label-invariant structural proposal "
                "raises heldout feasible-basin coverage in a "
                "dimension-equivariant policy coordinate."
            ),
        },
    })


def validate_frozen_proposal_payload(payload, *, expected_n0=TARGET_N0):
    """Validate the proposal artifact before it may enter a paper run."""

    failures = []
    expected = {
        "proposal_mode": "risk_objective_atlas",
        "structural_prior_profile": "low_frequency_only",
        "source_archive_oracle_aided": False,
        "target_labels_used": False,
        "target_oracle_used": False,
        "n0": int(expected_n0),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            failures.append(
                f"{key}={payload.get(key)!r}, expected {value!r}")
    contract_id = payload.get(
        "paper_frontend_contract_id", FRONTEND_CONTRACT_ID)
    lower_envelope = bool(payload.get(
        "universal_lower_envelope_sentinel", False))
    monotone_envelope = bool(payload.get("source_monotone_envelope", False))
    if contract_id == FRONTEND_CONTRACT_ID:
        if lower_envelope:
            failures.append(
                "the V1 front-end contract cannot enable the V2 lower-envelope "
                "sentinel")
        if monotone_envelope:
            failures.append(
                "the V1 front-end contract cannot enable the V3 source "
                "monotone envelope")
    elif contract_id == FRONTEND_LOWER_ENVELOPE_CHALLENGER_ID:
        if not lower_envelope:
            failures.append(
                "the V2 lower-envelope contract requires its universal "
                "sentinel")
        if monotone_envelope:
            failures.append(
                "the V2 lower-envelope contract cannot enable the V3 source "
                "monotone envelope")
    elif contract_id == FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID:
        if not monotone_envelope:
            failures.append(
                "the V3 source-envelope contract requires its DC envelope")
        if lower_envelope:
            failures.append(
                "the V3 source-envelope contract cannot enable the V2 "
                "lower-envelope sentinel")
    elif contract_id != FRONTEND_CONTRACT_ID:
        failures.append(f"unknown paper front-end contract {contract_id!r}")
    if int(payload.get("source_dimension", 0)) <= 0:
        failures.append("source_dimension must be positive")
    if int(payload.get("dimension", 0)) <= 0:
        failures.append("target dimension must be positive")
    designs = payload.get("designs")
    if not isinstance(designs, dict) or not designs:
        failures.append("at least one frozen target design is required")
    elif any(
        len(row.get("points", ())) != int(expected_n0)
        for row in designs.values()
    ):
        failures.append("every frozen design must contain exactly n0 points")
    if failures:
        raise ValueError(
            "paper proposal contract violation: " + "; ".join(failures))
    return {
        "contract_id": contract_id,
        "validated": True,
        "source_archive_calls": SOURCE_ARCHIVE_CALLS,
        "target_outcomes_used": False,
        "target_oracle_used": False,
        "design_count": len(designs),
    }


def validate_final_protocol(
    *,
    initial_design_mode,
    backend,
    terminal_profile,
    n0,
    target_search_calls,
    offline_source_calls,
):
    """Reject silent drift in a result labelled as the final method."""

    observed = {
        "initial_design_mode": str(initial_design_mode),
        "backend": str(backend),
        "terminal_profile": str(terminal_profile),
        "n0": int(n0),
        "target_search_calls": int(target_search_calls),
        "offline_source_calls": int(offline_source_calls),
    }
    expected = {
        "initial_design_mode": "source_informed",
        "backend": "saasbo",
        "terminal_profile": "v69",
        "n0": TARGET_N0,
        "target_search_calls": TARGET_SEARCH_CALLS,
        "offline_source_calls": SOURCE_ARCHIVE_CALLS,
    }
    mismatch = {
        key: {"observed": observed[key], "expected": value}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatch:
        raise ValueError(f"final paper protocol drift: {mismatch}")
    return {
        "contract_id": PAPER_METHOD_CONTRACT_ID,
        "validated": True,
        **observed,
    }
