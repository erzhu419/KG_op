#!/usr/bin/env python3
"""Freeze one hash-addressed, fail-closed paper evidence release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


FINAL_CONTRACT_ID = "or_transfer_frontend_saas_v1"
UNIFORM_VERIFIER_CONTRACT_ID = "uniform_two_policy_external_verifier_v1"
FINAL_HEADLINE_TRACK = "final_frozen_source_frontend_backend_d1000_n13"
FINAL_CONTROL_TRACK = "final_frozen_sobol_frontend_control_d1000_n13"
FINAL_HEADLINE_METHODS = {
    "frozen_crossdim_proposal_only",
    "stacked_transfer_gp_cbo:official_transfergpbo_code",
    "canonical_saasbo_every_iteration",
}
FINAL_CONTROL_METHODS = {
    "common_sobol_proposal_only",
    "stacked_transfer_gp_cbo:official_transfergpbo_code",
    "canonical_saasbo_every_iteration",
}
CONFIRMATORY_INFERENCE_FAMILIES = {
    "frontend_coverage_confirmatory",
    "online_backend_confirmatory",
    "archive_fair_transfer_confirmatory",
    "total_cost_sota_confirmatory",
    "hvd_mechanistic_confirmatory",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition, message, failures):
    if not condition:
        failures.append(str(message))


def _hvd_release_role(hvd_causal_gate):
    gate = hvd_causal_gate.get("gate", {})
    domain_gates = hvd_causal_gate.get("domain_gates", {})
    expected_promotion = bool(
        domain_gates
        and all(
            value.get("promote_in_domain") is True
            for value in domain_gates.values()
        )
    )
    observed_promotion = gate.get("promote_hvd_as_core")
    if observed_promotion is not expected_promotion:
        raise ValueError(
            "HVD promotion decision does not match the preregistered "
            "all-domain gate"
        )
    if observed_promotion:
        return "core_cumulative_risk_calibration_contribution"
    if gate.get("retain_as_domain_conditional_calibration_component"):
        return "domain_conditional_calibration_component"
    return "mechanistic_negative_control_and_certification_ablation"


def validate_release_inputs(
    method_contract,
    registry,
    audit,
    statistics,
    convergence,
    traffic,
    traffic_negative_control,
    proposal_coverage,
    proof_receipt,
    execution_snapshot,
    hvd_causal_gate,
    dimension_frontier,
    artifact_manifest,
):
    failures = []
    _require(
        method_contract.get("contract_id") == FINAL_CONTRACT_ID,
        "final method identity is not frozen to the registered contract",
        failures,
    )
    _require(
        method_contract.get("online_backend", {}).get("refit_schedule")
        == "every_iteration",
        "headline backend is not canonical every-iteration SAASBO",
        failures,
    )
    snapshot_support = method_contract.get(
        "supporting_evidence", {}).get("immutable_execution_snapshot", {})
    traffic_snapshot_support = method_contract.get(
        "supporting_evidence", {}).get(
            "external_traffic_execution_snapshot", {})
    _require(
        execution_snapshot.get("status") == "frozen"
        and execution_snapshot.get("repository_commit")
        == snapshot_support.get("repository_commit")
        and execution_snapshot.get("scolhkg_tree")
        == snapshot_support.get("scolhkg_tree")
        and execution_snapshot.get("proof_tree")
        == snapshot_support.get("proof_tree")
        and execution_snapshot.get("scripts_tree")
        == snapshot_support.get("scripts_tree")
        and execution_snapshot.get("method_contract_id")
        == FINAL_CONTRACT_ID
        and execution_snapshot.get("theory_contract_id")
        == "source_target_geometric_atlas_coverage_v1"
        and execution_snapshot.get(
            "runtime_checkpoints_or_model_weights_included") is False
        and execution_snapshot.get(
            "target_outcomes_used_to_select_snapshot") is False,
        "immutable execution snapshot differs from the frozen method "
        "contract",
        failures,
    )
    _require(
        audit.get("registry_id") == registry.get("registry_id"),
        "audit and experiment registry identities differ",
        failures,
    )
    traffic_execution = traffic.get("execution_provenance", {})
    _require(
        traffic_execution.get("status") == "frozen"
        and traffic_execution.get("repository_commit")
        == traffic_snapshot_support.get("repository_commit")
        and traffic_execution.get("scolhkg_tree")
        == traffic_snapshot_support.get("scolhkg_tree")
        and traffic_execution.get("proof_tree")
        == traffic_snapshot_support.get("proof_tree")
        and traffic_execution.get("scripts_tree")
        == traffic_snapshot_support.get("scripts_tree")
        and traffic_execution.get("legacy_traffic_tree")
        == traffic_snapshot_support.get("legacy_traffic_tree")
        and traffic_execution.get("traffic_decision_space_blob")
        == traffic_snapshot_support.get("traffic_decision_space_blob")
        and traffic_execution.get("traffic_baseline_blob")
        == traffic_snapshot_support.get("traffic_baseline_blob")
        and traffic_execution.get("method_contract_id")
        == FINAL_CONTRACT_ID
        and traffic_execution.get("theory_contract_id")
        == "source_target_geometric_atlas_coverage_v1"
        and traffic_execution.get(
            "runtime_checkpoints_or_model_weights_in_snapshot") is False
        and traffic_execution.get(
            "target_outcomes_used_to_select_snapshot") is False,
        "external traffic evidence is not bound to the registered sparse "
        "traffic execution snapshot",
        failures,
    )
    _require(
        audit.get("status") == "pass",
        "one or more registered experiment tracks are incomplete or failed",
        failures,
    )
    _require(
        all(row.get("status") == "pass"
            for row in audit.get("track_audits", ())),
        "one or more track audits failed",
        failures,
    )
    _require(
        statistics.get("status") == "complete",
        "paired statistics are incomplete",
        failures,
    )
    _require(
        all(row.get("status") == "pass"
            for row in statistics.get("comparison_audits", ())),
        "one or more preregistered comparisons are incomplete",
        failures,
    )
    registered_family_ids = {
        row.get("family_id")
        for row in registry.get("inference_families", ())
    }
    observed_families = list(statistics.get(
        "inference_families", ()))
    observed_family_ids = {
        row.get("family_id") for row in observed_families
    }
    _require(
        registered_family_ids == CONFIRMATORY_INFERENCE_FAMILIES
        and observed_family_ids == CONFIRMATORY_INFERENCE_FAMILIES
        and all(
            int(row.get("hypothesis_count", 0)) > 0
            and row.get("scope") == "global_stratum_only"
            for row in observed_families
        )
        and statistics.get("domain_strata_inference_role")
        == "unadjusted heterogeneity analysis, not confirmatory",
        "confirmatory inference families or Holm scopes drifted",
        failures,
    )
    final_track = next((
        track for track in registry.get("tracks", ())
        if track.get("track_id")
        == FINAL_HEADLINE_TRACK
    ), {})
    _require(
        bool(final_track),
        "experiment registry is missing the headline factorial track",
        failures,
    )
    control_track = next((
        track for track in registry.get("tracks", ())
        if track.get("track_id") == FINAL_CONTROL_TRACK
    ), {})
    _require(
        bool(control_track),
        "experiment registry is missing the frozen common-Sobol control",
        failures,
    )
    _require(
        set(final_track.get("expected_method_identities", ()))
        == FINAL_HEADLINE_METHODS
        and set(control_track.get("expected_method_identities", ()))
        == FINAL_CONTROL_METHODS,
        "frozen headline/control backend identities drifted",
        failures,
    )
    _require(
        convergence.get("contract_id")
        == "post_run_search_convergence_v1"
        and convergence.get("status") == "complete",
        "final search convergence artifact is incomplete",
        failures,
    )
    _require(
        convergence.get("track_id")
        == FINAL_HEADLINE_TRACK,
        "convergence artifact does not describe the headline track",
        failures,
    )
    _require(
        set(convergence.get("method_identities", ()))
        == set(final_track.get("expected_method_identities", ())),
        "convergence artifact does not contain every headline method",
        failures,
    )
    final_records = [
        row for row in audit.get("records", ())
        if row.get("track_id")
        == FINAL_HEADLINE_TRACK
    ]
    control_records = [
        row for row in audit.get("records", ())
        if row.get("track_id") == FINAL_CONTROL_TRACK
    ]
    expected_final_results = (
        len(final_track.get("expected_method_identities", ()))
        * len(final_track.get("expected_domains", ()))
        * len(final_track.get("expected_dimensions", (1,)))
        * len(final_track.get("expected_seeds", ()))
    )
    _require(
        len(final_records) == expected_final_results
        and expected_final_results > 0,
        "compact audit does not contain the complete headline factorial",
        failures,
    )
    expected_control_results = (
        len(control_track.get("expected_method_identities", ()))
        * len(control_track.get("expected_domains", ()))
        * len(control_track.get("expected_dimensions", (1,)))
        * len(control_track.get("expected_seeds", ()))
    )
    _require(
        len(control_records) == expected_control_results
        and expected_control_results > 0,
        "compact audit does not contain the complete frozen Sobol control",
        failures,
    )
    frozen_rows = final_records + control_records
    _require(
        bool(frozen_rows)
        and all(
            row.get("execution_provenance_status") == "frozen"
            and row.get("execution_repository_commit")
            == execution_snapshot.get("repository_commit")
            and row.get("execution_scolhkg_tree")
            == execution_snapshot.get("scolhkg_tree")
            and row.get("execution_proof_tree")
            == execution_snapshot.get("proof_tree")
            and row.get("execution_scripts_tree")
            == execution_snapshot.get("scripts_tree")
            and row.get("execution_theory_contract_id")
            == execution_snapshot.get("theory_contract_id")
            for row in frozen_rows
        ),
        "one or more final replay rows lack exact frozen execution "
        "provenance",
        failures,
    )
    _require(
        int(convergence.get("result_count", 0)) == len(final_records)
        and int(convergence.get("completed_trace_count", 0))
        == len(final_records),
        "convergence artifact does not contain every headline result",
        failures,
    )
    _require(
        int(convergence.get("trace_row_count", -1))
        == int(convergence.get("expected_trace_row_count", -2)),
        "convergence artifact does not cover every target search call",
        failures,
    )
    _require(
        int(convergence.get(
            "terminal_validation_failure_count", -1)) == 0,
        "convergence reconstruction failed a terminal truth check",
        failures,
    )
    _require(
        convergence.get("target_truth_used_post_run_only") is True
        and convergence.get(
            "target_truth_used_for_search_or_selection") is False
        and convergence.get("verification_samples_included") is False
        and convergence.get("policy_vectors_exported") is False,
        "convergence artifact violates the post-run compact-data contract",
        failures,
    )
    expected_result_receipt = hashlib.sha256(json.dumps(
        sorted(str(row["result_sha256"]) for row in final_records),
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    _require(
        convergence.get("result_receipts_sha256")
        == expected_result_receipt,
        "convergence result receipts differ from the compact audit",
        failures,
    )
    _require(
        proof_receipt.get("status") == "pass",
        "Lean proof receipt did not pass",
        failures,
    )
    _require(
        proof_receipt.get("forbidden_declaration_count") == 0,
        "Lean proof tree contains sorry, admit, or axiom",
        failures,
    )
    _require(
        proof_receipt.get("build", {}).get("executed") is True
        and proof_receipt.get("build", {}).get("returncode") == 0,
        "Lean proof receipt does not contain a successful lake build",
        failures,
    )
    _require(
        traffic.get("status") == "complete",
        "strict no-history traffic audit is incomplete",
        failures,
    )
    _require(
        int(traffic.get("n_seeds", 0)) >= 20,
        "strict no-history traffic audit has fewer than 20 search seeds",
        failures,
    )
    _require(
        int(traffic.get("source_calls_per_run", -1)) == 384
        and int(traffic.get("target_search_calls_per_run", -1)) == 13,
        "traffic source/search budgets differ from the registered contract",
        failures,
    )
    _require(
        traffic.get("policy_vectors_exported") is False,
        "traffic release artifact contains policy vectors",
        failures,
    )
    traffic_information = traffic.get("information_contract", {})
    _require(
        traffic_information.get("track")
        == "descriptor_conditional_external_holdout"
        and traffic_information.get("source_selection_mode")
        == "descriptor_nearest"
        and traffic_information.get(
            "heldout_task_family_identifier_used_by_proposal") is True
        and traffic_information.get(
            "target_labels_used_to_fit_proposal") is False
        and traffic_information.get("target_oracle_used") is False
        and traffic_information.get("historical_target_anchor_used") is False
        and traffic_information.get("evidence_phase")
        == "confirmatory_holdout"
        and traffic_information.get(
            "method_selected_using_target_domain_development_results") is True
        and traffic_information.get(
            "evaluation_outcomes_used_for_method_selection") is False
        and traffic_information.get(
            "confirmatory_holdout_seed_disjoint_from_development") is True
        and traffic_information.get(
            "excluded_nearest_source_analogue")
        == "FactorShockStatePolicyRZDT1"
        and traffic_information.get("source_split_heldout")
        == "FactorShockStatePolicyRZDT1"
        and traffic_information.get("source_domains")
        == ["QueueResourceControl", "InventorySupplyChain"],
        "traffic experiment is not the registered disjoint-seed, "
        "descriptor-conditioned external holdout",
        failures,
    )
    _require(
        traffic_negative_control.get("status") == "complete",
        "domain-blind traffic negative control is incomplete",
        failures,
    )
    _require(
        int(traffic_negative_control.get("n_seeds", 0)) >= 5,
        "domain-blind traffic negative control has fewer than 5 search seeds",
        failures,
    )
    _require(
        int(traffic_negative_control.get("source_calls_per_run", -1)) == 384
        and int(traffic_negative_control.get(
            "target_search_calls_per_run", -1)) == 13,
        "domain-blind traffic negative-control budgets differ from the "
        "registered contract",
        failures,
    )
    _require(
        traffic_negative_control.get("policy_vectors_exported") is False,
        "traffic negative-control artifact contains policy vectors",
        failures,
    )
    negative_information = traffic_negative_control.get(
        "information_contract", {})
    _require(
        negative_information.get("track")
        == "domain_blind_external_holdout"
        and negative_information.get(
            "heldout_task_family_identifier_used_by_proposal") is False
        and negative_information.get(
            "target_labels_used_to_fit_proposal") is False
        and negative_information.get("target_oracle_used") is False
        and negative_information.get(
            "historical_target_anchor_used") is False
        and negative_information.get(
            "excluded_nearest_source_analogue")
        == "QueueResourceControl"
        and negative_information.get("source_domains")
        == [
            "FactorShockStatePolicyRZDT1",
            "InventorySupplyChain",
        ],
        "traffic negative control is not the registered domain-blind "
        "nearest-analogue exclusion",
        failures,
    )
    _require(
        proposal_coverage.get("contract_id")
        == "source_target_geometric_atlas_coverage_v1",
        "proposal coverage audit uses the wrong theory contract",
        failures,
    )
    _require(
        proposal_coverage.get("status") in {
            "complete",
            "complete_with_conditional_global_bound",
        },
        "proposal coverage audit is incomplete",
        failures,
    )
    proposal_rows = list(proposal_coverage.get("rows", ()))
    _require(
        len(proposal_rows) == 3
        and int(proposal_coverage.get(
            "finite_library_condition_pass_count", 0)) == 3,
        "proposal coverage audit lacks three passing finite-library domains",
        failures,
    )
    _require(
        bool(proposal_rows)
        and all(
            row.get("deterministic_atlas") is True
            and row.get("target_truth_used_post_run_only") is True
            and row.get(
                "target_truth_used_for_proposal_or_selection") is False
            for row in proposal_rows
        ),
        "proposal coverage audit violates the frozen post-run truth contract",
        failures,
    )
    if proposal_coverage.get(
        "unconditional_global_coverage_claim_allowed"
    ) is not True:
        _require(
            proposal_coverage.get("global_theorem_claim_mode")
            == "conditional_theorem_only",
            "uncertified global proposal coverage is not labelled conditional",
            failures,
        )
    traffic_rows = list(traffic.get("rows", ()))
    _require(
        bool(traffic_rows)
        and all(
            row.get("certificate")
            == "one_sided_clopper_pearson_bonferroni"
            and row.get("fixed_shortlist_order") is True
            and row.get("verification_samples_update_optimizer") is False
            and row.get(
                "verification_samples_used_to_reorder_shortlist") is False
            for row in traffic_rows
        ),
        "traffic verifier is not a fixed fresh-seed external certificate",
        failures,
    )
    negative_traffic_rows = list(traffic_negative_control.get("rows", ()))
    _require(
        bool(negative_traffic_rows)
        and all(
            row.get("certificate")
            == "one_sided_clopper_pearson_bonferroni"
            and row.get("fixed_shortlist_order") is True
            and row.get("verification_samples_update_optimizer") is False
            and row.get(
                "verification_samples_used_to_reorder_shortlist") is False
            for row in negative_traffic_rows
        ),
        "traffic negative-control verifier is not a fixed fresh-seed "
        "external certificate",
        failures,
    )
    uniform_rows = [
        row for row in audit.get("records", ())
        if row.get("track_id")
        == "uniform_external_total_cost_d1000_n397"
    ]
    _require(
        len(uniform_rows) == 240,
        "uniform total-cost verifier does not contain 240 results",
        failures,
    )
    _require(
        all(
            row.get("optimization_calls_excluding_verification") == 397
            for row in uniform_rows
        ),
        "uniform total-cost comparison violates the 397-call budget",
        failures,
    )
    _require(
        len({
            row.get("verifier_signature") for row in uniform_rows
        }) == 1,
        "uniform total-cost comparison uses different verifiers",
        failures,
    )
    hvd_gate = hvd_causal_gate.get("gate", {})
    try:
        _hvd_release_role(hvd_causal_gate)
        hvd_decision_consistent = True
    except ValueError:
        hvd_decision_consistent = False
    _require(
        int(hvd_gate.get("complete_pair_count", 0)) == 60
        and hvd_gate.get("all_expected_pairs_present") is True
        and hvd_gate.get("all_rows_paired") is True
        and hvd_gate.get("false_certification_not_harmed") is True
        and hvd_decision_consistent,
        "HVD causal gate is incomplete, unsafe, or inconsistent with the "
        "preregistered all-domain decision rule",
        failures,
    )
    expected_frontier = dimension_frontier.get("expected", {})
    frontier_gates = list(
        dimension_frontier.get("gates", {}).values())
    _require(
        dimension_frontier.get("status") == "complete"
        and set(expected_frontier.get("dimensions", ())) >= {200, 1000}
        and set(expected_frontier.get("budgets", ())) >= {10, 20, 40, 80}
        and len(expected_frontier.get("seeds", ())) >= 20
        and bool(frontier_gates)
        and all(
            gate.get("all_rows_ok") is True
            and gate.get("false_certification_free") is True
            for gate in frontier_gates
        ),
        "dimension/budget frontier lacks the complete 20-seed registered "
        "matrix",
        failures,
    )
    output_rows = list(artifact_manifest.get("outputs", ()))
    render_input = artifact_manifest.get(
        "inputs", {}).get("audit_export_manifest") or {}
    rendered_statistics = artifact_manifest.get(
        "inputs", {}).get("paired_statistics") or {}
    _require(
        artifact_manifest.get("status") == "complete"
        and artifact_manifest.get("contracts", {}).get(
            "reads_checkpoints") is False
        and artifact_manifest.get("contracts", {}).get(
            "reads_pickle_or_model_weights") is False
        and artifact_manifest.get("contracts", {}).get(
            "post_run_truth_not_used_for_decisions") is True
        and artifact_manifest.get("contracts", {}).get(
            "rows_from_passed_registered_paper_audit") is True
        and artifact_manifest.get("contracts", {}).get(
            "paired_statistics_preregistered") is True
        and render_input.get("contract_id")
        == "audited_compact_render_input_v1"
        and rendered_statistics.get("status") == "complete"
        and bool(output_rows),
        "publication artifact manifest is incomplete, unaudited, or reads "
        "runtime state",
        failures,
    )
    render_input_path = Path(str(render_input.get("path", "")))
    _require(
        render_input_path.is_file()
        and _sha256(render_input_path) == render_input.get("sha256"),
        "audited render-input manifest is missing or changed",
        failures,
    )
    rendered_statistics_path = Path(str(
        rendered_statistics.get("path", "")))
    _require(
        rendered_statistics_path.is_file()
        and _sha256(rendered_statistics_path)
        == rendered_statistics.get("sha256"),
        "preregistered paired-statistics input is missing or changed",
        failures,
    )
    artifact_root = Path(str(artifact_manifest.get(
        "_manifest_path", ""))).parent
    for output in output_rows:
        path = artifact_root / str(output.get("name", ""))
        if not path.is_file():
            failures.append(f"rendered artifact is missing: {path}")
        elif _sha256(path) != output.get("sha256"):
            failures.append(f"rendered artifact hash changed: {path}")
    source_mode = audit.get("source_mode", "local_result_files")
    if source_mode == "remote_compact_record_shards":
        shard_receipts = list(audit.get("record_shard_receipts", ()))
        _require(
            bool(shard_receipts),
            "remote compact audit has no record-shard receipts",
            failures,
        )
        for receipt in shard_receipts:
            path = Path(str(receipt.get("path", "")))
            if not path.is_file():
                failures.append(f"compact record shard is missing: {path}")
                continue
            if _sha256(path) != receipt.get("sha256"):
                failures.append(
                    f"compact record shard changed after audit: {path}")
        _require(
            all(
                record.get("content_verified_at_extraction") is True
                and isinstance(record.get("result_sha256"), str)
                and len(record["result_sha256"]) == 64
                for record in audit.get("records", ())
            ),
            "remote compact audit contains an unverified result receipt",
            failures,
        )
    else:
        _require(
            source_mode == "local_result_files",
            f"unsupported compact audit source mode: {source_mode}",
            failures,
        )
        for record in audit.get("records", ()):
            path = Path(str(record.get("path", "")))
            if not path.is_file():
                failures.append(f"audited result is missing: {path}")
                continue
            if _sha256(path) != record.get("result_sha256"):
                failures.append(f"audited result changed after audit: {path}")
    return failures


def _record_receipts(audit):
    rows = []
    for record in audit.get("records", ()):
        rows.append({
            "track_id": record["track_id"],
            "method_identity": record["method_identity"],
            "domain": record["domain"],
            "target_dimension": record["target_dimension"],
            "seed": record["seed"],
            "source_calls": record["source_calls"],
            "target_search_calls": record["target_search_calls"],
            "target_verification_calls": record[
                "target_verification_calls"],
            "optimization_calls_excluding_verification": record[
                "optimization_calls_excluding_verification"],
            "execution_repository_commit": record.get(
                "execution_repository_commit"),
            "execution_scolhkg_tree": record.get(
                "execution_scolhkg_tree"),
            "execution_method_contract_id": record.get(
                "execution_method_contract_id"),
            "execution_theory_contract_id": record.get(
                "execution_theory_contract_id"),
            "result_sha256": record["result_sha256"],
        })
    return sorted(rows, key=lambda row: (
        row["track_id"],
        row["method_identity"],
        row["domain"],
        -1 if row["target_dimension"] is None else row["target_dimension"],
        -1 if row["seed"] is None else row["seed"],
    ))


def build_release(
    *,
    method_contract_path,
    registry_path,
    audit_path,
    statistics_path,
    convergence_path,
    traffic_path,
    traffic_negative_control_path,
    proposal_coverage_path,
    proof_receipt_path,
    execution_snapshot_path,
    hvd_causal_gate_path,
    dimension_frontier_path,
    artifact_manifest_path,
    repository_commit,
):
    method_contract = _load(method_contract_path)
    registry = _load(registry_path)
    audit = _load(audit_path)
    statistics = _load(statistics_path)
    convergence = _load(convergence_path)
    traffic = _load(traffic_path)
    traffic_negative_control = _load(traffic_negative_control_path)
    proposal_coverage = _load(proposal_coverage_path)
    proof_receipt = _load(proof_receipt_path)
    execution_snapshot = _load(execution_snapshot_path)
    hvd_causal_gate = _load(hvd_causal_gate_path)
    dimension_frontier = _load(dimension_frontier_path)
    artifact_manifest = _load(artifact_manifest_path)
    artifact_manifest["_manifest_path"] = str(artifact_manifest_path)
    failures = validate_release_inputs(
        method_contract,
        registry,
        audit,
        statistics,
        convergence,
        traffic,
        traffic_negative_control,
        proposal_coverage,
        proof_receipt,
        execution_snapshot,
        hvd_causal_gate,
        dimension_frontier,
        artifact_manifest,
    )
    if failures:
        raise ValueError(
            "paper release validation failed:\n- " + "\n- ".join(failures))
    hvd_release_role = _hvd_release_role(hvd_causal_gate)
    traffic_execution = traffic["execution_provenance"]
    paths = {
        "method_contract": Path(method_contract_path),
        "experiment_registry": Path(registry_path),
        "compact_audit": Path(audit_path),
        "paired_statistics": Path(statistics_path),
        "search_convergence": Path(convergence_path),
        "external_traffic_audit": Path(traffic_path),
        "external_traffic_negative_control": Path(
            traffic_negative_control_path),
        "proposal_coverage_audit": Path(proposal_coverage_path),
        "lean_proof_receipt": Path(proof_receipt_path),
        "execution_snapshot": Path(execution_snapshot_path),
        "hvd_causal_gate": Path(hvd_causal_gate_path),
        "dimension_budget_frontier": Path(dimension_frontier_path),
        "rendered_artifact_manifest": Path(artifact_manifest_path),
    }
    records = _record_receipts(audit)
    record_digest = hashlib.sha256(
        json.dumps(
            records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "status": "ready_for_manuscript_lock",
        "release_contract_id": "or_submission_evidence_release_v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repository_commit": str(repository_commit),
        "headline_method_contract_id": FINAL_CONTRACT_ID,
        "headline_method": (
            "source-learned dimension-equivariant structural proposal + "
            "canonical SAASBO backend + independent verifier"
        ),
        "execution_snapshot": {
            key: execution_snapshot[key]
            for key in (
                "repository_commit",
                "scolhkg_tree",
                "proof_tree",
                "scripts_tree",
                "method_contract_id",
                "theory_contract_id",
            )
        },
        "claim_boundaries": {
            **method_contract["claim_boundaries"],
            "cumulative_hvd": hvd_release_role,
        },
        "registry_id": registry["registry_id"],
        "registered_track_count": len(registry["tracks"]),
        "registered_comparison_count": len(
            registry.get("primary_comparisons", ())),
        "audited_result_count": len(records),
        "audited_result_receipt_sha256": record_digest,
        "audited_result_receipts": records,
        "failed_or_timeout_result_count": int(sum(
            row.get("status") != "ok"
            for row in audit.get("records", ()))),
        "false_certificate_count": int(sum(
            bool(row.get("false_certificate"))
            for row in audit.get("records", ()))),
        "external_traffic": {
            "n_seeds": int(traffic["n_seeds"]),
            "certified_seed_count": int(
                traffic["certified_seed_count"]),
            "source_calls_per_run": int(
                traffic["source_calls_per_run"]),
            "target_search_calls_per_run": int(
                traffic["target_search_calls_per_run"]),
            "target_verification_calls_per_run": int(
                traffic["target_verification_calls_per_run"]),
            "information_contract": traffic["information_contract"],
            "execution_snapshot": {
                key: traffic_execution[key]
                for key in (
                    "repository_commit",
                    "scolhkg_tree",
                    "proof_tree",
                    "scripts_tree",
                    "legacy_traffic_tree",
                    "traffic_decision_space_blob",
                    "traffic_baseline_blob",
                )
            },
        },
        "external_traffic_negative_control": {
            "n_seeds": int(traffic_negative_control["n_seeds"]),
            "certified_seed_count": int(
                traffic_negative_control["certified_seed_count"]),
            "source_calls_per_run": int(
                traffic_negative_control["source_calls_per_run"]),
            "target_search_calls_per_run": int(
                traffic_negative_control["target_search_calls_per_run"]),
            "target_verification_calls_per_run": int(
                traffic_negative_control[
                    "target_verification_calls_per_run"]),
            "information_contract": traffic_negative_control[
                "information_contract"],
        },
        "proposal_coverage": {
            "contract_id": proposal_coverage["contract_id"],
            "domain_count": int(proposal_coverage["domain_count"]),
            "finite_library_condition_pass_count": int(
                proposal_coverage[
                    "finite_library_condition_pass_count"]),
            "global_lipschitz_certified_count": int(
                proposal_coverage[
                    "global_lipschitz_certified_count"]),
            "global_theorem_claim_mode": proposal_coverage[
                "global_theorem_claim_mode"],
            "unconditional_global_coverage_claim_allowed": bool(
                proposal_coverage[
                    "unconditional_global_coverage_claim_allowed"]),
        },
        "search_convergence": {
            "contract_id": convergence["contract_id"],
            "track_id": convergence["track_id"],
            "result_count": int(convergence["result_count"]),
            "trace_row_count": int(convergence["trace_row_count"]),
            "target_truth_used_post_run_only": True,
            "verification_samples_included": False,
            "result_receipts_sha256": convergence[
                "result_receipts_sha256"],
        },
        "proof": {
            "lean_source_count": int(
                proof_receipt["lean_source_count"]),
            "lean_source_tree_sha256": proof_receipt[
                "lean_source_tree_sha256"],
            "lake_build_returncode": int(
                proof_receipt["build"]["returncode"]),
            "forbidden_declaration_count": int(
                proof_receipt["forbidden_declaration_count"]),
        },
        "hvd_causal_decision": {
            "complete_pair_count": int(
                hvd_causal_gate["gate"]["complete_pair_count"]),
            "promote_hvd_as_core": bool(
                hvd_causal_gate["gate"]["promote_hvd_as_core"]),
            "paper_role": hvd_release_role,
        },
        "dimension_budget_frontier": {
            "dimensions": dimension_frontier["expected"]["dimensions"],
            "budgets": dimension_frontier["expected"]["budgets"],
            "seed_count": len(
                dimension_frontier["expected"]["seeds"]),
        },
        "rendered_artifact_count": len(
            artifact_manifest["outputs"]),
        "artifact_sha256": {
            name: _sha256(path) for name, path in paths.items()
        },
        "statistics_holm_family": statistics["holm_family"],
        "statistics_rows": statistics["rows"],
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-contract", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--convergence", required=True)
    parser.add_argument("--traffic", required=True)
    parser.add_argument("--traffic-negative-control", required=True)
    parser.add_argument("--proposal-coverage", required=True)
    parser.add_argument("--proof-receipt", required=True)
    parser.add_argument("--execution-snapshot", required=True)
    parser.add_argument("--hvd-causal-gate", required=True)
    parser.add_argument("--dimension-frontier", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repository_root,
        text=True,
    ).strip()
    release = build_release(
        method_contract_path=args.method_contract,
        registry_path=args.registry,
        audit_path=args.audit,
        statistics_path=args.statistics,
        convergence_path=args.convergence,
        traffic_path=args.traffic,
        traffic_negative_control_path=args.traffic_negative_control,
        proposal_coverage_path=args.proposal_coverage,
        proof_receipt_path=args.proof_receipt,
        execution_snapshot_path=args.execution_snapshot,
        hvd_causal_gate_path=args.hvd_causal_gate,
        dimension_frontier_path=args.dimension_frontier,
        artifact_manifest_path=args.artifact_manifest,
        repository_commit=commit,
    )
    _atomic_json(args.out, release)
    print(json.dumps({
        "status": release["status"],
        "repository_commit": release["repository_commit"],
        "audited_result_count": release["audited_result_count"],
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
