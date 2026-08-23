import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import runners.run_structural_hypothesis_recursive_report_advance as runner  # noqa: E402


RUNNER = (
    ROOT / "runners/run_structural_hypothesis_recursive_report_advance.py"
)
CORE = (
    ROOT / "performance/structural_hypothesis_recursive_report_advance.py"
)
ADVANCE_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_recursive_report_advance_v1.json"
)
ADOPTION_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_report_adoption_v1.json"
)
SUCCESSOR_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_adopted_successor_materializer_v1.json"
)
BRIDGE_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_successor_bound_single_task_v1.json"
)
PUBLISHER_CONTRACT = (
    ROOT
    / "performance/manifests/"
    "structural_hypothesis_reingestion_publisher_v1.json"
)
HYPOTHESIS_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_loop_v1.json"
)
EXECUTOR_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_executor_v1.json"
)
MATERIALIZER_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_task_materializer_v1.json"
)
RUNTIME_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_single_task_runtime_v1.json"
)
BASE_MANIFEST = ROOT / "performance/manifests/v18b_exactkg_mcdiag.json"
ASSET_ROOT = (
    ROOT / "performance/task_inputs/structural_hypothesis_materializer_v1"
)

ADVANCE_STATUS = (
    "ADVANCED_AS_IMMUTABLE_LOCAL_REPORT_VERSION_"
    "NOT_CURRENT_NOT_PLANNED"
)
VERIFY_STATUS = "VERIFIED_" + ADVANCE_STATUS

PROFILES = (
    "none",
    "low_frequency_only",
    "orthogonality_only",
    "sparsity_only",
    "additivity_only",
    "leave_out_low_frequency",
    "leave_out_orthogonality",
    "leave_out_sparsity",
    "leave_out_additivity",
)
BASE_CSV_FIELDS = (
    "run_id", "track", "variant", "method", "implementation",
    "initial_design", "domain", "seed", "d", "N", "n0",
    "source_calls", "total_calls", "d_over_target_calls",
    "d_over_total_calls", "status", "true_feasible", "feasible_regret",
    "true_objective", "constraint_violation", "initial_has_true_feasible",
    "initial_true_feasible_count", "initial_best_feasible_regret",
    "adaptive_rescue", "adaptive_loss", "adaptive_improves_initial_best",
    "adaptive_regret_change", "posterior_feasible",
    "posterior_certificate_vacuous", "posterior_certified_count",
    "false_certificate_count", "certificate_precision",
    "certificate_recall", "decision_backend", "structural_prior_profile",
    "hvd_profile", "source_discrepancy_update", "recheck_top_k",
    "risk_penalty", "utility_weight", "adaptive_replication_voi",
    "adaptive_replication_count", "adaptive_new_point_count",
    "posterior_dominance_enabled", "posterior_dominance_switch_count",
    "wall_time_sec", "result_path",
)


def _digest(character):
    return "sha256:" + character * 64


def _canonical_digest(value):
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _raw_digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_observation(root):
    if not root.exists():
        return None
    return {
        str(path.relative_to(root)): (
            "directory" if path.is_dir() else path.read_bytes(),
            path.stat().st_mode,
            path.stat().st_nlink,
        )
        for path in root.rglob("*")
    }


def _json_file(path, payload=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {}), encoding="utf-8")
    return path.resolve()


def _unit_case(tmp_path, *, committed=False):
    directories = {
        name: (tmp_path / name).resolve()
        for name in (
            "publication",
            "adoption",
            "successor",
            "source-attempt",
            "assets",
            "completed-attempt",
        )
    }
    for path in directories.values():
        path.mkdir(parents=True)
    advance_root = (tmp_path / "advance").resolve()
    if committed:
        advance_root.mkdir()
    files = {
        name: _json_file(tmp_path / f"{name}.json")
        for name in (
            "adoption-contract",
            "successor-contract",
            "hypothesis-contract",
            "executor-contract",
            "runtime-contract",
            "publisher-contract",
            "materializer-contract",
            "bridge-contract",
            "base-manifest",
            "advance-contract",
        )
    }
    evidence = tmp_path / "base.csv"
    evidence.write_text("track,run_id\npriors,fake\n", encoding="utf-8")
    return {
        **directories,
        **files,
        "evidence": evidence.resolve(),
        "advance-root": advance_root,
        "advance_id": "advance-unit-0001",
        "adoption_id": "adoption-unit-0001",
        "successor_id": "successor-unit-0001",
        "task_id": "task:" + "a" * 24,
        "digests": {
            name: _digest(character)
            for name, character in zip(
                (
                    "adoption",
                    "pending",
                    "projection",
                    "successor",
                    "bundle",
                    "plan",
                    "task",
                    "provenance",
                    "authorization",
                    "attempt",
                    "receipt",
                    "journal",
                    "advance",
                    "reingestion",
                    "report",
                    "audit",
                    "evidence",
                ),
                "123456789abcdef01",
            )
        },
    }


def _unit_common_cli(case):
    digests = case["digests"]
    return [
        "--publication-root", str(case["publication"]),
        "--adoption-contract", str(case["adoption-contract"]),
        "--adoption-root", str(case["adoption"]),
        "--adoption-id", case["adoption_id"],
        "--successor-contract", str(case["successor-contract"]),
        "--successor-root", str(case["successor"]),
        "--successor-id", case["successor_id"],
        "--base-evidence-csv", str(case["evidence"]),
        "--source-attempt-root", str(case["source-attempt"]),
        "--hypothesis-contract", str(case["hypothesis-contract"]),
        "--executor-contract", str(case["executor-contract"]),
        "--runtime-contract", str(case["runtime-contract"]),
        "--publisher-contract", str(case["publisher-contract"]),
        "--materializer-contract", str(case["materializer-contract"]),
        "--bridge-contract", str(case["bridge-contract"]),
        "--base-manifest", str(case["base-manifest"]),
        "--asset-root", str(case["assets"]),
        "--completed-attempt-root", str(case["completed-attempt"]),
        "--advance-contract", str(case["advance-contract"]),
        "--advance-root", str(case["advance-root"]),
        "--advance-id", case["advance_id"],
        "--expected-adoption-digest", digests["adoption"],
        "--expected-pending-evidence-digest", digests["pending"],
        "--expected-first-pending-projection-digest", digests["projection"],
        "--expected-successor-digest", digests["successor"],
        "--expected-bundle-digest", digests["bundle"],
        "--expected-plan-digest", digests["plan"],
        "--task-id", case["task_id"],
        "--expected-task-digest", digests["task"],
        "--expected-provenance-binding-digest", digests["provenance"],
        "--expected-authorization-digest", digests["authorization"],
        "--expected-attempt-digest", digests["attempt"],
        "--expected-execution-receipt-digest", digests["receipt"],
        "--expected-execution-journal-head-digest", digests["journal"],
    ]


def _unit_verify_cli(case):
    digests = case["digests"]
    return [
        *_unit_common_cli(case),
        "--expected-advance-digest", digests["advance"],
        "--expected-reingestion-digest", digests["reingestion"],
        "--expected-output-report-body-digest", digests["report"],
        "--expected-output-audit-head", digests["audit"],
        "--expected-output-evidence-digest", digests["evidence"],
    ]


def _unit_result(case, *, verified=False):
    digests = case["digests"]
    return {
        "status": VERIFY_STATUS if verified else ADVANCE_STATUS,
        "advance_root": str(case["advance-root"]),
        "advance_digest": digests["advance"],
        "reingestion_digest": digests["reingestion"],
        "output_report_body_digest": digests["report"],
        "output_audit_head": digests["audit"],
        "output_evidence_digest": digests["evidence"],
        "typed_row_count": 1352,
        "pending_evidence_count": 28,
        "current_status": "NOT_CURRENT",
        "planning_status": "NOT_PLANNED",
    }


def _unit_expected_positional(case):
    return (
        case["publication"],
        case["adoption-contract"],
        case["adoption"],
        case["successor-contract"],
        case["successor"],
        case["evidence"],
        case["source-attempt"],
        case["hypothesis-contract"],
        case["executor-contract"],
        case["runtime-contract"],
        case["publisher-contract"],
        case["materializer-contract"],
        case["bridge-contract"],
        case["base-manifest"],
        case["assets"],
        case["completed-attempt"],
        case["advance-contract"],
        case["advance-root"],
    )


def _unit_expected_kwargs(case):
    digests = case["digests"]
    return {
        "advance_id": case["advance_id"],
        "adoption_id": case["adoption_id"],
        "successor_id": case["successor_id"],
        "expected_adoption_digest": digests["adoption"],
        "expected_pending_evidence_digest": digests["pending"],
        "expected_first_pending_projection_digest": digests["projection"],
        "expected_successor_digest": digests["successor"],
        "expected_bundle_digest": digests["bundle"],
        "expected_plan_digest": digests["plan"],
        "expected_task_digest": digests["task"],
        "expected_provenance_binding_digest": digests["provenance"],
        "expected_authorization_digest": digests["authorization"],
        "expected_attempt_digest": digests["attempt"],
        "expected_execution_receipt_digest": digests["receipt"],
        "expected_execution_journal_head_digest": digests["journal"],
        "task_id": case["task_id"],
    }


def _real_common_cli(case, *, advance_root=None, advance_id=None):
    expected = case["expected"]
    target = advance_root or case["advance_root"]
    target_id = advance_id or case["advance_id"]
    return [
        "--publication-root", str(case["publication"]),
        "--adoption-contract", str(ADOPTION_CONTRACT),
        "--adoption-root", str(case["adoption"]),
        "--adoption-id", case["adoption_id"],
        "--successor-contract", str(SUCCESSOR_CONTRACT),
        "--successor-root", str(case["successor"]),
        "--successor-id", case["successor_id"],
        "--base-evidence-csv", str(case["evidence"]),
        "--source-attempt-root", str(case["source_attempt"]),
        "--hypothesis-contract", str(HYPOTHESIS_CONTRACT),
        "--executor-contract", str(EXECUTOR_CONTRACT),
        "--runtime-contract", str(RUNTIME_CONTRACT),
        "--publisher-contract", str(PUBLISHER_CONTRACT),
        "--materializer-contract", str(MATERIALIZER_CONTRACT),
        "--bridge-contract", str(BRIDGE_CONTRACT),
        "--base-manifest", str(BASE_MANIFEST),
        "--asset-root", str(ASSET_ROOT),
        "--completed-attempt-root", str(case["completed_attempt"]),
        "--advance-contract", str(ADVANCE_CONTRACT),
        "--advance-root", str(target),
        "--advance-id", target_id,
        "--expected-adoption-digest", expected["adoption_digest"],
        "--expected-pending-evidence-digest",
        expected["pending_evidence_digest"],
        "--expected-first-pending-projection-digest",
        expected["first_pending_projection_digest"],
        "--expected-successor-digest", expected["successor_digest"],
        "--expected-bundle-digest", expected["bundle_digest"],
        "--expected-plan-digest", expected["plan_digest"],
        "--task-id", expected["task_id"],
        "--expected-task-digest", expected["task_digest"],
        "--expected-provenance-binding-digest",
        expected["provenance_binding_digest"],
        "--expected-authorization-digest", expected["authorization_digest"],
        "--expected-attempt-digest", expected["attempt_digest"],
        "--expected-execution-receipt-digest",
        expected["execution_receipt_digest"],
        "--expected-execution-journal-head-digest",
        expected["execution_journal_head_digest"],
    ]


def _real_verify_cli(case, advanced):
    return [
        *_real_common_cli(case),
        "--expected-advance-digest", advanced["advance_digest"],
        "--expected-reingestion-digest", advanced["reingestion_digest"],
        "--expected-output-report-body-digest",
        advanced["output_report_body_digest"],
        "--expected-output-audit-head", advanced["output_audit_head"],
        "--expected-output-evidence-digest",
        advanced["output_evidence_digest"],
    ]


def _real_source_positional(case):
    return (
        case["publication"],
        ADOPTION_CONTRACT,
        case["adoption"],
        SUCCESSOR_CONTRACT,
        case["successor"],
        case["evidence"],
        case["source_attempt"],
        HYPOTHESIS_CONTRACT,
        EXECUTOR_CONTRACT,
        RUNTIME_CONTRACT,
        PUBLISHER_CONTRACT,
        MATERIALIZER_CONTRACT,
        BRIDGE_CONTRACT,
        BASE_MANIFEST,
        ASSET_ROOT,
        case["completed_attempt"],
    )


def _real_core_expected(case):
    expected = case["expected"]
    return {
        "expected_adoption_digest": expected["adoption_digest"],
        "expected_pending_evidence_digest": expected[
            "pending_evidence_digest"
        ],
        "expected_first_pending_projection_digest": expected[
            "first_pending_projection_digest"
        ],
        "expected_successor_digest": expected["successor_digest"],
        "expected_bundle_digest": expected["bundle_digest"],
        "expected_plan_digest": expected["plan_digest"],
        "expected_task_digest": expected["task_digest"],
        "expected_provenance_binding_digest": expected[
            "provenance_binding_digest"
        ],
        "expected_authorization_digest": expected[
            "authorization_digest"
        ],
        "expected_attempt_digest": expected["attempt_digest"],
        "expected_execution_receipt_digest": expected[
            "execution_receipt_digest"
        ],
        "expected_execution_journal_head_digest": expected[
            "execution_journal_head_digest"
        ],
    }


def _synthetic_row(scope, profile, domain, seed):
    row = {field: "" for field in BASE_CSV_FIELDS}
    row.update({
        "run_id": scope["run_id"],
        "track": scope["track"],
        "variant": scope["variant_template"].format(profile=profile),
        "method": profile,
        "implementation": scope["implementation"],
        "initial_design": scope["initial_design"],
        "domain": domain,
        "seed": str(seed),
        "d": str(scope["d"]),
        "N": str(scope["N"]),
        "n0": str(scope["n0"]),
        "source_calls": str(scope["source_calls"]),
        "d_over_target_calls": "0.0",
        "d_over_total_calls": "0.0",
        "status": "ok",
        "true_feasible": "True" if seed < 9 else "False",
        "feasible_regret": "1.0",
        "true_objective": "0.0",
        "constraint_violation": "0.0",
        "initial_has_true_feasible": "True",
        "initial_true_feasible_count": "1",
        "initial_best_feasible_regret": "1.0",
        "adaptive_rescue": "False",
        "adaptive_loss": "False",
        "adaptive_improves_initial_best": "False",
        "adaptive_regret_change": "0.0",
        "posterior_feasible": "True",
        "posterior_certificate_vacuous": "False",
        "posterior_certified_count": "1",
        "false_certificate_count": "0",
        "certificate_precision": "1.0",
        "certificate_recall": "1.0",
        "decision_backend": scope["decision_backend"],
        "structural_prior_profile": profile,
        "adaptive_replication_count": "0",
        "adaptive_new_point_count": "0",
        "wall_time_sec": "0.0",
        "result_path": f"synthetic/{profile}/{domain}/seed{seed}",
    })
    row.update({
        key: str(value) for key, value in scope["fixed_row_values"].items()
    })
    return row


def _write_1350_row_base_csv(path, hypothesis_contract):
    """Write 270 in-scope rows plus 1,080 inert historical rows."""
    scope = hypothesis_contract["evidence_scope"]
    in_scope = [
        _synthetic_row(scope, profile, domain, seed)
        for profile in PROFILES
        for domain in scope["domains"]
        for seed in scope["seeds"]
    ]
    historical = []
    for ordinal in range(1080):
        source = dict(in_scope[ordinal % len(in_scope)])
        source["run_id"] = f"synthetic-historical-{ordinal:04d}"
        source["result_path"] = f"synthetic/history/{ordinal:04d}"
        historical.append(source)
    rows = [*historical, *in_scope]
    assert len(rows) == 1350
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)
    return rows


def _fake_native_result(task):
    args = task["args"]
    return {
        "line": task["line"],
        "heldout": task["heldout"],
        "seed": task["seed"],
        "n_search_simulations": args["N"],
        "n_simulations": args["N"],
        "n0": args["n0"],
        "source_target_adaptation_contract": {
            "source_simulator_calls": 384,
        },
        "true_feasible": True,
        "feasible_simple_regret": 0.0,
        "adaptive_loss": False,
        "decision_backend": args["decision_backend"],
        "structural_prior_profile": args["structural_prior_profile"],
        "hvd_ablation_profile": args["hvd_ablation_profile"],
        "source_discrepancy_update": args["source_discrepancy_update"],
        "certification_recheck_top_k": args[
            "certification_recheck_top_k"
        ],
        "decision_risk_penalty": args["decision_risk_penalty"],
        "decision_source_utility_weight": args[
            "decision_source_utility_weight"
        ],
        "adaptive_replication_voi_enabled": args[
            "adaptive_replication_voi"
        ],
        "posterior_dominance_enabled": args[
            "posterior_dominance_enabled"
        ],
        "posterior_dominance_switch_count": 0,
    }


def _valid_fake_preflight(runtime_core, runtime_contract, task, prepared):
    binding = runtime_contract["runtime_binding"]
    body = {
        "schema_version": runtime_core.PREFLIGHT_SCHEMA_VERSION,
        "status": "PASSED_LOCAL_PREFLIGHT",
        "attempt_digest": prepared["attempt_digest"],
        "authorization_digest": prepared["authorization_digest"],
        "task_id": task["task_id"],
        "task_digest": task["task_digest"],
        "requirements": runtime_contract["preflight"],
        "observed": {
            "affinity_cpu_ids": list(range(12)),
            "affinity_cpu_count": 12,
            "memory_available_bytes": 12884901888,
            "checkpoint_free_bytes": 2147483648,
            "fork_probe_passed": True,
            "fork_probe_process_count": 12,
            "thread_pools": [{
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "version": "test-only",
                "threading_layer": "pthreads",
                "num_threads": 1,
            }],
            "executor_callable": {
                "module": binding["executor_module"],
                "callable": binding["executor_callable"],
                "source_file": str(
                    (ROOT / "performance/benchmark_lodo_meta_prior.py").resolve()
                ),
                "source_sha256": binding[
                    "executor_callable_source_sha256"
                ],
                "code_sha256": binding["executor_callable_code_sha256"],
                "firstlineno": binding["executor_callable_firstlineno"],
            },
            "required_environment": dict(runtime_core._REQUIRED_ENVIRONMENT),
        },
        "nonclaims": [
            "local_preflight_is_not_external_authority",
            "local_preflight_does_not_guarantee_completion",
            "no_runtime_duration_claim",
            "no_peak_memory_claim",
        ],
    }
    return {
        **body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "preflight_digest": runtime_core._digest(body),
        },
    }


def _projection(cell):
    return {
        "profile": cell["profile"],
        "domain": cell["domain"],
        "line": "lodo",
        "seed": cell["seed"],
        "d": cell["d"],
        "N": cell["N"],
        "n0": cell["n0"],
    }


@pytest.fixture
def fake_completed_successor_case(monkeypatch, request):
    """Build the entire 1351/29 -> completed-seed1 chain without run_one."""
    from performance import structural_hypothesis_adopted_successor_materializer as successor_core
    from performance import structural_hypothesis_reingestion_publisher as publisher_core
    from performance import structural_hypothesis_report_adoption as adoption_core
    from performance import structural_hypothesis_single_task_runtime as runtime_core
    from performance import structural_hypothesis_successor_bound_single_task as bridge_core
    from performance.structural_hypothesis_loop import (
        run_structural_hypothesis_loop,
        verify_report_integrity,
    )
    from performance.structural_hypothesis_task_materializer import (
        materialize_task_bundle,
        verify_materialized_task_bundle,
    )
    from runners import run_structural_hypothesis_loop as loop_runner
    from runners import run_structural_hypothesis_single_task as single_runner

    state_home = Path(tempfile.mkdtemp(
        prefix="kgop-recursive-advance-test.", dir="/tmp"
    ))
    state_home.chmod(0o700)
    request.addfinalizer(lambda: shutil.rmtree(state_home, ignore_errors=True))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    for key, value in single_runner.REQUIRED_EXECUTION_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)

    hypothesis, contract_artifact = loop_runner._load_contract(
        HYPOTHESIS_CONTRACT
    )
    evidence = state_home / "base-evidence-1350.csv"
    generated = _write_1350_row_base_csv(evidence, hypothesis)
    base_rows, evidence_artifact = loop_runner._load_evidence(evidence)
    assert len(generated) == len(base_rows) == 1350
    initial_report = run_structural_hypothesis_loop(
        base_rows,
        hypothesis,
        input_artifacts={
            "evidence_csv": evidence_artifact,
            "contract_json": contract_artifact,
        },
    ).to_dict()
    assert verify_report_integrity(initial_report)
    assert len(initial_report["pending_evidence"]) == 30

    executor_contract = json.loads(
        EXECUTOR_CONTRACT.read_text(encoding="utf-8")
    )
    materializer_contract = json.loads(
        MATERIALIZER_CONTRACT.read_text(encoding="utf-8")
    )
    runtime_contract = json.loads(
        RUNTIME_CONTRACT.read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        runtime_core,
        "_load_real_executor",
        lambda _contract: _fake_native_result,
    )

    source_attempt = (
        state_home
        / "kg-op/structural-hypothesis-execution/v1/fake-source-seed0"
    )
    source_bundle = materialize_task_bundle(
        initial_report,
        hypothesis,
        executor_contract,
        materializer_contract,
        BASE_MANIFEST,
        ASSET_ROOT,
        source_attempt / "checkpoints",
    )
    assert verify_materialized_task_bundle(
        source_bundle,
        initial_report,
        hypothesis,
        executor_contract,
        materializer_contract,
        BASE_MANIFEST,
        ASSET_ROOT,
        source_attempt / "checkpoints",
    )
    source_task = source_bundle["plan"]["tasks"][0]
    assert source_task["cell"]["seed"] == 0
    source_prepared = runtime_core.prepare_single_task_attempt(
        initial_report,
        source_bundle,
        hypothesis,
        executor_contract,
        materializer_contract,
        runtime_contract,
        BASE_MANIFEST,
        ASSET_ROOT,
        source_attempt,
        task_id=source_task["task_id"],
        expected_bundle_digest=source_bundle["integrity"]["bundle_digest"],
        expected_plan_digest=source_bundle["plan"]["integrity"][
            "plan_digest"
        ],
        authorization_id="fake-source-seed0-authorization",
    )
    monkeypatch.setattr(
        runtime_core,
        "_run_preflight",
        lambda *args, **kwargs: _valid_fake_preflight(
            runtime_core, runtime_contract, source_task, source_prepared
        ),
    )
    source_executed = runtime_core.execute_single_task_attempt(
        source_attempt,
        runtime_contract,
        expected_authorization_digest=source_prepared[
            "authorization_digest"
        ],
    )

    publication_id = "fake-source-seed0-publication"
    publication = (
        state_home
        / "kg-op/structural-hypothesis-reingestion/v1"
        / publication_id
    )
    published = publisher_core.publish_single_task_reingestion(
        evidence,
        source_attempt,
        HYPOTHESIS_CONTRACT,
        EXECUTOR_CONTRACT,
        RUNTIME_CONTRACT,
        PUBLISHER_CONTRACT,
        BASE_MANIFEST,
        ASSET_ROOT,
        publication,
        publication_id=publication_id,
        expected_source_evidence_digest=initial_report["evidence_digest"],
        expected_plan_digest=source_bundle["plan"]["integrity"][
            "plan_digest"
        ],
        expected_authorization_digest=source_prepared[
            "authorization_digest"
        ],
        expected_execution_receipt_digest=source_executed[
            "receipt_digest"
        ],
        expected_execution_journal_head_digest=source_executed[
            "journal_head_digest"
        ],
        expected_execution_attempt_digest=source_executed[
            "attempt_digest"
        ],
    )
    published_report = json.loads(
        (publication / "output_report.json").read_text(encoding="utf-8")
    )
    combined_rows = json.loads(
        (publication / "combined_rows.json").read_text(encoding="utf-8")
    )
    assert len(combined_rows) == 1351
    assert len(published_report["pending_evidence"]) == 29

    adoption_id = "fake-source-seed0-adoption"
    adoption = (
        state_home
        / "kg-op/structural-hypothesis-report-adoption/v1"
        / adoption_id
    )
    adopted = adoption_core.adopt_structural_hypothesis_report(
        publication,
        ADOPTION_CONTRACT,
        adoption,
        evidence,
        source_attempt,
        HYPOTHESIS_CONTRACT,
        EXECUTOR_CONTRACT,
        RUNTIME_CONTRACT,
        PUBLISHER_CONTRACT,
        BASE_MANIFEST,
        ASSET_ROOT,
        adoption_id=adoption_id,
        expected_source_evidence_digest=initial_report["evidence_digest"],
        expected_plan_digest=source_bundle["plan"]["integrity"][
            "plan_digest"
        ],
        expected_authorization_digest=source_prepared[
            "authorization_digest"
        ],
        expected_execution_receipt_digest=source_executed[
            "receipt_digest"
        ],
        expected_execution_journal_head_digest=source_executed[
            "journal_head_digest"
        ],
        expected_execution_attempt_digest=source_executed[
            "attempt_digest"
        ],
        expected_publication_digest=published["publication_digest"],
        expected_reingestion_digest=published["reingestion_digest"],
        expected_output_report_body_digest=published[
            "output_report_body_digest"
        ],
        expected_output_audit_head=published["output_audit_head"],
        expected_output_evidence_digest=published[
            "output_evidence_digest"
        ],
        expected_publication_marker_raw_sha256=_raw_digest(
            publication / "publication.json"
        ),
        expected_combined_rows_raw_sha256=_raw_digest(
            publication / "combined_rows.json"
        ),
        expected_output_report_raw_sha256=_raw_digest(
            publication / "output_report.json"
        ),
        expected_reingestion_receipt_raw_sha256=_raw_digest(
            publication / "reingestion_receipt.json"
        ),
    )

    successor_id = "fake-successor-seed1"
    successor = (
        state_home
        / "kg-op/structural-hypothesis-adopted-successor/v1"
        / successor_id
    )
    completed_attempt = (
        state_home
        / "kg-op/structural-hypothesis-execution/v1"
        / successor_id
    )
    first_pending = published_report["pending_evidence"][0]
    projection = _projection(first_pending)
    assert (projection["domain"], projection["seed"]) == (
        "FactorShockStatePolicyRZDT1",
        1,
    )
    pending_digest = _canonical_digest(
        published_report["pending_evidence"]
    )
    projection_digest = _canonical_digest(projection)
    materialized = successor_core.materialize_adopted_successor(
        publication,
        ADOPTION_CONTRACT,
        adoption,
        SUCCESSOR_CONTRACT,
        successor,
        evidence,
        source_attempt,
        HYPOTHESIS_CONTRACT,
        EXECUTOR_CONTRACT,
        RUNTIME_CONTRACT,
        PUBLISHER_CONTRACT,
        MATERIALIZER_CONTRACT,
        BASE_MANIFEST,
        ASSET_ROOT,
        completed_attempt,
        adoption_id=adoption_id,
        successor_id=successor_id,
        expected_adoption_digest=adopted["adoption_digest"],
        expected_pending_evidence_digest=pending_digest,
        expected_first_pending_projection_digest=projection_digest,
    )
    inspected = bridge_core.inspect_successor_bound_single_task(
        publication,
        ADOPTION_CONTRACT,
        adoption,
        SUCCESSOR_CONTRACT,
        successor,
        evidence,
        source_attempt,
        HYPOTHESIS_CONTRACT,
        EXECUTOR_CONTRACT,
        RUNTIME_CONTRACT,
        PUBLISHER_CONTRACT,
        MATERIALIZER_CONTRACT,
        BRIDGE_CONTRACT,
        BASE_MANIFEST,
        ASSET_ROOT,
        completed_attempt,
        adoption_id=adoption_id,
        successor_id=successor_id,
        expected_adoption_digest=adopted["adoption_digest"],
        expected_pending_evidence_digest=pending_digest,
        expected_first_pending_projection_digest=projection_digest,
        expected_successor_digest=materialized["successor_digest"],
        expected_bundle_digest=materialized["bundle_digest"],
        expected_plan_digest=materialized["plan_digest"],
        task_id=materialized["first_task_id"],
        expected_task_digest=materialized["first_task_digest"],
    )
    prepared = bridge_core.prepare_successor_bound_single_task_attempt(
        publication,
        ADOPTION_CONTRACT,
        adoption,
        SUCCESSOR_CONTRACT,
        successor,
        evidence,
        source_attempt,
        HYPOTHESIS_CONTRACT,
        EXECUTOR_CONTRACT,
        RUNTIME_CONTRACT,
        PUBLISHER_CONTRACT,
        MATERIALIZER_CONTRACT,
        BRIDGE_CONTRACT,
        BASE_MANIFEST,
        ASSET_ROOT,
        completed_attempt,
        adoption_id=adoption_id,
        successor_id=successor_id,
        expected_adoption_digest=adopted["adoption_digest"],
        expected_pending_evidence_digest=pending_digest,
        expected_first_pending_projection_digest=projection_digest,
        expected_successor_digest=materialized["successor_digest"],
        expected_bundle_digest=materialized["bundle_digest"],
        expected_plan_digest=materialized["plan_digest"],
        task_id=materialized["first_task_id"],
        expected_task_digest=materialized["first_task_digest"],
        expected_provenance_binding_digest=inspected[
            "provenance_binding_digest"
        ],
        authorization_id=inspected["required_authorization_id"],
    )
    successor_task = json.loads(
        (completed_attempt / "bundle.json").read_text(encoding="utf-8")
    )["plan"]["tasks"][0]
    monkeypatch.setattr(
        runtime_core,
        "_run_preflight",
        lambda *args, **kwargs: _valid_fake_preflight(
            runtime_core, runtime_contract, successor_task, prepared
        ),
    )
    completed = runtime_core.execute_single_task_attempt(
        completed_attempt,
        runtime_contract,
        expected_authorization_digest=prepared["authorization_digest"],
    )
    assert completed["status"] == "EXECUTED_RECEIPT_WRITTEN"

    advance_id = "fake-recursive-advance-seed1"
    advance_root = (
        state_home
        / "kg-op/structural-hypothesis-recursive-report-advance/v1"
        / advance_id
    )
    yield {
        "state_home": state_home,
        "publication": publication,
        "adoption": adoption,
        "adoption_id": adoption_id,
        "successor": successor,
        "successor_id": successor_id,
        "evidence": evidence,
        "source_attempt": source_attempt,
        "completed_attempt": completed_attempt,
        "advance_id": advance_id,
        "advance_root": advance_root,
        "published_report": published_report,
        "adopted": adopted,
        "materialized": materialized,
        "inspected": inspected,
        "prepared": prepared,
        "completed": completed,
        "expected": {
            "adoption_digest": adopted["adoption_digest"],
            "pending_evidence_digest": pending_digest,
            "first_pending_projection_digest": projection_digest,
            "successor_digest": materialized["successor_digest"],
            "bundle_digest": materialized["bundle_digest"],
            "plan_digest": materialized["plan_digest"],
            "task_id": materialized["first_task_id"],
            "task_digest": materialized["first_task_digest"],
            "provenance_binding_digest": inspected[
                "provenance_binding_digest"
            ],
            "authorization_digest": prepared["authorization_digest"],
            "attempt_digest": prepared["attempt_digest"],
            "execution_receipt_digest": completed["receipt_digest"],
            "execution_journal_head_digest": completed[
                "journal_head_digest"
            ],
        },
    }


def test_runner_advance_forwards_frozen_surface_and_prints_canonical_json(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    observed = {}

    def advance(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return _unit_result(case)

    monkeypatch.setattr(
        runner,
        "_load_advance_core",
        lambda: SimpleNamespace(advance_recursive_report_version=advance),
    )
    argv = [
        "advance",
        *_unit_common_cli(case),
        "--confirm-immutable-local-report-advance",
    ]
    assert runner.main(argv) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload == _unit_result(case)
    assert captured.out == json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert "action=advance" in captured.err
    assert observed == {
        "args": _unit_expected_positional(case),
        "kwargs": {
            **_unit_expected_kwargs(case),
            "confirm_immutable_local_report_advance": True,
        },
    }


def test_runner_verify_forwards_five_output_anchors_and_is_read_only(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path, committed=True)
    observed = {}

    def verify(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return _unit_result(case, verified=True)

    monkeypatch.setattr(
        runner,
        "_load_advance_core",
        lambda: SimpleNamespace(verify_recursive_report_advance=verify),
    )
    before = _tree_observation(case["advance-root"])
    assert runner.main(["verify", *_unit_verify_cli(case)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == _unit_result(case, verified=True)
    assert "action=verify" in captured.err
    assert _tree_observation(case["advance-root"]) == before
    assert observed["args"] == _unit_expected_positional(case)
    assert observed["kwargs"] == {
        **_unit_expected_kwargs(case),
        "expected_advance_digest": case["digests"]["advance"],
        "expected_reingestion_digest": case["digests"]["reingestion"],
        "expected_output_report_body_digest": case["digests"]["report"],
        "expected_output_audit_head": case["digests"]["audit"],
        "expected_output_evidence_digest": case["digests"]["evidence"],
    }


def test_runner_requires_confirmation_and_rejects_bad_anchor_before_core(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    with pytest.raises(SystemExit) as error:
        runner._parser().parse_args(["advance", *_unit_common_cli(case)])
    assert error.value.code == 2

    monkeypatch.setattr(
        runner,
        "_load_advance_core",
        lambda: pytest.fail("malformed digest reached recursive core"),
    )
    argv = [
        "advance",
        *_unit_common_cli(case),
        "--confirm-immutable-local-report-advance",
    ]
    argv[argv.index("--expected-adoption-digest") + 1] = "sha256:BAD"
    assert runner.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "lowercase sha256 digest" in captured.err
    assert not case["advance-root"].exists()


def test_fake_only_real_core_advances_1351_29_to_1352_28_and_verifies(
    fake_completed_successor_case, monkeypatch, capsys
):
    from performance import structural_hypothesis_recursive_report_advance as core

    case = fake_completed_successor_case
    generations = []
    original_validate_generation = core._validate_generation

    def capture_generation(*args, **kwargs):
        generation = original_validate_generation(*args, **kwargs)
        generations.append(generation)
        return generation

    monkeypatch.setattr(core, "_validate_generation", capture_generation)
    before_sources = {
        name: _tree_observation(case[name])
        for name in (
            "publication",
            "adoption",
            "successor",
            "source_attempt",
            "completed_attempt",
        )
    }
    assert runner.main([
        "advance",
        *_real_common_cli(case),
        "--confirm-immutable-local-report-advance",
    ]) == 0
    advanced_io = capsys.readouterr()
    advanced = json.loads(advanced_io.out)
    assert advanced["status"] == ADVANCE_STATUS
    assert advanced["typed_row_count"] == 1352
    assert advanced["pending_evidence_count"] == 28
    assert advanced["current_status"] == "NOT_CURRENT"
    assert advanced["planning_status"] == "NOT_PLANNED"
    assert "action=advance" in advanced_io.err

    root = case["advance_root"]
    assert {path.name for path in root.iterdir()} == {
        "advance_contract.json",
        "source",
        "combined_rows.json",
        "output_report.json",
        "reingestion_receipt.json",
        "advance.json",
    }
    assert {path.name for path in (root / "source").iterdir()} == {
        "adoption.json",
        "successor.json",
        "execution",
    }
    assert {
        path.name for path in (root / "source/execution").iterdir()
    } == {"attempt.json", "authorization.json", "receipt.json", "journal"}
    assert {
        path.name for path in (root / "source/execution/journal").iterdir()
    } == {"0002_COMPLETED.json"}
    assert all(
        (path.stat().st_mode & 0o777) == 0o700
        for path in (
            root,
            root / "source",
            root / "source/execution",
            root / "source/execution/journal",
        )
    )
    assert all(
        (path.stat().st_mode & 0o777) == 0o600
        for path in root.rglob("*")
        if path.is_file()
    )

    combined = json.loads(
        (root / "combined_rows.json").read_text(encoding="utf-8")
    )
    output_report = json.loads(
        (root / "output_report.json").read_text(encoding="utf-8")
    )
    marker = json.loads(
        (root / "advance.json").read_text(encoding="utf-8")
    )
    assert len(combined) == 1352
    assert len(output_report["pending_evidence"]) == 28
    assert combined[:-1] == json.loads(
        (
            case["adoption"] / "publication/combined_rows.json"
        ).read_text(encoding="utf-8")
    )
    appended = combined[-1]
    assert type(appended["seed"]) is int
    assert type(appended["N"]) is int
    assert type(appended["adaptive_loss"]) is bool
    assert type(appended["source_discrepancy_update"]) is bool
    assert marker["transition"]["source_typed_row_count"] == 1351
    assert marker["transition"]["output_typed_row_count"] == 1352
    assert marker["transition"]["source_pending_count"] == 29
    assert marker["transition"]["output_pending_count"] == 28
    assert marker["transition"]["accepted_successful_rows"] == 1
    assert marker["transition"]["ignored_failed_attempts"] == 0
    assert marker["source_binding"]["provenance_binding_digest"] == (
        case["expected"]["provenance_binding_digest"]
    )
    assert marker["source_binding"]["execution_receipt_digest"] == (
        case["expected"]["execution_receipt_digest"]
    )

    before_verify = _tree_observation(root)
    assert runner.main(["verify", *_real_verify_cli(case, advanced)]) == 0
    verified_io = capsys.readouterr()
    verified = json.loads(verified_io.out)
    assert verified == {**advanced, "status": VERIFY_STATUS}
    assert "action=verify" in verified_io.err
    assert _tree_observation(root) == before_verify
    assert len(generations) == 4

    # The source chain was fully verified above.  Freeze that exact derived
    # generation so each following call isolates one capsule-side rejection.
    monkeypatch.setattr(
        core,
        "_validate_generation",
        lambda *args, **kwargs: generations[-1],
    )

    def verify_rejects():
        assert runner.main([
            "verify", *_real_verify_cli(case, advanced)
        ]) == 2
        rejected_io = capsys.readouterr()
        assert rejected_io.out == ""
        assert "Traceback" not in rejected_io.err

    combined_path = root / "combined_rows.json"
    combined_raw = combined_path.read_bytes()
    combined_path.write_bytes(combined_raw + b"\n")
    combined_path.chmod(0o600)
    verify_rejects()
    combined_path.write_bytes(combined_raw)
    combined_path.chmod(0o600)

    marker_path = root / "advance.json"
    marker_raw = marker_path.read_bytes()
    marker_path.write_bytes(marker_raw + b"\n")
    marker_path.chmod(0o600)
    verify_rejects()
    marker_path.write_bytes(marker_raw)
    marker_path.chmod(0o600)

    combined_path.chmod(0o644)
    verify_rejects()
    combined_path.chmod(0o600)

    extra = root / "unexpected.json"
    extra.write_text("{}\n", encoding="utf-8")
    extra.chmod(0o600)
    verify_rejects()
    extra.unlink()

    external_link = case["state_home"] / "combined-rows-hardlink"
    os.link(combined_path, external_link)
    try:
        verify_rejects()
    finally:
        external_link.unlink()

    combined_path.unlink()
    os.mkfifo(combined_path, mode=0o600)
    try:
        verify_rejects()
    finally:
        combined_path.unlink()
        combined_path.write_bytes(combined_raw)
        combined_path.chmod(0o600)

    symlink_target = case["state_home"] / "combined-rows-target.json"
    combined_path.replace(symlink_target)
    combined_path.symlink_to(symlink_target)
    try:
        verify_rejects()
    finally:
        combined_path.unlink()
        symlink_target.replace(combined_path)

    assert _tree_observation(root) == before_verify
    assert {
        name: _tree_observation(case[name]) for name in before_sources
    } == before_sources

    assert runner.main([
        "advance",
        *_real_common_cli(case),
        "--confirm-immutable-local-report-advance",
    ]) == 2
    rejected = capsys.readouterr()
    assert rejected.out == ""
    assert "fresh advance root" in rejected.err
    assert _tree_observation(root) == before_verify


def test_fake_only_real_core_rejects_every_independent_source_anchor(
    fake_completed_successor_case, capsys
):
    case = fake_completed_successor_case
    option_names = (
        "--expected-adoption-digest",
        "--expected-pending-evidence-digest",
        "--expected-first-pending-projection-digest",
        "--expected-successor-digest",
        "--expected-bundle-digest",
        "--expected-plan-digest",
        "--expected-task-digest",
        "--expected-provenance-binding-digest",
        "--expected-authorization-digest",
        "--expected-attempt-digest",
        "--expected-execution-receipt-digest",
        "--expected-execution-journal-head-digest",
    )
    prefix = (
        case["state_home"]
        / "kg-op/structural-hypothesis-recursive-report-advance/v1"
    )
    for ordinal, option in enumerate(option_names):
        advance_id = f"wrong-anchor-{ordinal:02d}"
        target = prefix / advance_id
        argv = [
            "advance",
            *_real_common_cli(
                case, advance_root=target, advance_id=advance_id
            ),
            "--confirm-immutable-local-report-advance",
        ]
        argv[argv.index(option) + 1] = _digest("f")
        assert runner.main(argv) == 2, option
        rejected = capsys.readouterr()
        assert rejected.out == "", option
        assert "Traceback" not in rejected.err, option
        assert not target.exists(), option
    assert not prefix.exists()


def test_advance_location_rejects_outside_and_symlink_roots(
    tmp_path, monkeypatch
):
    from performance import structural_hypothesis_recursive_report_advance as core

    state_home = Path(tempfile.mkdtemp(
        prefix="kgop-recursive-path-test.", dir="/tmp"
    ))
    state_home.chmod(0o700)
    try:
        monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
        outside = state_home / "outside" / "advance-path-test"
        with pytest.raises(core.RecursiveReportAdvanceError):
            core._validate_advance_location(
                outside, "advance-path-test", fresh=True
            )
        assert not outside.exists()

        prefix = (
            state_home
            / "kg-op/structural-hypothesis-recursive-report-advance/v1"
        )
        prefix.mkdir(parents=True, mode=0o700)
        for path in (state_home / "kg-op", state_home / "kg-op/structural-hypothesis-recursive-report-advance", prefix):
            path.chmod(0o700)
        target = state_home / "real-target"
        target.mkdir(mode=0o700)
        alias = prefix / "advance-path-alias"
        alias.symlink_to(target, target_is_directory=True)
        with pytest.raises(core.RecursiveReportAdvanceError):
            core._validate_advance_location(
                alias, "advance-path-alias", fresh=True
            )
    finally:
        shutil.rmtree(state_home, ignore_errors=True)


def test_reader_rejects_fifo_hardlink_and_lstat_open_generation_swap(
    monkeypatch
):
    from performance import structural_hypothesis_recursive_report_advance as core

    root = Path(tempfile.mkdtemp(
        prefix="kgop-recursive-reader-test.", dir="/tmp"
    ))
    root.chmod(0o700)
    try:
        fifo = root / "fifo.json"
        os.mkfifo(fifo, mode=0o600)
        with pytest.raises(core.RecursiveReportAdvanceError):
            core._read_regular(fifo, "fifo", exact_mode=0o600)

        hardlinked = root / "hardlinked.json"
        hardlinked.write_bytes(b"{}\n")
        hardlinked.chmod(0o600)
        alias = root / "hardlinked.alias"
        os.link(hardlinked, alias)
        with pytest.raises(core.RecursiveReportAdvanceError):
            core._read_regular(
                hardlinked, "hardlinked leaf", exact_mode=0o600
            )

        observed = root / "observed.json"
        replacement = root / "replacement.json"
        observed.write_bytes(b'{"same":true}\n')
        replacement.write_bytes(b'{"same":true}\n')
        observed.chmod(0o600)
        replacement.chmod(0o600)
        real_open = os.open
        swapped = False

        def swap_before_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if Path(path) == observed and not swapped:
                swapped = True
                observed.unlink()
                replacement.replace(observed)
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(core.os, "open", swap_before_open)
        with pytest.raises(core.RecursiveReportAdvanceError):
            core._read_regular(
                observed, "generation-swapped leaf", exact_mode=0o600
            )
        assert swapped is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_fake_only_receipt_semantics_generation_swap_and_commit_last(
    fake_completed_successor_case, monkeypatch, capsys
):
    from performance import structural_hypothesis_recursive_report_advance as core

    case = fake_completed_successor_case
    expected = _real_core_expected(case)
    generation = core._validate_generation(
        *_real_source_positional(case),
        adoption_id=case["adoption_id"],
        successor_id=case["successor_id"],
        expected=expected,
        task_id=case["expected"]["task_id"],
    )
    derived = generation["derived"]
    real_chain = core._ASSERT_RUNTIME_CHAIN(
        derived,
        expected_authorization_digest=case["expected"][
            "authorization_digest"
        ],
    )
    captured_objects, captured_raws = core._CAPTURE_ATTEMPT(
        case["completed_attempt"]
    )

    monkeypatch.setattr(
        core, "_DERIVE_SUCCESSOR_BOUND", lambda *args, **kwargs: derived
    )
    monkeypatch.setattr(
        core,
        "_VERIFY_COMPLETED_ATTEMPT",
        lambda *args, **kwargs: {"status": "VERIFIED_COMPLETED"},
    )

    def semantic_rejection(mutator):
        receipt = copy.deepcopy(captured_objects["execution_receipt"])
        mutator(receipt)
        body = {
            key: value
            for key, value in receipt.items()
            if key != "integrity"
        }
        receipt_digest = core._digest(body)
        receipt["integrity"] = {
            "algorithm": "sha256-canonical-json-v1",
            "receipt_digest": receipt_digest,
        }
        local_expected = {**expected}
        local_expected["expected_execution_receipt_digest"] = receipt_digest
        local_captured = copy.deepcopy(captured_objects)
        local_captured["execution_receipt"] = receipt
        monkeypatch.setattr(
            core,
            "_CAPTURE_ATTEMPT",
            lambda *args, **kwargs: (local_captured, captured_raws),
        )
        monkeypatch.setattr(
            core,
            "_VERIFY_CAPTURED_ATTEMPT",
            lambda *args, **kwargs: None,
        )
        with pytest.raises(
            core.RecursiveReportAdvanceError,
            match="exactly one successful authorized result",
        ):
            core._validate_generation(
                *_real_source_positional(case),
                adoption_id=case["adoption_id"],
                successor_id=case["successor_id"],
                expected=local_expected,
                task_id=case["expected"]["task_id"],
            )

    semantic_rejection(
        lambda receipt: (
            receipt["results"][0].update({"status": "FAILED"}),
            receipt.update({
                "summary": {"authorized": 1, "failed": 1, "succeeded": 0}
            }),
        )
    )
    semantic_rejection(
        lambda receipt: receipt["results"][0].update({
            "task_id": "task:" + "b" * 24
        })
    )
    semantic_rejection(
        lambda receipt: receipt["results"].append(
            dict(receipt["results"][0])
        )
    )

    # A different second capture is rejected before the advance root exists.
    swapped_generation = copy.deepcopy(generation)
    swapped_generation["combined_rows"][-1]["seed"] = 999
    sequence = iter((generation, swapped_generation))
    monkeypatch.setattr(
        core, "_validate_generation", lambda *args, **kwargs: next(sequence)
    )
    swap_id = "fake-generation-swap"
    swap_root = case["advance_root"].parent / swap_id
    assert runner.main([
        "advance",
        *_real_common_cli(case, advance_root=swap_root, advance_id=swap_id),
        "--confirm-immutable-local-report-advance",
    ]) == 2
    swapped_io = capsys.readouterr()
    assert swapped_io.out == ""
    assert "source generation changed" in swapped_io.err
    assert not swap_root.exists()

    # Inject failure at the last write.  All staged artifacts may exist, but
    # absence of advance.json means the root is incomplete and never reusable.
    monkeypatch.setattr(
        core, "_validate_generation", lambda *args, **kwargs: generation
    )
    real_write = core._write_new_bytes

    def fail_commit(path, raw):
        if Path(path).name == "advance.json":
            raise core.RecursiveReportAdvanceError(
                "injected final marker failure"
            )
        return real_write(path, raw)

    monkeypatch.setattr(core, "_write_new_bytes", fail_commit)
    assert runner.main([
        "advance",
        *_real_common_cli(case),
        "--confirm-immutable-local-report-advance",
    ]) == 2
    failed_io = capsys.readouterr()
    assert failed_io.out == ""
    assert "injected final marker failure" in failed_io.err
    assert case["advance_root"].is_dir()
    assert not (case["advance_root"] / "advance.json").exists()
    assert {
        path.name for path in case["advance_root"].iterdir()
    } == {
        "advance_contract.json",
        "source",
        "combined_rows.json",
        "output_report.json",
        "reingestion_receipt.json",
    }

    monkeypatch.setattr(core, "_write_new_bytes", real_write)
    assert runner.main([
        "advance",
        *_real_common_cli(case),
        "--confirm-immutable-local-report-advance",
    ]) == 2
    reuse_io = capsys.readouterr()
    assert reuse_io.out == ""
    assert "fresh advance root" in reuse_io.err
    assert not (case["advance_root"] / "advance.json").exists()


def test_surface_contains_no_execution_or_network_primitive():
    source = RUNNER.read_text(encoding="utf-8")
    assert "benchmark_lodo_meta_prior" not in source
    assert "run_one" not in source
    assert "subprocess" not in source
    assert "requests" not in source
    assert "socket" not in source
    assert "import scheduler" not in source.lower()
    assert "from scheduler" not in source.lower()
    choices = set(runner._parser()._subparsers._group_actions[0].choices)
    assert choices == {"advance", "verify"}

    core_source = CORE.read_text(encoding="utf-8")
    assert "benchmark_lodo_meta_prior" not in core_source
    assert "run_one(" not in core_source
    assert "import subprocess" not in core_source
    assert "import socket" not in core_source
    assert "import requests" not in core_source
    assert "_load_real_executor" not in core_source
    assert "materialize_adopted_successor(" not in core_source
    assert "prepare_successor_bound_single_task_attempt(" not in core_source
    assert "execute_single_task_attempt(" not in core_source
