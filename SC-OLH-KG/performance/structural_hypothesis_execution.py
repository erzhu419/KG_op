"""Pure, offline lifecycle for materializing and executing hypothesis cells.

This module deliberately contains no file, network, shell, scheduler, or real
executor adapter.  Callers must inject both task materialization and execution.
The hashes below are integrity commitments, not signatures or authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .aggregate_completed_matrix import (
    _normalize_sc_result as _normalize_sc_run_one_result_v1,
)
from .structural_hypothesis_loop import (
    REQUIRED_EVIDENCE_FIELDS,
    canonical_json_bytes,
    run_structural_hypothesis_loop,
    validate_contract as validate_hypothesis_contract,
    verify_report_integrity,
)


EXECUTOR_SCHEMA_VERSION = "sc-olh-kg.structural-hypothesis-executor/1"
PLAN_SCHEMA_VERSION = "sc-olh-kg.structural-hypothesis-execution-plan/1"
AUTHORIZATION_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-execution-authorization/1"
)
RECEIPT_SCHEMA_VERSION = "sc-olh-kg.structural-hypothesis-execution-receipt/1"
REINGESTION_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-execution-reingestion/1"
)

_SOURCE_CONTRACT_ID = "structural_hypothesis_loop_v1"
_SOURCE_CONTRACT_DIGEST = (
    "sha256:4242f6af8424acca5c93136f0d4eb354f8c2203431f1c5145290c4a3f248cf26"
)
_EXECUTOR_CONTRACT_DIGEST = (
    "sha256:ede48b8b1fb0bb788f91a3834d5a41f336e55b331183922237176aec12624030"
)
_DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
_FIXED_TASK_INPUT_KEYS = {
    "decision_backend",
    "structural_prior_profile",
    "hvd_ablation_profile",
    "source_discrepancy_update",
    "certification_recheck_top_k",
    "decision_risk_penalty",
    "decision_source_utility_weight",
    "adaptive_replication_voi",
    "posterior_dominance_enabled",
    "offline_only",
    "llm_prior_enabled",
}
_NONCLAIMS = (
    "proposal_is_not_authorization",
    "authorization_is_not_execution",
    "execution_is_not_reingestion",
    "local_digest_is_not_signature",
    "no_external_authority",
    "no_credential_access",
    "no_network_access",
    "no_scheduler_submission",
    "no_shell_execution",
    "no_bundled_real_task_template",
    "no_exact_historical_runtime_reconstruction",
    "injected_template_not_verified_as_exact_historical_runtime",
    "plan_cli_performs_no_actual_experiment_execution",
    "source_report_hash_chain_not_external_anchor",
    "no_runtime_readiness_claim",
    "no_scientific_verdict_from_plan",
    "no_paper_promotion",
)


class ExecutionValidationError(ValueError):
    """Raised when a V1 lifecycle artifact violates its frozen contract."""


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ExecutionValidationError(
            f"{label} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _json_clone(value: Any, label: str = "value") -> Any:
    """Return a canonical deep copy while rejecting non-native JSON values."""
    def check(item: Any, path: str) -> None:
        if item is None or type(item) in (str, int, bool):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ExecutionValidationError(f"{path} must be finite")
            return
        if type(item) is list:
            for index, child in enumerate(item):
                check(child, f"{path}[{index}]")
            return
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ExecutionValidationError(f"{path} keys must be strings")
            for key, child in item.items():
                check(child, f"{path}.{key}")
            return
        raise ExecutionValidationError(f"{path} is not native JSON")

    check(value, label)
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def validate_executor_contract(contract: Mapping[str, Any]) -> None:
    if not isinstance(contract, Mapping):
        raise ExecutionValidationError("executor contract must be an object")
    _exact_keys(
        contract,
        {
            "schema_version", "contract_id", "source_hypothesis_contract_id",
            "source_hypothesis_contract_digest", "executor_binding",
            "execution_scope", "fixed_task_inputs", "mechanics", "nonclaims",
        },
        "executor contract",
    )
    if contract["schema_version"] != EXECUTOR_SCHEMA_VERSION:
        raise ExecutionValidationError("unsupported executor schema_version")
    if type(contract["contract_id"]) is not str or not contract["contract_id"]:
        raise ExecutionValidationError("executor contract_id must be non-empty text")
    if contract["source_hypothesis_contract_id"] != _SOURCE_CONTRACT_ID:
        raise ExecutionValidationError("source hypothesis contract id differs")
    if contract["source_hypothesis_contract_digest"] != _SOURCE_CONTRACT_DIGEST:
        raise ExecutionValidationError("source hypothesis contract digest differs")

    expected_binding = {
        "module": "performance.benchmark_lodo_meta_prior",
        "callable": "run_one",
        "task_keys": ["args", "heldout", "line", "seed"],
        "cell_input_map": {
            "profile": "args.structural_prior_profile",
            "domain": "heldout",
            "line": "line",
            "seed": "seed",
        },
        "result_adapter": {
            "module": "performance.aggregate_completed_matrix",
            "callable": "_normalize_sc_result",
            "projection": "REQUIRED_EVIDENCE_FIELDS",
        },
    }
    if contract["executor_binding"] != expected_binding:
        raise ExecutionValidationError("executor_binding differs from frozen V1")

    scope = contract["execution_scope"]
    if not isinstance(scope, Mapping):
        raise ExecutionValidationError("execution_scope must be an object")
    _exact_keys(scope, {"profile", "line", "domains", "seeds", "d", "N", "n0"}, "execution_scope")
    if (
        scope["profile"] != "full"
        or scope["line"] != "lodo"
        or scope["domains"] != list(_DOMAINS)
        or scope["seeds"] != list(range(10))
        or type(scope["d"]) is not int or scope["d"] != 50
        or type(scope["N"]) is not int or scope["N"] != 20
        or type(scope["n0"]) is not int or scope["n0"] != 10
    ):
        raise ExecutionValidationError("execution_scope differs from frozen V1")

    fixed = contract["fixed_task_inputs"]
    if not isinstance(fixed, Mapping):
        raise ExecutionValidationError("fixed_task_inputs must be an object")
    _exact_keys(fixed, _FIXED_TASK_INPUT_KEYS, "fixed_task_inputs")
    expected_fixed = {
        "decision_backend": "risk_ts",
        "structural_prior_profile": "full",
        "hvd_ablation_profile": "factor_cumulative",
        "source_discrepancy_update": True,
        "certification_recheck_top_k": 0,
        "decision_risk_penalty": 5.0,
        "decision_source_utility_weight": 1.0,
        "adaptive_replication_voi": False,
        "posterior_dominance_enabled": False,
        "offline_only": True,
        "llm_prior_enabled": False,
    }
    if fixed != expected_fixed:
        raise ExecutionValidationError("fixed_task_inputs differs from frozen V1")

    expected_mechanics = {
        "plan_only_default": True,
        "real_task_template": "NOT_IMPLEMENTED",
        "network_access": False,
        "scheduler_access": False,
        "shell_execution": False,
    }
    if contract["mechanics"] != expected_mechanics:
        raise ExecutionValidationError("mechanics differs from frozen V1")
    if contract["nonclaims"] != list(_NONCLAIMS):
        raise ExecutionValidationError("nonclaims differ from frozen V1")
    if _digest(contract) != _EXECUTOR_CONTRACT_DIGEST:
        raise ExecutionValidationError("executor contract digest differs from frozen V1")


def _bind_source_report(
    report: Mapping[str, Any], hypothesis_contract: Mapping[str, Any]
) -> dict[str, Any]:
    validate_hypothesis_contract(hypothesis_contract)
    if _digest(hypothesis_contract) != _SOURCE_CONTRACT_DIGEST:
        raise ExecutionValidationError("hypothesis contract digest is not frozen V1")
    if not verify_report_integrity(report):
        raise ExecutionValidationError("source report integrity failed")
    if report.get("schema_version") != hypothesis_contract["schema_version"]:
        raise ExecutionValidationError("source report schema_version differs")
    if report.get("contract_id") != hypothesis_contract["contract_id"]:
        raise ExecutionValidationError("source report contract_id differs")
    if report.get("contract_digest") != _digest(hypothesis_contract):
        raise ExecutionValidationError("source report contract_digest differs")
    for report_key, contract_key in (
        ("evidence_scope", "evidence_scope"),
        ("gate", "gate"),
        ("nonclaims", "nonclaims"),
    ):
        if report.get(report_key) != hypothesis_contract[contract_key]:
            raise ExecutionValidationError(f"source report {report_key} differs")
    audit = report["audit"]
    return {
        "contract_id": report["contract_id"],
        "contract_digest": report["contract_digest"],
        "evidence_digest": report["evidence_digest"],
        "report_body_digest": audit["report_body_digest"],
        "audit_head": audit["head"],
    }


def _expected_pending_cells(
    hypothesis_contract: Mapping[str, Any], executor_contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    source_scope = hypothesis_contract["evidence_scope"]
    execution_scope = executor_contract["execution_scope"]
    if (
        execution_scope["domains"] != source_scope["domains"]
        or execution_scope["seeds"] != source_scope["seeds"]
        or any(execution_scope[key] != source_scope[key] for key in ("d", "N", "n0"))
    ):
        raise ExecutionValidationError("executor scope differs from hypothesis scope")
    fixed = executor_contract["fixed_task_inputs"]
    expected_crosswalk = {
        "decision_backend": source_scope["decision_backend"],
        "structural_prior_profile": execution_scope["profile"],
        "hvd_ablation_profile": source_scope["fixed_row_values"]["hvd_profile"],
        "source_discrepancy_update": source_scope["fixed_row_values"]["source_discrepancy_update"],
        "certification_recheck_top_k": source_scope["fixed_row_values"]["recheck_top_k"],
        "decision_risk_penalty": source_scope["fixed_row_values"]["risk_penalty"],
        "decision_source_utility_weight": source_scope["fixed_row_values"]["utility_weight"],
        "adaptive_replication_voi": source_scope["fixed_row_values"]["adaptive_replication_voi"],
        "posterior_dominance_enabled": source_scope["fixed_row_values"]["posterior_dominance_enabled"],
        "offline_only": True,
        "llm_prior_enabled": False,
    }
    if fixed != expected_crosswalk:
        raise ExecutionValidationError("task inputs do not crosswalk to evidence scope")
    result = []
    profile = execution_scope["profile"]
    for domain in execution_scope["domains"]:
        for seed in execution_scope["seeds"]:
            cell = {
                "track": source_scope["track"],
                "run_id": source_scope["run_id"],
                "variant": source_scope["variant_template"].format(profile=profile),
                "profile": profile,
                "domain": domain,
                "seed": seed,
                "d": source_scope["d"],
                "N": source_scope["N"],
                "n0": source_scope["n0"],
                "source_calls": source_scope["source_calls"],
                "implementation": source_scope["implementation"],
                "initial_design": source_scope["initial_design"],
                "decision_backend": source_scope["decision_backend"],
            }
            cell.update(source_scope["fixed_row_values"])
            result.append(cell)
    return result


def _validate_task(task: Any, cell: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    if type(task) is not dict:
        raise ExecutionValidationError("materializer must return an exact task dict")
    _exact_keys(task, {"args", "heldout", "line", "seed"}, "run_one task")
    if type(task["args"]) is not dict:
        raise ExecutionValidationError("run_one task args must be an exact dict")
    cloned = _json_clone(task, "run_one task")
    if cloned["heldout"] != cell["domain"] or cloned["line"] != "lodo":
        raise ExecutionValidationError("task domain/line differs from cell")
    if type(cloned["seed"]) is not int or cloned["seed"] != cell["seed"]:
        raise ExecutionValidationError("task seed differs from cell")
    args = cloned["args"]
    for key in ("d", "N", "n0"):
        if type(args.get(key)) is not int or args[key] != cell[key]:
            raise ExecutionValidationError(f"task args.{key} differs from cell")
    for key, expected in contract["fixed_task_inputs"].items():
        if key not in args or type(args[key]) is not type(expected) or args[key] != expected:
            raise ExecutionValidationError(f"task args.{key} differs from fixed input")
    for derived in ("source_calls", "total_calls", "posterior_dominance_switch_count"):
        if derived in args:
            raise ExecutionValidationError(
                f"task args.{derived} is derived evidence, not a verified input gate"
            )
    return cloned


def _validated_source_pending_cells(
    report: Mapping[str, Any],
    hypothesis_contract: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    decisions = report.get("decisions")
    verdict_counts = report.get("verdict_counts")
    dispositions = {
        "SUPPORTED_SCOPED", "REFUTED_SCOPED", "NEEDS_EVIDENCE",
        "INVALID_EVIDENCE",
    }
    if type(decisions) is not list or not isinstance(verdict_counts, Mapping):
        raise ExecutionValidationError("source decisions/counts are malformed")
    observed_counts = {disposition: 0 for disposition in dispositions}
    for decision in decisions:
        if (
            type(decision) is not dict
            or decision.get("disposition") not in dispositions
        ):
            raise ExecutionValidationError(
                "source decision disposition is malformed"
            )
        observed_counts[decision["disposition"]] += 1
    if (
        set(verdict_counts) != dispositions
        or any(type(verdict_counts[key]) is not int for key in dispositions)
        or verdict_counts != observed_counts
    ):
        raise ExecutionValidationError(
            "source verdict counts differ from decisions"
        )
    if verdict_counts["INVALID_EVIDENCE"] != 0 or any(
        item.get("disposition") == "INVALID_EVIDENCE"
        for item in decisions
    ):
        raise ExecutionValidationError(
            "source report contains INVALID_EVIDENCE"
        )
    if (
        report.get("status") != "COMPLETED_WITH_EVIDENCE_GAPS"
        or report.get("stop_reason") != "FINITE_GRAPH_EXHAUSTED"
        or verdict_counts["NEEDS_EVIDENCE"] <= 0
    ):
        raise ExecutionValidationError(
            "source report is not a completed evidence-gap report"
        )

    all_expected_cells = _expected_pending_cells(
        hypothesis_contract, executor_contract
    )
    pending = report.get("pending_evidence")
    if type(pending) is not list or not pending:
        raise ExecutionValidationError(
            "pending evidence must be a non-empty list"
        )
    expected_index = {
        _digest(cell): ordinal
        for ordinal, cell in enumerate(all_expected_cells)
    }
    pending_digests = [_digest(cell) for cell in pending]
    if (
        any(digest not in expected_index for digest in pending_digests)
        or [expected_index[digest] for digest in pending_digests]
        != sorted(expected_index[digest] for digest in pending_digests)
    ):
        raise ExecutionValidationError(
            "pending evidence is not an ordered V1 full-cell subset"
        )
    if len(set(pending_digests)) != len(pending):
        raise ExecutionValidationError(
            "pending evidence contains duplicate cells"
        )
    decision_pending = {}
    for decision in decisions:
        if decision["disposition"] != "NEEDS_EVIDENCE":
            continue
        missing = decision.get("missing_cells")
        if type(missing) is not list:
            raise ExecutionValidationError(
                "NEEDS_EVIDENCE decision lacks cells"
            )
        for cell in missing:
            if type(cell) is not dict:
                raise ExecutionValidationError(
                    "decision missing cell is malformed"
                )
            key = (cell.get("profile"), cell.get("domain"), cell.get("seed"))
            if key in decision_pending and decision_pending[key] != cell:
                raise ExecutionValidationError(
                    "decision missing cells conflict"
                )
            decision_pending[key] = cell
    expected_pending_by_key = {
        (cell["profile"], cell["domain"], cell["seed"]): cell
        for cell in pending
    }
    if decision_pending != expected_pending_by_key:
        raise ExecutionValidationError(
            "report pending cells differ from decisions"
        )
    return _json_clone(pending, "source pending evidence")


def build_execution_plan(
    report: Mapping[str, Any],
    hypothesis_contract: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
    task_materializer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an ordered proposal; never infer authorization or execute work."""
    validate_executor_contract(executor_contract)
    source_binding = _bind_source_report(report, hypothesis_contract)
    pending = _validated_source_pending_cells(
        report, hypothesis_contract, executor_contract
    )

    contract_digest = _digest(executor_contract)
    identity = {"source_report_binding": source_binding, "executor_contract_digest": contract_digest}
    tasks = []
    for ordinal, cell in enumerate(pending):
        cell_projection = {
            "profile": cell["profile"], "domain": cell["domain"],
            "line": executor_contract["execution_scope"]["line"],
            "seed": cell["seed"], "d": cell["d"], "N": cell["N"], "n0": cell["n0"],
        }
        task_id = "task:" + _digest({**identity, "cell": cell_projection}).split(":", 1)[1][:24]
        run_task = None
        task_digest = None
        status = "BLOCKED_NO_TASK_TEMPLATE"
        if task_materializer is not None:
            run_task = _validate_task(task_materializer(_json_clone(cell_projection)), cell, executor_contract)
            task_digest = _digest({
                **identity,
                "task_id": task_id,
                "cell": cell_projection,
                "run_one_task": run_task,
            })
            status = "READY_FOR_AUTHORIZATION"
        tasks.append({
            "task_id": task_id, "ordinal": ordinal, "cell": cell_projection,
            "status": status, "run_one_task": run_task, "task_digest": task_digest,
        })
    plan_status = "AWAITING_TASK_TEMPLATE" if task_materializer is None else "READY_FOR_AUTHORIZATION"
    plan_identity = {**identity, "status": plan_status, "tasks": tasks}
    plan_id = "plan:" + _digest(plan_identity).split(":", 1)[1][:24]
    body = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": plan_id,
        "status": plan_status,
        "executor_contract_id": executor_contract["contract_id"],
        "executor_contract_digest": contract_digest,
        "source_report_binding": source_binding,
        "executor_binding": _json_clone(executor_contract["executor_binding"]),
        "execution_scope": _json_clone(executor_contract["execution_scope"]),
        "proposal_count": len(tasks),
        "tasks": tasks,
        "nonclaims": list(_NONCLAIMS),
    }
    return {**body, "integrity": {"algorithm": "sha256-canonical-json-v1", "plan_digest": _digest(body)}}


def _verify_plan_against_executor(
    plan: Mapping[str, Any], executor_contract: Mapping[str, Any]
) -> bool:
    try:
        validate_executor_contract(executor_contract)
        if not isinstance(plan, Mapping):
            return False
        integrity = plan.get("integrity")
        if not isinstance(integrity, Mapping) or set(integrity) != {"algorithm", "plan_digest"}:
            return False
        if integrity["algorithm"] != "sha256-canonical-json-v1":
            return False
        body = {key: plan[key] for key in plan if key != "integrity"}
        if _digest(body) != integrity["plan_digest"]:
            return False
        _exact_keys(body, {
            "schema_version", "plan_id", "status", "executor_contract_id",
            "executor_contract_digest", "source_report_binding", "executor_binding",
            "execution_scope", "proposal_count", "tasks", "nonclaims",
        }, "plan")
        if body["schema_version"] != PLAN_SCHEMA_VERSION:
            return False
        if body["executor_contract_id"] != executor_contract["contract_id"]:
            return False
        if body["executor_contract_digest"] != _digest(executor_contract):
            return False
        if body["executor_binding"] != executor_contract["executor_binding"] or body["execution_scope"] != executor_contract["execution_scope"]:
            return False
        if body["nonclaims"] != list(_NONCLAIMS) or type(body["tasks"]) is not list:
            return False
        if (
            type(body["proposal_count"]) is not int
            or body["proposal_count"] != len(body["tasks"])
            or not 1 <= body["proposal_count"] <= 30
        ):
            return False
        source_binding = body["source_report_binding"]
        if type(source_binding) is not dict or set(source_binding) != {
            "contract_id", "contract_digest", "evidence_digest",
            "report_body_digest", "audit_head",
        }:
            return False
        if (
            source_binding["contract_id"] != _SOURCE_CONTRACT_ID
            or source_binding["contract_digest"] != _SOURCE_CONTRACT_DIGEST
            or any(
                type(source_binding[key]) is not str
                or not source_binding[key].startswith("sha256:")
                for key in (
                    "contract_digest", "evidence_digest",
                    "report_body_digest", "audit_head",
                )
            )
        ):
            return False
        expected_state = "BLOCKED_NO_TASK_TEMPLATE" if body["status"] == "AWAITING_TASK_TEMPLATE" else "READY_FOR_AUTHORIZATION"
        if body["status"] not in {"AWAITING_TASK_TEMPLATE", "READY_FOR_AUTHORIZATION"}:
            return False
        ids = []
        domains = executor_contract["execution_scope"]["domains"]
        seeds = executor_contract["execution_scope"]["seeds"]
        allowed_cells = [
            {
                "profile": "full", "domain": domain, "line": "lodo",
                "seed": seed, "d": 50, "N": 20, "n0": 10,
            }
            for domain in domains for seed in seeds
        ]
        allowed_ordinals = {
            _digest(cell): ordinal for ordinal, cell in enumerate(allowed_cells)
        }
        identity = {
            "source_report_binding": source_binding,
            "executor_contract_digest": body["executor_contract_digest"],
        }
        previous_allowed_ordinal = -1
        for ordinal, task in enumerate(body["tasks"]):
            if type(task) is not dict:
                return False
            _exact_keys(task, {"task_id", "ordinal", "cell", "status", "run_one_task", "task_digest"}, "plan task")
            cell_digest = _digest(task.get("cell"))
            allowed_ordinal = allowed_ordinals.get(cell_digest)
            if (
                task["ordinal"] != ordinal
                or task["status"] != expected_state
                or allowed_ordinal is None
                or task["cell"] != allowed_cells[allowed_ordinal]
                or allowed_ordinal <= previous_allowed_ordinal
            ):
                return False
            previous_allowed_ordinal = allowed_ordinal
            expected_task_id = "task:" + _digest({
                **identity, "cell": task["cell"],
            }).split(":", 1)[1][:24]
            if task["task_id"] != expected_task_id:
                return False
            if expected_state.startswith("BLOCKED"):
                if task["run_one_task"] is not None or task["task_digest"] is not None:
                    return False
            else:
                validated = _validate_task(
                    task["run_one_task"], task["cell"], executor_contract
                )
                expected_task_digest = _digest({
                    **identity,
                    "task_id": task["task_id"],
                    "cell": task["cell"],
                    "run_one_task": validated,
                })
                if task["task_digest"] != expected_task_digest:
                    return False
            ids.append(task["task_id"])
        plan_identity = {**identity, "status": body["status"], "tasks": body["tasks"]}
        expected_plan_id = "plan:" + _digest(plan_identity).split(":", 1)[1][:24]
        return (
            len(set(ids)) == len(body["tasks"])
            and body["plan_id"] == expected_plan_id
        )
    except (ExecutionValidationError, KeyError, TypeError, ValueError):
        return False


def verify_plan_integrity(
    plan: Mapping[str, Any],
    hypothesis_contract: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
    source_report: Mapping[str, Any],
) -> bool:
    """Verify a plan against both frozen contracts and its own commitments."""
    try:
        validate_hypothesis_contract(hypothesis_contract)
        if _digest(hypothesis_contract) != _SOURCE_CONTRACT_DIGEST:
            return False
        if not _verify_plan_against_executor(plan, executor_contract):
            return False
        if plan["source_report_binding"]["contract_id"] != hypothesis_contract["contract_id"]:
            return False
        if plan["source_report_binding"]["contract_digest"] != _digest(hypothesis_contract):
            return False
        if plan["source_report_binding"] != _bind_source_report(
            source_report, hypothesis_contract
        ):
            return False
        pending = _validated_source_pending_cells(
            source_report, hypothesis_contract, executor_contract
        )
        expected_projections = [
            {
                "profile": cell["profile"],
                "domain": cell["domain"],
                "line": executor_contract["execution_scope"]["line"],
                "seed": cell["seed"],
                "d": cell["d"],
                "N": cell["N"],
                "n0": cell["n0"],
            }
            for cell in pending
        ]
        return [task["cell"] for task in plan["tasks"]] == expected_projections
    except (ExecutionValidationError, KeyError, TypeError, ValueError):
        return False


def authorize_plan(
    plan: Mapping[str, Any],
    hypothesis_contract: Mapping[str, Any],
    source_report: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
    *,
    expected_plan_digest: str,
    authorization_id: str,
    authorized_task_ids: Sequence[str],
) -> dict[str, Any]:
    if not verify_plan_integrity(
        plan, hypothesis_contract, executor_contract, source_report
    ):
        raise ExecutionValidationError("plan integrity failed")
    if plan["status"] != "READY_FOR_AUTHORIZATION":
        raise ExecutionValidationError("blocked plan cannot be authorized")
    if expected_plan_digest != plan["integrity"]["plan_digest"]:
        raise ExecutionValidationError("expected plan digest differs")
    if type(authorization_id) is not str or not authorization_id:
        raise ExecutionValidationError("authorization_id must be non-empty text")
    if type(authorized_task_ids) not in (list, tuple) or not authorized_task_ids:
        raise ExecutionValidationError("authorized_task_ids must be an explicit non-empty sequence")
    if any(type(item) is not str for item in authorized_task_ids) or len(set(authorized_task_ids)) != len(authorized_task_ids):
        raise ExecutionValidationError("authorized_task_ids are malformed or duplicated")
    selected = set(authorized_task_ids)
    known = {task["task_id"] for task in plan["tasks"]}
    if not selected.issubset(known):
        raise ExecutionValidationError("authorization contains an unknown task id")
    ordered = [task for task in plan["tasks"] if task["task_id"] in selected]
    body = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorization_id": authorization_id,
        "consent_kind": "LOCAL_EXPLICIT_CONSENT_NOT_AUTHORITY",
        "plan_id": plan["plan_id"],
        "plan_digest": expected_plan_digest,
        "executor_contract_digest": plan["executor_contract_digest"],
        "authorized_tasks": [
            {"task_id": task["task_id"], "task_digest": task["task_digest"]}
            for task in ordered
        ],
        "nonclaims": ["local_digest_is_not_signature", "no_external_authority", "authorization_is_not_execution"],
    }
    return {**body, "integrity": {"algorithm": "sha256-canonical-json-v1", "authorization_digest": _digest(body)}}


def verify_authorization_integrity(authorization: Mapping[str, Any]) -> bool:
    try:
        if not isinstance(authorization, Mapping):
            return False
        integrity = authorization.get("integrity")
        if not isinstance(integrity, Mapping) or set(integrity) != {"algorithm", "authorization_digest"}:
            return False
        body = {key: authorization[key] for key in authorization if key != "integrity"}
        if integrity["algorithm"] != "sha256-canonical-json-v1" or _digest(body) != integrity["authorization_digest"]:
            return False
        _exact_keys(body, {"schema_version", "authorization_id", "consent_kind", "plan_id", "plan_digest", "executor_contract_digest", "authorized_tasks", "nonclaims"}, "authorization")
        if body["schema_version"] != AUTHORIZATION_SCHEMA_VERSION or body["consent_kind"] != "LOCAL_EXPLICIT_CONSENT_NOT_AUTHORITY":
            return False
        if (
            type(body["authorization_id"]) is not str
            or not body["authorization_id"]
            or type(body["plan_id"]) is not str
            or not body["plan_id"].startswith("plan:")
            or type(body["plan_digest"]) is not str
            or not body["plan_digest"].startswith("sha256:")
            or body["executor_contract_digest"] != _EXECUTOR_CONTRACT_DIGEST
            or body["nonclaims"] != [
                "local_digest_is_not_signature",
                "no_external_authority",
                "authorization_is_not_execution",
            ]
        ):
            return False
        tasks = body["authorized_tasks"]
        return (
            type(tasks) is list and bool(tasks)
            and all(
                type(item) is dict
                and set(item) == {"task_id", "task_digest"}
                and type(item["task_id"]) is str
                and item["task_id"].startswith("task:")
                and type(item["task_digest"]) is str
                and item["task_digest"].startswith("sha256:")
                for item in tasks
            )
            and len({item["task_id"] for item in tasks}) == len(tasks)
        )
    except (ExecutionValidationError, KeyError, TypeError, ValueError):
        return False


def _authorization_matches_plan(
    authorization: Mapping[str, Any], plan: Mapping[str, Any]
) -> bool:
    if not verify_authorization_integrity(authorization):
        return False
    if (
        authorization.get("plan_id") != plan.get("plan_id")
        or authorization.get("plan_digest")
        != plan.get("integrity", {}).get("plan_digest")
        or authorization.get("executor_contract_digest")
        != plan.get("executor_contract_digest")
    ):
        return False
    selected = {item["task_id"] for item in authorization["authorized_tasks"]}
    expected = [
        {"task_id": task["task_id"], "task_digest": task["task_digest"]}
        for task in plan["tasks"]
        if task["task_id"] in selected
    ]
    return authorization["authorized_tasks"] == expected


def normalize_run_one_result(
    raw_result: Any,
    run_one_task: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a native ``run_one(task)`` result through the existing reader.

    The benchmark returns a large native result mapping, while the hypothesis
    loop consumes the stable aggregate-row projection.  This adapter performs
    that already-established projection in memory; it does not write or scan a
    result directory.
    """
    if type(raw_result) is not dict:
        raise ExecutionValidationError(
            "run_one result must be an exact native result dict"
        )
    if type(run_one_task) is not dict or type(run_one_task.get("args")) is not dict:
        raise ExecutionValidationError("run_one task is malformed at result adaptation")
    if (
        raw_result.get("heldout") != run_one_task.get("heldout")
        or raw_result.get("line") != run_one_task.get("line")
        or type(raw_result.get("seed")) is not int
        or raw_result.get("seed") != run_one_task.get("seed")
    ):
        raise ExecutionValidationError(
            "native run_one result identity differs from its task"
        )
    if raw_result.get("status") not in (None, "ok"):
        raise ExecutionValidationError("native run_one result is non-ok")
    observed_variant = raw_result.get("experiment_variant")
    if observed_variant not in (None, cell["variant"]):
        raise ExecutionValidationError(
            "native run_one result variant differs from its cell"
        )

    native_row = dict(raw_result)
    native_row["experiment_variant"] = cell["variant"]
    payload = {
        "schema_version": 1,
        "experiment_variant": cell["variant"],
        "config": run_one_task["args"],
    }
    synthetic_root = Path(cell["run_id"])
    synthetic_path = (
        synthetic_root
        / "priors"
        / cell["profile"]
        / cell["domain"]
        / f"seed{cell['seed']}"
        / "result.json"
    )
    normalized = _normalize_sc_run_one_result_v1(
        payload, native_row, synthetic_path, synthetic_root
    )
    return {
        field: normalized.get(field)
        for field in REQUIRED_EVIDENCE_FIELDS
    }


def _normalized_evidence_row(raw: Any, cell: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    if type(raw) is not dict:
        raise ExecutionValidationError("executor result is not an exact evidence-row dict")
    if set(raw) != set(REQUIRED_EVIDENCE_FIELDS):
        raise ExecutionValidationError("executor result is not an exact normalized evidence row")
    row = _json_clone(raw, f"executor result for {task_id}")
    expected = {
        "track": cell["track"], "run_id": cell["run_id"], "variant": cell["variant"],
        "method": cell["profile"], "structural_prior_profile": cell["profile"],
        "domain": cell["domain"], "seed": cell["seed"], "d": cell["d"],
        "N": cell["N"], "n0": cell["n0"], "source_calls": cell["source_calls"],
        "implementation": cell["implementation"], "initial_design": cell["initial_design"],
        "decision_backend": cell["decision_backend"],
        "total_calls": cell["total_calls"], "hvd_profile": cell["hvd_profile"],
        "source_discrepancy_update": cell["source_discrepancy_update"],
        "recheck_top_k": cell["recheck_top_k"], "risk_penalty": cell["risk_penalty"],
        "utility_weight": cell["utility_weight"], "adaptive_replication_voi": cell["adaptive_replication_voi"],
        "posterior_dominance_enabled": cell["posterior_dominance_enabled"],
        "posterior_dominance_switch_count": cell["posterior_dominance_switch_count"],
    }
    for key, value in expected.items():
        if type(row.get(key)) is not type(value) or row[key] != value:
            raise ExecutionValidationError(f"evidence row {key} differs from authorized cell")
    if row.get("status") != "ok":
        raise ExecutionValidationError("evidence row status must be exact ok")
    for key in ("true_feasible", "adaptive_loss"):
        if type(row.get(key)) is not bool:
            raise ExecutionValidationError(f"evidence row {key} must be native bool")
    regret = row.get("feasible_regret")
    if regret is not None and (type(regret) not in (int, float) or not math.isfinite(regret)):
        raise ExecutionValidationError("evidence row feasible_regret must be null or finite")
    if regret is not None and regret < 0:
        raise ExecutionValidationError("evidence row feasible_regret must be nonnegative")
    if row["status"] == "ok" and row["true_feasible"] and regret is None:
        raise ExecutionValidationError("feasible successful row lacks finite regret")
    if not row["true_feasible"] and regret is not None:
        raise ExecutionValidationError(
            "infeasible successful row must not carry feasible_regret"
        )
    return row


def execute_authorized_plan(
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    expected_authorization_digest: str,
    executor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    hypothesis_contract: Mapping[str, Any],
    source_report: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not callable(executor):
        raise ExecutionValidationError("executor must be an injected callable")
    plan_snapshot = _json_clone(dict(plan), "execution plan snapshot")
    authorization_snapshot = _json_clone(
        dict(authorization), "authorization snapshot"
    )
    contract_snapshot = _json_clone(
        dict(executor_contract), "executor contract snapshot"
    )
    hypothesis_contract_snapshot = _json_clone(
        dict(hypothesis_contract), "hypothesis contract snapshot"
    )
    source_report_snapshot = _json_clone(
        dict(source_report), "source report snapshot"
    )
    if not verify_plan_integrity(
        plan_snapshot,
        hypothesis_contract_snapshot,
        contract_snapshot,
        source_report_snapshot,
    ):
        raise ExecutionValidationError("plan integrity failed before execution")
    if not _authorization_matches_plan(
        authorization_snapshot, plan_snapshot
    ):
        raise ExecutionValidationError("authorization integrity failed")
    if expected_authorization_digest != authorization_snapshot["integrity"]["authorization_digest"]:
        raise ExecutionValidationError("expected authorization digest differs")
    authorized = {
        item["task_id"]: item["task_digest"]
        for item in authorization_snapshot["authorized_tasks"]
    }
    task_by_id = {
        item["task_id"]: item for item in plan_snapshot["tasks"]
    }
    if any(task_id not in task_by_id or task_by_id[task_id]["task_digest"] != digest for task_id, digest in authorized.items()):
        raise ExecutionValidationError("authorization task binding differs from plan")
    results = []
    for task in plan_snapshot["tasks"]:
        if task["task_id"] not in authorized:
            continue
        phase = "EXECUTOR_CALLBACK"
        try:
            raw = executor(_json_clone(task["run_one_task"], "authorized task"))
            phase = "RESULT_ADAPTER"
            full_cell = _full_cell_from_plan_task(task, contract_snapshot)
            projected = normalize_run_one_result(
                raw, task["run_one_task"], full_cell
            )
            row = _normalized_evidence_row(
                projected, full_cell, task["task_id"]
            )
            results.append({
                "task_id": task["task_id"], "task_digest": task["task_digest"],
                "status": "SUCCEEDED", "evidence_row": row,
                "evidence_digest": _digest(row), "error": None,
            })
        except Exception as exc:  # callback failures become evidence-neutral receipts
            results.append({
                "task_id": task["task_id"], "task_digest": task["task_digest"],
                "status": "FAILED", "evidence_row": None, "evidence_digest": None,
                "error": {
                    "code": (
                        "EXECUTOR_EXCEPTION"
                        if phase == "EXECUTOR_CALLBACK"
                        else "RESULT_REJECTED"
                    ),
                    "type": type(exc).__name__,
                },
            })
    succeeded = sum(item["status"] == "SUCCEEDED" for item in results)
    body = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_id": "receipt:" + _digest({"plan": plan_snapshot["plan_id"], "authorization": expected_authorization_digest, "results": results}).split(":", 1)[1][:24],
        "status": "COMPLETED" if succeeded == len(results) else "COMPLETED_WITH_FAILURES",
        "plan_binding": {
            "plan_id": plan_snapshot["plan_id"],
            "plan_digest": plan_snapshot["integrity"]["plan_digest"],
            "executor_contract_digest": plan_snapshot["executor_contract_digest"],
            "source_report_binding": _json_clone(plan_snapshot["source_report_binding"]),
        },
        "authorization_binding": {"authorization_id": authorization_snapshot["authorization_id"], "authorization_digest": expected_authorization_digest},
        "results": results,
        "summary": {"authorized": len(results), "succeeded": succeeded, "failed": len(results) - succeeded},
        "nonclaims": ["execution_is_not_reingestion", "no_scientific_verdict_from_plan", "local_digest_is_not_signature"],
    }
    return {**body, "integrity": {"algorithm": "sha256-canonical-json-v1", "receipt_digest": _digest(body)}}


def _full_cell_from_plan_task(
    task: Mapping[str, Any], executor_contract: Mapping[str, Any]
) -> dict[str, Any]:
    # The exact evidence cell is committed by the task args and frozen V1 scope.
    cell = task["cell"]
    fixed = executor_contract["fixed_task_inputs"]
    return {
        "track": "priors", "run_id": "scolh_structural_backend_gate_n20_s10_20260716",
        "variant": "structural_backend/priors/full", "profile": "full",
        "domain": cell["domain"], "seed": cell["seed"], "d": cell["d"], "N": cell["N"], "n0": cell["n0"],
        "source_calls": 384, "implementation": "sc_olh", "initial_design": "source_informed",
        "decision_backend": fixed["decision_backend"], "total_calls": 404,
        "hvd_profile": fixed["hvd_ablation_profile"],
        "source_discrepancy_update": fixed["source_discrepancy_update"],
        "recheck_top_k": fixed["certification_recheck_top_k"],
        "risk_penalty": fixed["decision_risk_penalty"], "utility_weight": fixed["decision_source_utility_weight"],
        "adaptive_replication_voi": fixed["adaptive_replication_voi"],
        "posterior_dominance_enabled": fixed["posterior_dominance_enabled"],
        "posterior_dominance_switch_count": 0,
    }


def verify_receipt_integrity(
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any] | None = None,
    authorization: Mapping[str, Any] | None = None,
    source_report: Mapping[str, Any] | None = None,
    hypothesis_contract: Mapping[str, Any] | None = None,
    executor_contract: Mapping[str, Any] | None = None,
) -> bool:
    try:
        if not isinstance(receipt, Mapping):
            return False
        integrity = receipt.get("integrity")
        if not isinstance(integrity, Mapping) or set(integrity) != {"algorithm", "receipt_digest"}:
            return False
        body = {key: receipt[key] for key in receipt if key != "integrity"}
        if integrity["algorithm"] != "sha256-canonical-json-v1" or _digest(body) != integrity["receipt_digest"]:
            return False
        _exact_keys(body, {"schema_version", "receipt_id", "status", "plan_binding", "authorization_binding", "results", "summary", "nonclaims"}, "receipt")
        if (
            body["schema_version"] != RECEIPT_SCHEMA_VERSION
            or type(body["results"]) is not list
            or not 1 <= len(body["results"]) <= 30
        ):
            return False
        if type(body["plan_binding"]) is not dict or set(body["plan_binding"]) != {
            "plan_id", "plan_digest", "executor_contract_digest",
            "source_report_binding",
        }:
            return False
        if type(body["authorization_binding"]) is not dict or set(body["authorization_binding"]) != {
            "authorization_id", "authorization_digest",
        }:
            return False
        if (
            type(body["receipt_id"]) is not str
            or not body["receipt_id"].startswith("receipt:")
            or body["nonclaims"] != [
                "execution_is_not_reingestion",
                "no_scientific_verdict_from_plan",
                "local_digest_is_not_signature",
            ]
            or body["plan_binding"]["executor_contract_digest"]
            != _EXECUTOR_CONTRACT_DIGEST
        ):
            return False
        source_binding = body["plan_binding"]["source_report_binding"]
        if (
            type(source_binding) is not dict
            or set(source_binding) != {
                "contract_id", "contract_digest", "evidence_digest",
                "report_body_digest", "audit_head",
            }
            or source_binding["contract_id"] != _SOURCE_CONTRACT_ID
            or source_binding["contract_digest"] != _SOURCE_CONTRACT_DIGEST
            or type(body["plan_binding"]["plan_id"]) is not str
            or not body["plan_binding"]["plan_id"].startswith("plan:")
            or type(body["plan_binding"]["plan_digest"]) is not str
            or not body["plan_binding"]["plan_digest"].startswith("sha256:")
            or type(body["authorization_binding"]["authorization_id"])
            is not str
            or not body["authorization_binding"]["authorization_id"]
            or type(body["authorization_binding"]["authorization_digest"])
            is not str
            or not body["authorization_binding"][
                "authorization_digest"
            ].startswith("sha256:")
        ):
            return False
        ids = set()
        success = failed = 0
        for result in body["results"]:
            if type(result) is not dict or set(result) != {"task_id", "task_digest", "status", "evidence_row", "evidence_digest", "error"}:
                return False
            if (
                type(result["task_id"]) is not str
                or not result["task_id"].startswith("task:")
                or type(result["task_digest"]) is not str
                or not result["task_digest"].startswith("sha256:")
            ):
                return False
            if result["task_id"] in ids:
                return False
            ids.add(result["task_id"])
            if result["status"] == "SUCCEEDED":
                success += 1
                if type(result["evidence_row"]) is not dict or result["evidence_digest"] != _digest(result["evidence_row"]) or result["error"] is not None:
                    return False
            elif result["status"] == "FAILED":
                failed += 1
                if (
                    result["evidence_row"] is not None
                    or result["evidence_digest"] is not None
                    or type(result["error"]) is not dict
                    or set(result["error"]) != {"code", "type"}
                    or result["error"]["code"] not in {
                        "EXECUTOR_EXCEPTION", "RESULT_REJECTED",
                    }
                    or any(
                        type(result["error"][key]) is not str
                        for key in ("code", "type")
                    )
                ):
                    return False
            else:
                return False
        expected_status = (
            "COMPLETED" if success == len(body["results"])
            else "COMPLETED_WITH_FAILURES"
        )
        expected_receipt_id = "receipt:" + _digest({
            "plan": body["plan_binding"]["plan_id"],
            "authorization": body["authorization_binding"][
                "authorization_digest"
            ],
            "results": body["results"],
        }).split(":", 1)[1][:24]
        structurally_valid = (
            body["status"] == expected_status
            and body["receipt_id"] == expected_receipt_id
            and body["summary"] == {
                "authorized": len(body["results"]),
                "succeeded": success,
                "failed": failed,
            }
        )
        if not structurally_valid:
            return False

        supplied = (
            plan, authorization, source_report, hypothesis_contract,
            executor_contract,
        )
        if all(item is None for item in supplied):
            return True
        if any(item is None for item in supplied):
            return False
        if not verify_plan_integrity(
            plan, hypothesis_contract, executor_contract, source_report
        ):
            return False
        if not _authorization_matches_plan(authorization, plan):
            return False
        if body["plan_binding"] != {
            "plan_id": plan["plan_id"],
            "plan_digest": plan["integrity"]["plan_digest"],
            "executor_contract_digest": plan["executor_contract_digest"],
            "source_report_binding": plan["source_report_binding"],
        }:
            return False
        if body["authorization_binding"] != {
            "authorization_id": authorization["authorization_id"],
            "authorization_digest": authorization["integrity"][
                "authorization_digest"
            ],
        }:
            return False
        expected_tasks = authorization["authorized_tasks"]
        observed_tasks = [
            {
                "task_id": result["task_id"],
                "task_digest": result["task_digest"],
            }
            for result in body["results"]
        ]
        if observed_tasks != expected_tasks:
            return False
        task_by_id = {task["task_id"]: task for task in plan["tasks"]}
        for result in body["results"]:
            if result["status"] != "SUCCEEDED":
                continue
            task = task_by_id[result["task_id"]]
            full_cell = _full_cell_from_plan_task(task, executor_contract)
            _normalized_evidence_row(
                result["evidence_row"], full_cell, result["task_id"]
            )
        return True
    except (ExecutionValidationError, KeyError, TypeError, ValueError):
        return False


def _full_cell_from_hypothesis_contract(
    row: Mapping[str, Any], hypothesis_contract: Mapping[str, Any]
) -> dict[str, Any]:
    scope = hypothesis_contract["evidence_scope"]
    if (
        row.get("method") != "full"
        or row.get("domain") not in scope["domains"]
        or type(row.get("seed")) is not int
        or row["seed"] not in scope["seeds"]
    ):
        raise ExecutionValidationError("successful row is outside the full-cell scope")
    cell = {
        "track": scope["track"],
        "run_id": scope["run_id"],
        "variant": scope["variant_template"].format(profile="full"),
        "profile": "full",
        "domain": row["domain"],
        "seed": row["seed"],
        "d": scope["d"],
        "N": scope["N"],
        "n0": scope["n0"],
        "source_calls": scope["source_calls"],
        "implementation": scope["implementation"],
        "initial_design": scope["initial_design"],
        "decision_backend": scope["decision_backend"],
    }
    cell.update(scope["fixed_row_values"])
    return cell


def _wire_cell_identity(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    seed = row.get("seed")
    if type(seed) is str:
        try:
            parsed = int(seed)
        except ValueError:
            parsed = seed
        else:
            seed = parsed if seed == str(parsed) else seed
    return row.get("method"), row.get("domain"), seed


def reingest_successful_receipts(
    base_rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    hypothesis_contract: Mapping[str, Any],
    source_report: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
) -> dict[str, Any]:
    source_binding = _bind_source_report(source_report, hypothesis_contract)
    if not verify_plan_integrity(
        plan, hypothesis_contract, executor_contract, source_report
    ):
        raise ExecutionValidationError("execution plan integrity failed")
    if plan.get("source_report_binding") != source_binding:
        raise ExecutionValidationError("plan does not bind the source report")
    if not _authorization_matches_plan(authorization, plan):
        raise ExecutionValidationError(
            "authorization does not bind the execution plan"
        )
    if isinstance(base_rows, (str, bytes)) or not isinstance(base_rows, Sequence):
        raise ExecutionValidationError("base_rows must be a sequence")
    if _digest(base_rows) != source_report["evidence_digest"]:
        raise ExecutionValidationError("base_rows do not bind source evidence_digest")
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise ExecutionValidationError("receipts must be a sequence")
    if not receipts:
        raise ExecutionValidationError("at least one execution receipt is required")
    if any(not isinstance(row, Mapping) for row in base_rows):
        raise ExecutionValidationError("every base row must be an object")
    combined = [_json_clone(dict(row), "base row") for row in base_rows]
    existing = set()
    scope = hypothesis_contract["evidence_scope"]
    profiles = set(hypothesis_contract["profiles"])
    for row in combined:
        if row.get("track") == scope["track"] and row.get("run_id") == scope["run_id"] and row.get("method") in profiles:
            key = _wire_cell_identity(row)
            if key in existing:
                raise ExecutionValidationError("base_rows contain duplicate in-scope cells")
            existing.add(key)
    receipt_digests = []
    accepted = failed = 0
    plan_binding = None
    seen_receipts = set()
    seen_result_tasks = set()
    for receipt in receipts:
        if not verify_receipt_integrity(
            receipt,
            plan,
            authorization,
            source_report,
            hypothesis_contract,
            executor_contract,
        ):
            raise ExecutionValidationError("execution receipt integrity failed")
        receipt_digest = receipt["integrity"]["receipt_digest"]
        if receipt_digest in seen_receipts:
            raise ExecutionValidationError("duplicate execution receipt")
        seen_receipts.add(receipt_digest)
        binding = receipt["plan_binding"]
        if binding.get("source_report_binding") != source_binding:
            raise ExecutionValidationError("receipt does not bind source report")
        if binding.get("executor_contract_digest") != _EXECUTOR_CONTRACT_DIGEST:
            raise ExecutionValidationError("receipt binds an unknown executor contract")
        if plan_binding is None:
            plan_binding = {"plan_id": binding.get("plan_id"), "plan_digest": binding.get("plan_digest")}
        elif plan_binding != {"plan_id": binding.get("plan_id"), "plan_digest": binding.get("plan_digest")}:
            raise ExecutionValidationError("receipts bind conflicting plans")
        receipt_digests.append(receipt_digest)
        for result in receipt["results"]:
            if result["task_id"] in seen_result_tasks:
                raise ExecutionValidationError(
                    "receipts contain a duplicate task result"
                )
            seen_result_tasks.add(result["task_id"])
            if result["status"] != "SUCCEEDED":
                failed += 1
                continue
            row = _json_clone(result["evidence_row"], "successful evidence row")
            full_cell = _full_cell_from_hypothesis_contract(
                row, hypothesis_contract
            )
            row = _normalized_evidence_row(row, full_cell, result["task_id"])
            cell_projection = {
                "profile": "full", "domain": row["domain"], "line": "lodo",
                "seed": row["seed"], "d": row["d"], "N": row["N"],
                "n0": row["n0"],
            }
            expected_task_id = "task:" + _digest({
                "source_report_binding": source_binding,
                "executor_contract_digest": _EXECUTOR_CONTRACT_DIGEST,
                "cell": cell_projection,
            }).split(":", 1)[1][:24]
            if result["task_id"] != expected_task_id:
                raise ExecutionValidationError(
                    "successful evidence row differs from its task id"
                )
            key = _wire_cell_identity(row)
            if key in existing:
                raise ExecutionValidationError("duplicate or conflicting evidence cell")
            existing.add(key)
            combined.append(row)
            accepted += 1
    if accepted == 0:
        raise ExecutionValidationError(
            "reingestion requires at least one successful authorized row"
        )
    loop_result = run_structural_hypothesis_loop(combined, hypothesis_contract)
    report = loop_result.to_dict()
    reingestion_body = {
        "schema_version": REINGESTION_SCHEMA_VERSION,
        "status": "REINGESTED",
        "source_report_binding": source_binding,
        "plan_binding": plan_binding,
        "authorization_binding": {
            "authorization_id": authorization["authorization_id"],
            "authorization_digest": authorization["integrity"][
                "authorization_digest"
            ],
        },
        "executor_contract_digest": _digest(executor_contract),
        "execution_receipt_digests": receipt_digests,
        "accepted_successful_rows": accepted,
        "ignored_failed_attempts": failed,
        "combined_evidence_digest": report["evidence_digest"],
        "output_report_body_digest": report["audit"]["report_body_digest"],
        "nonclaims": ["failed_execution_is_not_scientific_refutation", "reingestion_is_not_external_verification"],
    }
    reingestion_receipt = {
        **reingestion_body,
        "integrity": {"algorithm": "sha256-canonical-json-v1", "reingestion_digest": _digest(reingestion_body)},
    }
    if not verify_reingestion_integrity(
        reingestion_receipt,
        source_report=source_report,
        base_rows=base_rows,
        plan=plan,
        authorization=authorization,
        receipts=receipts,
        output_report=report,
        hypothesis_contract=hypothesis_contract,
        executor_contract=executor_contract,
    ):
        raise ExecutionValidationError(
            "generated reingestion receipt failed integrity verification"
        )
    return {"report": report, "reingestion_receipt": reingestion_receipt}


def verify_reingestion_integrity(
    receipt: Mapping[str, Any],
    *,
    source_report: Mapping[str, Any] | None = None,
    base_rows: Sequence[Mapping[str, Any]] | None = None,
    plan: Mapping[str, Any] | None = None,
    authorization: Mapping[str, Any] | None = None,
    receipts: Sequence[Mapping[str, Any]] | None = None,
    output_report: Mapping[str, Any] | None = None,
    hypothesis_contract: Mapping[str, Any] | None = None,
    executor_contract: Mapping[str, Any] | None = None,
) -> bool:
    """Verify the fourth-stage artifact, optionally through its full chain."""
    try:
        if not isinstance(receipt, Mapping):
            return False
        integrity = receipt.get("integrity")
        if (
            not isinstance(integrity, Mapping)
            or set(integrity) != {"algorithm", "reingestion_digest"}
            or integrity.get("algorithm") != "sha256-canonical-json-v1"
        ):
            return False
        body = {key: receipt[key] for key in receipt if key != "integrity"}
        if _digest(body) != integrity.get("reingestion_digest"):
            return False
        _exact_keys(
            body,
            {
                "schema_version", "status", "source_report_binding",
                "plan_binding", "authorization_binding",
                "executor_contract_digest", "execution_receipt_digests",
                "accepted_successful_rows", "ignored_failed_attempts",
                "combined_evidence_digest", "output_report_body_digest",
                "nonclaims",
            },
            "reingestion receipt",
        )
        if (
            body["schema_version"] != REINGESTION_SCHEMA_VERSION
            or body["status"] != "REINGESTED"
            or body["executor_contract_digest"] != _EXECUTOR_CONTRACT_DIGEST
            or type(body["execution_receipt_digests"]) is not list
            or not body["execution_receipt_digests"]
            or len(set(body["execution_receipt_digests"]))
            != len(body["execution_receipt_digests"])
            or any(
                type(item) is not str or not item.startswith("sha256:")
                for item in body["execution_receipt_digests"]
            )
            or type(body["accepted_successful_rows"]) is not int
            or body["accepted_successful_rows"] <= 0
            or type(body["ignored_failed_attempts"]) is not int
            or body["ignored_failed_attempts"] < 0
            or body["nonclaims"] != [
                "failed_execution_is_not_scientific_refutation",
                "reingestion_is_not_external_verification",
            ]
        ):
            return False
        supplied = (
            source_report, base_rows, plan, authorization, receipts, output_report,
            hypothesis_contract, executor_contract,
        )
        if all(item is None for item in supplied):
            return True
        if any(item is None for item in supplied):
            return False
        source_binding = _bind_source_report(
            source_report, hypothesis_contract
        )
        if (
            isinstance(base_rows, (str, bytes))
            or not isinstance(base_rows, Sequence)
            or any(not isinstance(row, Mapping) for row in base_rows)
            or _digest(base_rows) != source_report["evidence_digest"]
        ):
            return False
        if not verify_plan_integrity(
            plan, hypothesis_contract, executor_contract, source_report
        ) or plan["source_report_binding"] != source_binding:
            return False
        if not _authorization_matches_plan(authorization, plan):
            return False
        if body["source_report_binding"] != source_binding:
            return False
        if body["plan_binding"] != {
            "plan_id": plan["plan_id"],
            "plan_digest": plan["integrity"]["plan_digest"],
        }:
            return False
        if body["authorization_binding"] != {
            "authorization_id": authorization["authorization_id"],
            "authorization_digest": authorization["integrity"][
                "authorization_digest"
            ],
        }:
            return False
        if not isinstance(receipts, Sequence) or isinstance(
            receipts, (str, bytes)
        ):
            return False
        if body["execution_receipt_digests"] != [
            item["integrity"]["receipt_digest"] for item in receipts
        ]:
            return False
        seen_tasks = set()
        successful = failed = 0
        for execution_receipt in receipts:
            if not verify_receipt_integrity(
                execution_receipt,
                plan,
                authorization,
                source_report,
                hypothesis_contract,
                executor_contract,
            ):
                return False
            for result in execution_receipt["results"]:
                if result["task_id"] in seen_tasks:
                    return False
                seen_tasks.add(result["task_id"])
                if result["status"] == "SUCCEEDED":
                    successful += 1
                else:
                    failed += 1
        if (
            body["accepted_successful_rows"] != successful
            or body["ignored_failed_attempts"] != failed
        ):
            return False
        if not verify_report_integrity(output_report):
            return False
        if not (
            body["combined_evidence_digest"]
            == output_report["evidence_digest"]
            and body["output_report_body_digest"]
            == output_report["audit"]["report_body_digest"]
        ):
            return False
        recomputed_rows = [
            _json_clone(dict(row), "reingestion verification base row")
            for row in base_rows
        ]
        for execution_receipt in receipts:
            for result in execution_receipt["results"]:
                if result["status"] == "SUCCEEDED":
                    recomputed_rows.append(
                        _json_clone(
                            result["evidence_row"],
                            "reingestion verification evidence row",
                        )
                    )
        recomputed_report = run_structural_hypothesis_loop(
            recomputed_rows, hypothesis_contract
        ).to_dict()
        return recomputed_report == output_report
    except (ExecutionValidationError, KeyError, TypeError, ValueError):
        return False


__all__ = [
    "AUTHORIZATION_SCHEMA_VERSION", "EXECUTOR_SCHEMA_VERSION",
    "ExecutionValidationError", "PLAN_SCHEMA_VERSION", "RECEIPT_SCHEMA_VERSION",
    "REINGESTION_SCHEMA_VERSION", "authorize_plan", "build_execution_plan",
    "execute_authorized_plan", "normalize_run_one_result",
    "reingest_successful_receipts",
    "validate_executor_contract", "verify_authorization_integrity",
    "verify_plan_integrity", "verify_receipt_integrity",
    "verify_reingestion_integrity",
]
