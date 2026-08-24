"""Inspect and advance one bounded local structural-hypothesis campaign.

The CLI is deliberately thin.  It parses one duplicate-free source
descriptor and forwards only explicit, action-specific anchors to the core.
Importing this module never imports the benchmark or crosses the callback
boundary.  The ``execute`` action is the sole action that can ask the core to
cross that boundary, and it requires an explicit confirmation flag.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

DEFAULT_CAMPAIGN_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_recursive_campaign_v1.json"
)
DEFAULT_RUNTIME_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_single_task_runtime_v1.json"
)

REQUIRED_EXECUTION_ENVIRONMENT = {
    "SCOLHKG_OFFLINE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_TASK_ID_PATTERN = re.compile(r"task:[0-9a-f]{24}\Z")
_AUTHORIZATION_ID_PATTERN = re.compile(
    r"recursive-campaign-v1:[0-9a-f]{64}\Z"
)

_STATUS_BY_ACTION = {
    "inspect": (
        "INSPECTED_RECURSIVE_CAMPAIGN_SOURCE_"
        "NONTERMINAL_NOT_AUTHORIZED"
    ),
    "authorize": (
        "RECURSIVE_CAMPAIGN_AUTHORIZED_ONE_CALLBACK_START_LEASED"
    ),
    "execute": (
        "RECURSIVE_CAMPAIGN_CALLBACK_START_CLAIMED_"
        "RUNTIME_COMPLETED_HARD_STOP"
    ),
}
_ADVANCE_STATUSES = {
    "ADVANCED_RECURSIVE_CAMPAIGN_ONE_STEP_NONTERMINAL_HARD_STOP",
    "ADVANCED_RECURSIVE_CAMPAIGN_ONE_STEP_TERMINAL_HARD_STOP",
}
_CALLBACK_INCOMPLETE_STATUS = (
    "RECURSIVE_CAMPAIGN_CALLBACK_START_CLAIMED_"
    "RUNTIME_INCOMPLETE_HARD_STOP"
)
_EXECUTE_COMPLETION_STATUSES = {
    "COMPLETED_SUCCESS_AWAITING_ADVANCE",
    "COMPLETED_FAILED_EVIDENCE_NEUTRAL_HARD_STOP",
}
_SUCCESS_PREVIEW_STATUSES = {
    "COMPLETED_SUCCESS_PREVIEW_INDEPENDENTLY_ANCHORED_HARD_STOP",
    (
        "COMPLETED_SUCCESS_PREVIEW_RECOVERED_LOCAL_ANCHORS_"
        "NOT_INDEPENDENT_HARD_STOP"
    ),
}
_FAILED_COMPLETION_STATUSES = {
    "COMPLETED_FAILED_EVIDENCE_NEUTRAL_INDEPENDENTLY_ANCHORED_HARD_STOP",
    (
        "COMPLETED_FAILED_EVIDENCE_NEUTRAL_RECOVERED_LOCAL_ANCHORS_"
        "NOT_INDEPENDENT_HARD_STOP"
    ),
}
_VERIFYABLE_STATUSES = {
    *_STATUS_BY_ACTION.values(),
    *_ADVANCE_STATUSES,
    _CALLBACK_INCOMPLETE_STATUS,
}
_PHASE_BY_STATUS = {
    _STATUS_BY_ACTION["authorize"]: "AUTHORIZED",
    _CALLBACK_INCOMPLETE_STATUS: "CALLBACK_INCOMPLETE",
    _STATUS_BY_ACTION["execute"]: "CALLBACK_COMPLETED",
    (
        "ADVANCED_RECURSIVE_CAMPAIGN_ONE_STEP_"
        "NONTERMINAL_HARD_STOP"
    ): "ADVANCED_NONTERMINAL",
    (
        "ADVANCED_RECURSIVE_CAMPAIGN_ONE_STEP_"
        "TERMINAL_HARD_STOP"
    ): "ADVANCED_TERMINAL",
}

_INSPECT_RESULT_KEYS = (
    "status",
    "source_kind",
    "source_state_digest",
    "bundle_digest",
    "plan_digest",
    "task_count",
    "task_id",
    "task_digest",
    "next_attempt_root",
    "checkpoint_root",
    "provenance_binding",
    "provenance_binding_digest",
    "required_authorization_id",
    "terminal_status",
)
_AUTHORIZE_RESULT_KEYS = (
    *_INSPECT_RESULT_KEYS,
    "campaign_root",
    "campaign_digest",
    "lease_digest",
    "authorization_digest",
    "attempt_digest",
    "authorization_status",
    "execution_status",
)
_EXECUTE_RESULT_KEYS = (
    "status",
    "campaign_root",
    "campaign_digest",
    "lease_digest",
    "callback_start_claim_digest",
    "provenance_binding_digest",
    "authorization_digest",
    "attempt_digest",
    "receipt_digest",
    "journal_head_digest",
    "task_id",
    "execution_status",
)
_ADVANCE_RESULT_KEYS = (
    "status",
    "phase",
    "campaign_root",
    "campaign_digest",
    "lease_digest",
    "callback_start_claim_digest",
    "provenance_binding_digest",
    "authorization_digest",
    "attempt_digest",
    "receipt_digest",
    "journal_head_digest",
    "advance_digest",
    "reingestion_digest",
    "output_evidence_digest",
    "output_report_body_digest",
    "output_audit_head",
    "next_pending_evidence_digest",
    "next_first_pending_projection_digest",
    "next_bundle_digest",
    "next_plan_digest",
    "remaining_task_count",
    "terminal_status",
    "next_attempt_root",
    "execution_status",
)
_RESULT_KEYS_BY_ACTION = {
    "inspect": _INSPECT_RESULT_KEYS,
    "authorize": _AUTHORIZE_RESULT_KEYS,
    "execute": _EXECUTE_RESULT_KEYS,
    "advance": _ADVANCE_RESULT_KEYS,
    "verify": _ADVANCE_RESULT_KEYS,
}

_AUTHORIZE_DIGEST_NAMES = (
    "expected_source_state_digest",
    "expected_bundle_digest",
    "expected_plan_digest",
    "expected_task_digest",
    "expected_provenance_binding_digest",
)
_EXECUTE_DIGEST_NAMES = (
    "expected_campaign_digest",
    "expected_lease_digest",
    "expected_provenance_binding_digest",
    "expected_authorization_digest",
    "expected_attempt_digest",
)
_ADVANCE_DIGEST_NAMES = (
    "expected_campaign_digest",
    "expected_lease_digest",
    "expected_provenance_binding_digest",
    "expected_authorization_digest",
    "expected_attempt_digest",
    "expected_receipt_digest",
    "expected_journal_head_digest",
    "expected_output_evidence_digest",
    "expected_output_report_body_digest",
    "expected_output_audit_head",
    "expected_reingestion_digest",
)
_ADVANCE_NEXT_DIGEST_NAMES = (
    "expected_next_pending_evidence_digest",
    "expected_next_first_pending_projection_digest",
    "expected_next_bundle_digest",
    "expected_next_plan_digest",
)
_VERIFY_OPTIONAL_DIGEST_NAMES = (
    "expected_callback_start_claim_digest",
    "expected_receipt_digest",
    "expected_journal_head_digest",
    "expected_advance_digest",
    "expected_output_evidence_digest",
    "expected_output_report_body_digest",
    "expected_output_audit_head",
    "expected_reingestion_digest",
    "expected_next_bundle_digest",
    "expected_next_plan_digest",
)


class DuplicateKeyError(ValueError):
    """Raised before the core sees an ambiguous JSON descriptor."""


class _FrozenArgumentParser(argparse.ArgumentParser):
    """Turn every CLI parse failure into the frozen one-line error path."""

    def error(self, message: str) -> None:
        raise ValueError(f"invalid command line: {message}")


def _load_campaign_core():
    from performance import structural_hypothesis_recursive_campaign

    return structural_hypothesis_recursive_campaign


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateKeyError(
                f"source descriptor contains duplicate key: {key}"
            )
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str):
    raise ValueError(f"source descriptor contains non-finite value: {value}")


def _require_absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path


def _require_source_file(path: Path, label: str) -> Path:
    _require_absolute(path, label)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an existing non-symlink file")
    return path


def _require_digest(value: Any, label: str) -> str:
    if type(value) is not str or not _DIGEST_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_execution_environment() -> None:
    mismatches = {
        key: os.environ.get(key)
        for key, required in REQUIRED_EXECUTION_ENVIRONMENT.items()
        if os.environ.get(key) != required
    }
    if mismatches:
        rendered = ", ".join(
            f"{key}={value!r} (required '1')"
            for key, value in sorted(mismatches.items())
        )
        raise ValueError(
            "offline/single-thread runtime environment is not pinned before "
            f"Python startup: {rendered}"
        )


def _load_source_descriptor(path: Path) -> dict[str, Any]:
    _require_source_file(path, "--source-descriptor")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("source descriptor must be UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError("source descriptor must contain one JSON object")
    return value


def _add_base(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-descriptor", type=Path, required=True)
    parser.add_argument(
        "--campaign-contract", type=Path, default=DEFAULT_CAMPAIGN_CONTRACT
    )
    parser.add_argument("--campaign-root", type=Path, required=True)


def _add_digest_options(
    parser: argparse.ArgumentParser,
    names: tuple[str, ...],
    *,
    required: bool,
) -> None:
    for name in names:
        parser.add_argument(
            "--" + name.replace("_", "-"), required=required
        )


def _parser() -> argparse.ArgumentParser:
    parser = _FrozenArgumentParser(
        description=(
            "Drive at most one explicitly authorized callback start and one "
            "immutable recursive structural-hypothesis advance."
        )
    )
    actions = parser.add_subparsers(dest="action", required=True)

    inspect = actions.add_parser(
        "inspect", help="inspect a source and proposed campaign without writes"
    )
    _add_base(inspect)
    inspect.add_argument("--campaign-id", required=True)
    inspect.add_argument("--next-attempt-root", type=Path)

    authorize = actions.add_parser(
        "authorize", help="publish one explicit one-task authorization"
    )
    _add_base(authorize)
    authorize.add_argument("--campaign-id", required=True)
    _add_digest_options(authorize, _AUTHORIZE_DIGEST_NAMES, required=True)
    authorize.add_argument("--task-id", required=True)
    authorize.add_argument("--authorization-id", required=True)
    authorize.add_argument(
        "--confirm-explicit-local-task-authorization",
        action="store_true",
        required=True,
    )

    execute = actions.add_parser(
        "execute", help="consume the single callback-start lease"
    )
    _add_base(execute)
    execute.add_argument(
        "--runtime-contract", type=Path, default=DEFAULT_RUNTIME_CONTRACT
    )
    _add_digest_options(execute, _EXECUTE_DIGEST_NAMES, required=True)
    execute.add_argument(
        "--confirm-real-local-execution",
        action="store_true",
        required=True,
    )

    advance = actions.add_parser(
        "advance", help="publish exactly one completed recursive transition"
    )
    _add_base(advance)
    advance.add_argument("--next-attempt-root", type=Path)
    _add_digest_options(advance, _ADVANCE_DIGEST_NAMES, required=True)
    _add_digest_options(
        advance, _ADVANCE_NEXT_DIGEST_NAMES, required=False
    )
    advance.add_argument(
        "--confirm-immutable-one-step-advance",
        action="store_true",
        required=True,
    )

    verify = actions.add_parser(
        "verify", help="verify a campaign capsule without writes"
    )
    _add_base(verify)
    verify.add_argument("--next-attempt-root", type=Path)
    _add_digest_options(
        verify,
        ("expected_campaign_digest", "expected_lease_digest"),
        required=True,
    )
    _add_digest_options(
        verify, _VERIFY_OPTIONAL_DIGEST_NAMES, required=False
    )
    return parser


def _validate_paths(args: argparse.Namespace) -> None:
    _require_source_file(args.source_descriptor, "--source-descriptor")
    _require_source_file(args.campaign_contract, "--campaign-contract")
    _require_absolute(args.campaign_root, "--campaign-root")
    if args.action == "inspect":
        if args.campaign_root.exists() or args.campaign_root.is_symlink():
            raise ValueError(
                "fresh campaign root is required; refusing to overwrite"
            )
    elif args.action == "authorize":
        if args.campaign_root.is_symlink() or (
            args.campaign_root.exists() and not args.campaign_root.is_dir()
        ):
            raise ValueError(
                "campaign root recovery target must be absent or an existing "
                "non-symlink directory"
            )
    elif args.campaign_root.is_symlink() or not args.campaign_root.is_dir():
        raise ValueError(
            "campaign root must be an existing non-symlink directory"
        )

    if args.action == "execute":
        _require_source_file(args.runtime_contract, "--runtime-contract")
    if getattr(args, "next_attempt_root", None) is not None:
        _require_absolute(args.next_attempt_root, "--next-attempt-root")


def _validate_values(args: argparse.Namespace) -> None:
    campaign_id = getattr(args, "campaign_id", None)
    if campaign_id is not None:
        if type(campaign_id) is not str or not _ID_PATTERN.fullmatch(
            campaign_id
        ):
            raise ValueError("--campaign-id must be a non-path local label")
        if args.campaign_root.name != campaign_id:
            raise ValueError(
                "--campaign-id must equal --campaign-root basename"
            )
    task_id = getattr(args, "task_id", None)
    if task_id is not None and not _TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(
            "--task-id must be an exact task:<24-hex> identifier"
        )
    authorization_id = getattr(args, "authorization_id", None)
    if authorization_id is not None and (
        type(authorization_id) is not str
        or not _AUTHORIZATION_ID_PATTERN.fullmatch(authorization_id)
    ):
        raise ValueError(
            "--authorization-id must be the exact recursive-campaign-v1 ID"
        )

    names: tuple[str, ...]
    if args.action == "authorize":
        names = _AUTHORIZE_DIGEST_NAMES
    elif args.action == "execute":
        names = _EXECUTE_DIGEST_NAMES
    elif args.action == "advance":
        next_values = (
            args.next_attempt_root,
            *(getattr(args, name) for name in _ADVANCE_NEXT_DIGEST_NAMES),
        )
        if any(value is not None for value in next_values) and not all(
            value is not None for value in next_values
        ):
            raise ValueError(
                "nonterminal advance requires --next-attempt-root and all "
                "four next-round digests; terminal advance requires none"
            )
        names = (
            *_ADVANCE_DIGEST_NAMES,
            *tuple(
                name
                for name in _ADVANCE_NEXT_DIGEST_NAMES
                if getattr(args, name) is not None
            ),
        )
    elif args.action == "verify":
        supplied_receipt = args.expected_receipt_digest is not None
        supplied_journal = args.expected_journal_head_digest is not None
        if supplied_receipt != supplied_journal:
            raise ValueError(
                "verification receipt and journal anchors must be supplied "
                "together or both omitted"
            )
        names = (
            "expected_campaign_digest",
            "expected_lease_digest",
            *tuple(
                name
                for name in _VERIFY_OPTIONAL_DIGEST_NAMES
                if getattr(args, name) is not None
            ),
        )
    else:
        names = ()
    for name in names:
        _require_digest(
            getattr(args, name), "--" + name.replace("_", "-")
        )


def _validated_result(payload: Any, args: argparse.Namespace) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("recursive campaign core returned a non-object")
    expected_keys = _RESULT_KEYS_BY_ACTION[args.action]
    if set(payload) != set(expected_keys):
        raise ValueError(
            "recursive campaign result keys differ from frozen V1"
        )
    status = payload.get("status")
    if type(status) is not str:
        raise ValueError("recursive campaign result has invalid status")
    if args.action in _STATUS_BY_ACTION:
        if status != _STATUS_BY_ACTION[args.action]:
            raise ValueError("recursive campaign status differs from frozen V1")
    elif args.action == "advance":
        if status not in _ADVANCE_STATUSES:
            raise ValueError("recursive campaign advance status is invalid")
    else:
        if not status.startswith("VERIFIED_") or (
            status.removeprefix("VERIFIED_") not in _VERIFYABLE_STATUSES
        ):
            raise ValueError("recursive campaign verification status is invalid")

    if args.action in {"advance", "verify"}:
        base_status = status.removeprefix("VERIFIED_")
        if payload.get("phase") != _PHASE_BY_STATUS.get(base_status):
            raise ValueError(
                "recursive campaign result phase differs from status"
            )

    if args.action != "inspect":
        root = payload.get("campaign_root")
        if type(root) is not str or Path(root) != args.campaign_root.resolve():
            raise ValueError("recursive campaign result root differs")
    for key, value in payload.items():
        if key.endswith("_digest") or key.endswith("_audit_head"):
            if value is not None:
                _require_digest(value, f"result {key}")

    if args.action in {"inspect", "authorize"}:
        if payload["source_kind"] not in {
            "recursive_successor_v1",
            "recursive_campaign_v1",
        }:
            raise ValueError("recursive campaign result source_kind differs")
        if (
            type(payload["task_count"]) is not int
            or not 1 <= payload["task_count"] <= 30
            or type(payload["task_id"]) is not str
            or not _TASK_ID_PATTERN.fullmatch(payload["task_id"])
            or type(payload["provenance_binding"]) is not dict
            or type(payload["required_authorization_id"]) is not str
            or not _AUTHORIZATION_ID_PATTERN.fullmatch(
                payload["required_authorization_id"]
            )
            or payload["terminal_status"] != "NONTERMINAL"
        ):
            raise ValueError(
                "recursive campaign inspection result is malformed"
            )
        attempt = payload["next_attempt_root"]
        checkpoint = payload["checkpoint_root"]
        if (
            type(attempt) is not str
            or not Path(attempt).is_absolute()
            or checkpoint != str(Path(attempt) / "checkpoints")
        ):
            raise ValueError(
                "recursive campaign result attempt binding differs"
            )
    if args.action == "authorize" and (
        payload["authorization_status"] != "AUTHORIZED"
        or payload["execution_status"] != "NOT_EXECUTED"
    ):
        raise ValueError(
            "recursive campaign authorization result state differs"
        )
    if args.action == "execute" and (
        type(payload["task_id"]) is not str
        or not _TASK_ID_PATTERN.fullmatch(payload["task_id"])
        or payload["execution_status"] not in _EXECUTE_COMPLETION_STATUSES
    ):
        raise ValueError("recursive campaign execution result state differs")

    if args.action in {"advance", "verify"}:
        _validate_uniform_phase_result(payload)
    # Re-serialize now so stdout can never fail after an authorized mutation.
    json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {key: payload[key] for key in expected_keys}


def _validate_uniform_phase_result(payload: dict[str, Any]) -> None:
    phase = payload["phase"]
    execution_status = payload["execution_status"]
    failed_completion = (
        phase == "CALLBACK_COMPLETED"
        and type(execution_status) is str
        and execution_status in _FAILED_COMPLETION_STATUSES
    )
    always = (
        "campaign_digest",
        "lease_digest",
        "provenance_binding_digest",
        "authorization_digest",
        "attempt_digest",
    )
    if any(payload[key] is None for key in always):
        raise ValueError("recursive campaign phase lacks a base anchor")
    if (
        type(execution_status) is not str
        or not execution_status
    ):
        raise ValueError("recursive campaign phase state is malformed")
    exact_phase_execution = {
        "AUTHORIZED": "NOT_EXECUTED",
        "CALLBACK_INCOMPLETE": "CALLBACK_CLAIMED_INCOMPLETE_NO_REENTRY",
        "ADVANCED_NONTERMINAL": "COMPLETED_AND_ADVANCED_HARD_STOP",
        "ADVANCED_TERMINAL": "COMPLETED_AND_ADVANCED_HARD_STOP",
    }
    if (
        phase in exact_phase_execution
        and execution_status != exact_phase_execution[phase]
    ):
        raise ValueError("recursive campaign phase execution status differs")
    if (
        type(payload["remaining_task_count"]) is not int
        or not 0 <= payload["remaining_task_count"] <= 30
        or payload["terminal_status"] not in {"NONTERMINAL", "TERMINAL"}
    ):
        raise ValueError("recursive campaign completion state is malformed")
    if phase in {"AUTHORIZED", "CALLBACK_INCOMPLETE"} or failed_completion:
        if (
            payload["remaining_task_count"] < 1
            or payload["terminal_status"] != "NONTERMINAL"
            or payload["next_attempt_root"] is not None
        ):
            raise ValueError("unpreviewed campaign boundary differs")

    claim = payload["callback_start_claim_digest"]
    receipt_fields = ("receipt_digest", "journal_head_digest")
    advance_fields = (
        "reingestion_digest",
        "output_evidence_digest",
        "output_report_body_digest",
        "output_audit_head",
    )
    next_fields = (
        "next_pending_evidence_digest",
        "next_first_pending_projection_digest",
        "next_bundle_digest",
        "next_plan_digest",
    )
    if phase == "AUTHORIZED":
        should_be_none = (
            "callback_start_claim_digest",
            *receipt_fields,
            "advance_digest",
            *advance_fields,
            *next_fields,
        )
    elif phase == "CALLBACK_INCOMPLETE":
        if claim is None:
            raise ValueError("incomplete phase lacks callback-start claim")
        should_be_none = (
            *receipt_fields,
            "advance_digest",
            *advance_fields,
            *next_fields,
        )
    elif phase == "CALLBACK_COMPLETED":
        if claim is None or any(payload[key] is None for key in receipt_fields):
            raise ValueError("completed phase lacks callback receipt anchors")
        if payload["execution_status"] in _FAILED_COMPLETION_STATUSES:
            should_be_none = (
                "advance_digest",
                *advance_fields,
                *next_fields,
            )
        elif payload["execution_status"] in _SUCCESS_PREVIEW_STATUSES:
            should_be_none = ("advance_digest",)
            if any(payload[key] is None for key in advance_fields):
                raise ValueError("completed preview lacks advance anchors")
        else:
            raise ValueError("completed phase execution status is invalid")
    else:
        if (
            claim is None
            or any(payload[key] is None for key in receipt_fields)
            or payload["advance_digest"] is None
            or any(payload[key] is None for key in advance_fields)
        ):
            raise ValueError("advanced phase lacks committed anchors")
        should_be_none = ()
    if any(payload[key] is not None for key in should_be_none):
        raise ValueError("recursive campaign phase exposes a future anchor")

    terminal = phase == "ADVANCED_TERMINAL" or (
        phase == "CALLBACK_COMPLETED"
        and not failed_completion
        and payload["terminal_status"] == "TERMINAL"
    )
    if terminal:
        if (
            payload["remaining_task_count"] != 0
            or payload["terminal_status"] != "TERMINAL"
            or payload["next_attempt_root"] is not None
            or any(payload[key] is not None for key in next_fields)
        ):
            raise ValueError("terminal recursive campaign boundary differs")
    elif failed_completion:
        if payload["next_attempt_root"] is not None:
            raise ValueError("failed completion boundary differs")
    elif phase in {"CALLBACK_COMPLETED", "ADVANCED_NONTERMINAL"}:
        attempt = payload["next_attempt_root"]
        if (
            payload["remaining_task_count"] < 1
            or payload["terminal_status"] != "NONTERMINAL"
            or type(attempt) is not str
            or not Path(attempt).is_absolute()
            or any(payload[key] is None for key in next_fields)
        ):
            raise ValueError(
                "nonterminal recursive campaign boundary differs"
            )
    elif phase not in {"AUTHORIZED", "CALLBACK_INCOMPLETE"}:
        raise ValueError("recursive campaign phase boundary is invalid")


def _optional_kwargs(
    args: argparse.Namespace, names: tuple[str, ...]
) -> dict[str, str]:
    return {
        name: getattr(args, name)
        for name in names
        if getattr(args, name) is not None
    }


def _invoke(
    core: Any, source: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    base = (source, args.campaign_contract, args.campaign_root)
    if args.action == "inspect":
        return core.inspect_recursive_campaign(
            *base,
            campaign_id=args.campaign_id,
            next_attempt_root=args.next_attempt_root,
        )
    if args.action == "authorize":
        return core.authorize_recursive_campaign_task(
            *base,
            campaign_id=args.campaign_id,
            **{name: getattr(args, name) for name in _AUTHORIZE_DIGEST_NAMES},
            task_id=args.task_id,
            authorization_id=args.authorization_id,
            confirm_explicit_local_task_authorization=True,
        )
    if args.action == "execute":
        return core.execute_recursive_campaign_task(
            *base,
            args.runtime_contract,
            **{name: getattr(args, name) for name in _EXECUTE_DIGEST_NAMES},
            confirm_real_local_execution=True,
        )
    if args.action == "advance":
        return core.advance_recursive_campaign(
            *base,
            args.next_attempt_root,
            **{name: getattr(args, name) for name in _ADVANCE_DIGEST_NAMES},
            **{
                name: getattr(args, name)
                for name in _ADVANCE_NEXT_DIGEST_NAMES
            },
            confirm_immutable_one_step_advance=True,
        )
    return core.verify_recursive_campaign(
        *base,
        expected_campaign_digest=args.expected_campaign_digest,
        expected_lease_digest=args.expected_lease_digest,
        **(
            {"next_attempt_root": args.next_attempt_root}
            if args.next_attempt_root is not None
            else {}
        ),
        **_optional_kwargs(args, _VERIFY_OPTIONAL_DIGEST_NAMES),
    )


def _write_summary(payload: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _human_summary(payload: dict[str, Any], action: str) -> str:
    parts = [
        "structural_hypothesis_recursive_campaign",
        f"action={action}",
        f"status={payload['status']}",
    ]
    for key in (
        "campaign_digest",
        "lease_digest",
        "task_id",
        "pending_evidence_count",
    ):
        if key in payload:
            parts.append(f"{key}={payload[key]}")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.action in {"authorize", "execute"}:
            # Runtime preparation and execution import the numerical stack.
            # Thread-pool variables must be fixed before importing the core.
            _require_execution_environment()
        _validate_paths(args)
        _validate_values(args)
        source = _load_source_descriptor(args.source_descriptor)
        payload = _invoke(_load_campaign_core(), source, args)
        summary = _validated_result(payload, args)
        _write_summary(summary)
        print(_human_summary(summary, args.action), file=sys.stderr)
        return 0
    except (
        DuplicateKeyError,
        ImportError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        print(
            "invalid structural-hypothesis recursive campaign: " + str(exc),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
