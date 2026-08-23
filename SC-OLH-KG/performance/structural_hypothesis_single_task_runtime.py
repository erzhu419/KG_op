"""Fail-closed local runtime for exactly one structural-hypothesis task.

Preparation strongly rebuilds the materialized bundle, authorizes only its
first pending task, and creates an immutable local attempt.  Execution is the
only operation that lazily imports the native benchmark ``run_one`` callable.
It never submits work, opens a network connection, reads a credential, or
reingests evidence.  A RUNNING attempt is deliberately not auto-resumable.

The commitments in this module are local integrity checks, not signatures or
external authority.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

from .structural_hypothesis_execution import (
    authorize_plan as _CAPTURED_AUTHORIZE_PLAN,
    execute_authorized_plan as _CAPTURED_EXECUTE_AUTHORIZED_PLAN,
    verify_authorization_integrity as _CAPTURED_VERIFY_AUTHORIZATION,
    verify_receipt_integrity as _CAPTURED_VERIFY_RECEIPT,
)
from .structural_hypothesis_loop import canonical_json_bytes
from .structural_hypothesis_task_materializer import (
    load_strict_json_file,
    validate_materializer_contract,
    verify_materialized_task_bundle as _CAPTURED_VERIFY_BUNDLE,
)


RUNTIME_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-single-task-runtime/1"
)
ATTEMPT_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-single-task-attempt/1"
)
JOURNAL_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-single-task-journal-event/1"
)
RAW_RESULT_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-native-result/1"
)
PREFLIGHT_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-single-task-preflight/1"
)

_RUNTIME_CONTRACT_ID = "structural_hypothesis_single_task_runtime_v1"
_RUNTIME_CONTRACT_DIGEST = (
    "sha256:d03529c64e6ea63b9997ded35fb6c0b44c6e17fb828f9e9db8960adb764a8c6b"
)
_HYPOTHESIS_CONTRACT_DIGEST = (
    "sha256:4242f6af8424acca5c93136f0d4eb354f8c2203431f1c5145290c4a3f248cf26"
)
_EXECUTOR_CONTRACT_DIGEST = (
    "sha256:ede48b8b1fb0bb788f91a3834d5a41f336e55b331183922237176aec12624030"
)
_MATERIALIZER_CONTRACT_DIGEST = (
    "sha256:30c65d77e6cbdbc13b95e9083604f6f99835b0982d52319b99d0040491c1d013"
)
_REQUIRED_ENVIRONMENT = {
    "SCOLHKG_OFFLINE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
_SOURCE_FILES = {
    "performance/benchmark_lodo_meta_prior.py": (
        "a7d531cb79525966890904c5b4e7117709fddded012638c8bdd4cc8a33e825a0"
    ),
    "algorithms/single_olhkg.py": (
        "0ad688a94a8388615ca544ab50fb7d49fd49e76a7cab3ccc05e3897a80ddcc62"
    ),
    "performance/aggregate_completed_matrix.py": (
        "cdf0111c10a18a0cf647fc1e6100f6149427a9ec206e66eeb079c755ab6ed2df"
    ),
    "performance/structural_hypothesis_execution.py": (
        "7c5cc27f8e97da9b51e57975f63e860a23463d5e7728d1089f32978146f27c9b"
    ),
}
_ALLOWED_DOMAINS = {
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
}
_ALLOWED_SEEDS = set(range(10))
_LAYOUT = {
    "attempt_binding": "attempt.json",
    "bundle_snapshot": "bundle.json",
    "authorization": "authorization.json",
    "input_directory": "inputs",
    "report_snapshot": "inputs/report.json",
    "hypothesis_contract_snapshot": "inputs/hypothesis_contract.json",
    "executor_contract_snapshot": "inputs/executor_contract.json",
    "materializer_contract_snapshot": "inputs/materializer_contract.json",
    "journal_directory": "journal",
    "authorized_event": "journal/0000_AUTHORIZED.json",
    "running_event": "journal/0001_RUNNING.json",
    "completed_event": "journal/0002_COMPLETED.json",
    "preflight": "preflight.json",
    "raw_result": "raw_result.json",
    "receipt": "receipt.json",
    "checkpoint_root": "checkpoints",
}
_ATTEMPT_NONCLAIMS = [
    "local_authorization_is_not_external_authority",
    "local_digest_is_not_signature",
    "authorized_is_not_executed",
    "execution_is_not_reingestion",
    "checkpoint_subtree_is_not_receipt_evidence",
    "checkpoint_subtree_is_not_resume_authority",
    "no_exactly_once_execution_claim",
]


class SingleTaskRuntimeValidationError(ValueError):
    """Raised when a single-task runtime input or state fails closed."""


class ResultPersistenceRejected(RuntimeError):
    """Marks a returned callback value that could not be durably recorded."""


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_clone(value: Any, label: str = "value") -> Any:
    def check(item: Any, path: str) -> None:
        if item is None or type(item) in (str, int, bool):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise SingleTaskRuntimeValidationError(
                    f"{path} must be finite"
                )
            return
        if type(item) is list:
            for index, child in enumerate(item):
                check(child, f"{path}[{index}]")
            return
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise SingleTaskRuntimeValidationError(
                    f"{path} keys must be strings"
                )
            for key, child in item.items():
                check(child, f"{path}.{key}")
            return
        raise SingleTaskRuntimeValidationError(
            f"{path} is not native JSON"
        )

    check(value, label)
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _require_digest(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(ch not in "0123456789abcdef" for ch in value[7:])
    ):
        raise SingleTaskRuntimeValidationError(
            f"{label} must be a lowercase sha256 digest"
        )
    return value


def validate_runtime_contract(contract: Mapping[str, Any]) -> None:
    if type(contract) is not dict:
        raise SingleTaskRuntimeValidationError(
            "runtime contract must be an exact object"
        )
    if contract.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise SingleTaskRuntimeValidationError(
            "unsupported runtime contract schema_version"
        )
    if contract.get("contract_id") != _RUNTIME_CONTRACT_ID:
        raise SingleTaskRuntimeValidationError("runtime contract id differs")
    source = contract.get("source_contracts")
    if source != {
        "hypothesis_contract_id": "structural_hypothesis_loop_v1",
        "hypothesis_contract_digest": _HYPOTHESIS_CONTRACT_DIGEST,
        "executor_contract_id": "structural_hypothesis_executor_v1",
        "executor_contract_digest": _EXECUTOR_CONTRACT_DIGEST,
        "materializer_contract_id": "structural_hypothesis_task_materializer_v1",
        "materializer_contract_digest": _MATERIALIZER_CONTRACT_DIGEST,
    }:
        raise SingleTaskRuntimeValidationError(
            "runtime source-contract binding differs"
        )
    binding = contract.get("runtime_binding")
    if not isinstance(binding, Mapping) or binding.get("source_files") != _SOURCE_FILES:
        raise SingleTaskRuntimeValidationError(
            "runtime source-file binding differs"
        )
    if (
        binding.get("executor_module")
        != "performance.benchmark_lodo_meta_prior"
        or binding.get("executor_callable") != "run_one"
        or binding.get("executor_callable_source_sha256")
        != "01c38f1978fb80df1d83c67db62ea1535c5812f0fadb78b4b32bcf45467c12ae"
        or binding.get("executor_callable_code_sha256")
        != "bea51ae2b210fbc9e55600e988c8b97bf652e27c930cd945ca593d851226ac97"
        or binding.get("executor_callable_firstlineno") != 686
        or binding.get("task_keys") != ["args", "heldout", "line", "seed"]
    ):
        raise SingleTaskRuntimeValidationError(
            "runtime executor binding differs"
        )
    if contract.get("required_environment") != _REQUIRED_ENVIRONMENT:
        raise SingleTaskRuntimeValidationError(
            "runtime environment binding differs"
        )
    if contract.get("artifact_layout") != _LAYOUT:
        raise SingleTaskRuntimeValidationError(
            "runtime artifact layout differs"
        )
    if _digest(contract) != _RUNTIME_CONTRACT_DIGEST:
        raise SingleTaskRuntimeValidationError(
            "runtime contract digest differs from frozen V1"
        )


def _validate_environment() -> None:
    mismatches = {
        key: os.environ.get(key)
        for key, expected in _REQUIRED_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    if mismatches:
        raise SingleTaskRuntimeValidationError(
            "offline and single-thread environment is not frozen V1"
        )


def _validate_source_files(contract: Mapping[str, Any]) -> None:
    validate_runtime_contract(contract)
    root = Path(__file__).resolve().parents[1]
    for relative, expected in _SOURCE_FILES.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise SingleTaskRuntimeValidationError(
                f"runtime source is missing or aliased: {relative}"
            )
        if _raw_sha256(path) != expected:
            raise SingleTaskRuntimeValidationError(
                f"runtime source SHA-256 differs: {relative}"
            )


def _path(value: str | Path, label: str) -> Path:
    try:
        path = Path(value)
    except TypeError as error:
        raise SingleTaskRuntimeValidationError(
            f"{label} is not a path"
        ) from error
    if not path.is_absolute() or ".." in path.parts:
        raise SingleTaskRuntimeValidationError(
            f"{label} must be an absolute canonical path"
        )
    if path != path.resolve(strict=False):
        raise SingleTaskRuntimeValidationError(
            f"{label} contains an alias or symlink component"
        )
    return path


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise SingleTaskRuntimeValidationError(
                f"symlink path component is forbidden: {current}"
            )
        if not current.exists():
            break


def _state_base_and_prefix(contract: Mapping[str, Any]) -> tuple[Path, Path]:
    policy = contract["attempt_root_policy"]
    if policy.get("kind") != "xdg-state-home-relative-v1":
        raise SingleTaskRuntimeValidationError(
            "unsupported attempt-root policy"
        )
    configured = os.environ.get("XDG_STATE_HOME", "")
    if configured:
        base = Path(configured)
        if not base.is_absolute() or ".." in base.parts:
            raise SingleTaskRuntimeValidationError(
                "XDG_STATE_HOME must be an absolute canonical path"
            )
    else:
        base = Path.home() / ".local" / "state"
    if base != base.resolve(strict=False):
        raise SingleTaskRuntimeValidationError(
            "state home contains an alias or symlink component"
        )
    relative = Path(policy["relative_prefix"])
    if relative.is_absolute() or ".." in relative.parts:
        raise SingleTaskRuntimeValidationError(
            "runtime state prefix is unsafe"
        )
    prefix = base / relative
    _reject_symlink_components(prefix)
    return base, prefix


def _secure_directory(path: Path, *, exact_mode: int | None = None) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise SingleTaskRuntimeValidationError(
            f"cannot inspect runtime directory: {path}"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SingleTaskRuntimeValidationError(
            f"runtime path is not a real directory: {path}"
        )
    if info.st_uid != os.geteuid():
        raise SingleTaskRuntimeValidationError(
            f"runtime directory has a different owner: {path}"
        )
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o022:
        raise SingleTaskRuntimeValidationError(
            f"runtime directory is group/world writable: {path}"
        )
    if exact_mode is not None and mode != exact_mode:
        raise SingleTaskRuntimeValidationError(
            f"runtime directory mode must be {exact_mode:04o}: {path}"
        )


def _ensure_secure_tree(base: Path, target: Path) -> None:
    if target != base and base not in target.parents:
        raise SingleTaskRuntimeValidationError(
            "runtime directory escapes state home"
        )
    _reject_symlink_components(target)
    missing = []
    cursor = target
    while not cursor.exists():
        if cursor.is_symlink():
            raise SingleTaskRuntimeValidationError(
                f"runtime directory alias is forbidden: {cursor}"
            )
        missing.append(cursor)
        if cursor == base:
            break
        cursor = cursor.parent
    cursor.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in reversed(missing):
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        os.chmod(path, 0o700, follow_symlinks=False)
    for path in (base, *[p for p in target.parents if base in p.parents], target):
        if path.exists() and (path == base or base in path.parents):
            _secure_directory(path)


def _validate_attempt_location(
    attempt_root: str | Path,
    contract: Mapping[str, Any],
    *,
    fresh: bool,
) -> Path:
    path = _path(attempt_root, "attempt_root")
    base, prefix = _state_base_and_prefix(contract)
    if prefix not in path.parents:
        raise SingleTaskRuntimeValidationError(
            "attempt_root is outside the frozen local state prefix"
        )
    if fresh:
        if path.exists() or path.is_symlink():
            raise SingleTaskRuntimeValidationError(
                "attempt_root already exists"
            )
        _ensure_secure_tree(base, path.parent)
    else:
        _reject_symlink_components(path)
        _secure_directory(path, exact_mode=0o700)
    return path


def _mkdir_new(path: Path, mode: int = 0o700) -> None:
    try:
        os.mkdir(path, mode)
        os.chmod(path, mode, follow_symlinks=False)
    except OSError as error:
        raise SingleTaskRuntimeValidationError(
            f"cannot create fresh runtime directory: {path}"
        ) from error
    _secure_directory(path, exact_mode=mode)


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one complete immutable JSON artifact without replacement."""
    rendered = (
        json.dumps(
            _json_clone(dict(payload), str(path)),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists() or path.is_symlink():
        raise SingleTaskRuntimeValidationError(
            f"runtime artifact already exists: {path.name}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_owned_json(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as error:
        raise SingleTaskRuntimeValidationError(
            f"missing runtime artifact: {path}"
        ) from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 1
    ):
        raise SingleTaskRuntimeValidationError(
            f"runtime artifact ownership or mode differs: {path}"
        )
    try:
        return load_strict_json_file(path)
    except ValueError as error:
        raise SingleTaskRuntimeValidationError(
            f"runtime artifact JSON is invalid: {path}"
        ) from error


def _lexists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _directory_names(path: Path) -> set[str]:
    _secure_directory(path, exact_mode=0o700)
    try:
        return {item.name for item in path.iterdir()}
    except OSError as error:
        raise SingleTaskRuntimeValidationError(
            f"cannot enumerate runtime directory: {path}"
        ) from error


def _validate_checkpoint_files(path: Path, *, must_be_empty: bool) -> None:
    names = _directory_names(path)
    if must_be_empty and names:
        raise SingleTaskRuntimeValidationError(
            "authorized task checkpoint directory is not empty"
        )
    for name in names:
        if not (
            name == "checkpoint_latest.pkl"
            or (
                name.startswith("checkpoint_stage_")
                and name.endswith(".pkl")
                and len(name[len("checkpoint_stage_"):-len(".pkl")]) == 5
                and name[len("checkpoint_stage_"):-len(".pkl")].isdigit()
            )
        ):
            raise SingleTaskRuntimeValidationError(
                f"unexpected checkpoint artifact: {name}"
            )
        candidate = path / name
        info = candidate.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise SingleTaskRuntimeValidationError(
                f"checkpoint artifact ownership or mode differs: {name}"
            )


def _checkpoint_task_components(
    task: Mapping[str, Any],
) -> tuple[str, int]:
    run_task = task.get("run_one_task")
    if type(run_task) is not dict:
        raise SingleTaskRuntimeValidationError(
            "first task run_one binding is malformed"
        )
    domain = run_task.get("heldout")
    seed = run_task.get("seed")
    if type(domain) is not str or domain not in _ALLOWED_DOMAINS:
        raise SingleTaskRuntimeValidationError(
            "first task checkpoint domain is outside frozen V1"
        )
    if type(seed) is not int or seed not in _ALLOWED_SEEDS:
        raise SingleTaskRuntimeValidationError(
            "first task checkpoint seed is outside frozen V1"
        )
    return domain, seed


def _validate_layout(
    root: Path,
    task: Mapping[str, Any],
    *,
    state: str,
    has_raw_result: bool = False,
    has_receipt: bool = False,
) -> None:
    _validate_fixed_directory_tree(root, task)
    inputs = root / _LAYOUT["input_directory"]
    journal = root / _LAYOUT["journal_directory"]
    checkpoints = root / _LAYOUT["checkpoint_root"]
    domain, seed = _checkpoint_task_components(task)
    domain_dir = checkpoints / domain
    task_dir = domain_dir / f"seed{seed}"
    root_names = {
        "attempt.json",
        "bundle.json",
        "authorization.json",
        "inputs",
        "journal",
        "checkpoints",
    }
    journal_names = {"0000_AUTHORIZED.json"}
    if state in {"PREFLIGHT", "RUNNING", "COMPLETED"}:
        root_names.add("preflight.json")
    if state in {"RUNNING", "COMPLETED"}:
        journal_names.add("0001_RUNNING.json")
    if state == "COMPLETED":
        journal_names.add("0002_COMPLETED.json")
    if has_raw_result:
        root_names.add("raw_result.json")
    if has_receipt:
        root_names.add("receipt.json")
    if _directory_names(root) != root_names:
        raise SingleTaskRuntimeValidationError(
            "attempt root contains missing or unexpected artifacts"
        )
    if _directory_names(journal) != journal_names:
        raise SingleTaskRuntimeValidationError(
            "attempt journal layout differs"
        )
    _validate_checkpoint_files(
        task_dir,
        must_be_empty=state in {"AUTHORIZED", "PREFLIGHT"},
    )


def _validate_fixed_directory_tree(
    root: Path, task: Mapping[str, Any]
) -> None:
    inputs = root / _LAYOUT["input_directory"]
    journal = root / _LAYOUT["journal_directory"]
    checkpoints = root / _LAYOUT["checkpoint_root"]
    domain, seed = _checkpoint_task_components(task)
    domain_dir = checkpoints / domain
    task_dir = domain_dir / f"seed{seed}"
    for directory in (root, inputs, journal, checkpoints, domain_dir, task_dir):
        _secure_directory(directory, exact_mode=0o700)
    if _directory_names(inputs) != {
        "report.json",
        "hypothesis_contract.json",
        "executor_contract.json",
        "materializer_contract.json",
    }:
        raise SingleTaskRuntimeValidationError(
            "attempt input snapshot layout differs"
        )
    if _directory_names(checkpoints) != {domain}:
        raise SingleTaskRuntimeValidationError(
            "checkpoint root layout differs"
        )
    if _directory_names(domain_dir) != {f"seed{seed}"}:
        raise SingleTaskRuntimeValidationError(
            "checkpoint domain layout differs"
        )


def _validate_bound_contracts(
    hypothesis_contract: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
    materializer_contract: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> None:
    validate_runtime_contract(runtime_contract)
    validate_materializer_contract(materializer_contract)
    if _digest(hypothesis_contract) != _HYPOTHESIS_CONTRACT_DIGEST:
        raise SingleTaskRuntimeValidationError(
            "hypothesis contract digest differs"
        )
    if _digest(executor_contract) != _EXECUTOR_CONTRACT_DIGEST:
        raise SingleTaskRuntimeValidationError(
            "executor contract digest differs"
        )
    if _digest(materializer_contract) != _MATERIALIZER_CONTRACT_DIGEST:
        raise SingleTaskRuntimeValidationError(
            "materializer contract digest differs"
        )
    _validate_source_files(runtime_contract)


def _validate_bundle(
    report: Mapping[str, Any],
    bundle: Mapping[str, Any],
    hypothesis_contract: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
    materializer_contract: Mapping[str, Any],
    base_manifest_path: str | Path,
    asset_root: str | Path,
    checkpoint_root: Path,
) -> None:
    if type(bundle) is not dict or not _CAPTURED_VERIFY_BUNDLE(
        bundle,
        report,
        hypothesis_contract,
        executor_contract,
        materializer_contract,
        base_manifest_path,
        asset_root,
        checkpoint_root,
    ):
        raise SingleTaskRuntimeValidationError(
            "materialized task bundle failed strong verification"
        )


def _selected_task(bundle: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    plan = bundle.get("plan")
    tasks = plan.get("tasks") if isinstance(plan, Mapping) else None
    if type(tasks) is not list or not tasks or type(tasks[0]) is not dict:
        raise SingleTaskRuntimeValidationError(
            "bundle has no first pending task"
        )
    first = tasks[0]
    if type(task_id) is not str or task_id != first.get("task_id"):
        raise SingleTaskRuntimeValidationError(
            "task_id must exactly name plan.tasks[0]"
        )
    if first.get("ordinal") != 0 or first.get("status") != "READY_FOR_AUTHORIZATION":
        raise SingleTaskRuntimeValidationError(
            "first pending task is not ordinal-zero ready mechanics"
        )
    return _json_clone(first, "first pending task")


def _validate_task_runtime(task: Mapping[str, Any], checkpoint_root: Path) -> Path:
    run_task = task.get("run_one_task")
    domain, seed = _checkpoint_task_components(task)
    args = run_task.get("args") if isinstance(run_task, Mapping) else None
    if type(args) is not dict:
        raise SingleTaskRuntimeValidationError("first task args are malformed")
    expected = {
        "offline_only": True,
        "llm_prior_enabled": False,
        "jobs": 1,
        "exact_kg_jobs": 12,
        "exact_kg_parallel_backend": "process_fork",
        "runtime_checkpoint_resume": True,
        "runtime_checkpoint_interval": 1,
    }
    if any(type(args.get(key)) is not type(value) or args.get(key) != value for key, value in expected.items()):
        raise SingleTaskRuntimeValidationError(
            "first task runtime settings differ from frozen V1"
        )
    if set(run_task) != {"args", "heldout", "line", "seed"}:
        raise SingleTaskRuntimeValidationError("run_one task ABI differs")
    expected_checkpoint = (
        checkpoint_root
        / domain
        / f"seed{seed}"
    )
    if args.get("runtime_checkpoint_dir") != str(expected_checkpoint):
        raise SingleTaskRuntimeValidationError(
            "task checkpoint directory differs from attempt binding"
        )
    return expected_checkpoint


def _event(
    *,
    sequence: int,
    state: str,
    previous_event_digest: str | None,
    attempt_digest: str,
    authorization_digest: str,
    task_id: str,
    task_digest: str,
    preflight_digest: str | None = None,
    raw_result_digest: str | None = None,
    receipt_digest: str | None = None,
    runtime_error_code: str | None = None,
) -> dict[str, Any]:
    body = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "sequence": sequence,
        "state": state,
        "previous_event_digest": previous_event_digest,
        "attempt_digest": attempt_digest,
        "authorization_digest": authorization_digest,
        "task_id": task_id,
        "task_digest": task_digest,
        "preflight_digest": preflight_digest,
        "raw_result_digest": raw_result_digest,
        "receipt_digest": receipt_digest,
        "runtime_error_code": runtime_error_code,
    }
    return {
        **body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "event_digest": _digest(body),
        },
    }


def _attempt_binding(
    *,
    bundle: Mapping[str, Any],
    task: Mapping[str, Any],
    authorization: Mapping[str, Any],
    base_manifest_path: Path,
    asset_root: Path,
    checkpoint_root: Path,
    state_root: Path,
) -> dict[str, Any]:
    body = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": "attempt:" + _digest({
            "bundle_digest": bundle["integrity"]["bundle_digest"],
            "task_digest": task["task_digest"],
            "authorization_digest": authorization["integrity"][
                "authorization_digest"
            ],
        }).split(":", 1)[1][:24],
        "runtime_contract_id": _RUNTIME_CONTRACT_ID,
        "runtime_contract_digest": _RUNTIME_CONTRACT_DIGEST,
        "bundle_binding": {
            "bundle_id": bundle["bundle_id"],
            "bundle_digest": bundle["integrity"]["bundle_digest"],
        },
        "plan_binding": {
            "plan_id": bundle["plan"]["plan_id"],
            "plan_digest": bundle["plan"]["integrity"]["plan_digest"],
        },
        "task_binding": {
            "task_id": task["task_id"],
            "task_digest": task["task_digest"],
            "ordinal": 0,
            "cell": _json_clone(task["cell"]),
        },
        "authorization_binding": {
            "authorization_id": authorization["authorization_id"],
            "authorization_digest": authorization["integrity"][
                "authorization_digest"
            ],
        },
        "local_input_paths": {
            "base_manifest": str(base_manifest_path),
            "asset_root": str(asset_root),
            "checkpoint_root": str(checkpoint_root),
            "state_root": str(state_root),
        },
        "artifact_layout": dict(_LAYOUT),
        "nonclaims": list(_ATTEMPT_NONCLAIMS),
    }
    return {
        **body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "attempt_digest": _digest(body),
        },
    }


def prepare_single_task_attempt(
    report: Mapping[str, Any],
    bundle: Mapping[str, Any],
    hypothesis_contract: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
    materializer_contract: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    base_manifest_path: str | Path,
    asset_root: str | Path,
    attempt_root: str | Path,
    *,
    task_id: str,
    expected_bundle_digest: str,
    expected_plan_digest: str,
    authorization_id: str,
) -> dict[str, Any]:
    """Authorize exactly ``plan.tasks[0]`` and create a fresh local attempt."""
    _validate_environment()
    _validate_bound_contracts(
        hypothesis_contract,
        executor_contract,
        materializer_contract,
        runtime_contract,
    )
    expected_bundle_digest = _require_digest(
        expected_bundle_digest, "expected_bundle_digest"
    )
    expected_plan_digest = _require_digest(
        expected_plan_digest, "expected_plan_digest"
    )
    root = _validate_attempt_location(attempt_root, runtime_contract, fresh=True)
    _state_base, state_root = _state_base_and_prefix(runtime_contract)
    checkpoint_root = root / _LAYOUT["checkpoint_root"]
    base_path = _path(base_manifest_path, "base_manifest_path")
    asset_path = _path(asset_root, "asset_root")
    if base_path.is_symlink() or not base_path.is_file():
        raise SingleTaskRuntimeValidationError(
            "base manifest must be a non-aliased file"
        )
    if asset_path.is_symlink() or not asset_path.is_dir():
        raise SingleTaskRuntimeValidationError(
            "asset root must be a non-aliased directory"
        )
    _validate_bundle(
        report,
        bundle,
        hypothesis_contract,
        executor_contract,
        materializer_contract,
        base_path,
        asset_path,
        checkpoint_root,
    )
    if bundle.get("status") != "MATERIALIZED_NOT_AUTHORIZED":
        raise SingleTaskRuntimeValidationError("bundle status differs")
    if bundle.get("integrity", {}).get("bundle_digest") != expected_bundle_digest:
        raise SingleTaskRuntimeValidationError("expected bundle digest differs")
    plan = bundle["plan"]
    if plan.get("integrity", {}).get("plan_digest") != expected_plan_digest:
        raise SingleTaskRuntimeValidationError("expected plan digest differs")
    task = _selected_task(bundle, task_id)
    checkpoint_task_dir = _validate_task_runtime(task, checkpoint_root)
    authorization = _CAPTURED_AUTHORIZE_PLAN(
        plan,
        hypothesis_contract,
        report,
        executor_contract,
        expected_plan_digest=expected_plan_digest,
        authorization_id=authorization_id,
        authorized_task_ids=[task_id],
    )
    if not _CAPTURED_VERIFY_AUTHORIZATION(authorization):
        raise SingleTaskRuntimeValidationError(
            "generated authorization failed verification"
        )
    attempt = _attempt_binding(
        bundle=bundle,
        task=task,
        authorization=authorization,
        base_manifest_path=base_path,
        asset_root=asset_path,
        checkpoint_root=checkpoint_root,
        state_root=state_root,
    )
    attempt_digest = attempt["integrity"]["attempt_digest"]
    authorization_digest = authorization["integrity"][
        "authorization_digest"
    ]
    authorized_event = _event(
        sequence=0,
        state="AUTHORIZED",
        previous_event_digest=None,
        attempt_digest=attempt_digest,
        authorization_digest=authorization_digest,
        task_id=task["task_id"],
        task_digest=task["task_digest"],
    )

    _mkdir_new(root)
    _mkdir_new(root / _LAYOUT["input_directory"])
    _mkdir_new(root / _LAYOUT["journal_directory"])
    _mkdir_new(checkpoint_root)
    _mkdir_new(checkpoint_root / str(task["run_one_task"]["heldout"]))
    _mkdir_new(checkpoint_task_dir)
    if any(checkpoint_task_dir.iterdir()):
        raise SingleTaskRuntimeValidationError(
            "fresh task checkpoint directory is not empty"
        )

    snapshots = {
        _LAYOUT["attempt_binding"]: attempt,
        _LAYOUT["bundle_snapshot"]: bundle,
        _LAYOUT["authorization"]: authorization,
        _LAYOUT["report_snapshot"]: report,
        _LAYOUT["hypothesis_contract_snapshot"]: hypothesis_contract,
        _LAYOUT["executor_contract_snapshot"]: executor_contract,
        _LAYOUT["materializer_contract_snapshot"]: materializer_contract,
    }
    for relative, payload in snapshots.items():
        _write_new_json(root / relative, payload)
    _write_new_json(root / _LAYOUT["authorized_event"], authorized_event)
    return {
        "status": "AUTHORIZED",
        "task_id": task_id,
        "authorization_digest": authorization_digest,
        "bundle_digest": expected_bundle_digest,
        "plan_digest": expected_plan_digest,
        "attempt_digest": attempt_digest,
    }


def _expected_attempt(
    root: Path,
    runtime_contract: Mapping[str, Any],
    base_manifest_path: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    attempt = _load_owned_json(root / _LAYOUT["attempt_binding"])
    bundle = _load_owned_json(root / _LAYOUT["bundle_snapshot"])
    authorization = _load_owned_json(root / _LAYOUT["authorization"])
    task_id = attempt.get("task_binding", {}).get("task_id")
    task = _selected_task(bundle, task_id)
    # Validate every directory component before following a path to an input
    # snapshot.  This prevents replacing inputs/journal/checkpoints with a
    # same-user symlink after preparation.
    _validate_fixed_directory_tree(root, task)
    report = _load_owned_json(root / _LAYOUT["report_snapshot"])
    hypothesis = _load_owned_json(
        root / _LAYOUT["hypothesis_contract_snapshot"]
    )
    executor_contract = _load_owned_json(
        root / _LAYOUT["executor_contract_snapshot"]
    )
    materializer = _load_owned_json(
        root / _LAYOUT["materializer_contract_snapshot"]
    )
    _validate_bound_contracts(
        hypothesis, executor_contract, materializer, runtime_contract
    )
    bound_paths = attempt.get("local_input_paths")
    if type(bound_paths) is not dict:
        raise SingleTaskRuntimeValidationError(
            "attempt local input binding is malformed"
        )
    bound_base = _path(bound_paths.get("base_manifest"), "bound base manifest")
    bound_assets = _path(bound_paths.get("asset_root"), "bound asset root")
    bound_state_root = _path(
        bound_paths.get("state_root"), "bound state root"
    )
    _current_state_base, current_state_root = _state_base_and_prefix(
        runtime_contract
    )
    if bound_state_root != current_state_root or bound_state_root not in root.parents:
        raise SingleTaskRuntimeValidationError(
            "resolved state root differs from prepared attempt binding"
        )
    checkpoint_root = root / _LAYOUT["checkpoint_root"]
    if bound_paths.get("checkpoint_root") != str(checkpoint_root):
        raise SingleTaskRuntimeValidationError(
            "attempt checkpoint root differs"
        )
    if base_manifest_path is not None and _path(
        base_manifest_path, "base_manifest_path"
    ) != bound_base:
        raise SingleTaskRuntimeValidationError(
            "base manifest path differs from prepared attempt"
        )
    if asset_root is not None and _path(asset_root, "asset_root") != bound_assets:
        raise SingleTaskRuntimeValidationError(
            "asset root path differs from prepared attempt"
        )
    _validate_bundle(
        report,
        bundle,
        hypothesis,
        executor_contract,
        materializer,
        bound_base,
        bound_assets,
        checkpoint_root,
    )
    checkpoint_task_dir = _validate_task_runtime(task, checkpoint_root)
    expected_authorization = _CAPTURED_AUTHORIZE_PLAN(
        bundle["plan"],
        hypothesis,
        report,
        executor_contract,
        expected_plan_digest=bundle["plan"]["integrity"]["plan_digest"],
        authorization_id=authorization.get("authorization_id"),
        authorized_task_ids=[task_id],
    )
    if authorization != expected_authorization:
        raise SingleTaskRuntimeValidationError(
            "saved authorization differs from exact first-task authorization"
        )
    expected_attempt = _attempt_binding(
        bundle=bundle,
        task=task,
        authorization=authorization,
        base_manifest_path=bound_base,
        asset_root=bound_assets,
        checkpoint_root=checkpoint_root,
        state_root=bound_state_root,
    )
    if attempt != expected_attempt:
        raise SingleTaskRuntimeValidationError(
            "saved attempt binding differs"
        )
    return {
        "attempt": attempt,
        "bundle": bundle,
        "authorization": authorization,
        "report": report,
        "hypothesis_contract": hypothesis,
        "executor_contract": executor_contract,
        "materializer_contract": materializer,
        "task": task,
        "checkpoint_task_dir": checkpoint_task_dir,
    }


def _load_event(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = _load_owned_json(path)
    if observed != expected:
        raise SingleTaskRuntimeValidationError(
            f"journal event differs: {path.name}"
        )
    return observed


def _raw_result_envelope(
    raw_result: Any,
    *,
    task: Mapping[str, Any],
    authorization_digest: str,
) -> dict[str, Any]:
    native = _json_clone(
        _native_result_json_value(raw_result, "native run_one result"),
        "native run_one result",
    )
    body = {
        "schema_version": RAW_RESULT_SCHEMA_VERSION,
        "task_id": task["task_id"],
        "task_digest": task["task_digest"],
        "authorization_digest": authorization_digest,
        "native_result": native,
        "native_result_digest": _digest(native),
    }
    return {
        **body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "raw_result_digest": _digest(body),
        },
    }


def _receipt_runtime_error_code(receipt: Mapping[str, Any]) -> str | None:
    results = receipt.get("results")
    if type(results) is not list or len(results) != 1:
        raise SingleTaskRuntimeValidationError(
            "single-task receipt result count differs"
        )
    result = results[0]
    if type(result) is not dict:
        raise SingleTaskRuntimeValidationError(
            "single-task receipt result is malformed"
        )
    if result.get("status") == "SUCCEEDED":
        return None
    error = result.get("error")
    if type(error) is not dict:
        raise SingleTaskRuntimeValidationError(
            "failed single-task receipt lacks an error"
        )
    if error.get("type") == "ResultPersistenceRejected":
        return "RESULT_PERSISTENCE_REJECTED"
    code = error.get("code")
    if code not in {"EXECUTOR_EXCEPTION", "RESULT_REJECTED"}:
        raise SingleTaskRuntimeValidationError(
            "single-task receipt error code differs"
        )
    return code


def _native_result_json_value(value: Any, path: str) -> Any:
    """Convert only the native NumPy/container types emitted by run_one."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise SingleTaskRuntimeValidationError(
                f"{path} has a non-string mapping key"
            )
        return {
            key: _native_result_json_value(child, f"{path}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _native_result_json_value(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    try:
        import numpy as np
    except ImportError as error:  # benchmark import itself requires NumPy
        raise SingleTaskRuntimeValidationError(
            "NumPy is unavailable during native result conversion"
        ) from error
    if isinstance(value, np.ndarray):
        return _native_result_json_value(value.tolist(), path)
    if isinstance(value, np.generic):
        return _native_result_json_value(value.item(), path)
    raise SingleTaskRuntimeValidationError(
        f"{path} contains unsupported native type {type(value).__name__}"
    )


def _callable_binding(run_one: Any) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    expected_file = root / "performance/benchmark_lodo_meta_prior.py"
    code = getattr(run_one, "__code__", None)
    try:
        source = inspect.getsource(run_one).encode("utf-8")
        source_sha256 = hashlib.sha256(source).hexdigest()
        filename = Path(code.co_filename).resolve(strict=True)
        firstlineno = int(code.co_firstlineno)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise SingleTaskRuntimeValidationError(
            "native run_one has no verifiable Python source binding"
        ) from error
    observed = {
        "module": getattr(run_one, "__module__", None),
        "callable": getattr(run_one, "__name__", None),
        "source_file": str(filename),
        "source_sha256": source_sha256,
        "code_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "firstlineno": firstlineno,
    }
    if observed != {
        "module": "performance.benchmark_lodo_meta_prior",
        "callable": "run_one",
        "source_file": str(expected_file),
        "source_sha256": (
            "01c38f1978fb80df1d83c67db62ea1535c5812f0fadb78b4b32bcf45467c12ae"
        ),
        "code_sha256": (
            "bea51ae2b210fbc9e55600e988c8b97bf652e27c930cd945ca593d851226ac97"
        ),
        "firstlineno": 686,
    }:
        raise SingleTaskRuntimeValidationError(
            "native run_one callable source binding differs"
        )
    return observed


def _load_real_executor(runtime_contract: Mapping[str, Any]):
    from performance.benchmark_lodo_meta_prior import run_one

    if not callable(run_one):
        raise SingleTaskRuntimeValidationError(
            "native run_one callable binding differs"
        )
    # Recheck the pinned files after import, then bind the actual callable to
    # the exact function definition rather than trusting module/name labels.
    _validate_source_files(runtime_contract)
    _callable_binding(run_one)
    return run_one


def _memory_available_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(
            encoding="ascii"
        ).splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) * 1024
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise SingleTaskRuntimeValidationError(
            "cannot read local MemAvailable preflight"
        ) from error
    raise SingleTaskRuntimeValidationError(
        "local MemAvailable preflight is unavailable"
    )


def _fork_probe(process_count: int) -> int:
    if not hasattr(os, "fork"):
        raise SingleTaskRuntimeValidationError(
            "local runtime has no os.fork"
        )
    if type(process_count) is not int or process_count != 12:
        raise SingleTaskRuntimeValidationError(
            "fork probe must use exactly 12 processes"
        )
    children = []
    fork_error = None
    for _index in range(process_count):
        try:
            child = os.fork()
        except OSError as error:
            fork_error = error
            break
        if child == 0:  # pragma: no cover - child exits without test machinery
            os._exit(0)
        children.append(child)
    statuses = []
    for child in children:
        try:
            statuses.append(os.waitpid(child, 0))
        except OSError as error:
            fork_error = fork_error or error
    if fork_error is not None or len(children) != process_count or any(
        waited != child
        or not os.WIFEXITED(status_code)
        or os.WEXITSTATUS(status_code) != 0
        for child, (waited, status_code) in zip(children, statuses)
    ):
        raise SingleTaskRuntimeValidationError(
            "12-process local fork probe failed"
        ) from fork_error
    return len(children)


def _run_preflight(
    *,
    task: Mapping[str, Any],
    checkpoint_dir: Path,
    run_one: Any,
    runtime_contract: Mapping[str, Any],
    attempt_digest: str,
    authorization_digest: str,
) -> dict[str, Any]:
    requirements = runtime_contract["preflight"]
    if requirements != {
        "affinity_cpu_minimum": 12,
        "memory_available_minimum_bytes": 12884901888,
        "checkpoint_free_minimum_bytes": 2147483648,
        "fork_probe_required": True,
        "fork_probe_process_count": 12,
        "threadpool_info_required": True,
        "threadpool_num_threads": 1,
    }:
        raise SingleTaskRuntimeValidationError(
            "runtime preflight contract differs"
        )
    try:
        affinity = sorted(int(item) for item in os.sched_getaffinity(0))
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise SingleTaskRuntimeValidationError(
            "cannot inspect local CPU affinity"
        ) from error
    memory_available = _memory_available_bytes()
    filesystem = os.statvfs(checkpoint_dir)
    disk_free = int(filesystem.f_bavail) * int(filesystem.f_frsize)
    if len(affinity) < requirements["affinity_cpu_minimum"]:
        raise SingleTaskRuntimeValidationError(
            "local CPU affinity is below the 12-process task contract"
        )
    if memory_available < requirements["memory_available_minimum_bytes"]:
        raise SingleTaskRuntimeValidationError(
            "local MemAvailable is below the 12 GiB runtime gate"
        )
    if disk_free < requirements["checkpoint_free_minimum_bytes"]:
        raise SingleTaskRuntimeValidationError(
            "checkpoint filesystem has less than 2 GiB free"
        )
    try:
        from threadpoolctl import threadpool_info

        raw_pools = threadpool_info()
    except (ImportError, RuntimeError) as error:
        raise SingleTaskRuntimeValidationError(
            "cannot inspect native thread pools"
        ) from error
    if type(raw_pools) is not list or not raw_pools:
        raise SingleTaskRuntimeValidationError(
            "native thread-pool inventory is empty"
        )
    pools = []
    for item in raw_pools:
        if not isinstance(item, Mapping):
            raise SingleTaskRuntimeValidationError(
                "native thread-pool inventory is malformed"
            )
        threads = item.get("num_threads")
        if type(threads) is not int or threads != 1:
            raise SingleTaskRuntimeValidationError(
                "native thread pool is not frozen to one thread"
            )
        pools.append({
            "user_api": item.get("user_api"),
            "internal_api": item.get("internal_api"),
            "prefix": item.get("prefix"),
            "version": item.get("version"),
            "threading_layer": item.get("threading_layer"),
            "num_threads": threads,
        })
    fork_count = _fork_probe(requirements["fork_probe_process_count"])
    callable_binding = _callable_binding(run_one)
    body = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "PASSED_LOCAL_PREFLIGHT",
        "attempt_digest": attempt_digest,
        "authorization_digest": authorization_digest,
        "task_id": task["task_id"],
        "task_digest": task["task_digest"],
        "requirements": _json_clone(requirements),
        "observed": {
            "affinity_cpu_ids": affinity,
            "affinity_cpu_count": len(affinity),
            "memory_available_bytes": memory_available,
            "checkpoint_free_bytes": disk_free,
            "fork_probe_passed": True,
            "fork_probe_process_count": fork_count,
            "thread_pools": pools,
            "executor_callable": callable_binding,
            "required_environment": dict(_REQUIRED_ENVIRONMENT),
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
            "preflight_digest": _digest(body),
        },
    }


def execute_single_task_attempt(
    attempt_root: str | Path,
    runtime_contract: Mapping[str, Any],
    *,
    expected_authorization_digest: str,
) -> dict[str, Any]:
    """Cross the real callback boundary once for a fresh AUTHORIZED attempt."""
    _validate_environment()
    validate_runtime_contract(runtime_contract)
    _validate_source_files(runtime_contract)
    expected_authorization_digest = _require_digest(
        expected_authorization_digest, "expected_authorization_digest"
    )
    root = _validate_attempt_location(
        attempt_root, runtime_contract, fresh=False
    )
    chain = _expected_attempt(root, runtime_contract)
    attempt = chain["attempt"]
    task = chain["task"]
    authorization = chain["authorization"]
    authorization_digest = authorization["integrity"][
        "authorization_digest"
    ]
    _validate_layout(root, task, state="AUTHORIZED")
    if authorization_digest != expected_authorization_digest:
        raise SingleTaskRuntimeValidationError(
            "expected authorization digest differs"
        )
    attempt_digest = attempt["integrity"]["attempt_digest"]
    authorized = _event(
        sequence=0,
        state="AUTHORIZED",
        previous_event_digest=None,
        attempt_digest=attempt_digest,
        authorization_digest=authorization_digest,
        task_id=task["task_id"],
        task_digest=task["task_digest"],
    )
    _load_event(root / _LAYOUT["authorized_event"], authorized)
    running_path = root / _LAYOUT["running_event"]
    completed_path = root / _LAYOUT["completed_event"]
    if running_path.exists() or running_path.is_symlink():
        raise SingleTaskRuntimeValidationError(
            "RUNNING attempt cannot be re-entered"
        )
    if completed_path.exists() or completed_path.is_symlink():
        raise SingleTaskRuntimeValidationError(
            "COMPLETED attempt cannot be executed again"
        )
    checkpoint_dir = chain["checkpoint_task_dir"]
    _secure_directory(checkpoint_dir, exact_mode=0o700)
    if any(checkpoint_dir.iterdir()):
        raise SingleTaskRuntimeValidationError(
            "fresh task checkpoint directory is not empty"
        )
    for relative in (
        _LAYOUT["preflight"],
        _LAYOUT["raw_result"],
        _LAYOUT["receipt"],
    ):
        path = root / relative
        if path.exists() or path.is_symlink():
            raise SingleTaskRuntimeValidationError(
                f"preexisting execution artifact is forbidden: {relative}"
            )
    # Import, callable binding, thread-pool inspection, fork probing and local
    # capacity checks all happen before RUNNING is durably published.  Thus an
    # import or preflight failure remains safely AUTHORIZED, not uncertain.
    real_executor = _load_real_executor(runtime_contract)
    preflight = _run_preflight(
        task=task,
        checkpoint_dir=checkpoint_dir,
        run_one=real_executor,
        runtime_contract=runtime_contract,
        attempt_digest=attempt_digest,
        authorization_digest=authorization_digest,
    )
    preflight_digest = preflight["integrity"]["preflight_digest"]
    _write_new_json(root / _LAYOUT["preflight"], preflight)
    running = _event(
        sequence=1,
        state="RUNNING",
        previous_event_digest=authorized["integrity"]["event_digest"],
        attempt_digest=attempt_digest,
        authorization_digest=authorization_digest,
        task_id=task["task_id"],
        task_digest=task["task_digest"],
        preflight_digest=preflight_digest,
    )
    # Immutable publication is the point after which re-entry is forbidden.
    _write_new_json(running_path, running)
    raw_holder: dict[str, Any] = {}
    persistence_failure = False

    def persist_raw(run_task: Mapping[str, Any]) -> Mapping[str, Any]:
        nonlocal persistence_failure
        if _json_clone(run_task, "callback task") != task["run_one_task"]:
            raise SingleTaskRuntimeValidationError(
                "executor callback task differs from authorized snapshot"
            )
        raw = real_executor(_json_clone(run_task, "native callback task"))
        envelope = None
        raw_path = root / _LAYOUT["raw_result"]
        try:
            envelope = _raw_result_envelope(
                raw,
                task=task,
                authorization_digest=authorization_digest,
            )
            _write_new_json(raw_path, envelope)
        except Exception as error:
            persistence_failure = True
            if envelope is not None and _lexists(raw_path):
                try:
                    if _load_owned_json(raw_path) == envelope:
                        raw_holder["envelope"] = envelope
                except SingleTaskRuntimeValidationError:
                    pass
            raise ResultPersistenceRejected from error
        raw_holder["envelope"] = envelope
        return raw

    receipt = _CAPTURED_EXECUTE_AUTHORIZED_PLAN(
        chain["bundle"]["plan"],
        authorization,
        expected_authorization_digest=authorization_digest,
        executor=persist_raw,
        hypothesis_contract=chain["hypothesis_contract"],
        source_report=chain["report"],
        executor_contract=chain["executor_contract"],
    )
    if not _CAPTURED_VERIFY_RECEIPT(
        receipt,
        chain["bundle"]["plan"],
        authorization,
        chain["report"],
        chain["hypothesis_contract"],
        chain["executor_contract"],
    ):
        raise SingleTaskRuntimeValidationError(
            "generated execution receipt failed full-chain verification"
        )
    raw_path = root / _LAYOUT["raw_result"]
    if persistence_failure and _lexists(raw_path) and "envelope" not in raw_holder:
        raise SingleTaskRuntimeValidationError(
            "result persistence failed with an unverified raw artifact"
        )
    runtime_error_code = _receipt_runtime_error_code(receipt)
    if persistence_failure != (
        runtime_error_code == "RESULT_PERSISTENCE_REJECTED"
    ):
        raise SingleTaskRuntimeValidationError(
            "result persistence classification differs from callback boundary"
        )
    _write_new_json(root / _LAYOUT["receipt"], receipt)
    receipt_digest = receipt["integrity"]["receipt_digest"]
    raw_digest = (
        raw_holder.get("envelope", {})
        .get("integrity", {})
        .get("raw_result_digest")
    )
    completed = _event(
        sequence=2,
        state="COMPLETED",
        previous_event_digest=running["integrity"]["event_digest"],
        attempt_digest=attempt_digest,
        authorization_digest=authorization_digest,
        task_id=task["task_id"],
        task_digest=task["task_digest"],
        preflight_digest=preflight_digest,
        raw_result_digest=raw_digest,
        receipt_digest=receipt_digest,
        runtime_error_code=runtime_error_code,
    )
    _write_new_json(completed_path, completed)
    return {
        "status": "EXECUTED_RECEIPT_WRITTEN",
        "authorization_digest": authorization_digest,
        "receipt_digest": receipt_digest,
        "journal_head_digest": completed["integrity"]["event_digest"],
        "attempt_digest": attempt_digest,
    }


def _validate_raw_envelope(
    envelope: Mapping[str, Any],
    task: Mapping[str, Any],
    authorization_digest: str,
) -> dict[str, Any]:
    if type(envelope) is not dict:
        raise SingleTaskRuntimeValidationError(
            "raw result envelope is malformed"
        )
    body = {key: envelope[key] for key in envelope if key != "integrity"}
    integrity = envelope.get("integrity")
    if (
        type(integrity) is not dict
        or integrity != {
            "algorithm": "sha256-canonical-json-v1",
            "raw_result_digest": _digest(body),
        }
        or body.get("schema_version") != RAW_RESULT_SCHEMA_VERSION
        or body.get("task_id") != task["task_id"]
        or body.get("task_digest") != task["task_digest"]
        or body.get("authorization_digest") != authorization_digest
        or body.get("native_result_digest")
        != _digest(body.get("native_result"))
    ):
        raise SingleTaskRuntimeValidationError(
            "raw result envelope integrity failed"
        )
    return _json_clone(body["native_result"], "saved native result")


def _validate_preflight_artifact(
    preflight: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    attempt_digest: str,
    authorization_digest: str,
) -> str:
    if type(preflight) is not dict:
        raise SingleTaskRuntimeValidationError(
            "preflight artifact is malformed"
        )
    body = {key: preflight[key] for key in preflight if key != "integrity"}
    integrity = preflight.get("integrity")
    if (
        type(integrity) is not dict
        or integrity != {
            "algorithm": "sha256-canonical-json-v1",
            "preflight_digest": _digest(body),
        }
        or body.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or body.get("status") != "PASSED_LOCAL_PREFLIGHT"
        or body.get("attempt_digest") != attempt_digest
        or body.get("authorization_digest") != authorization_digest
        or body.get("task_id") != task["task_id"]
        or body.get("task_digest") != task["task_digest"]
        or body.get("requirements") != runtime_contract["preflight"]
    ):
        raise SingleTaskRuntimeValidationError(
            "preflight artifact integrity failed"
        )
    observed = body.get("observed")
    pools = observed.get("thread_pools") if isinstance(observed, Mapping) else None
    if (
        type(observed) is not dict
        or type(observed.get("affinity_cpu_ids")) is not list
        or type(observed.get("affinity_cpu_count")) is not int
        or observed["affinity_cpu_count"]
        != len(observed["affinity_cpu_ids"])
        or observed["affinity_cpu_count"] < 12
        or type(observed.get("memory_available_bytes")) is not int
        or observed["memory_available_bytes"] < 12884901888
        or type(observed.get("checkpoint_free_bytes")) is not int
        or observed["checkpoint_free_bytes"] < 2147483648
        or observed.get("fork_probe_passed") is not True
        or observed.get("fork_probe_process_count") != 12
        or observed.get("required_environment") != _REQUIRED_ENVIRONMENT
        or type(pools) is not list
        or not pools
        or any(
            type(pool) is not dict or pool.get("num_threads") != 1
            for pool in pools
        )
    ):
        raise SingleTaskRuntimeValidationError(
            "preflight observations do not satisfy frozen thresholds"
        )
    binding = runtime_contract["runtime_binding"]
    source_file = (
        Path(__file__).resolve().parents[1]
        / "performance/benchmark_lodo_meta_prior.py"
    )
    callable_binding = observed.get("executor_callable")
    if callable_binding != {
        "module": binding["executor_module"],
        "callable": binding["executor_callable"],
        "source_file": str(source_file),
        "source_sha256": binding["executor_callable_source_sha256"],
        "code_sha256": binding["executor_callable_code_sha256"],
        "firstlineno": binding["executor_callable_firstlineno"],
    }:
        raise SingleTaskRuntimeValidationError(
            "preflight callable binding differs"
        )
    return integrity["preflight_digest"]


def verify_single_task_attempt(
    attempt_root: str | Path,
    runtime_contract: Mapping[str, Any],
    base_manifest_path: str | Path,
    asset_root: str | Path,
    *,
    expected_authorization_digest: str,
    expected_receipt_digest: str | None = None,
    expected_journal_head_digest: str | None = None,
    expected_attempt_digest: str | None = None,
) -> dict[str, Any]:
    """Verify an AUTHORIZED/RUNNING attempt or a fully pinned COMPLETED one."""
    validate_runtime_contract(runtime_contract)
    _validate_source_files(runtime_contract)
    expected_authorization_digest = _require_digest(
        expected_authorization_digest, "expected_authorization_digest"
    )
    root = _validate_attempt_location(
        attempt_root, runtime_contract, fresh=False
    )
    chain = _expected_attempt(
        root,
        runtime_contract,
        base_manifest_path=base_manifest_path,
        asset_root=asset_root,
    )
    attempt = chain["attempt"]
    task = chain["task"]
    authorization = chain["authorization"]
    authorization_digest = authorization["integrity"][
        "authorization_digest"
    ]
    attempt_digest = attempt["integrity"]["attempt_digest"]
    if authorization_digest != expected_authorization_digest:
        raise SingleTaskRuntimeValidationError(
            "expected authorization digest differs"
        )
    authorized = _event(
        sequence=0,
        state="AUTHORIZED",
        previous_event_digest=None,
        attempt_digest=attempt_digest,
        authorization_digest=authorization_digest,
        task_id=task["task_id"],
        task_digest=task["task_digest"],
    )
    _load_event(root / _LAYOUT["authorized_event"], authorized)
    running_path = root / _LAYOUT["running_event"]
    completed_path = root / _LAYOUT["completed_event"]
    preflight_path = root / _LAYOUT["preflight"]
    raw_path = root / _LAYOUT["raw_result"]
    receipt_path = root / _LAYOUT["receipt"]
    has_running = _lexists(running_path)
    has_completed = _lexists(completed_path)
    has_preflight = _lexists(preflight_path)
    has_raw = _lexists(raw_path)
    has_receipt = _lexists(receipt_path)
    if not has_running and not has_completed:
        if has_raw or has_receipt:
            raise SingleTaskRuntimeValidationError(
                "AUTHORIZED attempt has execution artifacts"
            )
        if any(
            value is not None
            for value in (
                expected_receipt_digest,
                expected_journal_head_digest,
                expected_attempt_digest,
            )
        ):
            raise SingleTaskRuntimeValidationError(
                "AUTHORIZED attempt cannot satisfy final expectations"
            )
        preflight_digest = None
        status = "VERIFIED_AUTHORIZED_NOT_EXECUTED"
        if has_preflight:
            preflight_digest = _validate_preflight_artifact(
                _load_owned_json(preflight_path),
                task=task,
                runtime_contract=runtime_contract,
                attempt_digest=attempt_digest,
                authorization_digest=authorization_digest,
            )
            status = "VERIFIED_PREFLIGHT_PASSED_NO_CALLBACK"
        _validate_layout(
            root,
            task,
            state="PREFLIGHT" if has_preflight else "AUTHORIZED",
        )
        result = {
            "status": status,
            "authorization_digest": authorization_digest,
            "attempt_digest": attempt_digest,
        }
        if preflight_digest is not None:
            result["preflight_digest"] = preflight_digest
        return result
    preflight = _load_owned_json(preflight_path)
    preflight_digest = _validate_preflight_artifact(
        preflight,
        task=task,
        runtime_contract=runtime_contract,
        attempt_digest=attempt_digest,
        authorization_digest=authorization_digest,
    )
    running = _event(
        sequence=1,
        state="RUNNING",
        previous_event_digest=authorized["integrity"]["event_digest"],
        attempt_digest=attempt_digest,
        authorization_digest=authorization_digest,
        task_id=task["task_id"],
        task_digest=task["task_digest"],
        preflight_digest=preflight_digest,
    )
    _load_event(running_path, running)
    if not has_completed:
        _validate_layout(
            root,
            task,
            state="RUNNING",
            has_raw_result=has_raw,
            has_receipt=has_receipt,
        )
        return {
            "status": "VERIFIED_RUNNING_INCOMPLETE_NO_REENTRY",
            "authorization_digest": authorization_digest,
            "attempt_digest": attempt_digest,
            "journal_head_digest": running["integrity"]["event_digest"],
        }
    supplied_final = (
        expected_receipt_digest,
        expected_journal_head_digest,
        expected_attempt_digest,
    )
    if any(item is None for item in supplied_final):
        raise SingleTaskRuntimeValidationError(
            "COMPLETED verification requires all independent final digests"
        )
    expected_receipt_digest = _require_digest(
        expected_receipt_digest, "expected_receipt_digest"
    )
    expected_journal_head_digest = _require_digest(
        expected_journal_head_digest, "expected_journal_head_digest"
    )
    expected_attempt_digest = _require_digest(
        expected_attempt_digest, "expected_attempt_digest"
    )
    if not has_receipt:
        raise SingleTaskRuntimeValidationError(
            "COMPLETED attempt lacks its receipt"
        )
    _validate_layout(
        root,
        task,
        state="COMPLETED",
        has_raw_result=has_raw,
        has_receipt=True,
    )
    receipt = _load_owned_json(receipt_path)
    if not _CAPTURED_VERIFY_RECEIPT(
        receipt,
        chain["bundle"]["plan"],
        authorization,
        chain["report"],
        chain["hypothesis_contract"],
        chain["executor_contract"],
    ):
        raise SingleTaskRuntimeValidationError(
            "saved receipt failed full-chain verification"
        )
    receipt_digest = receipt["integrity"]["receipt_digest"]
    runtime_error_code = _receipt_runtime_error_code(receipt)
    raw_digest = None
    if has_raw:
        envelope = _load_owned_json(raw_path)
        native = _validate_raw_envelope(
            envelope, task, authorization_digest
        )
        if runtime_error_code != "RESULT_PERSISTENCE_REJECTED":
            replayed = _CAPTURED_EXECUTE_AUTHORIZED_PLAN(
                chain["bundle"]["plan"],
                authorization,
                expected_authorization_digest=authorization_digest,
                executor=lambda _task: _json_clone(native),
                hypothesis_contract=chain["hypothesis_contract"],
                source_report=chain["report"],
                executor_contract=chain["executor_contract"],
            )
            if replayed != receipt:
                raise SingleTaskRuntimeValidationError(
                    "saved raw result does not reproduce the receipt"
                )
        raw_digest = envelope["integrity"]["raw_result_digest"]
    elif receipt["results"][0].get("error", {}).get("code") != "EXECUTOR_EXCEPTION":
        raise SingleTaskRuntimeValidationError(
            "non-executor-failure receipt lacks its raw result"
        )
    completed = _event(
        sequence=2,
        state="COMPLETED",
        previous_event_digest=running["integrity"]["event_digest"],
        attempt_digest=attempt_digest,
        authorization_digest=authorization_digest,
        task_id=task["task_id"],
        task_digest=task["task_digest"],
        preflight_digest=preflight_digest,
        raw_result_digest=raw_digest,
        receipt_digest=receipt_digest,
        runtime_error_code=runtime_error_code,
    )
    _load_event(completed_path, completed)
    journal_head_digest = completed["integrity"]["event_digest"]
    if (
        receipt_digest != expected_receipt_digest
        or journal_head_digest != expected_journal_head_digest
        or attempt_digest != expected_attempt_digest
    ):
        raise SingleTaskRuntimeValidationError(
            "independent completed-attempt digest differs"
        )
    return {
        "status": "VERIFIED_COMPLETED",
        "authorization_digest": authorization_digest,
        "receipt_digest": receipt_digest,
        "journal_head_digest": journal_head_digest,
        "attempt_digest": attempt_digest,
    }


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "JOURNAL_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "RAW_RESULT_SCHEMA_VERSION",
    "ResultPersistenceRejected",
    "RUNTIME_SCHEMA_VERSION",
    "SingleTaskRuntimeValidationError",
    "execute_single_task_attempt",
    "prepare_single_task_attempt",
    "validate_runtime_contract",
    "verify_single_task_attempt",
]
