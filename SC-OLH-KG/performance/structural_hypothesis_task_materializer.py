"""Read-only V1 materialization of structural-hypothesis ``run_one`` tasks.

This module resolves the current local runner's task ABI, but never imports or
invokes ``benchmark_lodo_meta_prior.run_one``.  Materialization is neither
authorization nor execution.  All paths supplied here are read-only inputs or
strings embedded in the proposed tasks; this module creates no checkpoint or
result path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from baselines.transfer_archive import FrozenTransferArchive
from core.designs import (
    integer_design_fingerprint,
    load_frozen_source_informed_design,
)

from .structural_hypothesis_execution import (
    build_execution_plan,
    validate_executor_contract,
    verify_plan_integrity,
)
from .structural_hypothesis_loop import (
    canonical_json_bytes,
    validate_contract as validate_hypothesis_contract,
)
from . import run_lodo_manifest_shard as _RUNNER_MODULE


_CAPTURED_BUILD_RUN_ONE_TASK = _RUNNER_MODULE.build_run_one_task
_CAPTURED_LOAD_CONFIG = _RUNNER_MODULE.load_config


MATERIALIZER_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-task-materializer/1"
)
TASK_BUNDLE_SCHEMA_VERSION = (
    "sc-olh-kg.structural-hypothesis-materialized-task-bundle/1"
)

_HYPOTHESIS_CONTRACT_ID = "structural_hypothesis_loop_v1"
_HYPOTHESIS_CONTRACT_DIGEST = (
    "sha256:4242f6af8424acca5c93136f0d4eb354f8c2203431f1c5145290c4a3f248cf26"
)
_EXECUTOR_CONTRACT_ID = "structural_hypothesis_executor_v1"
_EXECUTOR_CONTRACT_DIGEST = (
    "sha256:ede48b8b1fb0bb788f91a3834d5a41f336e55b331183922237176aec12624030"
)
_MATERIALIZER_CONTRACT_ID = "structural_hypothesis_task_materializer_v1"
_MATERIALIZER_CONTRACT_DIGEST = (
    "sha256:30c65d77e6cbdbc13b95e9083604f6f99835b0982d52319b99d0040491c1d013"
)
_DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
_DESIGN_KEYS = {
    "design_kind",
    "designs",
    "dimension",
    "heldout_target_domain",
    "n0",
    "n_seeds",
    "schema_version",
    "seed_start",
    "source_archive_fingerprint",
    "source_archive_oracle_aided",
    "target_labels_used",
    "target_oracle_used",
}
_ARCHIVE_KEYS = {
    "schema_version",
    "source_seed",
    "observation_mode",
    "fingerprint",
    "tasks",
}
_NONCLAIMS = (
    "materialization_is_not_authorization",
    "materialization_is_not_execution",
    "materialization_is_not_reingestion",
    "no_exact_historical_runtime_reconstruction",
    "local_recovered_bytes_have_no_external_signature",
    "current_runner_compatible_replay_only",
    "local_digest_is_not_signature",
    "no_external_authority",
    "no_currentness_claim",
    "no_runtime_readiness_claim",
    "no_scientific_claim",
    "no_network_access",
    "no_scheduler_access",
    "no_credential_access",
    "no_shell_execution",
    "core_materializer_does_not_create_checkpoint_or_result_paths",
)


class MaterializationValidationError(ValueError):
    """Raised when a V1 materialization input violates its frozen contract."""


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise MaterializationValidationError(
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
                raise MaterializationValidationError(f"{path} must be finite")
            return
        if type(item) is list:
            for index, child in enumerate(item):
                check(child, f"{path}[{index}]")
            return
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise MaterializationValidationError(
                    f"{path} keys must be strings"
                )
            for key, child in item.items():
                check(child, f"{path}.{key}")
            return
        raise MaterializationValidationError(f"{path} is not native JSON")

    check(value, label)
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _reject_constant(value: str) -> None:
    raise MaterializationValidationError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MaterializationValidationError(
                f"duplicate JSON key is forbidden: {key}"
            )
        result[key] = value
    return result


def load_strict_json_file(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object while rejecting duplicate keys and NaN."""
    input_path = Path(path)
    try:
        raw = input_path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except MaterializationValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationValidationError(
            f"cannot load strict JSON object {input_path}: {type(error).__name__}"
        ) from error
    if type(payload) is not dict:
        raise MaterializationValidationError(
            f"strict JSON root must be an object: {input_path}"
        )
    return payload


def _load_bound_json(
    path: str | Path,
    *,
    expected_raw_sha256: str,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    input_path = Path(path)
    try:
        raw = input_path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except MaterializationValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationValidationError(
            f"cannot load {label}: {type(error).__name__}"
        ) from error
    if type(payload) is not dict:
        raise MaterializationValidationError(f"{label} root must be an object")
    if _raw_sha256(raw) != expected_raw_sha256:
        raise MaterializationValidationError(f"{label} raw SHA-256 differs")
    return payload, raw


def _safe_relative_path(value: Any, label: str) -> Path:
    if type(value) is not str or not value:
        raise MaterializationValidationError(f"{label} must be relative text")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise MaterializationValidationError(f"{label} escapes the asset root")
    return path


def _bound_asset_path(root: Path, relative: Path, label: str) -> Path:
    """Reject missing files and symlink aliases beneath the registered root."""
    candidate = root / relative
    try:
        canonical_root = root.resolve(strict=True)
        canonical_candidate = candidate.resolve(strict=True)
    except OSError as error:
        raise MaterializationValidationError(
            f"cannot resolve {label}: {type(error).__name__}"
        ) from error
    expected = canonical_root / relative
    if (
        canonical_candidate != expected
        or candidate.is_symlink()
        or not canonical_candidate.is_file()
    ):
        raise MaterializationValidationError(
            f"{label} must be a non-aliased file under the asset root"
        )
    return candidate


def validate_materializer_contract(contract: Mapping[str, Any]) -> None:
    """Validate the single frozen V1 downstream materializer contract."""
    if type(contract) is not dict:
        raise MaterializationValidationError(
            "materializer contract must be an exact object"
        )
    cloned = _json_clone(contract, "materializer contract")
    _exact_keys(
        cloned,
        {
            "schema_version",
            "contract_id",
            "source_contracts",
            "materializer_binding",
            "base_manifest",
            "asset_bundle",
            "execution_scope",
            "argv_template",
            "artifact_mechanics",
            "nonclaims",
        },
        "materializer contract",
    )
    if cloned["schema_version"] != MATERIALIZER_SCHEMA_VERSION:
        raise MaterializationValidationError(
            "unsupported materializer schema_version"
        )
    if cloned["contract_id"] != _MATERIALIZER_CONTRACT_ID:
        raise MaterializationValidationError("materializer contract_id differs")
    if cloned["source_contracts"] != {
        "hypothesis_contract_id": _HYPOTHESIS_CONTRACT_ID,
        "hypothesis_contract_digest": _HYPOTHESIS_CONTRACT_DIGEST,
        "executor_contract_id": _EXECUTOR_CONTRACT_ID,
        "executor_contract_digest": _EXECUTOR_CONTRACT_DIGEST,
    }:
        raise MaterializationValidationError("source contract pins differ")
    if cloned["materializer_binding"] != {
        "module": "performance.run_lodo_manifest_shard",
        "callable": "build_run_one_task",
        "runner_relative_path": "performance/run_lodo_manifest_shard.py",
        "runner_raw_sha256": (
            "a2113a7b2e2b27a8e4f4e894def33c852ab922a2563616e1a2de461920124fa6"
        ),
        "invokes_run_one": False,
        "definition_time_callable_capture": True,
    }:
        raise MaterializationValidationError("materializer binding differs")
    if cloned["base_manifest"] != {
        "filename": "v18b_exactkg_mcdiag.json",
        "raw_sha256": (
            "0ce0cb5d5e254453719c85224d11fe292764b0a60268367ddcc22cb82e714dc2"
        ),
        "canonical_json_digest": (
            "sha256:54699cb648472ca414c4465068d175c37bfac76c8314dd207796218b8d45e16e"
        ),
        "config_canonical_json_digest": (
            "sha256:2d71510a81d7bac514acc7ce5d1529e2c1a041cccc4b85916cb584ad6f5a97f0"
        ),
        "resolved_config_key_count": 393,
        "resolved_config_canonical_json_digest": (
            "sha256:3d91e4a6a33a1d95767d1407d2af2bd4b977537bad76e905bbaeb936dde59c68"
        ),
    }:
        raise MaterializationValidationError("base manifest binding differs")
    scope = cloned["execution_scope"]
    if scope != {
        "profile": "full",
        "line": "lodo",
        "domains": list(_DOMAINS),
        "seeds": list(range(10)),
        "d": 50,
        "N": 20,
        "n0": 10,
        "source_calls": 384,
    }:
        raise MaterializationValidationError("materializer execution scope differs")
    mechanics = cloned["artifact_mechanics"]
    if mechanics != {
        "status": "MATERIALIZED_NOT_AUTHORIZED",
        "embedded_plan_status": "READY_FOR_AUTHORIZATION",
        "task_selection": "exact_source_report_pending_cells_in_order",
        "task_count_minimum": 1,
        "task_count_maximum": 30,
        "base_resolved_config_key_count": 393,
        "task_args_key_count": 438,
        "normalized_task_args_digest_by_domain": {
            "FactorShockStatePolicyRZDT1": (
                "sha256:dd61a43a0e5885b6ee30fd1397021b01604418ab13d58aa396f079acbced50b2"
            ),
            "InventorySupplyChain": (
                "sha256:cce23c0eb15304a5c5280c18975f2e45f1260d10a185863b0d9a472ef93ec82b"
            ),
            "QueueResourceControl": (
                "sha256:7018a17d788d6ee12fed0408102bb379c19f1da642d7439bfab46a929450c106"
            ),
        },
        "runtime_checkpoint_interval": 1,
        "reads_local_files_only": True,
        "core_materializer_writes_task_or_checkpoint_files": False,
        "performs_authorization": False,
        "performs_execution": False,
        "performs_reingestion": False,
    }:
        raise MaterializationValidationError("artifact mechanics differ")
    if cloned["nonclaims"] != list(_NONCLAIMS):
        raise MaterializationValidationError("materializer nonclaims differ")
    argv = cloned["argv_template"]
    if type(argv) is not list or any(type(item) is not str for item in argv):
        raise MaterializationValidationError("argv_template must be text tokens")
    placeholders = {
        item for item in argv if item.startswith("{") or item.endswith("}")
    }
    if placeholders != {
        "{base_manifest}",
        "{domain}",
        "{seed}",
        "{materialization_only_out}",
        "{checkpoint_dir}",
        "{design_file}",
    } or any(argv.count(item) != 1 for item in placeholders):
        raise MaterializationValidationError("argv_template placeholders differ")
    required_sequences = (
        ("--runtime-checkpoint-interval", "1"),
        ("--structural-prior-profile", "full"),
        ("--hvd-profile", "factor_cumulative"),
        ("--decision-backend", "risk_ts"),
        ("--initial-design", "source_informed"),
    )
    if any(
        not any(argv[index:index + 2] == list(sequence)
                for index in range(len(argv) - 1))
        for sequence in required_sequences
    ):
        raise MaterializationValidationError("argv_template fixed flags differ")
    bundle = cloned["asset_bundle"]
    if type(bundle) is not dict or set(bundle) != {
        "bundle_id", "design_schema", "domains",
    }:
        raise MaterializationValidationError("asset bundle contract is malformed")
    if bundle["bundle_id"] != (
        "transfer_source_informed_official_n20_s20_20260716"
    ):
        raise MaterializationValidationError("asset bundle id differs")
    if type(bundle["domains"]) is not dict or tuple(bundle["domains"]) != _DOMAINS:
        raise MaterializationValidationError("asset domain order differs")
    for domain, spec in bundle["domains"].items():
        if type(spec) is not dict:
            raise MaterializationValidationError(
                f"asset spec for {domain} is malformed"
            )
        _safe_relative_path(spec.get("design_relative_path"), "design path")
        _safe_relative_path(
            spec.get("companion_relative_path"), "companion path"
        )
    if _digest(cloned) != _MATERIALIZER_CONTRACT_DIGEST:
        raise MaterializationValidationError(
            "materializer contract digest differs from frozen V1"
        )


def _validate_source_contracts(
    hypothesis_contract: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
    materializer_contract: Mapping[str, Any],
) -> None:
    validate_hypothesis_contract(hypothesis_contract)
    if (
        hypothesis_contract.get("contract_id") != _HYPOTHESIS_CONTRACT_ID
        or _digest(hypothesis_contract) != _HYPOTHESIS_CONTRACT_DIGEST
    ):
        raise MaterializationValidationError(
            "hypothesis contract is not frozen V1"
        )
    validate_executor_contract(executor_contract)
    if (
        executor_contract.get("contract_id") != _EXECUTOR_CONTRACT_ID
        or _digest(executor_contract) != _EXECUTOR_CONTRACT_DIGEST
    ):
        raise MaterializationValidationError("executor contract is not frozen V1")
    validate_materializer_contract(materializer_contract)


def _validate_base_manifest(
    path: str | Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    binding = contract["base_manifest"]
    payload, _ = _load_bound_json(
        path,
        expected_raw_sha256=binding["raw_sha256"],
        label="base manifest",
    )
    _exact_keys(
        payload,
        {"config", "heldout", "line", "schema_version"},
        "base manifest",
    )
    if type(payload["config"]) is not dict:
        raise MaterializationValidationError("base manifest config is malformed")
    if _digest(payload) != binding["canonical_json_digest"]:
        raise MaterializationValidationError(
            "base manifest canonical digest differs"
        )
    if _digest(payload["config"]) != binding["config_canonical_json_digest"]:
        raise MaterializationValidationError(
            "base manifest config digest differs"
        )
    try:
        resolved_config = _CAPTURED_LOAD_CONFIG(path)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise MaterializationValidationError(
            "base manifest default resolution failed"
        ) from error
    if (
        type(resolved_config) is not dict
        or len(resolved_config) != binding["resolved_config_key_count"]
        or _digest(resolved_config)
        != binding["resolved_config_canonical_json_digest"]
    ):
        raise MaterializationValidationError(
            "base manifest resolved config differs"
        )
    return {
        "raw_sha256": binding["raw_sha256"],
        "canonical_json_digest": binding["canonical_json_digest"],
        "config_canonical_json_digest": binding[
            "config_canonical_json_digest"
        ],
        "resolved_config_key_count": binding["resolved_config_key_count"],
        "resolved_config_canonical_json_digest": binding[
            "resolved_config_canonical_json_digest"
        ],
    }


def _validate_runner_source(contract: Mapping[str, Any]) -> dict[str, Any]:
    binding = contract["materializer_binding"]
    runner_path = Path(__file__).with_name("run_lodo_manifest_shard.py")
    try:
        raw = runner_path.read_bytes()
    except OSError as error:
        raise MaterializationValidationError(
            f"cannot read runner source: {type(error).__name__}"
        ) from error
    if _raw_sha256(raw) != binding["runner_raw_sha256"]:
        raise MaterializationValidationError("runner source raw SHA-256 differs")
    if (
        _RUNNER_MODULE.build_run_one_task is not _CAPTURED_BUILD_RUN_ONE_TASK
        or _RUNNER_MODULE.load_config is not _CAPTURED_LOAD_CONFIG
        or _CAPTURED_BUILD_RUN_ONE_TASK.__module__
        != "performance.run_lodo_manifest_shard"
        or _CAPTURED_BUILD_RUN_ONE_TASK.__name__ != "build_run_one_task"
        or _CAPTURED_LOAD_CONFIG.__module__
        != "performance.run_lodo_manifest_shard"
        or _CAPTURED_LOAD_CONFIG.__name__ != "load_config"
    ):
        raise MaterializationValidationError(
            "runner callable binding differs from definition-time capture"
        )
    return {
        "module": binding["module"],
        "callable": binding["callable"],
        "raw_sha256": binding["runner_raw_sha256"],
        "invokes_run_one": False,
        "definition_time_callable_capture": True,
    }


def _revalidate_raw_inputs(
    *,
    base_manifest_path: str,
    asset_root: str,
    contract: Mapping[str, Any],
) -> None:
    """Ensure every bound raw input still has its initially accepted bytes."""
    _load_bound_json(
        base_manifest_path,
        expected_raw_sha256=contract["base_manifest"]["raw_sha256"],
        label="base manifest post-materialization",
    )
    root = Path(asset_root)
    for domain in _DOMAINS:
        spec = contract["asset_bundle"]["domains"][domain]
        for kind, relative_key, sha_key in (
            ("design", "design_relative_path", "design_raw_sha256"),
            ("companion", "companion_relative_path", "companion_raw_sha256"),
        ):
            path = _bound_asset_path(
                root,
                _safe_relative_path(spec[relative_key], f"{kind} path"),
                f"{domain} {kind} post-materialization path",
            )
            _load_bound_json(
                path,
                expected_raw_sha256=spec[sha_key],
                label=f"{domain} {kind} post-materialization",
            )
    _validate_runner_source(contract)


def _validate_design_payload(
    payload: Mapping[str, Any],
    *,
    path: Path,
    domain: str,
    archive_fingerprint: str,
    schema: Mapping[str, Any],
) -> dict[str, str]:
    _exact_keys(payload, _DESIGN_KEYS, f"{domain} design")
    expected_scalars = {
        "schema_version": schema["schema_version"],
        "design_kind": schema["design_kind"],
        "dimension": schema["dimension"],
        "n0": schema["n0"],
        "seed_start": schema["seed_start"],
        "n_seeds": schema["n_seeds"],
        "source_archive_oracle_aided": schema[
            "source_archive_oracle_aided"
        ],
        "target_labels_used": schema["target_labels_used"],
        "target_oracle_used": schema["target_oracle_used"],
        "heldout_target_domain": domain,
        "source_archive_fingerprint": archive_fingerprint,
    }
    if any(payload.get(key) != value for key, value in expected_scalars.items()):
        raise MaterializationValidationError(
            f"{domain} design schema or binding differs"
        )
    designs = payload["designs"]
    seed_keys = schema["seed_keys"]
    if type(designs) is not dict or set(designs) != set(seed_keys):
        raise MaterializationValidationError(
            f"{domain} design must contain exact seeds 0..19"
        )
    fingerprints: dict[str, str] = {}
    for seed in range(20):
        key = str(seed)
        row = designs[key]
        if type(row) is not dict:
            raise MaterializationValidationError(
                f"{domain} seed {seed} design is malformed"
            )
        _exact_keys(row, {"fingerprint", "points"}, "design seed")
        points = row["points"]
        if (
            type(points) is not list
            or len(points) != 10
            or any(type(point) is not list or len(point) != 50 for point in points)
            or any(
                type(coordinate) is not int
                for point in points
                for coordinate in point
            )
            or len({tuple(point) for point in points}) != 10
        ):
            raise MaterializationValidationError(
                f"{domain} seed {seed} points violate the 10x50 integer schema"
            )
        fingerprint = integer_design_fingerprint(points)
        if row["fingerprint"] != fingerprint:
            raise MaterializationValidationError(
                f"{domain} seed {seed} design fingerprint differs"
            )
        try:
            loaded_points, loaded_contract = load_frozen_source_informed_design(
                path,
                heldout=domain,
                seed=seed,
                n0=10,
                dimension=50,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise MaterializationValidationError(
                f"{domain} seed {seed} frozen design load failed"
            ) from error
        if (
            [list(point) for point in loaded_points] != points
            or loaded_contract.get("fingerprint") != fingerprint
            or loaded_contract.get("source_archive_fingerprint")
            != archive_fingerprint
            or loaded_contract.get("uses_source_archive") is not True
            or loaded_contract.get("source_archive_oracle_aided") is not False
            or loaded_contract.get("target_labels_used") is not False
            or loaded_contract.get("target_oracle_used") is not False
        ):
            raise MaterializationValidationError(
                f"{domain} seed {seed} frozen design contract differs"
            )
        fingerprints[key] = fingerprint
    return fingerprints


def _validate_assets(
    asset_root: str | Path, contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    root = Path(asset_root)
    asset_contract = contract["asset_bundle"]
    schema = asset_contract["design_schema"]
    summaries = []
    for domain in _DOMAINS:
        spec = asset_contract["domains"][domain]
        design_path = _bound_asset_path(
            root,
            _safe_relative_path(spec["design_relative_path"], "design path"),
            f"{domain} design path",
        )
        companion_path = _bound_asset_path(
            root,
            _safe_relative_path(
                spec["companion_relative_path"], "companion path"
            ),
            f"{domain} companion path",
        )
        design_payload, _ = _load_bound_json(
            design_path,
            expected_raw_sha256=spec["design_raw_sha256"],
            label=f"{domain} source design",
        )
        fingerprints = _validate_design_payload(
            design_payload,
            path=design_path,
            domain=domain,
            archive_fingerprint=spec["archive_fingerprint"],
            schema=schema,
        )
        companion_payload, _ = _load_bound_json(
            companion_path,
            expected_raw_sha256=spec["companion_raw_sha256"],
            label=f"{domain} companion archive",
        )
        _exact_keys(
            companion_payload,
            _ARCHIVE_KEYS,
            f"{domain} companion archive",
        )
        try:
            archive = FrozenTransferArchive.load(companion_path)
            archive.validate(
                expected_domains=spec["source_domains"],
                expected_dimension=50,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise MaterializationValidationError(
                f"{domain} companion archive load failed"
            ) from error
        if (
            archive.fingerprint != spec["archive_fingerprint"]
            or companion_payload["fingerprint"] != spec["archive_fingerprint"]
            or archive.source_seed != 0
            or archive.observation_mode != "replicated"
            or list(archive.source_domains) != spec["source_domains"]
            or archive.simulator_calls != spec["source_simulator_calls"]
            or any(
                archive.profiles_per_domain.get(source_domain)
                != spec["source_profiles_per_domain"]
                for source_domain in spec["source_domains"]
            )
        ):
            raise MaterializationValidationError(
                f"{domain} companion archive contract differs"
            )
        summaries.append({
            "domain": domain,
            "design_raw_sha256": spec["design_raw_sha256"],
            "companion_raw_sha256": spec["companion_raw_sha256"],
            "archive_fingerprint": spec["archive_fingerprint"],
            "source_domains": list(spec["source_domains"]),
            "source_profiles_per_domain": spec[
                "source_profiles_per_domain"
            ],
            "source_simulator_calls": spec["source_simulator_calls"],
            "design_seed_fingerprints": fingerprints,
        })
    return summaries


def _path_text(
    value: str | Path, label: str, *, require_absolute: bool = False
) -> str:
    try:
        path = Path(value)
    except TypeError as error:
        raise MaterializationValidationError(f"{label} is not a path") from error
    if require_absolute and not path.is_absolute():
        raise MaterializationValidationError(f"{label} must be absolute")
    text = str(path)
    if not text:
        raise MaterializationValidationError(f"{label} must not be empty")
    return text


def _render_argv(
    *,
    cell: Mapping[str, Any],
    contract: Mapping[str, Any],
    base_manifest_path: str,
    asset_root: str,
    checkpoint_root: str,
) -> list[str]:
    domain = cell["domain"]
    seed = cell["seed"]
    spec = contract["asset_bundle"]["domains"][domain]
    checkpoint_dir = Path(checkpoint_root) / domain / f"seed{seed}"
    materialization_only_out = (
        Path(checkpoint_root)
        / "_materialization_only"
        / domain
        / f"seed{seed}"
        / "result.json"
    )
    replacements = {
        "{base_manifest}": base_manifest_path,
        "{domain}": domain,
        "{seed}": str(seed),
        "{materialization_only_out}": str(materialization_only_out),
        "{checkpoint_dir}": str(checkpoint_dir),
        "{design_file}": str(
            Path(asset_root) / spec["design_relative_path"]
        ),
    }
    return [replacements.get(token, token) for token in contract["argv_template"]]


def _validate_resolved_plan(
    plan: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    hypothesis_contract: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
    materializer_contract: Mapping[str, Any],
    checkpoint_root: str,
) -> None:
    if not verify_plan_integrity(
        plan, hypothesis_contract, executor_contract, report
    ):
        raise MaterializationValidationError(
            "resolved execution plan integrity failed"
        )
    if plan.get("status") != "READY_FOR_AUTHORIZATION":
        raise MaterializationValidationError(
            "resolved plan is not ready mechanics"
        )
    mechanics = materializer_contract["artifact_mechanics"]
    if (
        plan.get("proposal_count") != len(report.get("pending_evidence", ()))
        or not mechanics["task_count_minimum"]
        <= plan["proposal_count"]
        <= mechanics["task_count_maximum"]
    ):
        raise MaterializationValidationError(
            "resolved plan does not match the exact pending-cell set"
        )
    expected_count = mechanics["task_args_key_count"]
    expected_interval = mechanics["runtime_checkpoint_interval"]
    for task in plan["tasks"]:
        if task.get("status") != "READY_FOR_AUTHORIZATION":
            raise MaterializationValidationError(
                "resolved task is not ready mechanics"
            )
        run_task = task.get("run_one_task")
        if type(run_task) is not dict or type(run_task.get("args")) is not dict:
            raise MaterializationValidationError("resolved run_one task is malformed")
        args = run_task["args"]
        domain = run_task["heldout"]
        seed = run_task["seed"]
        expected_checkpoint = str(
            Path(checkpoint_root) / domain / f"seed{seed}"
        )
        if (
            len(args) != expected_count
            or args.get("runtime_checkpoint_interval") != expected_interval
            or args.get("runtime_checkpoint_dir") != expected_checkpoint
            or args.get("initial_design") != "source_informed"
            or args.get("initial_design_source_archive_fingerprint")
            != materializer_contract["asset_bundle"]["domains"][domain][
                "archive_fingerprint"
            ]
            or "initial_design_file" in args
        ):
            raise MaterializationValidationError(
                "resolved task differs from materializer mechanics"
            )
        normalized_args = _json_clone(args, "resolved task args")
        normalized_args["runtime_checkpoint_dir"] = "{checkpoint_dir}"
        if _digest(normalized_args) != materializer_contract[
            "artifact_mechanics"
        ]["normalized_task_args_digest_by_domain"][domain]:
            raise MaterializationValidationError(
                "resolved task args differ from the frozen domain template"
            )


def materialize_task_bundle(
    report: Mapping[str, Any],
    hypothesis_contract: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
    materializer_contract: Mapping[str, Any],
    base_manifest_path: str | Path,
    asset_root: str | Path,
    checkpoint_root: str | Path,
) -> dict[str, Any]:
    """Materialize a bound, non-authorized plan without invoking ``run_one``."""
    try:
        _validate_source_contracts(
            hypothesis_contract, executor_contract, materializer_contract
        )
        base_manifest_text = _path_text(
            base_manifest_path, "base_manifest_path"
        )
        asset_root_text = _path_text(asset_root, "asset_root")
        checkpoint_root_text = _path_text(
            checkpoint_root, "checkpoint_root", require_absolute=True
        )
        base_binding = _validate_base_manifest(
            base_manifest_text, materializer_contract
        )
        runner_binding = _validate_runner_source(materializer_contract)
        asset_binding = _validate_assets(
            asset_root_text, materializer_contract
        )

        def materialize_cell(cell: Mapping[str, Any]) -> Mapping[str, Any]:
            argv = _render_argv(
                cell=cell,
                contract=materializer_contract,
                base_manifest_path=base_manifest_text,
                asset_root=asset_root_text,
                checkpoint_root=checkpoint_root_text,
            )
            try:
                return _CAPTURED_BUILD_RUN_ONE_TASK(argv)
            except SystemExit as error:
                raise MaterializationValidationError(
                    "runner rejected the frozen materializer argv"
                ) from error

        plan = build_execution_plan(
            report,
            hypothesis_contract,
            executor_contract,
            task_materializer=materialize_cell,
        )
        _validate_resolved_plan(
            plan,
            report=report,
            hypothesis_contract=hypothesis_contract,
            executor_contract=executor_contract,
            materializer_contract=materializer_contract,
            checkpoint_root=checkpoint_root_text,
        )
        _revalidate_raw_inputs(
            base_manifest_path=base_manifest_text,
            asset_root=asset_root_text,
            contract=materializer_contract,
        )
        identity = {
            "materializer_contract_digest": _MATERIALIZER_CONTRACT_DIGEST,
            "plan_digest": plan["integrity"]["plan_digest"],
            "input_bindings": {
                "base_manifest": base_binding,
                "runner_source": runner_binding,
                "asset_bundle": {
                    "bundle_id": materializer_contract["asset_bundle"][
                        "bundle_id"
                    ],
                    "domains": asset_binding,
                },
                "checkpoint_root": checkpoint_root_text,
            },
        }
        bundle_id = "task-bundle:" + _digest(identity).split(":", 1)[1][:24]
        body = {
            "schema_version": TASK_BUNDLE_SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "status": "MATERIALIZED_NOT_AUTHORIZED",
            "materializer_contract_id": _MATERIALIZER_CONTRACT_ID,
            "materializer_contract_digest": _MATERIALIZER_CONTRACT_DIGEST,
            "source_contracts": {
                "hypothesis_contract_id": _HYPOTHESIS_CONTRACT_ID,
                "hypothesis_contract_digest": _HYPOTHESIS_CONTRACT_DIGEST,
                "executor_contract_id": _EXECUTOR_CONTRACT_ID,
                "executor_contract_digest": _EXECUTOR_CONTRACT_DIGEST,
            },
            "input_bindings": identity["input_bindings"],
            "task_count": plan["proposal_count"],
            "plan": plan,
            "nonclaims": list(_NONCLAIMS),
        }
        return {
            **body,
            "integrity": {
                "algorithm": "sha256-canonical-json-v1",
                "bundle_digest": _digest(body),
            },
        }
    except MaterializationValidationError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise MaterializationValidationError(
            f"task materialization failed: {type(error).__name__}"
        ) from error


def verify_materialized_task_bundle(
    bundle: Mapping[str, Any],
    report: Mapping[str, Any],
    hypothesis_contract: Mapping[str, Any],
    executor_contract: Mapping[str, Any],
    materializer_contract: Mapping[str, Any],
    base_manifest_path: str | Path,
    asset_root: str | Path,
    checkpoint_root: str | Path,
) -> bool:
    """Strong-verify a bundle by rebuilding every task from bound raw inputs."""
    try:
        if type(bundle) is not dict:
            return False
        actual = _json_clone(bundle, "materialized task bundle")
        expected = materialize_task_bundle(
            report,
            hypothesis_contract,
            executor_contract,
            materializer_contract,
            base_manifest_path,
            asset_root,
            checkpoint_root,
        )
        return actual == expected
    except (
        MaterializationValidationError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
    ):
        return False


__all__ = [
    "MATERIALIZER_SCHEMA_VERSION",
    "TASK_BUNDLE_SCHEMA_VERSION",
    "MaterializationValidationError",
    "load_strict_json_file",
    "materialize_task_bundle",
    "validate_materializer_contract",
    "verify_materialized_task_bundle",
]
