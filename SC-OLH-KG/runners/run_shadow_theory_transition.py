"""Materialize one verified competition result as a shadow theory transition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

DEFAULT_COMPETITION_CONTRACT = (
    ROOT
    / "performance"
    / "manifests"
    / "theory_operation_competition_v1.json"
)
DEFAULT_TRANSITION_CONTRACT = (
    ROOT / "performance" / "manifests" / "shadow_theory_transition_v1.json"
)


class DuplicateKeyError(ValueError):
    """Raised before either core sees an ambiguous JSON object."""


class _StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(f"invalid command line: {message}")


def _reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str):
    raise ValueError(f"non-finite JSON number: {value}")


def _require_input_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an existing non-symlink file")
    return path


def _artifact(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "bytes": len(raw),
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_input_file(path, label)
    raw = path.read_bytes()
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload, _artifact(path, raw)


def _protect_output(out: Path, inputs: tuple[Path, ...]) -> Path:
    if not out.is_absolute():
        raise ValueError("--out must be an absolute path")
    if out.is_symlink():
        raise ValueError("--out cannot be a symlink")
    resolved_out = out.resolve()
    for source in inputs:
        if resolved_out == source.resolve():
            raise ValueError("--out cannot overwrite an input file")
        if out.exists() and out.samefile(source):
            raise ValueError("--out cannot be a hard link to an input file")
    return out


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = _StrictArgumentParser(
        description=(
            "Materialize a verified theory-operation result as a local, "
            "unadopted shadow theory transition."
        )
    )
    parser.add_argument("--competition-input", type=Path, required=True)
    parser.add_argument(
        "--competition-contract", type=Path, default=DEFAULT_COMPETITION_CONTRACT
    )
    parser.add_argument("--competition-report", type=Path, required=True)
    parser.add_argument(
        "--transition-contract", type=Path, default=DEFAULT_TRANSITION_CONTRACT
    )
    parser.add_argument("--expected-competition-contract-digest", required=True)
    parser.add_argument("--expected-competition-report-digest", required=True)
    parser.add_argument("--expected-transition-contract-digest", required=True)
    parser.add_argument(
        "--out", type=Path, help="atomically copy the canonical transition report"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        competition_input_path = _require_input_file(
            args.competition_input, "--competition-input"
        )
        competition_contract_path = _require_input_file(
            args.competition_contract, "--competition-contract"
        )
        competition_report_path = _require_input_file(
            args.competition_report, "--competition-report"
        )
        transition_contract_path = _require_input_file(
            args.transition_contract, "--transition-contract"
        )
        inputs = (
            competition_input_path,
            competition_contract_path,
            competition_report_path,
            transition_contract_path,
        )
        if args.out is not None:
            _protect_output(args.out, inputs)

        competition_input, competition_input_artifact = _load_json(
            competition_input_path, "--competition-input"
        )
        competition_contract, competition_contract_artifact = _load_json(
            competition_contract_path, "--competition-contract"
        )
        competition_report, competition_report_artifact = _load_json(
            competition_report_path, "--competition-report"
        )
        transition_contract, transition_contract_artifact = _load_json(
            transition_contract_path, "--transition-contract"
        )

        from performance.shadow_theory_transition import (  # noqa: E402
            materialize_shadow_theory_transition,
        )
        from performance.theory_operation_competition import (  # noqa: E402
            canonical_json_bytes,
        )

        competition_input_artifacts = {
            "contract_json": competition_contract_artifact,
            "evidence_json": competition_input_artifact,
        }
        transition_input_artifacts = {
            "competition_contract_json": competition_contract_artifact,
            "competition_input_json": competition_input_artifact,
            "competition_report_json": competition_report_artifact,
            "transition_contract_json": transition_contract_artifact,
        }
        result = materialize_shadow_theory_transition(
            competition_input,
            competition_contract,
            competition_report,
            transition_contract,
            expected_competition_contract_digest=(
                args.expected_competition_contract_digest
            ),
            expected_competition_report_digest=(
                args.expected_competition_report_digest
            ),
            expected_competition_input_artifacts=competition_input_artifacts,
            expected_transition_contract_digest=(
                args.expected_transition_contract_digest
            ),
            input_artifacts=transition_input_artifacts,
        )
        report = result.to_dict() if hasattr(result, "to_dict") else result
        if not isinstance(report, dict):
            raise ValueError("transition core must return a JSON object")
        payload = canonical_json_bytes(report) + b"\n"
        sys.stdout.buffer.write(payload)
        if args.out is not None:
            _write_atomic(args.out, payload)
    except (
        DuplicateKeyError,
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        print(f"invalid shadow-theory-transition input: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
