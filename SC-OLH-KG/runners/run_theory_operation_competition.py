"""Run the shadow-only theory-operation competition on strict local JSON.

This adapter has no execution action.  It reads one evidence document and one
versioned contract, delegates the scientific decision to
``performance.theory_operation_competition``, and emits canonical JSON.  An
optional output is an atomic copy of the same bytes; no input may be used as
that output, including through a hard link.
"""

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

DEFAULT_CONTRACT = (
    ROOT
    / "performance"
    / "manifests"
    / "theory_operation_competition_v1.json"
)


class DuplicateKeyError(ValueError):
    """Raised before the core sees an ambiguous JSON object."""


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
            "Compare bounded theory-operation candidates using only local, "
            "versioned shadow evidence."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--out",
        type=Path,
        help="atomically copy the canonical report to this path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        evidence_path = _require_input_file(args.input, "--input")
        contract_path = _require_input_file(args.contract, "--contract")
        if args.out is not None:
            _protect_output(args.out, (evidence_path, contract_path))

        evidence, evidence_artifact = _load_json(evidence_path, "--input")
        contract, contract_artifact = _load_json(contract_path, "--contract")

        from performance.theory_operation_competition import (  # noqa: E402
            canonical_json_bytes,
            run_theory_operation_competition,
        )

        result = run_theory_operation_competition(
            evidence,
            contract,
            input_artifacts={
                "contract_json": contract_artifact,
                "evidence_json": evidence_artifact,
            },
        )
        report = result.to_dict() if hasattr(result, "to_dict") else result
        if not isinstance(report, dict):
            raise ValueError("competition core must return a JSON object")
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
        print(f"invalid theory-operation-competition input: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
