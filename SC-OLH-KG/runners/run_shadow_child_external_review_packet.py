"""Build one non-authoritative external-review packet for a shadow child."""

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
    ROOT / "performance/manifests/theory_operation_competition_v1.json"
)
DEFAULT_TRANSITION_CONTRACT = (
    ROOT / "performance/manifests/shadow_theory_transition_v1.json"
)
DEFAULT_QUALIFICATION_CONTRACT = (
    ROOT / "performance/manifests/shadow_child_probe_qualification_v1.json"
)
DEFAULT_REVIEW_CONTRACT = (
    ROOT / "performance/manifests/shadow_child_external_review_packet_v1.json"
)


class DuplicateKeyError(ValueError):
    """Raised before any core sees an ambiguous JSON object."""


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


def _require_distinct_inputs(inputs: tuple[Path, ...]) -> None:
    resolved_paths: set[Path] = set()
    inodes: set[tuple[int, int]] = set()
    for path in inputs:
        resolved = path.resolve()
        if resolved in resolved_paths:
            raise ValueError("all nine input paths must resolve distinctly")
        resolved_paths.add(resolved)
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in inodes:
            raise ValueError("all nine input files must have distinct inodes")
        inodes.add(inode)


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
            "Build a local, non-authoritative external-review packet for one "
            "exact-replayed shadow-child qualification."
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
    parser.add_argument("--transition-report", type=Path, required=True)
    parser.add_argument("--qualification-input", type=Path, required=True)
    parser.add_argument(
        "--qualification-contract",
        type=Path,
        default=DEFAULT_QUALIFICATION_CONTRACT,
    )
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument(
        "--review-contract", type=Path, default=DEFAULT_REVIEW_CONTRACT
    )
    parser.add_argument("--expected-competition-contract-digest", required=True)
    parser.add_argument("--expected-competition-report-digest", required=True)
    parser.add_argument("--expected-transition-contract-digest", required=True)
    parser.add_argument("--expected-transition-report-digest", required=True)
    parser.add_argument("--expected-qualification-input-digest", required=True)
    parser.add_argument("--expected-qualification-contract-digest", required=True)
    parser.add_argument("--expected-qualification-report-digest", required=True)
    parser.add_argument("--expected-review-contract-digest", required=True)
    parser.add_argument(
        "--out", type=Path, help="atomically copy the canonical review packet"
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
        transition_report_path = _require_input_file(
            args.transition_report, "--transition-report"
        )
        qualification_input_path = _require_input_file(
            args.qualification_input, "--qualification-input"
        )
        qualification_contract_path = _require_input_file(
            args.qualification_contract, "--qualification-contract"
        )
        qualification_report_path = _require_input_file(
            args.qualification_report, "--qualification-report"
        )
        review_contract_path = _require_input_file(
            args.review_contract, "--review-contract"
        )
        inputs = (
            competition_input_path,
            competition_contract_path,
            competition_report_path,
            transition_contract_path,
            transition_report_path,
            qualification_input_path,
            qualification_contract_path,
            qualification_report_path,
            review_contract_path,
        )
        _require_distinct_inputs(inputs)
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
        transition_report, transition_report_artifact = _load_json(
            transition_report_path, "--transition-report"
        )
        qualification_input, qualification_input_artifact = _load_json(
            qualification_input_path, "--qualification-input"
        )
        qualification_contract, qualification_contract_artifact = _load_json(
            qualification_contract_path, "--qualification-contract"
        )
        qualification_report, qualification_report_artifact = _load_json(
            qualification_report_path, "--qualification-report"
        )
        review_contract, review_contract_artifact = _load_json(
            review_contract_path, "--review-contract"
        )

        from performance.shadow_child_external_review_packet import (  # noqa: E402
            build_shadow_child_external_review_packet,
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
        qualification_input_artifacts = {
            "competition_contract_json": competition_contract_artifact,
            "competition_input_json": competition_input_artifact,
            "competition_report_json": competition_report_artifact,
            "qualification_contract_json": qualification_contract_artifact,
            "qualification_input_json": qualification_input_artifact,
            "transition_contract_json": transition_contract_artifact,
            "transition_report_json": transition_report_artifact,
        }
        review_input_artifacts = {
            "competition_contract_json": competition_contract_artifact,
            "competition_input_json": competition_input_artifact,
            "competition_report_json": competition_report_artifact,
            "qualification_contract_json": qualification_contract_artifact,
            "qualification_input_json": qualification_input_artifact,
            "qualification_report_json": qualification_report_artifact,
            "review_contract_json": review_contract_artifact,
            "transition_contract_json": transition_contract_artifact,
            "transition_report_json": transition_report_artifact,
        }
        result = build_shadow_child_external_review_packet(
            competition_input,
            competition_contract,
            competition_report,
            transition_contract,
            transition_report,
            qualification_input,
            qualification_contract,
            qualification_report,
            review_contract,
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
            expected_transition_report_digest=(
                args.expected_transition_report_digest
            ),
            expected_transition_input_artifacts=transition_input_artifacts,
            expected_qualification_input_digest=(
                args.expected_qualification_input_digest
            ),
            expected_qualification_contract_digest=(
                args.expected_qualification_contract_digest
            ),
            expected_qualification_report_digest=(
                args.expected_qualification_report_digest
            ),
            expected_qualification_input_artifacts=qualification_input_artifacts,
            expected_review_contract_digest=args.expected_review_contract_digest,
            input_artifacts=review_input_artifacts,
        )
        report = result.to_dict() if hasattr(result, "to_dict") else result
        if not isinstance(report, dict):
            raise ValueError("external-review-packet core must return a JSON object")
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
        print(f"invalid shadow-child-external-review-packet input: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
