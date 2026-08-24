"""Qualify one verified interval/multi-Q shadow child on fresh evidence."""

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
DEFAULT_PROBE_CONTRACT = (
    ROOT / "performance/manifests/shadow_child_failure_boundary_probe_v1.json"
)
DEFAULT_RESTRICTION_CONTRACT = (
    ROOT / "performance/manifests/shadow_robust_interval_restriction_v1.json"
)
DEFAULT_ADJUDICATION_CONTRACT = (
    ROOT / "performance/manifests/shadow_post_restriction_adjudication_v1.json"
)
DEFAULT_ADAPTER_CONTRACT = (
    ROOT / "performance/manifests/shadow_interval_multi_q_recompetition_adapter_v1.json"
)
DEFAULT_INTERVAL_COMPETITION_CONTRACT = (
    ROOT
    / "performance/manifests/shadow_interval_multi_q_theory_operation_competition_v2.json"
)
DEFAULT_INTERVAL_TRANSITION_CONTRACT = (
    ROOT
    / "performance/manifests/shadow_interval_multi_q_theory_transition_v1.json"
)
DEFAULT_POST_TRANSITION_QUALIFICATION_CONTRACT = (
    ROOT
    / "performance/manifests/shadow_interval_multi_q_post_transition_qualification_v1.json"
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
        resolved = Path(os.path.abspath(os.fspath(path)))
        if resolved in resolved_paths:
            raise ValueError("all twenty-nine input paths must resolve distinctly")
        resolved_paths.add(resolved)
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in inodes:
            raise ValueError("all twenty-nine input files must have distinct inodes")
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
            "Exact-replay the bounded interval/multi-Q chain and qualify one "
            "materialized shadow child under a fresh evaluator epoch."
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
        "--qualification-contract", type=Path, default=DEFAULT_QUALIFICATION_CONTRACT
    )
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--review-contract", type=Path, default=DEFAULT_REVIEW_CONTRACT)
    parser.add_argument("--review-report", type=Path, required=True)
    parser.add_argument("--probe-input", type=Path, required=True)
    parser.add_argument("--probe-contract", type=Path, default=DEFAULT_PROBE_CONTRACT)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--restriction-input", type=Path, required=True)
    parser.add_argument(
        "--restriction-contract", type=Path, default=DEFAULT_RESTRICTION_CONTRACT
    )
    parser.add_argument("--restriction-report", type=Path, required=True)
    parser.add_argument("--adjudication-input", type=Path, required=True)
    parser.add_argument(
        "--adjudication-contract", type=Path, default=DEFAULT_ADJUDICATION_CONTRACT
    )
    parser.add_argument("--adjudication-report", type=Path, required=True)
    parser.add_argument("--adapter-input", type=Path, required=True)
    parser.add_argument("--adapter-contract", type=Path, default=DEFAULT_ADAPTER_CONTRACT)
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--interval-competition-input", type=Path, required=True)
    parser.add_argument(
        "--interval-competition-contract",
        type=Path,
        default=DEFAULT_INTERVAL_COMPETITION_CONTRACT,
    )
    parser.add_argument("--interval-competition-report", type=Path, required=True)
    parser.add_argument(
        "--interval-transition-contract",
        type=Path,
        default=DEFAULT_INTERVAL_TRANSITION_CONTRACT,
    )
    parser.add_argument("--interval-transition-report", type=Path, required=True)
    parser.add_argument("--post-transition-qualification-input", type=Path, required=True)
    parser.add_argument(
        "--post-transition-qualification-contract",
        type=Path,
        default=DEFAULT_POST_TRANSITION_QUALIFICATION_CONTRACT,
    )
    for name in (
        "competition-contract",
        "competition-report",
        "transition-contract",
        "transition-report",
        "qualification-input",
        "qualification-contract",
        "qualification-report",
        "review-contract",
        "review-report",
        "probe-input",
        "probe-contract",
        "probe-report",
        "restriction-input",
        "restriction-contract",
        "restriction-report",
        "adjudication-input",
        "adjudication-contract",
        "adjudication-report",
        "adapter-input",
        "adapter-contract",
        "adapter-report",
        "interval-competition-input",
        "interval-competition-contract",
        "interval-competition-report",
        "interval-transition-contract",
        "interval-transition-report",
        "post-transition-qualification-input",
        "post-transition-qualification-contract",
    ):
        parser.add_argument(f"--expected-{name}-digest", required=True)
    parser.add_argument(
        "--out", type=Path, help="atomically copy the canonical qualification report"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        labels = (
            ("competition_input", "--competition-input"),
            ("competition_contract", "--competition-contract"),
            ("competition_report", "--competition-report"),
            ("transition_contract", "--transition-contract"),
            ("transition_report", "--transition-report"),
            ("qualification_input", "--qualification-input"),
            ("qualification_contract", "--qualification-contract"),
            ("qualification_report", "--qualification-report"),
            ("review_contract", "--review-contract"),
            ("review_report", "--review-report"),
            ("probe_input", "--probe-input"),
            ("probe_contract", "--probe-contract"),
            ("probe_report", "--probe-report"),
            ("restriction_input", "--restriction-input"),
            ("restriction_contract", "--restriction-contract"),
            ("restriction_report", "--restriction-report"),
            ("adjudication_input", "--adjudication-input"),
            ("adjudication_contract", "--adjudication-contract"),
            ("adjudication_report", "--adjudication-report"),
            ("adapter_input", "--adapter-input"),
            ("adapter_contract", "--adapter-contract"),
            ("adapter_report", "--adapter-report"),
            ("interval_competition_input", "--interval-competition-input"),
            ("interval_competition_contract", "--interval-competition-contract"),
            ("interval_competition_report", "--interval-competition-report"),
            ("interval_transition_contract", "--interval-transition-contract"),
            ("interval_transition_report", "--interval-transition-report"),
            (
                "post_transition_qualification_input",
                "--post-transition-qualification-input",
            ),
            (
                "post_transition_qualification_contract",
                "--post-transition-qualification-contract",
            ),
        )
        paths = {
            name: _require_input_file(getattr(args, name), label)
            for name, label in labels
        }
        inputs = tuple(paths[name] for name, _ in labels)
        _require_distinct_inputs(inputs)
        if args.out is not None:
            _protect_output(args.out, inputs)

        loaded = {}
        artifacts = {}
        for name, label in labels:
            loaded[name], artifacts[name] = _load_json(paths[name], label)

        from performance.shadow_interval_multi_q_post_transition_qualification import (  # noqa: E402
            qualify_shadow_interval_multi_q_post_transition,
        )
        from performance.theory_operation_competition import (  # noqa: E402
            canonical_json_bytes,
        )

        competition_input_artifacts = {
            "contract_json": artifacts["competition_contract"],
            "evidence_json": artifacts["competition_input"],
        }
        transition_input_artifacts = {
            "competition_contract_json": artifacts["competition_contract"],
            "competition_input_json": artifacts["competition_input"],
            "competition_report_json": artifacts["competition_report"],
            "transition_contract_json": artifacts["transition_contract"],
        }
        qualification_input_artifacts = {
            "competition_contract_json": artifacts["competition_contract"],
            "competition_input_json": artifacts["competition_input"],
            "competition_report_json": artifacts["competition_report"],
            "qualification_contract_json": artifacts["qualification_contract"],
            "qualification_input_json": artifacts["qualification_input"],
            "transition_contract_json": artifacts["transition_contract"],
            "transition_report_json": artifacts["transition_report"],
        }
        review_input_artifacts = {
            **qualification_input_artifacts,
            "qualification_report_json": artifacts["qualification_report"],
            "review_contract_json": artifacts["review_contract"],
        }
        probe_input_artifacts = {
            **review_input_artifacts,
            "probe_contract_json": artifacts["probe_contract"],
            "probe_input_json": artifacts["probe_input"],
            "review_report_json": artifacts["review_report"],
        }
        restriction_input_artifacts = {
            **probe_input_artifacts,
            "probe_report_json": artifacts["probe_report"],
            "restriction_contract_json": artifacts["restriction_contract"],
            "restriction_input_json": artifacts["restriction_input"],
        }
        adjudication_input_artifacts = {
            **restriction_input_artifacts,
            "restriction_report_json": artifacts["restriction_report"],
            "adjudication_contract_json": artifacts["adjudication_contract"],
            "adjudication_input_json": artifacts["adjudication_input"],
        }
        adapter_input_artifacts = {
            **adjudication_input_artifacts,
            "adjudication_report_json": artifacts["adjudication_report"],
            "adapter_contract_json": artifacts["adapter_contract"],
            "adapter_input_json": artifacts["adapter_input"],
        }
        interval_competition_input_artifacts = {
            **adapter_input_artifacts,
            "adapter_report_json": artifacts["adapter_report"],
            "interval_competition_contract_json": artifacts[
                "interval_competition_contract"
            ],
            "interval_competition_input_json": artifacts[
                "interval_competition_input"
            ],
        }
        interval_transition_input_artifacts = {
            **interval_competition_input_artifacts,
            "interval_competition_report_json": artifacts[
                "interval_competition_report"
            ],
            "interval_transition_contract_json": artifacts[
                "interval_transition_contract"
            ],
        }
        post_transition_qualification_input_artifacts = {
            **interval_transition_input_artifacts,
            "interval_transition_report_json": artifacts[
                "interval_transition_report"
            ],
            "post_transition_qualification_contract_json": artifacts[
                "post_transition_qualification_contract"
            ],
            "post_transition_qualification_input_json": artifacts[
                "post_transition_qualification_input"
            ],
        }

        result = qualify_shadow_interval_multi_q_post_transition(
            *(loaded[name] for name, _ in labels),
            expected_competition_contract_digest=(
                args.expected_competition_contract_digest
            ),
            expected_competition_report_digest=args.expected_competition_report_digest,
            expected_competition_input_artifacts=competition_input_artifacts,
            expected_transition_contract_digest=args.expected_transition_contract_digest,
            expected_transition_report_digest=args.expected_transition_report_digest,
            expected_transition_input_artifacts=transition_input_artifacts,
            expected_qualification_input_digest=args.expected_qualification_input_digest,
            expected_qualification_contract_digest=(
                args.expected_qualification_contract_digest
            ),
            expected_qualification_report_digest=args.expected_qualification_report_digest,
            expected_qualification_input_artifacts=qualification_input_artifacts,
            expected_review_contract_digest=args.expected_review_contract_digest,
            expected_review_report_digest=args.expected_review_report_digest,
            expected_review_input_artifacts=review_input_artifacts,
            expected_probe_input_digest=args.expected_probe_input_digest,
            expected_probe_contract_digest=args.expected_probe_contract_digest,
            expected_probe_report_digest=args.expected_probe_report_digest,
            expected_probe_input_artifacts=probe_input_artifacts,
            expected_restriction_input_digest=args.expected_restriction_input_digest,
            expected_restriction_contract_digest=(
                args.expected_restriction_contract_digest
            ),
            expected_restriction_report_digest=args.expected_restriction_report_digest,
            expected_restriction_input_artifacts=restriction_input_artifacts,
            expected_adjudication_input_digest=args.expected_adjudication_input_digest,
            expected_adjudication_contract_digest=(
                args.expected_adjudication_contract_digest
            ),
            expected_adjudication_report_digest=args.expected_adjudication_report_digest,
            expected_adjudication_input_artifacts=adjudication_input_artifacts,
            expected_adapter_input_digest=args.expected_adapter_input_digest,
            expected_adapter_contract_digest=args.expected_adapter_contract_digest,
            expected_adapter_report_digest=args.expected_adapter_report_digest,
            expected_adapter_input_artifacts=adapter_input_artifacts,
            expected_interval_competition_input_digest=(
                args.expected_interval_competition_input_digest
            ),
            expected_interval_competition_contract_digest=(
                args.expected_interval_competition_contract_digest
            ),
            expected_interval_competition_report_digest=(
                args.expected_interval_competition_report_digest
            ),
            expected_interval_competition_input_artifacts=(
                interval_competition_input_artifacts
            ),
            expected_interval_transition_contract_digest=(
                args.expected_interval_transition_contract_digest
            ),
            expected_interval_transition_report_digest=(
                args.expected_interval_transition_report_digest
            ),
            expected_interval_transition_input_artifacts=(
                interval_transition_input_artifacts
            ),
            expected_post_transition_qualification_input_digest=(
                args.expected_post_transition_qualification_input_digest
            ),
            expected_post_transition_qualification_contract_digest=(
                args.expected_post_transition_qualification_contract_digest
            ),
            input_artifacts=post_transition_qualification_input_artifacts,
        )
        report = result.to_dict() if hasattr(result, "to_dict") else result
        if not isinstance(report, dict):
            raise ValueError("post-transition qualification core must return a JSON object")
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
        print(
            f"invalid shadow-interval-multi-q-post-transition-qualification input: {exc}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
