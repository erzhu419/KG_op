"""Run the versioned structural-prior hypothesis loop on a local CSV file.

The runner is deliberately a read-only, offline adapter unless ``--out`` is
provided.  Scientific decisions are delegated to
``performance.structural_hypothesis_loop``; this module only loads inputs,
records byte-level CSV provenance, and renders the JSON report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from performance.structural_hypothesis_loop import (  # noqa: E402
    ContractValidationError,
    REQUIRED_EVIDENCE_FIELDS,
    run_structural_hypothesis_loop,
)


DEFAULT_CONTRACT = (
    ROOT
    / "performance"
    / "manifests"
    / "structural_hypothesis_loop_v1.json"
)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _artifact(path: Path, raw: bytes) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"contract JSON has duplicate key {key!r}")
        result[key] = value
    return result


def _load_contract(path: Path) -> tuple[dict, dict]:
    raw = _read_bytes(path)
    payload = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
    )
    if not isinstance(payload, dict):
        raise ValueError("contract JSON must be an object")
    return payload, _artifact(path, raw)


def _load_evidence(path: Path) -> tuple[list[dict[str, str]], dict]:
    raw = _read_bytes(path)
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    fieldnames = reader.fieldnames
    if not fieldnames or any(not name for name in fieldnames):
        raise ValueError("evidence CSV must have non-empty column names")
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError("evidence CSV has duplicate column names")
    missing_fields = sorted(REQUIRED_EVIDENCE_FIELDS.difference(fieldnames))
    if missing_fields:
        raise ValueError(
            f"evidence CSV lacks required columns: {missing_fields}"
        )
    rows = []
    for row in reader:
        if None in row:
            raise ValueError("evidence CSV row has more values than its header")
        rows.append(dict(row))
    return rows, _artifact(path, raw)


def _contains_invalid_evidence(report: dict) -> bool:
    counts = report.get("verdict_counts")
    if not isinstance(counts, dict):
        return False
    count = counts.get("INVALID_EVIDENCE")
    return type(count) is int and count > 0


def _summary(report: dict) -> str:
    status = str(report.get("status", "completed"))
    counts = report.get("verdict_counts")
    if not isinstance(counts, dict):
        summary = report.get("summary")
        counts = summary.get("verdict_counts") if isinstance(summary, dict) else None
    if isinstance(counts, dict) and counts:
        rendered = ",".join(
            f"{key}={counts[key]}" for key in sorted(counts)
        )
        return f"structural_hypothesis_loop status={status} {rendered}"
    return f"structural_hypothesis_loop status={status}"


def _write_report(report: dict, out: Path | None) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if out is None:
        sys.stdout.write(payload)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{out.name}.", suffix=".tmp", dir=out.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _same_existing_file(first: Path, second: Path) -> bool:
    return first.exists() and second.exists() and first.samefile(second)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the four structural meta-priors against a versioned "
            "local retrospective-evidence contract."
        )
    )
    parser.add_argument("--evidence-csv", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--out",
        type=Path,
        help="write the complete JSON report here instead of stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence_path = args.evidence_csv
    contract_path = args.contract

    try:
        if args.out is not None:
            output = args.out.resolve()
            if output in {evidence_path.resolve(), contract_path.resolve()}:
                raise ValueError("--out cannot overwrite an input file")
            if (
                _same_existing_file(args.out, evidence_path)
                or _same_existing_file(args.out, contract_path)
            ):
                raise ValueError("--out cannot be a hard link to an input file")
        contract, contract_artifact = _load_contract(contract_path)
        rows, evidence_artifact = _load_evidence(evidence_path)
        result = run_structural_hypothesis_loop(
            rows,
            contract,
            input_artifacts={
                "evidence_csv": evidence_artifact,
                "contract_json": contract_artifact,
            },
        )
        report = result.to_dict()
        _write_report(report, args.out)
    except (
        ContractValidationError,
        csv.Error,
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        print(f"invalid structural-hypothesis-loop input: {exc}", file=sys.stderr)
        return 2

    print(_summary(report), file=sys.stderr)
    return 2 if _contains_invalid_evidence(report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
