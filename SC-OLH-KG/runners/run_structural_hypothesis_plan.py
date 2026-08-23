"""Build or verify a plan-only structural-hypothesis execution artifact.

This CLI never imports the benchmark executor, calls ``run_one``, opens a
network connection, submits to a scheduler, or launches a shell.  Its only
state change is an explicitly requested atomic ``--out`` write.
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

from performance.structural_hypothesis_execution import (  # noqa: E402
    build_execution_plan,
    verify_plan_integrity,
)
from performance.structural_hypothesis_loop import (  # noqa: E402
    canonical_json_bytes,
    validate_contract as validate_hypothesis_contract,
)


DEFAULT_CONTRACT = (
    ROOT
    / "performance"
    / "manifests"
    / "structural_hypothesis_executor_v1.json"
)
DEFAULT_HYPOTHESIS_CONTRACT = (
    ROOT
    / "performance"
    / "manifests"
    / "structural_hypothesis_loop_v1.json"
)


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON has duplicate key {key!r}")
        result[key] = value
    return result


def _load_object(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload


def _same_existing_file(first: Path, second: Path) -> bool:
    return first.exists() and second.exists() and first.samefile(second)


def _verify_source_contract_binding(
    hypothesis_contract: dict[str, Any], executor_contract: dict[str, Any]
) -> None:
    validate_hypothesis_contract(hypothesis_contract)
    digest = "sha256:" + hashlib.sha256(
        canonical_json_bytes(hypothesis_contract)
    ).hexdigest()
    if (
        hypothesis_contract.get("contract_id")
        != executor_contract.get("source_hypothesis_contract_id")
        or digest
        != executor_contract.get("source_hypothesis_contract_digest")
    ):
        raise ValueError(
            "hypothesis contract does not match executor source binding"
        )


def _validate_output_path(out: Path | None, inputs: tuple[Path, ...]) -> None:
    if out is None:
        return
    resolved_out = out.resolve()
    for source in inputs:
        if resolved_out == source.resolve():
            raise ValueError("--out cannot overwrite an input file")
        if _same_existing_file(out, source):
            raise ValueError("--out cannot be a hard link to an input file")


def _write_json(payload: dict[str, Any], out: Path | None) -> None:
    rendered = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if out is None:
        sys.stdout.write(rendered)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{out.name}.", suffix=".tmp", dir=out.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _summary(plan: dict[str, Any], action: str) -> str:
    status = plan.get("status", "UNKNOWN")
    tasks = plan.get("tasks")
    cell_count = len(tasks) if isinstance(tasks, list) else 0
    integrity = plan.get("integrity")
    digest = (
        integrity.get("plan_digest", "missing")
        if isinstance(integrity, dict)
        else "missing"
    )
    return (
        f"structural_hypothesis_plan action={action} status={status} "
        f"cells={cell_count} plan_digest={digest}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the offline, plan-only run_one(task) proposal "
            "for pending structural-hypothesis evidence."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    plan = subparsers.add_parser(
        "plan", help="build a blocked proposal without materializing tasks"
    )
    plan.add_argument("--report", type=Path, required=True)
    plan.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    plan.add_argument(
        "--hypothesis-contract",
        type=Path,
        default=DEFAULT_HYPOTHESIS_CONTRACT,
    )
    plan.add_argument("--out", type=Path)

    verify = subparsers.add_parser(
        "verify", help="verify an existing plan against the V1 contract"
    )
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument(
        "--report",
        type=Path,
        required=True,
        help="the exact source report bound by the plan",
    )
    verify.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    verify.add_argument(
        "--hypothesis-contract",
        type=Path,
        default=DEFAULT_HYPOTHESIS_CONTRACT,
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        contract = _load_object(args.contract, "executor contract")
        hypothesis_contract = _load_object(
            args.hypothesis_contract, "hypothesis contract"
        )
        _verify_source_contract_binding(hypothesis_contract, contract)
        if args.action == "plan":
            _validate_output_path(
                args.out,
                (args.report, args.contract, args.hypothesis_contract),
            )
            report = _load_object(args.report, "hypothesis report")
            plan = build_execution_plan(
                report,
                hypothesis_contract,
                contract,
                task_materializer=None,
            )
            if not verify_plan_integrity(
                plan, hypothesis_contract, contract, report
            ):
                raise ValueError("generated plan failed integrity verification")
            _write_json(plan, args.out)
        else:
            plan = _load_object(args.plan, "execution plan")
            report = _load_object(args.report, "hypothesis report")
            if not verify_plan_integrity(
                plan, hypothesis_contract, contract, report
            ):
                raise ValueError("execution plan failed integrity verification")
        print(_summary(plan, args.action), file=sys.stderr)
        return 0
    except (
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"invalid structural-hypothesis plan input: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
