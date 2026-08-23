"""Materialize or verify the frozen local structural-hypothesis task bundle.

This command has no authorization or execution operation.  In particular it
does not call ``run_one``, contact a scheduler, or perform network access.  A
successful ``materialize`` operation writes one new JSON artifact atomically
and refuses to replace any existing path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from performance.structural_hypothesis_task_materializer import (  # noqa: E402
    materialize_task_bundle,
    verify_materialized_task_bundle,
)


DEFAULT_HYPOTHESIS_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_loop_v1.json"
)
DEFAULT_EXECUTOR_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_executor_v1.json"
)
DEFAULT_MATERIALIZER_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_task_materializer_v1.json"
)


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON has duplicate key {key!r}")
        result[key] = value
    return result


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(
        path.read_bytes().decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload


def _validate_paths(
    *,
    checkpoint_root: Path,
    out: Path | None,
    inputs: tuple[Path, ...],
) -> None:
    if not checkpoint_root.is_absolute():
        raise ValueError("--checkpoint-root must be an absolute path")
    if out is None:
        return
    if out.exists() or out.is_symlink():
        raise ValueError("--out already exists; refusing to overwrite it")
    resolved_out = out.resolve()
    resolved_checkpoint_root = checkpoint_root.resolve()
    if (
        resolved_out == resolved_checkpoint_root
        or resolved_checkpoint_root in resolved_out.parents
    ):
        raise ValueError("--out must be outside --checkpoint-root")
    for source in inputs:
        resolved_source = source.resolve()
        if (
            resolved_out == resolved_source
            or source.is_dir() and resolved_source in resolved_out.parents
        ):
            raise ValueError("--out cannot alias or descend from an input path")


def _write_new_json(payload: dict[str, Any], out: Path) -> None:
    """Publish one complete JSON file without a replace race."""
    rendered = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
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
        # A hard-link publication is atomic on this filesystem and, unlike
        # os.replace(), fails if another writer created the output meanwhile.
        os.link(temporary, out)
        temporary.unlink()
        directory_fd = os.open(out.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--hypothesis-contract",
        type=Path,
        default=DEFAULT_HYPOTHESIS_CONTRACT,
    )
    parser.add_argument(
        "--executor-contract",
        type=Path,
        default=DEFAULT_EXECUTOR_CONTRACT,
    )
    parser.add_argument(
        "--materializer-contract",
        type=Path,
        default=DEFAULT_MATERIALIZER_CONTRACT,
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help=(
            "absolute path embedded in tasks; materialization does not create "
            "or inspect checkpoint contents"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize or verify complete, local run_one(task) mechanics "
            "without authorization or execution."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    materialize = subparsers.add_parser(
        "materialize", help="write a new not-authorized task bundle"
    )
    _add_common_inputs(materialize)
    materialize.add_argument("--out", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify", help="verify an existing task bundle and all local inputs"
    )
    verify.add_argument("--bundle", type=Path, required=True)
    _add_common_inputs(verify)
    return parser


def _summary(bundle: dict[str, Any], action: str) -> str:
    tasks = bundle.get("plan", {}).get("tasks", [])
    integrity = bundle.get("integrity", {})
    return (
        "structural_hypothesis_task_materializer "
        f"action={action} status={bundle.get('status', 'UNKNOWN')} "
        f"tasks={len(tasks) if isinstance(tasks, list) else 0} "
        f"bundle_digest={integrity.get('bundle_digest', 'missing')}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        path_inputs = (
            args.report,
            args.hypothesis_contract,
            args.executor_contract,
            args.materializer_contract,
            args.base_manifest,
            args.asset_root,
            args.checkpoint_root,
        )
        if args.action == "verify":
            path_inputs = (args.bundle, *path_inputs)
        _validate_paths(
            checkpoint_root=args.checkpoint_root,
            out=args.out if args.action == "materialize" else None,
            inputs=path_inputs,
        )

        report = _load_object(args.report, "hypothesis report")
        hypothesis_contract = _load_object(
            args.hypothesis_contract, "hypothesis contract"
        )
        executor_contract = _load_object(
            args.executor_contract, "executor contract"
        )
        materializer_contract = _load_object(
            args.materializer_contract, "materializer contract"
        )

        call = (
            report,
            hypothesis_contract,
            executor_contract,
            materializer_contract,
            args.base_manifest,
            args.asset_root,
            args.checkpoint_root,
        )
        if args.action == "materialize":
            bundle = materialize_task_bundle(*call)
            if not verify_materialized_task_bundle(bundle, *call):
                raise ValueError("generated task bundle failed verification")
            _write_new_json(bundle, args.out)
        else:
            bundle = _load_object(args.bundle, "materialized task bundle")
            if not verify_materialized_task_bundle(bundle, *call):
                raise ValueError("materialized task bundle failed verification")
        print(_summary(bundle, args.action), file=sys.stderr)
        return 0
    except (
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"invalid structural-hypothesis materializer input: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
