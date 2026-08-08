#!/usr/bin/env python3
"""Parallel post-decision temporal audit for frozen OPSD result cells."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from performance.audit_external_energy_temporal_blocks import (
    AUDIT_CONTRACT,
    audit_result,
)


MATRIX_CONTRACT = "opsd_postdecision_temporal_audit_matrix_v1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _output_path(output_dir, index, source_path):
    return Path(output_dir) / (
        f"cell{int(index):04d}__{Path(source_path).stem}__temporal.json"
    )


def _existing_ok(output, source_path):
    output = Path(output)
    if not output.exists():
        return False
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        payload.get("contract_id") == AUDIT_CONTRACT
        and payload.get("source_result_sha256") == _sha256(source_path)
        and payload.get("status") in {"complete", "not_certified"}
    )


def _run_cell(
    index,
    source_path,
    output_dir,
    data_path,
    chronological_blocks,
    maximum_sampled_starts,
):
    output = _output_path(output_dir, index, source_path)
    if _existing_ok(output, source_path):
        return {"index": int(index), "status": "skipped", "out": str(output)}
    started = time.perf_counter()
    try:
        payload = audit_result(
            source_path,
            data_path=data_path,
            chronological_blocks=chronological_blocks,
            maximum_sampled_starts=maximum_sampled_starts,
        )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "contract_id": "opsd_temporal_audit_cell_error_v1",
            "status": "error",
            "source_result_path": str(source_path),
            "source_result_sha256": _sha256(source_path),
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_time_sec": float(time.perf_counter() - started),
        }
        _atomic_json(output, payload)
        return {
            "index": int(index),
            "status": "error",
            "out": str(output),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    payload["wall_time_sec"] = float(time.perf_counter() - started)
    _atomic_json(output, payload)
    return {"index": int(index), "status": "done", "out": str(output)}


def run_matrix(
    *,
    input_roots,
    output_dir,
    data_path,
    start=0,
    end=None,
    workers=1,
    expected_source_count=None,
    chronological_blocks=4,
    maximum_sampled_starts=512,
):
    sources = sorted({
        path.resolve()
        for root in map(Path, input_roots)
        for path in root.rglob("cell*.json")
    })
    if expected_source_count is not None and len(sources) != int(
        expected_source_count
    ):
        raise ValueError(
            "temporal audit source count mismatch: "
            f"expected {int(expected_source_count)}, observed {len(sources)}"
        )
    start = max(0, int(start))
    stop = len(sources) if end is None else min(len(sources), int(end))
    if stop < start:
        raise ValueError("temporal audit end precedes start")
    selected = list(enumerate(sources[start:stop], start=start))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), len(selected) or 1))
    started = time.perf_counter()
    results = []
    if workers == 1:
        for completed, (index, source) in enumerate(selected, start=1):
            results.append(_run_cell(
                index,
                source,
                output_dir,
                data_path,
                chronological_blocks,
                maximum_sampled_starts,
            ))
            print(
                f"ENERGY_TEMPORAL_AUDIT_PROGRESS {completed}/{len(selected)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_cell,
                    index,
                    source,
                    output_dir,
                    data_path,
                    chronological_blocks,
                    maximum_sampled_starts,
                ): index
                for index, source in selected
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                print(
                    f"ENERGY_TEMPORAL_AUDIT_PROGRESS {completed}/{len(selected)}",
                    flush=True,
                )
    error_count = int(sum(row["status"] == "error" for row in results))
    summary = {
        "schema_version": 1,
        "contract_id": MATRIX_CONTRACT,
        "status": "complete" if error_count == 0 else "complete_with_errors",
        "source_cell_count": len(sources),
        "expected_source_count": (
            None if expected_source_count is None
            else int(expected_source_count)
        ),
        "shard_start": start,
        "shard_end": stop,
        "shard_cell_count": len(selected),
        "workers": workers,
        "completed_count": int(sum(
            row["status"] == "done" for row in results)),
        "skipped_count": int(sum(
            row["status"] == "skipped" for row in results)),
        "error_count": error_count,
        "cell_errors": [
            {
                "index": row["index"],
                "error_type": row.get("error_type"),
                "error": row.get("error"),
            }
            for row in results if row["status"] == "error"
        ],
        "chronological_blocks": int(chronological_blocks),
        "maximum_sampled_starts": int(maximum_sampled_starts),
        "postdecision_only": True,
        "inferential_certificate_claimed": False,
        "wall_time_sec": float(time.perf_counter() - started),
    }
    _atomic_json(
        Path(output_dir) / f"shard_{start:04d}_{stop:04d}.summary.json",
        summary,
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--expected-source-count", type=int)
    parser.add_argument("--chronological-blocks", type=int, default=4)
    parser.add_argument("--maximum-sampled-starts", type=int, default=512)
    args = parser.parse_args()
    summary = run_matrix(
        input_roots=args.input_root,
        output_dir=args.output_dir,
        data_path=args.data,
        start=args.start,
        end=args.end,
        workers=args.workers,
        expected_source_count=args.expected_source_count,
        chronological_blocks=args.chronological_blocks,
        maximum_sampled_starts=args.maximum_sampled_starts,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["error_count"] == 0 else 2)


if __name__ == "__main__":
    main()
