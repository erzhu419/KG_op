#!/usr/bin/env python3
"""Parallel matrix for the frozen target-only OPSD functional SCBO arm."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time


os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_external_energy_functional_scbo import (  # noqa: E402
    CONTRACT_ID,
    run_task,
)
from performance.benchmark_external_energy_v2 import (  # noqa: E402
    TARGET_MARKETS,
    _atomic_json,
)


def build_cells(
    *, markets=TARGET_MARKETS, target_seeds=(0, 1, 2, 3, 4),
):
    return [
        {
            "target_market": str(market),
            "target_seed": int(seed),
        }
        for market in markets
        for seed in target_seeds
    ]


def _cell_name(index, cell):
    return (
        f"cell{int(index):04d}__target-{cell['target_market']}__"
        f"seed-{int(cell['target_seed']):02d}__"
        "target_only_dct_space_scbo.json"
    )


def _existing_ok(
    path, *, method_commit, execution_commit, freeze_commit, cell,
):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        payload.get("contract_id") == CONTRACT_ID
        and payload.get("status") == "ok"
        and payload.get("functional_method_commit") == str(method_commit)
        and payload.get("execution_commit") == str(execution_commit)
        and payload.get("confirmatory_freeze_commit") == str(freeze_commit)
        and payload.get("target_market") == cell["target_market"]
        and int(payload.get("target_seed", -1)) == int(cell["target_seed"])
    )


def _run_cell(
    index,
    cell,
    *,
    data_path,
    output_dir,
    checkpoint_dir,
    method_commit,
    execution_commit,
    freeze_commit,
):
    output = Path(output_dir) / _cell_name(index, cell)
    if _existing_ok(
        output,
        method_commit=method_commit,
        execution_commit=execution_commit,
        freeze_commit=freeze_commit,
        cell=cell,
    ):
        return {"index": int(index), "status": "skipped", "out": str(output)}
    checkpoint = Path(checkpoint_dir) / f"cell{int(index):04d}.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        payload = run_task(
            data_path=data_path,
            target_market=cell["target_market"],
            target_seed=cell["target_seed"],
            checkpoint_path=str(checkpoint),
            checkpoint_resume=checkpoint.is_file(),
        )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "contract_id": "opsd_functional_scbo_cell_error_v1",
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "cell_index": int(index),
            "cell": dict(cell),
            "functional_method_commit": str(method_commit),
            "execution_commit": str(execution_commit),
            "confirmatory_freeze_commit": str(freeze_commit),
            "wall_time_sec": float(time.perf_counter() - started),
        }
        _atomic_json(output, payload)
        return {
            "index": int(index),
            "status": "error",
            "out": str(output),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    payload["functional_method_commit"] = str(method_commit)
    payload["execution_commit"] = str(execution_commit)
    payload["confirmatory_freeze_commit"] = str(freeze_commit)
    payload["confirmatory_cell_index"] = int(index)
    payload["wall_time_sec"] = float(time.perf_counter() - started)
    _atomic_json(output, payload)
    if checkpoint.is_file():
        checkpoint.unlink()
    return {
        "index": int(index),
        "status": "done",
        "out": str(output),
        "wall_time_sec": payload["wall_time_sec"],
    }


def run_matrix(
    *,
    data_path,
    output_dir,
    checkpoint_dir,
    method_commit,
    execution_commit,
    freeze_commit,
    start=0,
    end=None,
    workers=1,
    markets=TARGET_MARKETS,
    target_seeds=(0, 1, 2, 3, 4),
):
    cells = build_cells(markets=markets, target_seeds=target_seeds)
    start = max(0, int(start))
    stop = len(cells) if end is None else min(len(cells), int(end))
    if stop < start:
        raise ValueError("matrix end precedes start")
    selected = list(enumerate(cells[start:stop], start=start))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), len(selected) or 1))
    started = time.perf_counter()
    results = []
    if workers == 1:
        for completed, (index, cell) in enumerate(selected, start=1):
            results.append(_run_cell(
                index,
                cell,
                data_path=data_path,
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                method_commit=method_commit,
                execution_commit=execution_commit,
                freeze_commit=freeze_commit,
            ))
            print(
                f"ENERGY_FUNCTIONAL_SCBO_PROGRESS {completed}/{len(selected)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_cell,
                    index,
                    cell,
                    data_path=data_path,
                    output_dir=output_dir,
                    checkpoint_dir=checkpoint_dir,
                    method_commit=method_commit,
                    execution_commit=execution_commit,
                    freeze_commit=freeze_commit,
                ): index
                for index, cell in selected
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                print(
                    f"ENERGY_FUNCTIONAL_SCBO_PROGRESS {completed}/{len(selected)}",
                    flush=True,
                )
    error_count = int(sum(row["status"] == "error" for row in results))
    summary = {
        "schema_version": 1,
        "contract_id": "opsd_region_heldout_functional_scbo_matrix_v1",
        "status": (
            "complete" if error_count == 0 else "complete_with_cell_errors"),
        "functional_method_commit": str(method_commit),
        "execution_commit": str(execution_commit),
        "confirmatory_freeze_commit": str(freeze_commit),
        "matrix_cell_count": int(len(cells)),
        "shard_start": int(start),
        "shard_end": int(stop),
        "shard_cell_count": int(len(selected)),
        "workers": int(workers),
        "completed_count": int(sum(
            row["status"] == "done" for row in results)),
        "skipped_count": int(sum(
            row["status"] == "skipped" for row in results)),
        "error_count": error_count,
        "cell_errors": [
            {
                "index": int(row["index"]),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message"),
            }
            for row in results if row["status"] == "error"
        ],
        "wall_time_sec": float(time.perf_counter() - started),
    }
    _atomic_json(
        Path(output_dir) / f"shard_{start:04d}_{stop:04d}.summary.json",
        summary,
    )
    return summary


def _emit_terminal_status(summary):
    error_count = int(summary.get("error_count", 0))
    if error_count:
        print(
            f"ENERGY_FUNCTIONAL_SCBO_FAILED cell_errors={error_count}",
            flush=True,
        )
        return False
    print("DONE", flush=True)
    return True


def _csv(value, cast=str):
    return tuple(
        cast(item.strip()) for item in str(value).split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--method-commit", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--markets", default=",".join(TARGET_MARKETS))
    parser.add_argument("--target-seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    markets = _csv(args.markets)
    unknown = set(markets) - set(TARGET_MARKETS)
    if unknown:
        raise ValueError(f"unknown markets: {sorted(unknown)}")
    summary = run_matrix(
        data_path=args.data,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        method_commit=args.method_commit,
        execution_commit=args.execution_commit,
        freeze_commit=args.freeze_commit,
        start=args.start,
        end=args.end,
        workers=args.workers,
        markets=markets,
        target_seeds=_csv(args.target_seeds, int),
    )
    if not _emit_terminal_status(summary):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
