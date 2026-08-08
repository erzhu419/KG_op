#!/usr/bin/env python3
"""Restartable matrix runner for the frozen forecast-indexed energy domain."""

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

from performance.benchmark_external_energy_v2 import (  # noqa: E402
    TARGET_MARKETS,
    _atomic_json,
)
from performance.benchmark_external_energy_v3 import (  # noqa: E402
    ARMS,
    CONTRACT_ID,
    DESIGN_CONTRACT_ID,
    materialize_source_atlas,
    run_task,
)


MATRIX_CONTRACT_ID = "opsd_forecast_indexed_region_holdout_matrix_v3"


def design_filename(target_market):
    return f"source_atlas__target-{str(target_market)}.json"


def build_target_cells(
    *,
    markets=TARGET_MARKETS,
    target_seeds=(0, 1, 2, 3, 4),
    arms=ARMS,
):
    return [
        {
            "target_market": str(market),
            "target_seed": int(seed),
            "arm": str(arm),
        }
        for market in markets
        for seed in target_seeds
        for arm in arms
    ]


def _cell_filename(index, cell):
    return (
        f"cell{int(index):04d}__target-{cell['target_market']}__"
        f"seed-{int(cell['target_seed']):02d}__{cell['arm']}.json"
    )


def _valid_design(
    path,
    *,
    target_market,
    method_freeze_commit,
    execution_commit,
):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        row.get("contract_id") == DESIGN_CONTRACT_ID
        and row.get("status") == "frozen_before_target_outcomes"
        and row.get("target_market") == str(target_market)
        and row.get("method_freeze_commit") == str(method_freeze_commit)
        and row.get("execution_commit") == str(execution_commit)
    )


def _materialize_design(
    target_market,
    *,
    data_path,
    output_dir,
    method_freeze_commit,
    execution_commit,
    year,
    dimension,
    horizon,
    alpha,
    n0,
    library_size,
    source_replications,
    family_seed,
):
    output = Path(output_dir) / design_filename(target_market)
    if _valid_design(
        output,
        target_market=target_market,
        method_freeze_commit=method_freeze_commit,
        execution_commit=execution_commit,
    ):
        return {"target_market": target_market, "status": "skipped"}
    payload = materialize_source_atlas(
        data_path=data_path,
        target_market=target_market,
        year=year,
        dimension=dimension,
        horizon=horizon,
        alpha=alpha,
        n0=n0,
        library_size=library_size,
        source_replications=source_replications,
        family_seed=family_seed,
    )
    payload["method_freeze_commit"] = str(method_freeze_commit)
    payload["execution_commit"] = str(execution_commit)
    _atomic_json(output, payload)
    return {"target_market": target_market, "status": "done"}


def materialize_design_matrix(
    *,
    data_path,
    output_dir,
    method_freeze_commit,
    execution_commit,
    markets=TARGET_MARKETS,
    workers=1,
    year=2018,
    dimension=1000,
    horizon=168,
    alpha=0.05,
    n0=10,
    library_size=64,
    source_replications=3,
    family_seed=20260808,
):
    markets = tuple(str(value) for value in markets)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), len(markets) or 1))
    kwargs = {
        "data_path": data_path,
        "output_dir": output_dir,
        "method_freeze_commit": method_freeze_commit,
        "execution_commit": execution_commit,
        "year": year,
        "dimension": dimension,
        "horizon": horizon,
        "alpha": alpha,
        "n0": n0,
        "library_size": library_size,
        "source_replications": source_replications,
        "family_seed": family_seed,
    }
    results = []
    if workers == 1:
        for completed, market in enumerate(markets, start=1):
            results.append(_materialize_design(market, **kwargs))
            print(
                f"ENERGY_V3_DESIGN_PROGRESS {completed}/{len(markets)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_materialize_design, market, **kwargs): market
                for market in markets
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                print(
                    f"ENERGY_V3_DESIGN_PROGRESS {completed}/{len(markets)}",
                    flush=True,
                )
    summary = {
        "schema_version": 1,
        "contract_id": "opsd_forecast_indexed_design_matrix_v3",
        "status": "complete",
        "method_freeze_commit": str(method_freeze_commit),
        "execution_commit": str(execution_commit),
        "market_count": int(len(markets)),
        "completed_count": int(sum(row["status"] == "done" for row in results)),
        "skipped_count": int(sum(row["status"] == "skipped" for row in results)),
    }
    _atomic_json(Path(output_dir) / "design_matrix.summary.json", summary)
    return summary


def _valid_result(
    path,
    *,
    cell,
    method_freeze_commit,
    execution_commit,
):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        row.get("contract_id") == CONTRACT_ID
        and row.get("status") == "ok"
        and row.get("target_market") == cell["target_market"]
        and int(row.get("target_seed", -1)) == int(cell["target_seed"])
        and row.get("arm") == cell["arm"]
        and row.get("method_freeze_commit") == str(method_freeze_commit)
        and row.get("execution_commit") == str(execution_commit)
    )


def _run_target_cell(
    index,
    cell,
    *,
    data_path,
    design_dir,
    output_dir,
    checkpoint_dir,
    method_freeze_commit,
    execution_commit,
    run_arguments,
):
    output = Path(output_dir) / _cell_filename(index, cell)
    if _valid_result(
        output,
        cell=cell,
        method_freeze_commit=method_freeze_commit,
        execution_commit=execution_commit,
    ):
        return {"index": int(index), "status": "skipped", "out": str(output)}
    design_path = (
        Path(design_dir) / design_filename(cell["target_market"])
        if cell["arm"] == "source_atlas" else None
    )
    if design_path is not None and not design_path.is_file():
        raise FileNotFoundError(
            f"energy V3 target cell lacks frozen design: {design_path}")
    checkpoint = Path(checkpoint_dir) / f"cell{int(index):04d}.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        payload = run_task(
            data_path=data_path,
            target_market=cell["target_market"],
            target_seed=cell["target_seed"],
            arm=cell["arm"],
            design_path=design_path,
            checkpoint_path=(
                str(checkpoint)
                if cell["arm"] == "target_only_dct_space_scbo" else ""
            ),
            checkpoint_resume=bool(checkpoint.is_file()),
            **run_arguments,
        )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "contract_id": "opsd_forecast_indexed_cell_error_v3",
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "cell_index": int(index),
            "cell": dict(cell),
            "method_freeze_commit": str(method_freeze_commit),
            "execution_commit": str(execution_commit),
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
    payload["method_freeze_commit"] = str(method_freeze_commit)
    payload["execution_commit"] = str(execution_commit)
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


def run_target_matrix(
    *,
    data_path,
    design_dir,
    output_dir,
    checkpoint_dir,
    method_freeze_commit,
    execution_commit,
    start=0,
    end=None,
    workers=1,
    markets=TARGET_MARKETS,
    target_seeds=(0, 1, 2, 3, 4),
    arms=ARMS,
    **run_arguments,
):
    cells = build_target_cells(
        markets=markets, target_seeds=target_seeds, arms=arms)
    start = max(0, int(start))
    stop = len(cells) if end is None else min(len(cells), int(end))
    if stop < start:
        raise ValueError("matrix end precedes start")
    selected = list(enumerate(cells[start:stop], start=start))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), len(selected) or 1))
    started = time.perf_counter()
    common = {
        "data_path": data_path,
        "design_dir": design_dir,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "method_freeze_commit": method_freeze_commit,
        "execution_commit": execution_commit,
        "run_arguments": run_arguments,
    }
    results = []
    if workers == 1:
        for completed, (index, cell) in enumerate(selected, start=1):
            results.append(_run_target_cell(index, cell, **common))
            print(
                f"ENERGY_V3_TARGET_PROGRESS {completed}/{len(selected)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_target_cell, index, cell, **common): index
                for index, cell in selected
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                print(
                    f"ENERGY_V3_TARGET_PROGRESS {completed}/{len(selected)}",
                    flush=True,
                )
    error_count = int(sum(row["status"] == "error" for row in results))
    summary = {
        "schema_version": 1,
        "contract_id": MATRIX_CONTRACT_ID,
        "status": (
            "complete" if error_count == 0 else "complete_with_cell_errors"),
        "method_freeze_commit": str(method_freeze_commit),
        "execution_commit": str(execution_commit),
        "matrix_cell_count": int(len(cells)),
        "shard_start": int(start),
        "shard_end": int(stop),
        "shard_cell_count": int(len(selected)),
        "workers": int(workers),
        "completed_count": int(sum(row["status"] == "done" for row in results)),
        "skipped_count": int(sum(row["status"] == "skipped" for row in results)),
        "error_count": error_count,
        "wall_time_sec": float(time.perf_counter() - started),
    }
    _atomic_json(
        Path(output_dir) / f"shard_{start:04d}_{stop:04d}.summary.json",
        summary,
    )
    return summary


def _csv(value, cast=str):
    return tuple(
        cast(item.strip()) for item in str(value).split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("designs", "targets"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--design-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--method-freeze-commit", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--markets", default=",".join(TARGET_MARKETS))
    parser.add_argument("--target-seeds", default="0,1,2,3,4")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=168)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--library-size", type=int, default=64)
    parser.add_argument("--source-replications", type=int, default=3)
    parser.add_argument("--family-seed", type=int, default=20260808)
    parser.add_argument("--verification-budgets", default="80,80,80")
    args = parser.parse_args()
    markets = _csv(args.markets)
    if args.phase == "designs":
        summary = materialize_design_matrix(
            data_path=args.data,
            output_dir=args.design_dir,
            method_freeze_commit=args.method_freeze_commit,
            execution_commit=args.execution_commit,
            markets=markets,
            workers=args.workers,
            year=args.year,
            dimension=args.d,
            horizon=args.horizon,
            alpha=args.alpha,
            n0=args.n0,
            library_size=args.library_size,
            source_replications=args.source_replications,
            family_seed=args.family_seed,
        )
    else:
        summary = run_target_matrix(
            data_path=args.data,
            design_dir=args.design_dir,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            method_freeze_commit=args.method_freeze_commit,
            execution_commit=args.execution_commit,
            start=args.start,
            end=args.end,
            workers=args.workers,
            markets=markets,
            target_seeds=_csv(args.target_seeds, int),
            arms=_csv(args.arms),
            year=args.year,
            dimension=args.d,
            horizon=args.horizon,
            alpha=args.alpha,
            n0=args.n0,
            N=args.N,
            library_size=args.library_size,
            source_replications=args.source_replications,
            family_seed=args.family_seed,
            verification_budgets=_csv(args.verification_budgets, int),
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if int(summary.get("error_count", 0)) > 0:
        print("ENERGY_V3_FAILED", flush=True)
        raise SystemExit(1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
