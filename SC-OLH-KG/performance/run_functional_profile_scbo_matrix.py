#!/usr/bin/env python3
"""Parallel, restartable matrix for target-only functional-profile SCBO."""

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

from performance.benchmark_functional_profile_scbo import (  # noqa: E402
    CONTRACT_ID,
    run_task,
)
from performance.benchmark_profile_stress_suite import _atomic_json  # noqa: E402
from performance.run_profile_stress_matrix import (  # noqa: E402
    derived_design_seed,
    derived_target_seed,
)
from problems.randomized_profiles import PROFILE_STRESS_REGIMES  # noqa: E402


def _base_cells(
    *,
    task_freeze_commit,
    dimensions,
    task_count,
    regimes,
    configurations,
):
    cells = []
    for dimension in map(int, dimensions):
        for regime in regimes:
            for replicate in range(int(task_count)):
                target_seed = derived_target_seed(
                    task_freeze_commit, regime, replicate)
                design_seed = derived_design_seed(
                    task_freeze_commit, regime, replicate)
                for configuration_id, overrides in configurations:
                    cells.append({
                        "regime": str(regime),
                        "target_seed": int(target_seed),
                        "design_seed": int(design_seed),
                        "replicate_index": int(replicate),
                        "dimension": int(dimension),
                        "n0": int(overrides.get("n0", 10)),
                        "N": int(overrides.get("N", 13)),
                        "coefficient_count": int(
                            overrides.get("coefficient_count", 8)),
                        "schema_mode": str(
                            overrides.get("schema_mode", "declared")),
                        "configuration_id": str(configuration_id),
                    })
    return cells


def build_primary_cells(
    *, task_freeze_commit, dimensions=(200, 1000, 10000),
    task_count=20, regimes=tuple(PROFILE_STRESS_REGIMES),
):
    return _base_cells(
        task_freeze_commit=task_freeze_commit,
        dimensions=dimensions,
        task_count=task_count,
        regimes=regimes,
        configurations=(("target13-k8", {
            "N": 13, "coefficient_count": 8,
        }),),
    )


def build_rank_sensitivity_cells(
    *, task_freeze_commit, dimension=1000, task_count=20,
    regimes=tuple(PROFILE_STRESS_REGIMES),
):
    return _base_cells(
        task_freeze_commit=task_freeze_commit,
        dimensions=(int(dimension),),
        task_count=task_count,
        regimes=regimes,
        configurations=tuple(
            (f"target13-k{rank}", {
                "N": 13, "coefficient_count": rank,
            })
            for rank in (4, 8, 16)
        ),
    )


def build_equal_preverification_cost_cells(
    *, task_freeze_commit, dimension=1000, task_count=20,
    regimes=tuple(PROFILE_STRESS_REGIMES),
):
    """Match the profile-front-end audit's 384 source plus 10 target calls."""

    return _base_cells(
        task_freeze_commit=task_freeze_commit,
        dimensions=(int(dimension),),
        task_count=task_count,
        regimes=regimes,
        configurations=(("target394-k8", {
            "N": 394, "coefficient_count": 8,
        }),),
    )


def _cell_name(index, cell):
    return (
        f"cell{int(index):05d}__d{int(cell['dimension'])}__"
        f"{cell['regime']}__r{int(cell['replicate_index']):03d}__"
        f"K{int(cell['coefficient_count'])}__N{int(cell['N'])}__"
        f"config-{cell['configuration_id']}.json"
    )


def _existing_ok(
    path, *, task_freeze_commit, functional_method_commit, cell,
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
        and payload.get("task_freeze_commit") == str(task_freeze_commit)
        and payload.get("functional_method_commit")
            == str(functional_method_commit)
        and payload.get("matrix_configuration_id")
            == str(cell["configuration_id"])
        and int(payload.get("design_seed", -1)) == int(cell["design_seed"])
    )


def _run_cell(
    index,
    cell,
    output_dir,
    task_freeze_commit,
    functional_method_commit,
    checkpoint_dir,
):
    output = Path(output_dir) / _cell_name(index, cell)
    if _existing_ok(
        output,
        task_freeze_commit=task_freeze_commit,
        functional_method_commit=functional_method_commit,
        cell=cell,
    ):
        return {"index": int(index), "status": "skipped", "out": str(output)}
    checkpoint = Path(checkpoint_dir) / f"cell{int(index):05d}.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        payload = run_task(
            regime=cell["regime"],
            target_seed=cell["target_seed"],
            design_seed=cell["design_seed"],
            dimension=cell["dimension"],
            n0=cell["n0"],
            N=cell["N"],
            coefficient_count=cell["coefficient_count"],
            schema_mode=cell["schema_mode"],
            checkpoint_path=str(checkpoint),
            checkpoint_resume=checkpoint.is_file(),
            progress_logging=bool(int(cell["N"]) >= 100),
        )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "contract_id": "target_only_functional_profile_scbo_cell_error_v1",
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "cell_index": int(index),
            "cell": dict(cell),
            "task_freeze_commit": str(task_freeze_commit),
            "functional_method_commit": str(functional_method_commit),
            "matrix_configuration_id": str(cell["configuration_id"]),
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
    payload["task_freeze_commit"] = str(task_freeze_commit)
    payload["functional_method_commit"] = str(functional_method_commit)
    payload["confirmatory_replicate_index"] = int(cell["replicate_index"])
    payload["matrix_configuration_id"] = str(cell["configuration_id"])
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


def _build_cells(
    matrix, *, task_freeze_commit, dimensions, task_count, regimes,
):
    if matrix == "primary":
        return build_primary_cells(
            task_freeze_commit=task_freeze_commit,
            dimensions=dimensions,
            task_count=task_count,
            regimes=regimes,
        )
    if len(tuple(dimensions)) != 1:
        raise ValueError(f"{matrix} requires exactly one dimension")
    if matrix == "rank_sensitivity":
        return build_rank_sensitivity_cells(
            task_freeze_commit=task_freeze_commit,
            dimension=tuple(dimensions)[0],
            task_count=task_count,
            regimes=regimes,
        )
    if matrix == "equal_preverification_cost":
        return build_equal_preverification_cost_cells(
            task_freeze_commit=task_freeze_commit,
            dimension=tuple(dimensions)[0],
            task_count=task_count,
            regimes=regimes,
        )
    raise ValueError(f"unknown functional SCBO matrix: {matrix}")


def run_matrix(
    *,
    output_dir,
    checkpoint_dir,
    task_freeze_commit,
    functional_method_commit,
    matrix="primary",
    start=0,
    end=None,
    workers=1,
    dimensions=(200, 1000, 10000),
    task_count=20,
    regimes=tuple(PROFILE_STRESS_REGIMES),
):
    cells = _build_cells(
        matrix,
        task_freeze_commit=task_freeze_commit,
        dimensions=tuple(dimensions),
        task_count=task_count,
        regimes=tuple(regimes),
    )
    start = max(0, int(start))
    stop = len(cells) if end is None else min(len(cells), int(end))
    if stop < start:
        raise ValueError("matrix end precedes start")
    selected = list(enumerate(cells[start:stop], start=start))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), len(selected) or 1))
    results = []
    started = time.perf_counter()
    if workers == 1:
        for completed, (index, cell) in enumerate(selected, start=1):
            results.append(_run_cell(
                index,
                cell,
                output_dir,
                task_freeze_commit,
                functional_method_commit,
                checkpoint_dir,
            ))
            print(
                f"FUNCTIONAL_SCBO_PROGRESS {completed}/{len(selected)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_cell,
                    index,
                    cell,
                    output_dir,
                    task_freeze_commit,
                    functional_method_commit,
                    checkpoint_dir,
                ): index
                for index, cell in selected
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                print(
                    f"FUNCTIONAL_SCBO_PROGRESS {completed}/{len(selected)}",
                    flush=True,
                )
    error_count = int(sum(row["status"] == "error" for row in results))
    summary = {
        "schema_version": 1,
        "contract_id": "target_only_functional_profile_scbo_matrix_v1",
        "status": (
            "complete" if error_count == 0 else "complete_with_cell_errors"),
        "task_freeze_commit": str(task_freeze_commit),
        "functional_method_commit": str(functional_method_commit),
        "matrix": str(matrix),
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
    summary_path = (
        Path(output_dir) / f"shard_{start:05d}_{stop:05d}.summary.json")
    _atomic_json(summary_path, summary)
    print("DONE", flush=True)
    return summary


def _csv(value, cast=str):
    return tuple(
        cast(item.strip()) for item in str(value).split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--task-freeze-commit", required=True)
    parser.add_argument("--functional-method-commit", required=True)
    parser.add_argument(
        "--matrix",
        choices=("primary", "rank_sensitivity", "equal_preverification_cost"),
        default="primary",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dimensions", default="200,1000,10000")
    parser.add_argument("--task-count", type=int, default=20)
    parser.add_argument(
        "--regimes", default=",".join(PROFILE_STRESS_REGIMES))
    args = parser.parse_args()
    regimes = _csv(args.regimes)
    unknown = set(regimes) - set(PROFILE_STRESS_REGIMES)
    if unknown:
        raise ValueError(f"unknown regimes: {sorted(unknown)}")
    run_matrix(
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        task_freeze_commit=args.task_freeze_commit,
        functional_method_commit=args.functional_method_commit,
        matrix=args.matrix,
        start=args.start,
        end=args.end,
        workers=args.workers,
        dimensions=_csv(args.dimensions, int),
        task_count=args.task_count,
        regimes=regimes,
    )


if __name__ == "__main__":
    main()
