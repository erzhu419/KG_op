#!/usr/bin/env python3
"""Parallel, restartable execution of a frozen profile-stress matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time


os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_profile_stress_suite import (  # noqa: E402
    ARMS,
    _atomic_json,
    run_task,
)
from problems.randomized_profiles import PROFILE_STRESS_REGIMES  # noqa: E402


def derived_target_seed(freeze_commit, regime, replicate):
    token = f"{freeze_commit}:{regime}:{int(replicate)}".encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16) % 1_000_000_000


def build_primary_cells(
    *,
    freeze_commit,
    dimensions=(200, 1000, 10000),
    task_count=20,
    arms=ARMS,
    regimes=tuple(PROFILE_STRESS_REGIMES),
):
    cells = []
    for dimension in map(int, dimensions):
        for regime in regimes:
            for replicate in range(int(task_count)):
                seed = derived_target_seed(freeze_commit, regime, replicate)
                for arm in arms:
                    cells.append({
                        "regime": regime,
                        "target_seed": seed,
                        "replicate_index": replicate,
                        "arm": arm,
                        "dimension": dimension,
                        "N": 10,
                        "schema_mode": "declared",
                        "descriptor_mode": "domain_blind",
                        "configuration_id": "primary",
                    })
    return cells


def _paired_task_cells(
    *,
    freeze_commit,
    configurations,
    dimension=1000,
    task_count=20,
    arms=("source_atlas", "generic_dct_maximin"),
    regimes=tuple(PROFILE_STRESS_REGIMES),
):
    cells = []
    for configuration_id, overrides in configurations:
        for regime in regimes:
            for replicate in range(int(task_count)):
                seed = derived_target_seed(freeze_commit, regime, replicate)
                for arm in arms:
                    cells.append({
                        "regime": regime,
                        "target_seed": seed,
                        "replicate_index": replicate,
                        "arm": arm,
                        "dimension": int(dimension),
                        "N": int(overrides.get("N", overrides.get("n0", 10))),
                        "schema_mode": str(
                            overrides.get("schema_mode", "declared")),
                        "descriptor_mode": str(
                            overrides.get("descriptor_mode", "domain_blind")),
                        "configuration_id": str(configuration_id),
                        **{
                            key: value for key, value in overrides.items()
                            if key not in {"schema_mode", "descriptor_mode"}
                        },
                    })
    return cells


def build_sensitivity_cells(
    *, freeze_commit, dimension=1000, task_count=20,
    regimes=tuple(PROFILE_STRESS_REGIMES),
):
    """Registered one-factor-at-a-time atlas sensitivity matrix."""

    configurations = [("baseline", {})]
    axes = {
        "active_rank": (4, 8, 16, 32),
        "safe_mass": (0.03, 0.18),
        "alpha": (0.01, 0.10),
        "n0": (5, 20),
        "source_task_count": (1, 4),
        "library_size": (32, 128),
        "source_replications": (1, 5),
        "atlas_max_frequency": (4, 16),
        "atlas_frequency_penalty": (0.0, 1.0),
        "atlas_safety_metric_weight": (0.25, 4.0),
        "atlas_objective_metric_weight": (0.25, 4.0),
        "atlas_first_center_safety_weight": (0.25, 0.75),
    }
    for name, values in axes.items():
        for value in values:
            configurations.append((f"{name}-{value}", {name: value}))
    return _paired_task_cells(
        freeze_commit=freeze_commit,
        configurations=configurations,
        dimension=dimension,
        task_count=task_count,
        regimes=regimes,
    )


def build_equal_preverification_cost_cells(
    *, freeze_commit, dimension=1000, task_count=20,
    regimes=tuple(PROFILE_STRESS_REGIMES),
):
    """Match 384 source plus 10 target calls with 394 target-only calls."""

    cells = []
    for regime in regimes:
        for replicate in range(int(task_count)):
            seed = derived_target_seed(freeze_commit, regime, replicate)
            cells.append({
                "regime": regime,
                "target_seed": seed,
                "replicate_index": replicate,
                "arm": "source_atlas",
                "dimension": int(dimension),
                "n0": 10,
                "N": 10,
                "schema_mode": "declared",
                "descriptor_mode": "domain_blind",
                "configuration_id": "source384-target10",
            })
            for arm in (
                "generic_dct_maximin", "random_low_frequency",
                "natural_blockwise", "raw_sobol",
            ):
                cells.append({
                    "regime": regime,
                    "target_seed": seed,
                    "replicate_index": replicate,
                    "arm": arm,
                    "dimension": int(dimension),
                    "n0": 10,
                    "N": 394,
                    "schema_mode": "declared",
                    "descriptor_mode": "domain_blind",
                    "configuration_id": "target394",
                })
    return cells


def build_schema_descriptor_cells(
    *, freeze_commit, dimension=1000, task_count=20,
    regimes=tuple(PROFILE_STRESS_REGIMES),
):
    """Cross declared/blind schemas with conditioned/blind descriptors."""

    configurations = []
    for schema_mode in ("declared", "schema_blind"):
        for descriptor_mode in ("domain_blind", "conditioned"):
            configurations.append((
                f"schema-{schema_mode}__descriptor-{descriptor_mode}",
                {
                    "schema_mode": schema_mode,
                    "descriptor_mode": descriptor_mode,
                },
            ))
    return _paired_task_cells(
        freeze_commit=freeze_commit,
        configurations=configurations,
        dimension=dimension,
        task_count=task_count,
        regimes=regimes,
    )


def _cell_name(index, cell):
    return (
        f"cell{int(index):05d}__d{int(cell['dimension'])}__"
        f"{cell['regime']}__r{int(cell['replicate_index']):03d}__"
        f"{cell['arm']}__schema-{cell['schema_mode']}__"
        f"descriptor-{cell['descriptor_mode']}__"
        f"config-{cell.get('configuration_id', 'primary')}.json"
    )


def _existing_ok(path, *, freeze_commit, cell):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        payload.get("contract_id") == "randomized_ordered_profile_stress_v2"
        and payload.get("status") == "ok"
        and payload.get("confirmatory_freeze_commit") == str(freeze_commit)
        and payload.get("matrix_configuration_id")
            == str(cell.get("configuration_id", "primary"))
    )


def _run_cell(index, cell, output_dir, freeze_commit):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    output = Path(output_dir) / _cell_name(index, cell)
    if _existing_ok(output, freeze_commit=freeze_commit, cell=cell):
        return {"index": index, "status": "skipped", "out": str(output)}
    started = time.perf_counter()
    run_arguments = {
        "regime": cell["regime"],
        "target_seed": cell["target_seed"],
        "arm": cell["arm"],
        "dimension": cell["dimension"],
        "schema_mode": cell["schema_mode"],
        "descriptor_mode": cell["descriptor_mode"],
    }
    for name in (
        "active_rank", "alpha", "safe_mass", "n0", "N",
        "source_task_count", "library_size", "source_replications",
        "atlas_max_frequency", "atlas_frequency_penalty",
        "atlas_safety_metric_weight", "atlas_objective_metric_weight",
        "atlas_first_center_safety_weight",
    ):
        if name in cell:
            run_arguments[name] = cell[name]
    try:
        payload = run_task(**run_arguments)
    except Exception as error:
        payload = {
            "schema_version": 1,
            "contract_id": "randomized_ordered_profile_stress_cell_error_v1",
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "cell_index": int(index),
            "cell": dict(cell),
            "confirmatory_freeze_commit": str(freeze_commit),
            "confirmatory_replicate_index": int(cell["replicate_index"]),
            "matrix_configuration_id": str(
                cell.get("configuration_id", "primary")),
            "wall_time_sec": float(time.perf_counter() - started),
        }
        _atomic_json(output, payload)
        return {
            "index": index,
            "status": "error",
            "out": str(output),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "wall_time_sec": payload["wall_time_sec"],
        }
    payload["confirmatory_freeze_commit"] = str(freeze_commit)
    payload["confirmatory_replicate_index"] = int(cell["replicate_index"])
    payload["matrix_configuration_id"] = str(
        cell.get("configuration_id", "primary"))
    payload["wall_time_sec"] = float(time.perf_counter() - started)
    _atomic_json(output, payload)
    return {
        "index": index,
        "status": "done",
        "out": str(output),
        "wall_time_sec": payload["wall_time_sec"],
    }


def run_matrix(
    *,
    output_dir,
    freeze_commit,
    start=0,
    end=None,
    workers=1,
    dimensions=(200, 1000, 10000),
    task_count=20,
    arms=ARMS,
    regimes=tuple(PROFILE_STRESS_REGIMES),
    matrix="primary",
):
    matrix = str(matrix)
    if matrix == "primary":
        cells = build_primary_cells(
            freeze_commit=freeze_commit,
            dimensions=dimensions,
            task_count=task_count,
            arms=arms,
            regimes=regimes,
        )
    elif matrix == "sensitivity":
        if len(tuple(dimensions)) != 1:
            raise ValueError("sensitivity matrix requires one dimension")
        cells = build_sensitivity_cells(
            freeze_commit=freeze_commit,
            dimension=tuple(dimensions)[0],
            task_count=task_count,
            regimes=regimes,
        )
    elif matrix == "equal_preverification_cost":
        if len(tuple(dimensions)) != 1:
            raise ValueError("equal-cost matrix requires one dimension")
        cells = build_equal_preverification_cost_cells(
            freeze_commit=freeze_commit,
            dimension=tuple(dimensions)[0],
            task_count=task_count,
            regimes=regimes,
        )
    elif matrix == "schema_descriptor":
        if len(tuple(dimensions)) != 1:
            raise ValueError("schema matrix requires one dimension")
        cells = build_schema_descriptor_cells(
            freeze_commit=freeze_commit,
            dimension=tuple(dimensions)[0],
            task_count=task_count,
            regimes=regimes,
        )
    else:
        raise ValueError(f"unknown profile stress matrix: {matrix}")
    start = max(0, int(start))
    stop = len(cells) if end is None else min(len(cells), int(end))
    if stop < start:
        raise ValueError("matrix end precedes start")
    selected = list(enumerate(cells[start:stop], start=start))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), len(selected) or 1))
    results = []
    started = time.perf_counter()
    if workers == 1:
        for completed, (index, cell) in enumerate(selected, start=1):
            results.append(_run_cell(index, cell, output_dir, freeze_commit))
            print(
                f"PROFILE_STRESS_PROGRESS {completed}/{len(selected)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_cell, index, cell, output_dir, freeze_commit
                ): index
                for index, cell in selected
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                print(
                    f"PROFILE_STRESS_PROGRESS {completed}/{len(selected)}",
                    flush=True,
                )
    error_count = int(sum(row["status"] == "error" for row in results))
    summary = {
        "schema_version": 1,
        "contract_id": "randomized_ordered_profile_stress_matrix_v2",
        "status": (
            "complete" if error_count == 0 else "complete_with_cell_errors"),
        "freeze_commit": str(freeze_commit),
        "matrix": matrix,
        "matrix_cell_count": len(cells),
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
                "index": int(row["index"]),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message"),
            }
            for row in results if row["status"] == "error"
        ],
        "wall_time_sec": float(time.perf_counter() - started),
    }
    summary_path = Path(output_dir) / f"shard_{start:05d}_{stop:05d}.summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    print("DONE", flush=True)
    return summary


def _csv(value, cast=str):
    return tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--matrix",
        choices=(
            "primary", "sensitivity", "equal_preverification_cost",
            "schema_descriptor",
        ),
        default="primary",
    )
    parser.add_argument("--dimensions", default="200,1000,10000")
    parser.add_argument("--task-count", type=int, default=20)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument(
        "--regimes", default=",".join(PROFILE_STRESS_REGIMES))
    args = parser.parse_args()
    unknown_arms = set(_csv(args.arms)) - set(ARMS)
    unknown_regimes = set(_csv(args.regimes)) - set(PROFILE_STRESS_REGIMES)
    if unknown_arms or unknown_regimes:
        raise ValueError(
            f"unknown arms/regimes: {sorted(unknown_arms)}, "
            f"{sorted(unknown_regimes)}")
    run_matrix(
        output_dir=args.output_dir,
        freeze_commit=args.freeze_commit,
        start=args.start,
        end=args.end,
        workers=args.workers,
        dimensions=_csv(args.dimensions, int),
        task_count=args.task_count,
        arms=_csv(args.arms),
        regimes=_csv(args.regimes),
        matrix=args.matrix,
    )


if __name__ == "__main__":
    main()
