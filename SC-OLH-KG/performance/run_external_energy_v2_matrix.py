#!/usr/bin/env python3
"""Parallel, restartable execution of the frozen OPSD energy V2 matrix."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.benchmark_external_energy_v2 import (  # noqa: E402
    ARMS,
    TARGET_MARKETS,
    _atomic_json,
    materialize_source_atlas,
    run_task,
)


MATRIX_CONTRACT_ID = "opsd_region_heldout_profile_design_matrix_v2"
DESIGN_CONTRACT_ID = "opsd_region_heldout_source_atlas_design_v2"


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


def design_filename(target_market):
    return f"source_atlas__target-{str(target_market)}.json"


def target_filename(index, cell):
    return (
        f"cell{int(index):04d}__target-{cell['target_market']}__"
        f"seed-{int(cell['target_seed']):02d}__{cell['arm']}.json"
    )


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _existing_design_ok(path, *, freeze_commit, target_market):
    payload = _read_json(path)
    return bool(
        payload
        and payload.get("contract_id") == DESIGN_CONTRACT_ID
        and payload.get("status") == "frozen_before_target_outcomes"
        and payload.get("confirmatory_freeze_commit") == str(freeze_commit)
        and payload.get("target_market") == str(target_market)
        and payload.get("target_outcomes_used") is False
        and payload.get("target_oracle_used") is False
    )


def _materialize_design(
    target_market,
    *,
    data_path,
    output_dir,
    freeze_commit,
    run_arguments,
):
    output = Path(output_dir) / design_filename(target_market)
    if _existing_design_ok(
        output,
        freeze_commit=freeze_commit,
        target_market=target_market,
    ):
        return {"status": "skipped", "out": str(output)}
    started = time.perf_counter()
    payload = materialize_source_atlas(
        data_path=data_path,
        target_market=target_market,
        **run_arguments,
    )
    payload["confirmatory_freeze_commit"] = str(freeze_commit)
    payload["wall_time_sec"] = float(time.perf_counter() - started)
    _atomic_json(output, payload)
    return {
        "status": "done",
        "out": str(output),
        "wall_time_sec": payload["wall_time_sec"],
    }


def materialize_design_matrix(
    *,
    data_path,
    output_dir,
    freeze_commit,
    markets=TARGET_MARKETS,
    workers=1,
    year=2018,
    dimension=1000,
    alpha=0.05,
    n0=10,
    library_size=64,
    source_replications=3,
    family_seed=20260808,
):
    markets = tuple(map(str, markets))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), len(markets) or 1))
    run_arguments = {
        "year": int(year),
        "dimension": int(dimension),
        "alpha": float(alpha),
        "n0": int(n0),
        "library_size": int(library_size),
        "source_replications": int(source_replications),
        "family_seed": int(family_seed),
    }
    started = time.perf_counter()
    results = []
    if workers == 1:
        for completed, market in enumerate(markets, start=1):
            results.append(_materialize_design(
                market,
                data_path=data_path,
                output_dir=output_dir,
                freeze_commit=freeze_commit,
                run_arguments=run_arguments,
            ))
            print(f"ENERGY_DESIGN_PROGRESS {completed}/{len(markets)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _materialize_design,
                    market,
                    data_path=data_path,
                    output_dir=output_dir,
                    freeze_commit=freeze_commit,
                    run_arguments=run_arguments,
                ): market
                for market in markets
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                print(
                    f"ENERGY_DESIGN_PROGRESS {completed}/{len(markets)}",
                    flush=True,
                )
    summary = {
        "schema_version": 1,
        "contract_id": "opsd_region_heldout_source_design_matrix_v2",
        "status": "complete",
        "freeze_commit": str(freeze_commit),
        "market_count": len(markets),
        "workers": workers,
        "completed_count": sum(row["status"] == "done" for row in results),
        "skipped_count": sum(row["status"] == "skipped" for row in results),
        "wall_time_sec": float(time.perf_counter() - started),
    }
    _atomic_json(Path(output_dir) / "design_matrix.summary.json", summary)
    print("DONE", flush=True)
    return summary


def _existing_target_ok(path, *, freeze_commit, cell):
    payload = _read_json(path)
    return bool(
        payload
        and payload.get("contract_id")
            == "opsd_region_heldout_profile_design_v2"
        and payload.get("status") == "ok"
        and payload.get("confirmatory_freeze_commit") == str(freeze_commit)
        and payload.get("target_market") == cell["target_market"]
        and int(payload.get("target_seed", -1)) == int(cell["target_seed"])
        and payload.get("arm") == cell["arm"]
    )


def _run_target_cell(
    index,
    cell,
    *,
    data_path,
    design_dir,
    output_dir,
    freeze_commit,
    run_arguments,
):
    output = Path(output_dir) / target_filename(index, cell)
    if _existing_target_ok(output, freeze_commit=freeze_commit, cell=cell):
        return {"index": index, "status": "skipped", "out": str(output)}
    design_path = None
    if cell["arm"] == "source_atlas":
        design_path = Path(design_dir) / design_filename(cell["target_market"])
        if not _existing_design_ok(
            design_path,
            freeze_commit=freeze_commit,
            target_market=cell["target_market"],
        ):
            raise RuntimeError(
                "missing or mismatched frozen source design for "
                f"{cell['target_market']}: {design_path}"
            )
    started = time.perf_counter()
    payload = run_task(
        data_path=data_path,
        target_market=cell["target_market"],
        target_seed=cell["target_seed"],
        arm=cell["arm"],
        design_path=design_path,
        **run_arguments,
    )
    payload["confirmatory_freeze_commit"] = str(freeze_commit)
    payload["confirmatory_cell_index"] = int(index)
    payload["wall_time_sec"] = float(time.perf_counter() - started)
    _atomic_json(output, payload)
    return {
        "index": index,
        "status": "done",
        "out": str(output),
        "wall_time_sec": payload["wall_time_sec"],
    }


def run_target_matrix(
    *,
    data_path,
    design_dir,
    output_dir,
    freeze_commit,
    start=0,
    end=None,
    workers=1,
    markets=TARGET_MARKETS,
    target_seeds=(0, 1, 2, 3, 4),
    arms=ARMS,
    year=2018,
    dimension=1000,
    alpha=0.05,
    n0=10,
    N=13,
    library_size=64,
    source_replications=3,
    family_seed=20260808,
    verification_budgets=(80, 80, 80),
    familywise_delta=0.05,
    amortization_targets=20,
):
    cells = build_target_cells(
        markets=markets,
        target_seeds=target_seeds,
        arms=arms,
    )
    start = max(0, int(start))
    stop = len(cells) if end is None else min(len(cells), int(end))
    if stop < start:
        raise ValueError("matrix end precedes start")
    selected = list(enumerate(cells[start:stop], start=start))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), len(selected) or 1))
    run_arguments = {
        "year": int(year),
        "dimension": int(dimension),
        "alpha": float(alpha),
        "n0": int(n0),
        "N": int(N),
        "library_size": int(library_size),
        "source_replications": int(source_replications),
        "family_seed": int(family_seed),
        "verification_budgets": tuple(map(int, verification_budgets)),
        "familywise_delta": float(familywise_delta),
        "amortization_targets": int(amortization_targets),
    }
    results = []
    started = time.perf_counter()
    if workers == 1:
        for completed, (index, cell) in enumerate(selected, start=1):
            results.append(_run_target_cell(
                index,
                cell,
                data_path=data_path,
                design_dir=design_dir,
                output_dir=output_dir,
                freeze_commit=freeze_commit,
                run_arguments=run_arguments,
            ))
            print(
                f"ENERGY_TARGET_PROGRESS {completed}/{len(selected)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_target_cell,
                    index,
                    cell,
                    data_path=data_path,
                    design_dir=design_dir,
                    output_dir=output_dir,
                    freeze_commit=freeze_commit,
                    run_arguments=run_arguments,
                ): index
                for index, cell in selected
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                print(
                    f"ENERGY_TARGET_PROGRESS {completed}/{len(selected)}",
                    flush=True,
                )
    summary = {
        "schema_version": 1,
        "contract_id": MATRIX_CONTRACT_ID,
        "status": "complete",
        "freeze_commit": str(freeze_commit),
        "matrix_cell_count": len(cells),
        "shard_start": start,
        "shard_end": stop,
        "shard_cell_count": len(selected),
        "workers": workers,
        "completed_count": sum(row["status"] == "done" for row in results),
        "skipped_count": sum(row["status"] == "skipped" for row in results),
        "wall_time_sec": float(time.perf_counter() - started),
    }
    _atomic_json(
        Path(output_dir) / f"shard_{start:04d}_{stop:04d}.summary.json",
        summary,
    )
    print("DONE", flush=True)
    return summary


def _csv(value, cast=str):
    return tuple(
        cast(item.strip()) for item in str(value).split(",") if item.strip()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("designs", "targets"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--design-dir", required=True)
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--markets", default=",".join(TARGET_MARKETS))
    parser.add_argument("--target-seeds", default="0,1,2,3,4")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--N", type=int, default=13)
    parser.add_argument("--library-size", type=int, default=64)
    parser.add_argument("--source-replications", type=int, default=3)
    parser.add_argument("--family-seed", type=int, default=20260808)
    parser.add_argument("--verification-budgets", default="80,80,80")
    parser.add_argument("--familywise-delta", type=float, default=0.05)
    parser.add_argument("--amortization-targets", type=int, default=20)
    args = parser.parse_args()
    markets = _csv(args.markets)
    unknown_markets = set(markets) - set(TARGET_MARKETS)
    unknown_arms = set(_csv(args.arms)) - set(ARMS)
    if unknown_markets or unknown_arms:
        raise ValueError(
            f"unknown markets/arms: {sorted(unknown_markets)}, "
            f"{sorted(unknown_arms)}"
        )
    common = {
        "data_path": args.data,
        "freeze_commit": args.freeze_commit,
        "markets": markets,
        "workers": args.workers,
        "year": args.year,
        "dimension": args.d,
        "alpha": args.alpha,
        "n0": args.n0,
        "library_size": args.library_size,
        "source_replications": args.source_replications,
        "family_seed": args.family_seed,
    }
    if args.phase == "designs":
        materialize_design_matrix(
            output_dir=args.design_dir,
            **common,
        )
    else:
        run_target_matrix(
            design_dir=args.design_dir,
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
            target_seeds=_csv(args.target_seeds, int),
            arms=_csv(args.arms),
            N=args.N,
            verification_budgets=_csv(args.verification_budgets, int),
            familywise_delta=args.familywise_delta,
            amortization_targets=args.amortization_targets,
            **common,
        )


if __name__ == "__main__":
    main()
