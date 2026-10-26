#!/usr/bin/env python3
"""Scheduler-only release acceptance; historical cohorts are read-only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_one(index, root, output, lean_root):
    label = ("saved_evidence", "lean_cold")[index]
    directory = output / label
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = dict(stage=label, status="running", automatic_retries=0,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  new_simulator_calls=0, steps=[])
    write(directory / "receipt.json", report)
    python = sys.executable
    if index == 0:
        commands = [
            [python, "-m", "reproducibility.mainline_release.export_cells",
             "--root", str(root), "--output", str(output / "cells.json.gz")],
            [python, "-m", "reproducibility.mainline_release.statistics",
             "--cells", str(output / "cells.json.gz"),
             "--output", str(output / "statistics.json.gz")],
        ]
    else:
        commands = [[python, str(root / "reproducibility/mainline_release/lean_delivery.py"),
                     "build", "--proof", str(lean_root / "proof"),
                     "--toolchain", str(lean_root / "toolchain"),
                     "--output", str(directory / "cold")]]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        for i, command in enumerate(commands):
            wall = time.monotonic()
            with (directory / f"step_{i}.log").open("w") as log:
                result = subprocess.run(command, cwd=root, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            report["steps"].append(dict(command=command, returncode=result.returncode,
                                        wall_seconds=time.monotonic() - wall))
            if result.returncode:
                report["status"] = "failed"
                return result.returncode
        report["status"] = "complete"
        return 0
    except Exception as error:
        report.update(status="failed", error=repr(error))
        return 1
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=time.monotonic() - started,
                      child_cpu_seconds=after.ru_utime + after.ru_stime -
                      before.ru_utime - before.ru_stime,
                      cpu_accounting="isolated item process children; includes nested steps",
                      child_peak_rss_kib=after.ru_maxrss)
        write(directory / "receipt.json", report)
        print(json.dumps(dict(stage=label, status=report["status"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lean-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "SC-OLH-KG"
    indices = list(range(args.start, args.end))
    with ProcessPoolExecutor(max_workers=min(args.workers, len(indices))) as pool:
        results = list(pool.map(partial(run_one, root=root, output=args.output,
                                        lean_root=args.lean_root), indices))
    raise SystemExit(int(any(results)))


if __name__ == "__main__":
    main()
