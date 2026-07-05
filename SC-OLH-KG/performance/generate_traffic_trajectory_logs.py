"""Generate fresh-seed traffic trajectory CSV logs on the SUMO server.

This runner is intentionally separate from optimization.  It takes final
recommendations (or explicit signal vectors), reruns SUMO with fresh seeds, and
asks `sumo_sim.simulate` to append finite state-action occupancy/exposure rows
matching `TrafficTrajectoryEncoder`'s CSV contract.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
GPR_KG_CODE = REPO_ROOT / "Final_Submission" / "GPR_KG_Code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GPR_KG_CODE))

from benchmark_quality import json_safe, parse_csv  # noqa: E402
from encoders.policy_state_encoder import TrafficTrajectoryEncoder  # noqa: E402
from experiments.ingolstadt21.ingolstadt21_problem import Ingolstadt21Problem  # noqa: E402


def _safe_name(text):
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_")


def _parse_x(text):
    vals = [int(float(v)) for v in str(text).replace(",", " ").split() if v.strip()]
    if not vals:
        raise ValueError("empty x vector")
    return tuple(vals)


def _candidate_from_summary(path, source_indexes):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("final_pareto_set") or []
    method = data.get("method", "unknown")
    partition = data.get("partition_method", "unknown")
    run_seed = data.get("seed", "unknown")
    out = []
    for idx in source_indexes:
        if idx < 0 or idx >= len(rows):
            continue
        x = tuple(int(v) for v in rows[idx])
        out.append({
            "policy_id": _safe_name(f"{method}_{partition}_seed{run_seed}_src{idx}"),
            "method": method,
            "partition": partition,
            "run_seed": run_seed,
            "source_index": idx,
            "x": x,
            "summary_path": str(path),
        })
    return out


def _load_candidates(args):
    candidates = []
    source_indexes = parse_csv(args.source_indexes, int) or [0]
    for text in parse_csv(args.x):
        x = _parse_x(text)
        candidates.append({
            "policy_id": _safe_name(f"explicit_{len(candidates)}"),
            "method": "explicit",
            "partition": "manual",
            "run_seed": "",
            "source_index": 0,
            "x": x,
            "summary_path": "",
        })
    for pattern in parse_csv(args.summary_glob):
        for path in sorted(glob.glob(pattern)):
            candidates.extend(_candidate_from_summary(path, source_indexes))
    if args.max_policies is not None:
        candidates = candidates[: int(args.max_policies)]
    seen = set()
    unique = []
    for row in candidates:
        key = tuple(row["x"])
        if key in seen and args.dedupe:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _load_candidates_with_wait(args):
    deadline = time.time() + max(0.0, float(getattr(args, "wait_for_data_sec", 0.0)))
    interval = max(1.0, float(getattr(args, "wait_interval_sec", 30.0)))
    while True:
        candidates = _load_candidates(args)
        if candidates or time.time() >= deadline:
            return candidates
        print(
            "[traffic-trajectory] no candidate summaries yet; "
            f"sleeping {interval:.1f}s",
            flush=True,
        )
        time.sleep(interval)


def run(args):
    problem = Ingolstadt21Problem()
    candidates = _load_candidates_with_wait(args)
    if not candidates:
        return {
            "schema_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "missing_data",
            "reason": "no candidate summaries or explicit x vectors found",
            "trajectory_log": args.out_csv,
        }
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not args.resume:
        out_csv.unlink()
    rows = []
    t0 = time.time()
    for cidx, candidate in enumerate(candidates):
        x = np.asarray(candidate["x"], dtype=float)
        for rep in range(int(args.R)):
            seed = int(args.seed_start) + cidx * 100000 + rep
            policy_id = f"{candidate['policy_id']}_fresh{rep}"
            y = problem._sim.simulate(
                var_map=problem.var_map,
                x=x,
                route_file=problem._route_files[0],
                T0=problem.T0,
                A0=problem.A0,
                E0=problem.E0,
                seed=seed,
                trajectory_log_path=str(out_csv),
                policy_id=policy_id,
                trajectory_interval=args.trajectory_interval,
            )
            rows.append({
                "candidate_index": cidx,
                "policy_id": policy_id,
                "seed": seed,
                "method": candidate["method"],
                "partition": candidate["partition"],
                "run_seed": candidate["run_seed"],
                "source_index": candidate["source_index"],
                "f1": float(y[0]),
                "f2": float(y[1]),
                "f3": float(y[2]),
                "x": " ".join(map(str, candidate["x"])),
            })
            print(
                f"[traffic-trajectory] candidate={cidx} rep={rep} "
                f"policy={policy_id} f={y.tolist()}",
                flush=True,
            )
    status = TrafficTrajectoryEncoder.missing_data_status(out_csv)
    encoded = None
    if status["status"] == "available":
        encoder = TrafficTrajectoryEncoder.from_csv(out_csv)
        encoded = {
            "n_policies": len(encoder.policy_features),
            "feature_dim": 8,
            "policy_ids": sorted(encoder.policy_features)[:20],
        }
    summary = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "encoded" if encoded else status["status"],
        "trajectory_log": str(out_csv),
        "n_candidates": len(candidates),
        "R": int(args.R),
        "seed_start": int(args.seed_start),
        "trajectory_interval": int(args.trajectory_interval),
        "wall_time_sec": float(time.time() - t0),
        "simulations": rows,
        "encoder_summary": encoded,
    }
    return summary


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-glob", default="")
    parser.add_argument("--source-indexes", default="0")
    parser.add_argument("--x", default="")
    parser.add_argument("--max-policies", type=int, default=None)
    parser.add_argument("--dedupe", action="store_true", default=True)
    parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=900000)
    parser.add_argument("--trajectory-interval", type=int, default=60)
    parser.add_argument("--wait-for-data-sec", type=float, default=0.0)
    parser.add_argument("--wait-interval-sec", type=float, default=30.0)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-sim-csv", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(json_safe(result), indent=2), encoding="utf-8")
    if args.out_sim_csv and result.get("simulations"):
        write_csv(args.out_sim_csv, result["simulations"])
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
