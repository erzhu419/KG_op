#!/usr/bin/env python3
"""Submit, wait for, and merge ingolstadt21 OOS validation shards."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scheduler_node_policy import allowed_node_flags, parse_cpu_nodes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULER = Path.home() / ".claude/skills/scheduler/scheduler.py"
DEFAULT_DEPLOY = Path("/home/erzhu419/mine_code/KG_op_scheduler_deploy")
DEFAULT_REMOTE_ROOT = Path("/home/zhengliang01/scheduleurm_work/KG_op_scheduler_deploy")
SUMO_PKG = "/home/zhengliang01/scheduleurm_work/python_pkgs/eclipse_sumo_1_25"
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"


def parse_csv(text):
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def scheduler_module(path):
    spec = importlib.util.spec_from_file_location("scheduleurm", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_cmd(cmd, dry_run=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    if dry_run:
        return ""
    return subprocess.check_output([str(c) for c in cmd], text=True)


def submit(args):
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    cwd = args.deploy / "Final_Submission/GPR_KG_Code"
    out_dir = f"results/ingolstadt21/{args.out_prefix}_{run_id}"
    nodes = parse_cpu_nodes(args.nodes)
    task_ids = []
    for shard in range(args.shards):
        node = nodes[shard % len(nodes)]
        json_out = f"{out_dir}/oos_R{args.R}_shard{shard}_of{args.shards}.json"
        parts = [
            "export LC_ALL=C LANG=C",
            f"export SUMO_PKG={SUMO_PKG}",
            "export SUMO_HOME=$SUMO_PKG/sumo",
            "export PYTHONPATH=$SUMO_PKG:$SUMO_PKG/sumo/tools:$PYTHONPATH",
            "export PATH=$SUMO_HOME/bin:$PATH",
            "export LD_LIBRARY_PATH=$SUMO_PKG/libsumo.libs:$SUMO_PKG/eclipse_sumo.libs:$LD_LIBRARY_PATH",
            f"mkdir -p {out_dir}",
            (
                f"{PYTHON} -m experiments.ingolstadt21.validate_oos_feasibility "
                f"--R {args.R} --seed-start {args.seed_start} "
                f"--num-shards {args.shards} --shard-index {shard} "
                f"--jobs {args.jobs_per_shard} --backend {args.backend} "
                f"--progress-every {args.progress_every} --dedupe {args.dedupe} "
                f"--seed-mode {args.seed_mode} "
                f"--out {json_out}"
            ),
            "echo DONE",
        ]
        if args.method:
            parts[-2] += f" --method {args.method}"
        if args.partition:
            parts[-2] += f" --partition {args.partition}"
        if args.max_points is not None:
            parts[-2] += f" --max-points {args.max_points}"
        if args.source_indexes:
            parts[-2] += f" --source-indexes {args.source_indexes}"
        if args.resume:
            parts[-2] += " --resume"
        cmd = "; ".join(parts)
        submit_cmd = [
            sys.executable, args.scheduler, "submit",
            "--description", f"KG_op ingolstadt21 OOS R{args.R} shard {shard}/{args.shards} {run_id}",
            "--cmd", cmd,
            "--cwd", str(cwd),
            "--signature", f"KG_op/sumo/oos_R{args.R}/{run_id}/shard{shard}",
            "--project", "KG-SUMO",
            "--vram", "0",
            "--cpu", str(args.cpu_per_shard),
            "--ram-mb", str(args.ram_mb_per_shard),
            "--require-node", node,
            *allowed_node_flags(nodes),
            "--allow-no-ckpt",
            "--allow-no-resume",
            "--allow-duplicate",
        ]
        out = run_cmd(submit_cmd, dry_run=args.dry_run)
        if out:
            print(out, end="")
            task_ids.append(out.split()[1])
    if args.dispatch:
        run_cmd([sys.executable, args.scheduler, "dispatch"], dry_run=args.dry_run)
    if args.wait and task_ids:
        run_cmd(
            [sys.executable, args.scheduler, "wait-for", "--task-id", *task_ids,
             "--poll", str(args.poll), "--timeout", str(args.timeout), "--verbose"],
            dry_run=args.dry_run,
        )
    if args.merge and not args.dry_run:
        merge(args, run_id=run_id, task_ids=task_ids)
    print(json.dumps({"run_id": run_id, "task_ids": task_ids}, indent=2))
    return run_id, task_ids


def fetch_json(sched, node, path):
    for _ in range(8):
        rc, out, err = sched.run_on(node, f"cat {path}", timeout=30, check=False)
        if rc == 0 and out.strip().startswith("{"):
            return json.loads(out)
        time.sleep(2)
    raise RuntimeError(f"failed to fetch {node}:{path}: {err[:200]}")


def load_task_record(task_id, scheduler):
    out = subprocess.check_output([sys.executable, str(scheduler), "show", task_id], text=True)
    return json.loads(out.split("\n# tail log:", 1)[0])


def median(values):
    values = sorted(float(v) for v in values)
    if not values:
        return float("nan")
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge(args, run_id=None, task_ids=None):
    run_id = run_id or args.run_id
    if not run_id:
        raise SystemExit("--run-id is required for merge")
    sched = scheduler_module(args.scheduler)
    remote_dir = args.remote_root / "Final_Submission/GPR_KG_Code/results/ingolstadt21" / f"{args.out_prefix}_{run_id}"
    nodes = parse_cpu_nodes(args.nodes)
    shards = []
    for shard in range(args.shards):
        node = nodes[shard % len(nodes)]
        path = remote_dir / f"oos_R{args.R}_shard{shard}_of{args.shards}.json"
        data = fetch_json(sched, node, path)
        print(f"fetched shard {shard}: {len(data.get('candidates', []))} candidates")
        shards.append(data)
    rows = []
    partials = []
    for data in shards:
        rows.extend(data.get("candidates", []))
        partials.extend(data.get("partial_observations", []))
    rows.sort(key=lambda r: int(r.get("candidate_index", 10**9)))
    combined = dict(shards[0])
    combined.update({
        "source": "experiments.ingolstadt21.validate_oos_feasibility.scheduler_shards",
        "scheduler_run_id": run_id,
        "num_shards": args.shards,
        "shard_index": "combined",
        "shard_wall_times_sec": [d.get("wall_time_sec") for d in shards],
        "candidates": rows,
        "partial_observations": partials,
        "wall_time_sec": max(float(d.get("wall_time_sec") or 0.0) for d in shards),
    })
    if task_ids:
        tasks = []
        for tid in task_ids:
            rec = load_task_record(tid, args.scheduler)
            tasks.append({
                "id": tid,
                "node": rec.get("node"),
                "status": rec.get("status"),
                "runtime_sec": float((rec.get("finished_at") or 0) - (rec.get("started_at") or 0)),
            })
        combined["scheduler_tasks"] = tasks
        combined["scheduler_wall_time_sec"] = max(t["runtime_sec"] for t in tasks)

    out_root = ROOT / "Final_Submission/GPR_KG_Code/results/ingolstadt21"
    json_path = out_root / f"oos_feasibility_validation_R{args.R}_scheduler_{run_id}.json"
    audit_path = out_root / f"oos_feasibility_validation_R{args.R}_scheduler_{run_id}_audit.json"
    csv_path = out_root / f"oos_feasibility_validation_R{args.R}_scheduler_{run_id}_summary.csv"
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["partition"]), []).append(row)
    group_summary = []
    for (method, partition), group_rows in sorted(groups.items()):
        probs = [r["validation"]["feasible_probability"] for r in group_rows]
        group_summary.append({
            "method": method,
            "partition": partition,
            "n": len(group_rows),
            "median_p": median(probs),
            "max_p": max(probs),
            "point_pass": sum(r["validation"]["passes_chance_constraint_point_estimate"] for r in group_rows),
            "wilson_pass": sum(r["validation"]["passes_chance_constraint_wilson_lower"] for r in group_rows),
        })
    best = max(rows, key=lambda r: r["validation"]["feasible_probability"]) if rows else None
    audit = {
        "source": str(json_path),
        "target_probability": combined.get("target_probability", 0.95),
        "n_candidates": len(rows),
        "point_pass_count": sum(r["validation"]["passes_chance_constraint_point_estimate"] for r in rows),
        "wilson_pass_count": sum(r["validation"]["passes_chance_constraint_wilson_lower"] for r in rows),
        "best_feasible_probability_candidate": best,
        "group_summary": group_summary,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    csv_rows = []
    for row in rows:
        val = row["validation"]
        csv_rows.append({
            "candidate_index": row.get("candidate_index"),
            "method": row["method"],
            "partition": row["partition"],
            "run_seed": row["run_seed"],
            "source_index": row["source_index"],
            "R": val["R"],
            "p_feasible": val["feasible_probability"],
            "wilson_low": val["wilson_95"][0],
            "wilson_high": val["wilson_95"][1],
            "mean_f1": val["mean"][0],
            "mean_f2": val["mean"][1],
            "mean_f3": val["mean"][2],
            "point_pass": val["passes_chance_constraint_point_estimate"],
            "wilson_pass": val["passes_chance_constraint_wilson_lower"],
            "mean_sim_time_sec": val["mean_sim_time_sec"],
            "x": " ".join(map(str, row["x"])),
        })
    write_csv(csv_path, csv_rows)
    print(json.dumps({
        "json": str(json_path),
        "audit": str(audit_path),
        "summary_csv": str(csv_path),
        "n_candidates": len(rows),
        "point_pass_count": audit["point_pass_count"],
        "wilson_pass_count": audit["wilson_pass_count"],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("submit", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--scheduler", default=str(DEFAULT_SCHEDULER))
        p.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
        p.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT)
        p.add_argument("--run-id", default="")
        p.add_argument("--out-prefix", default="server_oos")
        p.add_argument("--nodes", default="node001,node002,node003,node004,node005,node006")
        p.add_argument("--shards", type=int, default=6)
        p.add_argument("--R", type=int, default=100)
        p.add_argument("--seed-start", type=int, default=50000)
        p.add_argument("--seed-mode", default="common", choices=["common", "blocked"])
        p.add_argument("--jobs-per-shard", type=int, default=48)
        p.add_argument("--cpu-per-shard", type=int, default=48)
        p.add_argument("--ram-mb-per-shard", type=int, default=49152)
        p.add_argument("--backend", default="libsumo", choices=["auto", "libsumo", "traci"])
        p.add_argument("--progress-every", type=int, default=50)
        p.add_argument("--method", default="")
        p.add_argument("--partition", default="")
        p.add_argument("--max-points", type=int, default=None)
        p.add_argument(
            "--source-indexes",
            default="",
            help="Comma-separated final_pareto_set source indices; 0 means final-only",
        )
        p.add_argument("--dedupe", default="by_x", choices=["by_x", "none"])
        p.add_argument("--resume", action="store_true")
        p.add_argument("--dry-run", action="store_true")
    sub.choices["submit"].add_argument("--dispatch", action="store_true")
    sub.choices["submit"].add_argument("--wait", action="store_true")
    sub.choices["submit"].add_argument("--merge", action="store_true")
    sub.choices["submit"].add_argument("--poll", type=int, default=30)
    sub.choices["submit"].add_argument("--timeout", type=int, default=14400)
    args = parser.parse_args()
    if args.cmd == "submit":
        submit(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
