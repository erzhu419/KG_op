#!/usr/bin/env python3
"""Window shards of the frozen population, dispatched in small CPU process pools."""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance import inspect_public_battery_announced_population as p

CAMPAIGN = p.OUTPUT / "parallel"
STATE = {}
JOURNALS = ("cells.jsonl", "executions.jsonl", "economic_replays.jsonl", "first_failures.jsonl")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(folder, name, value):
    # The collector reads progress while its owner updates it.
    temporary = folder / f".{name}.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(folder / name)


def saved_rows(folder):
    rows = {p.key(r): r for r in p.lines(folder / "cells.jsonl")}
    replays = p.lines(folder / "economic_replays.jsonl")
    executions = p.lines(folder / "executions.jsonl")
    validated = {p.key(r) for r in replays}
    for r in executions:
        if (r["evaluation_origin"] == "new_population" and p.key(r) not in rows
                or r["evaluation_origin"] == "economic_replay" and p.key(r) not in validated):
            raise ValueError("unresolved saved execution; retain its calls before any replay")
    for r in replays:
        rows[p.key(r)]["cost_GBP"] = r["cost_GBP"]
    return rows, replays, executions


def remaining_work(protocol, profiles, rows):
    """Interleave seasons and candidates within each original window start."""
    work = []
    for minute in protocol["population"]["window_start_minutes"]:
        for index, (name, _, _) in enumerate(profiles):
            for day in protocol["population"]["development_samples"]:
                for rule in protocol["population"]["dispatch_rules"]:
                    identity = dict(period=day, window_start_minute=minute, profile_id=name, dispatch_rule=rule)
                    old = rows.get(p.key(identity))
                    if old is not None and (not old["window_success"] or old.get("cost_GBP") is not None):
                        continue
                    work.append({**identity, "profile_index": index,
                                 "evaluation_origin": "economic_replay" if old is not None else "new_population"})
    return work


def split_work(work, shard_count):
    quotient, remainder = divmod(len(work), shard_count)
    shards, start = [], 0
    for i in range(shard_count):
        stop = start + quotient + int(i < remainder)
        shards.append({"shard_index": i, "start": start, "stop": stop})
        start = stop
    return shards


def prepare(campaign, shard_count):
    if (campaign / "work_plan.json").exists():
        raise ValueError("parallel work is already registered; use its saved plan")
    campaign.mkdir(parents=True, exist_ok=True)
    baseline = campaign / "baseline"
    baseline.mkdir()
    for name in JOURNALS:
        if (p.OUTPUT / name).exists():
            shutil.copy2(p.OUTPUT / name, baseline / name)
    protocol = p.read(p.PROTOCOL)
    eq = p.read(p.OUTPUT / "equivalence/summary.json")
    tests = p.read(p.OUTPUT / "unit_test_accounting.json")
    if not eq["gate_pass"] or tests["pytest_exit_code"]:
        raise ValueError("saved prerequisite blocks parallel launch")
    _, _, profiles, _, _, _, _ = p.inputs(protocol)
    rows, replays, executions = saved_rows(baseline)
    work = remaining_work(protocol, profiles, rows)
    if len(executions) + eq["controller_window_replays"] + len(work) > protocol["execution_budget"]["maximum_actual_controller_executions"]:
        raise ValueError("remaining work exceeds the unchanged recorded-execution budget")
    if not 1 <= shard_count <= len(work):
        raise ValueError("each logical shard needs at least one remaining cell")
    write_json(campaign, "work_items.json", work)
    plan = {"protocol_id": protocol["protocol_id"], "registered_at_utc": now(),
            "baseline_assessed_unique_cells": len(rows), "baseline_recorded_executions": len(executions) + eq["controller_window_replays"],
            "baseline_validated_economic_replays": len(replays), "remaining_cell_evaluations": len(work),
            "remaining_new_cells": sum(r["evaluation_origin"] == "new_population" for r in work),
            "remaining_economic_replays": sum(r["evaluation_origin"] == "economic_replay" for r in work),
            "shard_count": shard_count, "shards": split_work(work, shard_count),
            "execution_amendment": "Node pools evaluate disjoint window shards concurrently, interleaving candidates and seasons. The frozen roster, window starts, seed, controller, failure event, economics and gates are unchanged. Aggregate records are restored to original roster order.",
            "prerequisites_reexecuted": False, "maximum_workers_per_node": 96,
            "new_public_data_requests": 0, "confirmation_year_access": False, "source_comparison_gate": "HOLD"}
    write_json(campaign, "work_plan.json", plan)
    print(json.dumps({k: v for k, v in plan.items() if k != "shards"}), flush=True)


def load_state(campaign):
    protocol = p.read(p.PROTOCOL)
    data, contract, profiles, nodes, windows, _, _ = p.inputs(protocol)
    rows, _, _ = saved_rows(campaign / "baseline")
    STATE.update(campaign=campaign, protocol=protocol, data=data, contract=contract,
                 profiles=profiles, nodes=nodes, windows=windows, baseline=rows,
                 work=p.read(campaign / "work_items.json"), plan=p.read(campaign / "work_plan.json"))


def physical_cpu_ids():
    """Use one hardware thread per physical core in the allowed Linux CPU set."""
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        core = (root.joinpath("physical_package_id").read_text().strip(), root.joinpath("core_id").read_text().strip())
        cores.setdefault(core, cpu)
    return list(cores.values())


def run_shard(index):
    s = STATE
    registration = s["plan"]["shards"][index]
    selected = s["work"][registration["start"]:registration["stop"]]
    out = s["campaign"] / "shards" / f"{index:05d}"
    out.mkdir(parents=True, exist_ok=True)
    executions = p.lines(out / "executions.jsonl")
    validated = {p.key(r): r for name in ("cells.jsonl", "economic_replays.jsonl") for r in p.lines(out / name)}
    starts = p.lines(out / "started.jsonl")
    if {p.key(r) for r in starts} != {p.key(r) for r in executions} or any(p.key(r) not in validated for r in executions):
        raise ValueError(f"shard {index} has an interrupted or unresolved execution; no silent rerun")
    try:
        for item in selected:
            identity = {k: item[k] for k in p.IDENTITY}
            k = p.key(item)
            if k in validated:
                continue
            p.append(out, "started.jsonl", {**item, "started_at_utc": now(),
                                          "task_id": os.environ.get("SCHEDULEURM_TASK_ID"), "worker_pid": os.getpid()})
            window = s["windows"][(identity["period"], identity["window_start_minute"])]
            pair = s["profiles"][item["profile_index"]][2]
            result, hours, nominations = p.execute(identity, pair, s["nodes"], window, s["data"], s["contract"])
            row = {**identity, **result, "evaluation_origin": item["evaluation_origin"]}
            p.append(out, "executions.jsonl", row)
            executions.append(row)
            context = p.failure_context(result, hours, nominations)
            if context:
                p.append(out, "first_failures.jsonl", {**identity, **context})
            p.validate_result(result, window["metadata"], nominations, 1e-8)
            row.update(p.economics(result, hours, window["start"], window["prices"], s["data"]["asset"]))
            replay = item["evaluation_origin"] == "economic_replay"
            if replay:
                p.compare_saved(result, s["baseline"][k], row)
            p.append(out, "economic_replays.jsonl" if replay else "cells.jsonl", row)
            validated[k] = row
        if set(validated) != {p.key(r) for r in selected}:
            raise ValueError(f"shard {index} contains an unexpected or missing cell")
        summary = {"shard_index": index, "status": "complete", "completed_cells": len(validated),
                   "calls": p.call_totals(executions), "task_id": os.environ.get("SCHEDULEURM_TASK_ID"), "finished_at_utc": now()}
        write_json(out, "summary.json", summary)
        return summary
    except Exception as error:
        write_json(out, "summary.json", {"shard_index": index, "status": "discrepancy", "error": str(error),
                                     "recorded_executions": len(executions), "calls": p.call_totals(executions)})
        raise


def run(campaign, start, stop, workers):
    load_state(campaign)
    plan = STATE["plan"]
    if not 0 <= start < stop <= plan["shard_count"]:
        raise ValueError("node task must select registered logical shards")
    cpus = physical_cpu_ids()
    workers = min(workers, 96, len(cpus), stop - start)
    if workers < 1:
        raise ValueError("no worker cores available")
    ctx = mp.get_context("fork")
    # Several scheduler tasks may share a node. Let Linux balance their workers
    # across one hardware thread of every physical core instead of pinning each
    # pool to the same first few cores.
    os.sched_setaffinity(0, set(cpus))
    name = f"node_task_{start:05d}_{stop:05d}.json"
    status = {"status": "running", "shard_start": start, "shard_stop": stop, "workers": workers,
              "physical_cpus": cpus, "task_id": os.environ.get("SCHEDULEURM_TASK_ID"), "started_at_utc": now()}
    write_json(campaign, name, status)
    completed = 0
    try:
        with ctx.Pool(workers) as pool:
            for result in pool.imap_unordered(run_shard, range(start, stop), chunksize=1):
                completed += 1
                if completed % 10 == 0 or completed == stop - start:
                    status.update(completed_shards=completed, updated_at_utc=now())
                    write_json(campaign, name, status)
                    print(json.dumps({"completed_shards": completed, "total_shards": stop - start, "workers": workers}), flush=True)
        status.update(status="complete", completed_shards=completed, finished_at_utc=now())
        write_json(campaign, name, status)
    except Exception as error:
        write_json(campaign, name, {**status, "status": "discrepancy", "completed_shards": completed, "error": str(error)})
        raise


def merge_shard(campaign, index, expected, rows, replays, executions, failures, seen):
    out = campaign / "shards" / f"{index:05d}"
    saved = p.read(out / "summary.json")
    if saved["status"] != "complete":
        raise ValueError(f"shard {index} did not complete")
    current = p.lines(out / "executions.jsonl")
    wanted = {p.key(r): r["evaluation_origin"] for r in expected}
    actual = {p.key(r): r["evaluation_origin"] for r in current}
    if actual != wanted or len(current) != len(wanted) or seen.intersection(actual):
        raise ValueError(f"shard {index} overlaps, omits or changes registered evaluations")
    starts = p.lines(out / "started.jsonl")
    if len(starts) != len(current) or {p.key(r) for r in starts} != set(actual):
        raise ValueError(f"shard {index} has unaccounted started evaluations")
    new = p.lines(out / "cells.jsonl")
    economic = p.lines(out / "economic_replays.jsonl")
    validated = {p.key(r): r["evaluation_origin"] for r in new + economic}
    if validated != wanted or len(new) + len(economic) != len(current):
        raise ValueError(f"shard {index} has unresolved validation")
    for r in new:
        if p.key(r) in rows or r["evaluation_origin"] != "new_population":
            raise ValueError("new cell duplicates prior evidence")
        rows[p.key(r)] = r
    for r in economic:
        old = rows[p.key(r)]
        if not old["window_success"] or old.get("cost_GBP") is not None or r["cost_GBP"] is None:
            raise ValueError("economic replay duplicates or fails a saved successful window")
        old["cost_GBP"] = r["cost_GBP"]
    executions.extend(current)
    replays.extend(economic)
    failures.extend(p.lines(out / "first_failures.jsonl"))
    seen.update(actual)


def write_journal(folder, name, records):
    temporary = folder / f".{name}.parallel.tmp"
    with temporary.open("w") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    temporary.replace(folder / name)


def collect(campaign, wait):
    plan = p.read(campaign / "work_plan.json")
    while True:
        jobs = [p.read(path) for path in campaign.glob("node_task_*.json")]
        if any(j["status"] == "discrepancy" for j in jobs):
            write_json(campaign, "collection_status.json", {"status": "discrepancy", "node_tasks": jobs})
            raise ValueError("a node task reported a discrepancy; retain shard journals")
        count = sum(1 for _ in (campaign / "shards").glob("*/summary.json"))
        if count == plan["shard_count"] and jobs and all(j["status"] == "complete" for j in jobs):
            break
        if not wait:
            raise ValueError(f"only {count}/{plan['shard_count']} logical shards finished")
        time.sleep(10)
    rows, replays, executions = saved_rows(campaign / "baseline")
    failures = p.lines(campaign / "baseline/first_failures.jsonl")
    work, seen = p.read(campaign / "work_items.json"), set()
    for shard in plan["shards"]:
        merge_shard(campaign, shard["shard_index"], work[shard["start"]:shard["stop"]], rows, replays, executions, failures, seen)
    if len(seen) != len(work) or len(executions) + 8 > p.read(p.PROTOCOL)["execution_budget"]["maximum_actual_controller_executions"]:
        raise ValueError("global shard coverage or recorded-execution budget changed")
    protocol = p.read(p.PROTOCOL)
    _, _, profiles, _, _, _, _ = p.inputs(protocol)
    ordering = {(day, minute, name, rule): (i, d, m, r)
                for i, (name, _, _) in enumerate(profiles)
                for d, day in enumerate(protocol["population"]["development_samples"])
                for m, minute in enumerate(protocol["population"]["window_start_minutes"])
                for r, rule in enumerate(protocol["population"]["dispatch_rules"])}
    for name, records in (("cells.jsonl", list(rows.values())), ("executions.jsonl", executions),
                          ("economic_replays.jsonl", replays), ("first_failures.jsonl", failures)):
        write_journal(p.OUTPUT, name, sorted(records, key=lambda r: ordering[p.key(r)]))
    batch = p.read(p.OUTPUT / "batch_044_157.json")
    batch.update(status="complete", completion_mode="parallel_window_shards", completed_at_utc=now())
    write_json(p.OUTPUT, "batch_044_157.json", batch)
    result = p.summary(protocol, rows, replays, executions, p.read(p.OUTPUT / "equivalence/summary.json"), profiles, (44, 157), "population_batch_complete")
    result.update(parallel_logical_shards=plan["shard_count"], parallel_node_tasks=jobs, parallel_work_plan=str(campaign / "work_plan.json"))
    p.publish(p.OUTPUT, result, rows, executions, protocol, 0)
    write_json(campaign, "collection_status.json", {"status": "complete", "merged_cell_evaluations": len(seen), "finished_at_utc": now()})
    print(json.dumps({k: result[k] for k in ("population_complete", "assessed_unique_cells", "known_successful_costs", "pending_successful_economic_replays")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("--shards", type=int, default=3456)
    worker = modes.add_parser("run")
    worker.add_argument("--shard-start", type=int, required=True)
    worker.add_argument("--shard-stop", type=int, required=True)
    worker.add_argument("--workers", type=int, required=True)
    aggregator = modes.add_parser("collect")
    aggregator.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.campaign, args.shards)
    elif args.mode == "run":
        run(args.campaign, args.shard_start, args.shard_stop, args.workers)
    else:
        collect(args.campaign, args.wait)
