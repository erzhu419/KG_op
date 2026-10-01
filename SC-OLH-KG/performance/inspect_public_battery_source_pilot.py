#!/usr/bin/env python3
"""Fixed 2023 source data and bounded announced-controller interface pilot."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.profile_atlas import profile_cosine_coordinate
from performance.inspect_independent_energy_data import indexed, retrieve, write_csv
from performance.inspect_public_battery_announced_library import calibration, decode_library, forecast_stress, instant, read, read_csv
from performance import inspect_public_battery_announced_population as p
from performance.run_public_battery_population_parallel import write_json
from performance.inspect_public_battery_dispatch import decode
from problems.public_battery_causal_service import receipt_marks

PROTOCOL = ROOT / "performance/manifests/public_battery_source_archive_pilot_v1_20261002.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_source_archive_pilot_v1_20261002"


def ordered_coordinates(nodes, pairs):
    """Keep two independent ordered channels; never join them as one function."""
    return np.array([np.concatenate([profile_cosine_coordinate(channel, nodes=nodes) for channel in pair])
                     for pair in pairs])


def source_windows(protocol, out, starts):
    data = read(out / "source_data_protocol.json")
    contract = read(ROOT / protocol["task_protocol"])["contract"]
    roster, nodes, pairs, _ = decode_library(read(ROOT / protocol["library_protocol"]))
    marks = read(out / "source_request_marks.json")
    fit, windows = calibration(data), {}
    for case in data["development"]:
        day = case["start"][:10]
        rows = {instant(r["start_time_utc"]): r for r in read_csv(ROOT / case["csv"])}
        sample = next(r["marks"] for r in marks if r["period"] == day)
        for minute in starts:
            start = instant(case["start"]) + timedelta(minutes=minute)
            announcements, deliveries, metadata = p.window_workload(sample, minute, contract)
            stress_lead = [forecast_stress(rows, start + timedelta(minutes=t+60), start + timedelta(minutes=t), fit)
                           for t in range(0, 10140, 30)]
            windows[(day, minute)] = dict(start=start, prices=rows, stresses=[r[0] for r in stress_lead],
                announcements=announcements, marks=[{**r, "minute":r["delivery_minute"]} for r in deliveries],
                metadata=metadata, minimum_forecast_publication_lead_minutes=min(r[1] for r in stress_lead))
    return data, contract, roster, nodes, pairs, windows


def prepare(protocol, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "data_summary.json").exists():
        saved = read(out / "data_summary.json")
        if saved["data_interface_gate_pass"]:
            print(json.dumps(saved), flush=True)
            return
        raise ValueError("saved data discrepancy requires diagnosis before another request")
    base = read(ROOT / protocol["data_protocol"])
    interface = read(ROOT / protocol["interface_protocol"])
    session, log = requests.Session(), p.lines(out / "data_requests.jsonl")

    def fetch(endpoint, params, path):
        new = not path.exists()
        if new and sum(r["status"] == "started" for r in log) >= protocol["data_budget"]["maximum_public_requests"]:
            raise ValueError("registered request budget exhausted")
        if new:
            entry = dict(endpoint=endpoint, params=params, file=str(path.relative_to(out)), status="started",
                         started_at_utc=datetime.now(timezone.utc).isoformat(), owner_pid=os.getpid(),
                         task_id=os.environ.get("SCHEDULEURM_TASK_ID"))
            log.append(entry)
            p.append(out, "data_requests.jsonl", entry)
        rows, meta = retrieve(session, interface["api_base"], endpoint, params, path)
        entry = dict(file=str(path.relative_to(out)), status="complete", new_public_request=new, **meta)
        log.append(entry)
        p.append(out, "data_requests.jsonl", entry)
        if sum(r.get("response_bytes", 0) for r in log if r.get("new_public_request")) > protocol["data_budget"]["maximum_response_bytes"]:
            raise ValueError("registered public-response budget exhausted")
        return rows

    cases, source_marks, coverage = [], [], []
    for source in protocol["source_periods"]:
        case = deepcopy(source)
        start, stop = instant(case["start"]), instant(case["stop"])
        folder = out / start.date().isoformat()
        folder.mkdir(exist_ok=True)
        end = stop + timedelta(minutes=120)
        times = {"from":start.isoformat(), "to":end.isoformat()}
        forecasts = indexed(fetch(interface["forecast_endpoint"], times, folder / "forecast.json"), start, end)
        prices, left, part = {}, start, 0
        while left < end:
            right = min(end, left + timedelta(days=6))
            params = dict(dataProviders=interface["price_provider"], format="json", **{"from":left.isoformat(),"to":right.isoformat()})
            prices.update(indexed(fetch(interface["price_endpoint"], params, folder / f"price_{part}.json"), left, right))
            left, part = right, part+1
        expected = {start + timedelta(minutes=30*i) for i in range(int((end-start).total_seconds()/1800))}
        if set(forecasts) != expected or set(prices) != expected:
            raise ValueError("source forecast/settlement coverage is incomplete")
        aligned = []
        for stamp in sorted(expected):
            f, price = forecasts[stamp], prices[stamp]
            if f["boundary"] != "N" or price["dataProvider"] != interface["price_provider"]:
                raise ValueError("source forecast boundary or price provider differs from development")
            value, cash = float(f[interface["forecast_field"]]), float(price["price"])
            if not np.isfinite(value) or not np.isfinite(cash) or value <= 0:
                raise ValueError("source contains missing/nonfinite forecast or price")
            aligned.append(dict(start_time_utc=stamp.isoformat(), forecast_publish_time_utc=instant(f["publishTime"]).isoformat(),
                                forecast_MW=value, price_GBP_per_MWh=cash))
        csv = folder / "aligned_half_hours.csv"
        write_csv(csv, aligned)
        case["csv"] = str(csv.relative_to(ROOT))
        cases.append(case)
        units, counts = [], []
        for unit in base["asset"]["workload_bm_units"]:
            left, rows = start - timedelta(days=1), []
            while left < stop:
                right = min(stop, left + timedelta(days=2))
                params = {"bmUnit":unit, "from":left.isoformat(), "to":right.isoformat(), "format":"json"}
                piece = fetch(base["acceptances_endpoint"], params, folder / f"{unit}_{left.date()}_{right.date()}.json")
                rows.extend(r for r in piece if left <= instant(r["timeFrom"]) < right)
                left = right
            segments = decode(rows, unit)
            for s in segments:
                if s.start >= s.stop or s.received > s.start or any(t.second or t.microsecond for t in (s.start,s.stop,s.received)):
                    raise ValueError("source BOA does not satisfy the supported ordinary whole-minute receipt interface")
            units.append(segments)
            counts.append(len(segments))
        marks = receipt_marks(units, start, range(0,14400,60))
        source_marks.append(dict(period=start.date().isoformat(), marks=marks))
        coverage.append(dict(period=start.date().isoformat(), aligned_half_hours=len(aligned), unit_BOA_segments=counts,
                             original_hourly_receipt_marks=len(marks), active_receipt_marks=sum(r["direction"] is not None for r in marks)))
    data = deepcopy(base)
    data.update(development=cases, role="fixed 2023 source interface, unchanged 2022 calibration and asset", confirmation_year_access=False)
    write_json(out, "source_data_protocol.json", data)
    write_json(out, "source_request_marks.json", source_marks)
    _, _, roster, nodes, pairs, windows = source_windows(protocol, out, protocol["source_window_starts"])
    coords = ordered_coordinates(nodes, pairs)
    visibility = [dict(period=day, window_start_minute=minute, planned_requests=w["metadata"]["planned_requests"],
                       minimum_forecast_publication_lead_minutes=w["minimum_forecast_publication_lead_minutes"])
                  for (day,minute), w in windows.items()]
    write_json(out, "source_window_visibility.json", visibility)
    write_json(out, "structural_coordinates.json", dict(profile_ids=[r["profile_id"] for r in roster],
        coordinates=coords.tolist(), dimension=int(coords.shape[1]), channels=2, calibration_year=2022,
        source_or_target_outcomes_used=False, normalization_scope="all original 154 joint profiles, separately ordered channels"))
    result = dict(protocol_id=protocol["protocol_id"], data_interface_gate_pass=True, coverage=coverage,
        source_windows=len(windows), original_joint_candidates=len(roster), coordinates=int(coords.shape[1]),
        minimum_forecast_publication_lead_minutes=min(r["minimum_forecast_publication_lead_minutes"] for r in visibility),
        public_request_attempt_start_records=sum(r["status"] == "started" for r in log),
        completed_new_public_responses=sum(bool(r.get("new_public_request")) for r in log),
        uncompleted_request_start_records=sum(r["status"] == "started" for r in log)
                                         - sum(bool(r.get("new_public_request")) for r in log),
        public_response_bytes=sum(r.get("response_bytes",0) for r in log if r.get("new_public_request")),
        new_controller_window_evaluations=0, source_comparison_gate="HOLD", confirmation_year_access=False)
    write_json(out, "data_summary.json", result)
    print(json.dumps(result), flush=True)


def pilot_work(protocol):
    return [dict(period=case["start"][:10], window_start_minute=minute, profile_id=profile, dispatch_rule=rule)
            for case in protocol["source_periods"] for minute in protocol["pilot"]["window_start_minutes"]
            for profile in protocol["pilot"]["profiles"] for rule in protocol["pilot"]["dispatch_rules"]]


def run(protocol, out, index):
    if not read(out / "data_summary.json")["data_interface_gate_pass"]:
        raise ValueError("source data interface has not passed")
    identity = pilot_work(protocol)[index]
    folder = out / "pilot_cells" / f"{index:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "validated.json").exists():
        print(json.dumps(dict(index=index,status="saved_validated_result")), flush=True)
        return
    if (folder / "started.json").exists():
        raise ValueError("unresolved started pilot; preserve attempts before replay")
    data, contract, roster, nodes, pairs, windows = source_windows(protocol, out, protocol["pilot"]["window_start_minutes"])
    pair = pairs[next(i for i,r in enumerate(roster) if r["profile_id"] == identity["profile_id"])]
    window = windows[(identity["period"], identity["window_start_minute"])]
    write_json(folder,"started.json",{**identity,"started_at_utc":datetime.now(timezone.utc).isoformat(),
                                     "task_id":os.environ.get("SCHEDULEURM_TASK_ID"),"worker_pid":os.getpid()})
    result, hours, nominations = p.execute(identity, pair, nodes, window, data, contract)
    write_json(folder,"executed.json",{**identity,**result})
    context = p.failure_context(result,hours,nominations)
    if context:
        write_json(folder,"first_failure.json",context)
    p.validate_result(result, window["metadata"], nominations, 1e-8)
    value = p.economics(result,hours,window["start"],window["prices"],data["asset"])
    write_json(folder,"validated.json",{**identity,**result,**value})
    print(json.dumps(dict(index=index,window_success=result["window_success"],cost_GBP=value["cost_GBP"])),flush=True)


def collect(protocol, out):
    work = pilot_work(protocol)
    rows = [read(out / "pilot_cells" / f"{i:03d}" / "validated.json") for i in range(len(work))]
    if [p.key(r) for r in rows] != [p.key(r) for r in work]:
        raise ValueError("pilot results do not match the complete frozen roster")
    write_json(out,"pilot_results.json",rows)
    groups = [dict(period=case["start"][:10], profile_id=profile,
                   successful_windows=sum(r["window_success"] for r in rows
                       if r["period"] == case["start"][:10] and r["profile_id"] == profile),
                   assessed_windows=4)
              for case in protocol["source_periods"] for profile in protocol["pilot"]["profiles"]]
    summary = dict(protocol_id=protocol["protocol_id"], status="source_interface_pilot_complete",
        completed_controller_window_evaluations=len(rows), successful_windows=sum(r["window_success"] for r in rows),
        known_full_costs=sum(r["cost_GBP"] is not None for r in rows), calls=p.call_totals(rows),
        groups=groups, data_accounting=read(out / "data_summary.json"),
        maximum_energy_balance_error_MWh=max(abs(e) for r in rows for e in r["energy_balance_error_MWh"]),
        all_original_library_candidates_retained=154, source_comparison_gate="HOLD", confirmation_year_access=False,
        original_objective_atlas_interface_ready=False, source_designs_frozen=0,
        decision="define_source_selection_for_undefined_failed_costs_before_full_archive_and_target_comparison",
        limitations=protocol["limitations"])
    write_json(out,"summary.json",summary)
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=PROTOCOL)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("mode",choices=["prepare","run","collect"])
    parser.add_argument("--index",type=int)
    args=parser.parse_args()
    protocol=read(args.protocol)
    if args.mode == "prepare": prepare(protocol,args.output)
    elif args.mode == "run": run(protocol,args.output,args.index)
    else: collect(protocol,args.output)
