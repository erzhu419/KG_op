#!/usr/bin/env python3
"""Registered announced population, with saved-case equivalence and serial resume."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.inspect_public_battery_announced_development import failure_context, validate_result, window_workload
from performance.inspect_public_battery_announced_dynamic import call_totals, economics, write
from performance.inspect_public_battery_announced_library import (
    calibration, decode_library, forecast_stress, instant, read, read_csv, target_fractions,
)
from problems.public_battery_causal_service import simulate_service

PROTOCOL = ROOT / "performance/manifests/public_battery_announced_population_screen_v1_20261001.json"
OUTPUT = ROOT / "paper_artifacts/public_battery_announced_population_screen_v1_20261001"
IDENTITY = ("period", "window_start_minute", "profile_id", "dispatch_rule")


def key(row):
    return tuple(row[k] for k in IDENTITY)


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append(out, name, row):
    with (out / name).open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def inputs(protocol):
    data = read(ROOT / protocol["data_protocol"])
    dynamic = read(ROOT / protocol["basis_protocol"])
    basis = read(ROOT / protocol["library_protocol"])
    roster, nodes, channels, _ = decode_library(basis)
    profiles = [(r["profile_id"], r["family"], p) for r, p in zip(roster, channels)]
    profiles += [(f"constant_{value:.2f}", "extra_reference", np.full((2, len(nodes)), value))
                 for value in protocol["population"]["extra_shared_references"]]
    contract = read(ROOT / protocol["task_protocol"])["contract"]
    boundary = read(ROOT / dynamic["basis_outputs"] / "boundary_half_hours.json")
    source = read(ROOT / dynamic["screen_outputs"] / "sample_request_marks.json")
    fit, windows = calibration(data), {}
    for case in data["development"]:
        day = case["start"][:10]
        rows = {instant(r["start_time_utc"]): r for r in read_csv(ROOT / case["csv"]) + boundary[day]}
        marks = next(s["marks"] for s in source if s["period"] == day)
        for minute in protocol["population"]["window_start_minutes"]:
            start = instant(case["start"]) + timedelta(minutes=minute)
            announcements, deliveries, metadata = window_workload(marks, minute, contract)
            stresses = [forecast_stress(rows, start + timedelta(minutes=t + 60),
                                       start + timedelta(minutes=t), fit)[0] for t in range(0, 10140, 30)]
            windows[(day, minute)] = {"start": start, "prices": rows, "stresses": stresses,
                                      "announcements": announcements, "marks": [{**r, "minute": r["delivery_minute"]} for r in deliveries],
                                      "metadata": metadata}
    old = lines(ROOT / protocol["static_screen_outputs"] / "windows.jsonl")
    functional = read(ROOT / protocol["basis_outputs"] / "runs.json")
    repaired = {key(r): r for r in read(ROOT / protocol["repair_outputs"] / "runs.json")}
    reused = {key(r): {**r, "provenance": protocol["static_screen_outputs"] + "/windows.jsonl"} for r in old}
    reused.update({key(r): {**repaired.get(key(r), r), "provenance":
                  (protocol["repair_outputs"] if key(r) in repaired else protocol["basis_outputs"]) + "/runs.json"}
                  for r in functional})
    costs = {key(r): r for r in read(ROOT / dynamic["basis_outputs"] / "pilot_economics.json")["runs"]}
    return data, contract, profiles, nodes, windows, reused, costs


def execute(identity, pair, nodes, window, data, contract):
    targets = target_fractions(nodes, pair, window["stresses"]).T
    return simulate_service([49, 49], [[0, 0], [0, 0]], targets, window["marks"], identity["dispatch_rule"],
        10200, [49, 49], data["asset"], data["failure"]["numerical_energy_tolerance_MWh"],
        absolute_request_peaks_MW=contract["absolute_peak_MW"], announcement_ledger=window["announcements"],
        stop_on_service_failure=True)


def compare_saved(result, saved, value, old_cost=None):
    if (result["window_success"] != saved["window_success"] or result["planned_requests"] != saved["planned_requests"]
            or result["first_dispatch_failure"] != saved["first_dispatch_failure"]):
        raise ValueError("saved-case success, denominator or first failure changed")
    if saved["window_success"]:
        error = max(float(np.max(np.abs(np.array(result[k]) - saved[k]))) for k in
                    ("final_or_last_safe_SOC_MWh", "gross_import_MWh", "gross_export_MWh"))
        if error > 1e-8 or (old_cost is not None and abs(value["cost_GBP"] - old_cost) > 1e-6):
            raise ValueError("successful saved-case stocks, gross flows or cost changed")
        return error
    if value["cost_GBP"] is not None:
        raise ValueError("failed window received a full cost")
    return None


def equivalence(protocol, out, data, contract, profiles, nodes, windows, reused, costs):
    folder = out / "equivalence"
    if (folder / "summary.json").exists():
        result = read(folder / "summary.json")
        if not result["gate_pass"]:
            raise ValueError("saved stopping-equivalence discrepancy blocks population launch")
        return result
    folder.mkdir(exist_ok=True)
    if (folder / "runs.json").exists():
        raise ValueError("partial equivalence records exist; preserve their replay counts before repair")
    pairs = {name: pair for name, _, pair in profiles}
    cases = [dict(period="2024-04-15", window_start_minute=0, profile_id=p, dispatch_rule=r)
             for p in ("constant_pair_0003_0007", "constant_pair_0004_0007")
             for r in protocol["population"]["dispatch_rules"]]
    cases += [dict(period=day, window_start_minute=0, profile_id=p, dispatch_rule=r)
              for day, p in (("2024-07-15", "constant_pair_0003_0007"),
                             ("2024-04-15", "functional_pair_0010_0037"))
              for r in protocol["population"]["dispatch_rules"]]
    runs, hours_saved, noms_saved, comparisons = [], [], [], []
    try:
        for identity in cases:
            window = windows[(identity["period"], 0)]
            result, hours, noms = execute(identity, pairs[identity["profile_id"]], nodes, window, data, contract)
            runs.append({**identity, **result})
            hours_saved.extend({**identity, **r} for r in hours)
            noms_saved.extend({**identity, **r} for r in noms)
            for name, records in (("runs.json", runs), ("hourly_dispatch.json", hours_saved), ("nominations.json", noms_saved)):
                write(folder, name, records)
            validate_result(result, window["metadata"], noms, 1e-8)
            value = economics(result, hours, window["start"], window["prices"], data["asset"])
            saved = reused[key(identity)]
            error = compare_saved(result, saved, value, costs.get(key(identity), {}).get("cost_GBP"))
            comparisons.append({**identity, "equivalent": True, "saved_completed_minutes": saved["completed_minutes"],
                                "prefix_completed_minutes": result["completed_minutes"], "saved_case_energy_error_MWh": error,
                                "cost": value, "saved_full_cost_GBP": costs.get(key(identity), {}).get("cost_GBP")})
            write(folder, "comparisons.json", comparisons)
            print(json.dumps({"phase": "equivalence", **identity, "passed": len(comparisons),
                              "completed_minutes": result["completed_minutes"]}), flush=True)
        summary = {"gate_pass": True, "controller_window_replays": len(runs), "calls": call_totals(runs),
                   "saved_provenance_calls": call_totals([reused[key(c)] for c in cases]),
                   "successful_cases": sum(r["window_success"] for r in runs), "passed_cases": len(comparisons),
                   "maximum_successful_energy_difference_MWh": max(c["saved_case_energy_error_MWh"] for c in comparisons
                                                                     if c["saved_case_energy_error_MWh"] is not None),
                   "maximum_successful_cost_difference_GBP": max(abs(c["cost"]["cost_GBP"] - c["saved_full_cost_GBP"])
                                                               for c in comparisons if c["saved_full_cost_GBP"] is not None)}
        write(folder, "summary.json", summary)
        return summary
    except Exception as error:
        write(folder, "summary.json", {"gate_pass": False, "error": str(error), "controller_window_replays": len(runs),
                                        "calls": call_totals(runs), "passed_cases": len(comparisons)})
        raise


def summary(protocol, rows, replay_rows, execution_rows, eq, profiles, active_range, status):
    groups = []
    for profile, family, _ in profiles:
        chosen = [r for r in rows.values() if r["profile_id"] == profile]
        if not chosen:
            continue
        groups.append({"profile_id": profile, "family": family, "assessed_cells": len(chosen),
                       "successful_cells": sum(r["window_success"] for r in chosen),
                       "complete_candidate_population": len(chosen) == 438})
    new = [r for r in rows.values() if r["evaluation_origin"] == "new_population"]
    new_executions = [r for r in execution_rows if r["evaluation_origin"] == "new_population"]
    economic_executions = [r for r in execution_rows if r["evaluation_origin"] == "economic_replay"]
    known_costs = sum(r["window_success"] and r.get("cost_GBP") is not None for r in rows.values())
    return {"protocol_id": protocol["protocol_id"], "updated_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
            "registered_unique_cells": protocol["population"]["reported_unique_cells"], "assessed_unique_cells": len(rows),
            "new_unique_controller_cells": len(new), "reused_cells": len(rows) - len(new),
            "equivalence_controller_replays": eq["controller_window_replays"], "economic_controller_replays": len(economic_executions),
            "validated_economic_replays": len(replay_rows), "new_controller_executions": len(new_executions),
            "actual_controller_executions": len(execution_rows) + eq["controller_window_replays"],
            "new_population_calls": call_totals(new_executions), "equivalence_calls": eq["calls"],
            "economic_replay_calls": call_totals(economic_executions),
            "successful_unique_cells": sum(r["window_success"] for r in rows.values()), "known_successful_costs": known_costs,
            "pending_successful_economic_replays": sum(r["window_success"] and r.get("cost_GBP") is None for r in rows.values()),
            "active_profile_index_range": list(active_range), "completed_candidate_populations": sum(g["complete_candidate_population"] for g in groups),
            "groups": groups, "population_complete": len(rows) == protocol["population"]["reported_unique_cells"],
            "minimum_completed_physical_windows": sum(r["physical_complete"] for r in rows.values()),
            "prefix_stopped_failures": sum(r.get("stopped_on_service_failure", False) for r in new),
            **protocol["accounting_constraints"], "library_matrix_launch": True,
            "source_comparison_gate": "HOLD", "limitations": protocol["limitations"]}


def publish(out, result, rows, execution_rows, protocol, elapsed_seconds):
    """Publish saved progress and accounting without executing any controller."""
    interruption_path = out / "execution_interruptions.json"
    interruptions = read(interruption_path) if interruption_path.exists() else []
    result["actual_controller_execution_count_complete"] = not interruptions
    result["execution_interruptions"] = interruptions
    write(out, "summary.json", result)
    write(out, "progress.json", result)
    new = [r for r in rows.values() if r["evaluation_origin"] == "new_population"]
    batches = []
    for path in sorted(out.glob("batch_*.json")):
        registration = read(path)
        selected = set(registration["profile_ids"])
        cells = [r for r in new if r["profile_id"] in selected]
        executions = [r for r in execution_rows if r["profile_id"] in selected]
        population_calls = call_totals([r for r in executions if r["evaluation_origin"] == "new_population"])
        economic = [r for r in executions if r["evaluation_origin"] == "economic_replay"]
        batches.append({"profile_index_range": registration["profile_indices"],
                        "new_unique_controller_cells": len(cells),
                        "successful_cells": sum(r["window_success"] for r in cells),
                        "first_failure_contexts": sum(not r["window_success"] for r in cells),
                        "minute_physics_evaluations": population_calls["minute_physics_evaluations"],
                        "equivalence_replays": 0, "economic_replays": len(economic),
                        "economic_minute_physics_evaluations": call_totals(economic)["minute_physics_evaluations"],
                        "elapsed_seconds": registration.get("elapsed_seconds", elapsed_seconds),
                        "status": registration["status"]})
    accounting = read(out / "artifact_accounting.json") if (out / "artifact_accounting.json").exists() else {}
    failure_count = sum(not r["window_success"] for r in new)
    if result["status"] != "running":
        failure_path = out / "first_failures.jsonl"
        failure_count = sum(1 for _ in failure_path.open()) if failure_path.exists() else 0
        if result["status"] == "population_batch_complete" and failure_count != sum(not r["window_success"] for r in new):
            raise ValueError("population result and first-failure journal counts differ")
    accounting.update(population_unique_results=result["assessed_unique_cells"],
        new_unique_controller_cells=result["new_unique_controller_cells"], reused_cells=result["reused_cells"],
        equivalence_replays=result["equivalence_controller_replays"], economic_replays=result["economic_controller_replays"],
        actual_controller_executions=result["actual_controller_executions"], first_failure_contexts=failure_count,
        completed_candidate_populations=result["completed_candidate_populations"],
        successful_new_cells=sum(r["window_success"] for r in new),
        successful_unique_cells=result["successful_unique_cells"], known_successful_costs=result["known_successful_costs"],
        pending_successful_economic_replays=result["pending_successful_economic_replays"],
        maximum_new_energy_balance_error_MWh=max((abs(e) for r in new for e in r["energy_balance_error_MWh"]), default=0.),
        actual_controller_execution_count_complete=result["actual_controller_execution_count_complete"],
        execution_interruptions=interruptions,
        population_complete=result["population_complete"], new_public_data_requests=result["new_public_data_requests"],
        confirmation_year_access=result["confirmation_year_access"], last_batch=batches[-1])
    write(out, "artifact_accounting.json", accounting)

    samples, rules = protocol["population"]["development_samples"], protocol["population"]["dispatch_rules"]
    required = protocol["gate"]["required_successes_per_sample"]
    counts = {}
    for row in rows.values():
        group = counts.setdefault((row["profile_id"], row["period"], row["dispatch_rule"]), [0, 0])
        group[0] += 1
        group[1] += int(row["window_success"])
    eligible = {rule: [] for rule in rules}
    for group in result["groups"]:
        if group["family"] == "extra_reference" or not group["complete_candidate_population"]:
            continue
        for rule in rules:
            if all(counts[(group["profile_id"], day, rule)][0] == protocol["gate"]["denominator_per_sample"]
                   and counts[(group["profile_id"], day, rule)][1] >= required for day in samples):
                eligible[rule].append(group["profile_id"])
    joint = sorted(set.intersection(*(set(eligible[rule]) for rule in rules)))
    complete = (result["status"] == "population_batch_complete" and result["population_complete"]
                and result["pending_successful_economic_replays"] == 0)
    write(out, "population_reliability.json", {"population_complete": result["population_complete"],
        "assessed_unique_cells": result["assessed_unique_cells"], "registered_unique_cells": result["registered_unique_cells"],
        "required_successes_per_sample": required, "denominator_per_sample": protocol["gate"]["denominator_per_sample"],
        "reliable_joint_candidates_by_rule": eligible, "joint_candidates_reliable_under_both_rules": joint,
        "joint_candidate_gate_pass": complete and bool(joint),
        "source_comparison_gate": result["source_comparison_gate"]})
    running = result["status"] == "running"
    active = result["active_profile_index_range"]
    next_range = active if running else ([active[1], 157] if not complete and active[1] < 157 else None)
    command = (f"python3 performance/inspect_public_battery_announced_population.py --profile-start {next_range[0]} "
               f"--profile-stop {next_range[1]}") if next_range else None
    if result["status"] == "discrepancy":
        command = None
    write(out, "next_batch.json", {"status": "population_complete" if complete else result["status"],
        "next_profile_index_range": next_range, "next_command": command,
        "remaining_unique_cells": result["registered_unique_cells"] - result["assessed_unique_cells"],
        "missing_successful_economic_replays": result["pending_successful_economic_replays"],
        "source_comparison_gate": result["source_comparison_gate"], "confirmation_year_access": result["confirmation_year_access"],
        "scheduler_backend": "scheduleurm", "scheduler_task_id": os.environ.get("SCHEDULEURM_TASK_ID"),
        "next_action": "assess_reliable_joint_subset_before_source_archive_registration" if complete else "finish_registered_population"})
    if out.resolve() == OUTPUT.resolve():
        table = "\n".join(f"| [{b['profile_index_range'][0]},{b['profile_index_range'][1]}) | "
                          f"{b['new_unique_controller_cells']:,} | {b['successful_cells']:,} | "
                          f"{b['minute_physics_evaluations']:,} | {b['status']} |" for b in batches)
        state = ("完整人口与旧成功成本已全部评估，下一步依据保存的达标候选集合评估 source-archive 研究。" if complete
                 else f"剩余原索引 [{active[0]},{active[1]}) 已安排连续串行执行；每个候选完成后自动更新本记录与记账。")
        if result["status"] == "discrepancy":
            state = f"执行已因验证异常停止：{result.get('error', '见 summary.json')}。实际调用与原始记录保留，恢复前需处理异常。"
        note = ("# 早停验证与人口筛查进度\n\n"
                "8 条保存案例的早停等价验证和 4 项接口测试已通过；首批续跑新增控制器调用为 0。后续沿用已保存的验证结果。\n\n"
                "每个候选覆盖三季各 73 个窗口及两种规则。表中仅计新单元，经济重跑另记。\n\n"
                "| 原索引 | 新增单元 | 新单元成功 | 新单元物理分钟调用 | 状态 |\n"
                "| --- | ---: | ---: | ---: | --- |\n" + table + "\n\n"
                f"累计覆盖 **{result['assessed_unique_cells']:,}/{result['registered_unique_cells']:,}** 个唯一单元，"
                f"已完整记账的控制器执行 **{result['actual_controller_executions']:,} 次**："
                f"{result['new_controller_executions']:,} 条新路径、{result['equivalence_controller_replays']} 条等价重跑、"
                f"{result['economic_controller_replays']:,} 条经济重跑。完整成本已保存 **{result['known_successful_costs']:,} 条**，"
                f"旧成功成本缺口 **{result['pending_successful_economic_replays']:,} 条**。新失败上下文 **{failure_count:,} 条**。\n\n"
                f"已完整评估候选中，有 **{len(joint)} 个联合候选**在三季、两种规则下均达到每季至少 {required}/73。"
                "达标集合见 [可靠性汇总](../../paper_artifacts/public_battery_announced_population_screen_v1_20261001/population_reliability.json)。\n\n"
                + state + " 进度与调用见 [人口汇总](../../paper_artifacts/public_battery_announced_population_screen_v1_20261001/summary.json)"
                "和 [产物记账](../../paper_artifacts/public_battery_announced_population_screen_v1_20261001/artifact_accounting.json)。\n\n"
                "## 局限\n\n"
                "开发窗口重叠，服务通知、重置库存及 MID 代理均为研究设定；失败路径的完整成本未定义。"
                + (f"调度迁移时撤下的旧进程可能有 0–{sum(i['unrecorded_inflight_controller_attempts']['maximum'] for i in interruptions)} 条未完成路径，"
                   "其实际调用数未知，未混入已完成记录；见 execution_interruptions.json。" if interruptions else "")
                +
                "source 比较仍为 **HOLD**，2025 未读取；本阶段不构成 OR 投稿就绪结论。\n")
        (ROOT / "docs/submission_review_20260929/public_battery_announced_population_next_20261001.md").write_text(note)


def inspect(protocol, out, start_index, stop_index):
    out.mkdir(parents=True, exist_ok=True)
    write(out, "effective_protocol.json", protocol)
    if not (out / "unit_test_accounting.json").exists():
        import pytest

        class TestAccounting:
            def pytest_sessionfinish(self, session, exitstatus):
                write(out, "unit_test_accounting.json", {"pytest_exit_code": int(exitstatus), "collected_tests": len(session.items),
                      **session.items[0].module.COUNTS, "test_controller_window_evaluations": 0, "test_nomination_policy_calls": 0,
                      "test_declaration_path_evaluations": 0, "test_trajectory_evaluations": 0})
        code = pytest.main(["-q", str(ROOT / "tests/test_public_battery_population.py")], plugins=[TestAccounting()])
        if code:
            raise ValueError("population stopping tests failed")
    elif read(out / "unit_test_accounting.json")["pytest_exit_code"]:
        raise ValueError("saved stopping tests failed; repair before population launch")
    data, contract, profiles, nodes, windows, reused, costs = inputs(protocol)
    if not 0 <= start_index < stop_index <= len(profiles):
        raise ValueError("batch indices must select the original candidate roster")
    eq = equivalence(protocol, out, data, contract, profiles, nodes, windows, reused, costs)
    rows = {key(r): r for r in lines(out / "cells.jsonl")}
    if not rows:
        for k, row in reused.items():
            origin = {**row, "evaluation_origin": "reused_saved_cell", "cost_GBP": costs.get(k, {}).get("cost_GBP")}
            append(out, "cells.jsonl", origin)
            rows[k] = origin
    replay_rows = lines(out / "economic_replays.jsonl")
    execution_rows = lines(out / "executions.jsonl")
    validated_replays = {key(r) for r in replay_rows}
    unfinished = [r for r in execution_rows if (r["evaluation_origin"] == "new_population" and key(r) not in rows)
                  or (r["evaluation_origin"] == "economic_replay" and key(r) not in validated_replays)]
    if unfinished:
        raise ValueError("recorded execution has unresolved validation; preserve its calls before any repair replay")
    for row in replay_rows:
        rows[key(row)]["cost_GBP"] = row["cost_GBP"]
    batch_file = f"batch_{start_index:03d}_{stop_index:03d}.json"
    before, started = len(rows), time.monotonic()
    if not (out / batch_file).exists():
        write(out, batch_file, {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "profile_indices": [start_index, stop_index],
              "profile_ids": [p[0] for p in profiles[start_index:stop_index]], "maximum_selected_unique_cells": (stop_index - start_index) * 438,
              "classification": "original-roster batch; no outcome-based selection", "status": "running"})
    elif read(out / batch_file)["status"] == "interrupted":
        write(out, batch_file, {**read(out / batch_file), "status": "running",
              "resumed_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        publish(out, summary(protocol, rows, replay_rows, execution_rows, eq, profiles, (start_index, stop_index), "running"),
                rows, execution_rows, protocol, time.monotonic() - started)
        for profile, family, pair in profiles[start_index:stop_index]:
            for day in protocol["population"]["development_samples"]:
                for minute in protocol["population"]["window_start_minutes"]:
                    window = windows[(day, minute)]
                    for rule in protocol["population"]["dispatch_rules"]:
                        identity = dict(period=day, window_start_minute=minute, profile_id=profile, dispatch_rule=rule)
                        k = key(identity)
                        saved = rows.get(k)
                        replay = saved is not None and saved["window_success"] and saved.get("cost_GBP") is None
                        if saved is not None and not replay:
                            continue
                        if len(execution_rows) + eq["controller_window_replays"] >= protocol["execution_budget"]["maximum_actual_controller_executions"]:
                            raise ValueError("registered population controller-execution budget exhausted")
                        result, hours, nominations = execute(identity, pair, nodes, window, data, contract)
                        row = {**identity, **result, "evaluation_origin": "economic_replay" if replay else "new_population"}
                        context = failure_context(result, hours, nominations)
                        if context:
                            append(out, "first_failures.jsonl", {**identity, **context})
                        # The started execution is retained before validation; no silent rerun on resume.
                        append(out, "executions.jsonl", row)
                        execution_rows.append(row)
                        validate_result(result, window["metadata"], nominations, 1e-8)
                        value = economics(result, hours, window["start"], window["prices"], data["asset"])
                        row.update(**value)
                        if replay:
                            compare_saved(result, saved, value)
                            append(out, "economic_replays.jsonl", row)
                            replay_rows.append(row)
                            rows[k]["cost_GBP"] = value["cost_GBP"]
                        else:
                            append(out, "cells.jsonl", row)
                            rows[k] = row
                        if (len(rows) - before) % 50 == 0:
                            print(json.dumps({"profile": profile, "new_cells_in_batch": len(rows) - before,
                                              "assessed_cells": len(rows), "elapsed_seconds": time.monotonic() - started}), flush=True)
            publish(out, summary(protocol, rows, replay_rows, execution_rows, eq, profiles, (start_index, stop_index), "running"),
                    rows, execution_rows, protocol, time.monotonic() - started)
        result = summary(protocol, rows, replay_rows, execution_rows, eq, profiles, (start_index, stop_index), "population_batch_complete")
        result["batch_elapsed_seconds"] = time.monotonic() - started
        result["new_cells_in_batch"] = len(rows) - before
        registration = read(out / batch_file)
        registration["status"] = "complete"
        if "elapsed_seconds" not in registration:
            registration.update(status="complete", new_cells=len(rows) - before, elapsed_seconds=result["batch_elapsed_seconds"])
        else:
            registration.update(last_resume_new_cells=len(rows) - before, last_resume_elapsed_seconds=result["batch_elapsed_seconds"])
        write(out, batch_file, registration)
        publish(out, result, rows, execution_rows, protocol, result["batch_elapsed_seconds"])
        print(json.dumps(result), flush=True)
    except Exception as error:
        result = {**summary(protocol, rows, replay_rows, execution_rows, eq, profiles, (start_index, stop_index), "discrepancy"),
                  "error": str(error)}
        write(out, batch_file, {**read(out / batch_file), "status": "discrepancy", "error": str(error)})
        publish(out, result, rows, execution_rows, protocol, time.monotonic() - started)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--profile-start", type=int, default=0)
    parser.add_argument("--profile-stop", type=int, default=4)
    args = parser.parse_args()
    inspect(read(args.protocol), args.output, args.profile_start, args.profile_stop)
