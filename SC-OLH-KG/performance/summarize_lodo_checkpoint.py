#!/usr/bin/env python3
"""Summarize partial LODO meta-prior checkpoint JSONL files.

The benchmark writes one JSON object per completed
heldout/line/seed/basis cell.  This helper intentionally works on partial
files so long scheduler runs can be inspected before the final JSON/CSV
artifacts are written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from performance.benchmark_lodo_meta_prior import flatten_summary, summarize  # noqa: E402
from performance.benchmark_quality import json_safe, write_csv  # noqa: E402


def load_rows(paths):
    rows = []
    bad = 0
    seen = set()

    def add_row(row, path, lineno, inferred_N):
        nonlocal bad
        if not isinstance(row, dict):
            bad += 1
            return
        if row.get("N") is None and inferred_N is not None:
            row["N"] = inferred_N
        key = (
            int(row.get("N", -1)),
            str(row.get("heldout", "")),
            str(row.get("line", "")),
            int(row.get("seed", -1)),
            str(row.get("basis_label", "")),
        )
        if key in seen:
            return
        seen.add(key)
        row["_checkpoint_path"] = str(path)
        row["_checkpoint_lineno"] = lineno
        rows.append(row)

    for path in paths:
        path = Path(path)
        n_match = re.search(r"(?:^|_)N(\d+)(?:_|$)", path.name)
        inferred_N = int(n_match.group(1)) if n_match else None
        text = path.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                for idx, row in enumerate(payload["rows"], start=1):
                    add_row(row, path, idx, inferred_N)
                continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            add_row(row, path, lineno, inferred_N)
    return rows, bad


def nested(summary, metric, field="mean"):
    value = summary.get(metric)
    if isinstance(value, dict):
        return value.get(field)
    return value


def fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def compact_rows(summary):
    rows = []
    for key, item in sorted(summary.items()):
        rows.append({
            "variant": key,
            "N": item.get("N", ""),
            "heldout": item.get("heldout", ""),
            "basis": item.get("basis_label", ""),
            "line": item.get("line", ""),
            "n": item.get("n_runs", 0),
            "true_feasible_rate": item.get("true_feasible_rate"),
            "posterior_feasible_rate": item.get("posterior_feasible_rate"),
            "false_feasible_rate": item.get("false_feasible_rate"),
            "regret_median": nested(item, "feasible_simple_regret", "median"),
            "regret_mean": nested(item, "feasible_simple_regret", "mean"),
            "violation_median": nested(item, "constraint_violation", "median"),
            "violation_mean": nested(item, "constraint_violation", "mean"),
            "pool_true_feasible": nested(item, "pool_has_true_feasible_rate", "mean"),
            "selected_true_feasible": nested(
                item,
                "pool_selected_true_feasible_rate",
                "mean",
            ),
            "wall_time_median_sec": nested(item, "wall_time_sec", "median"),
        })
    return rows


def print_compact(rows):
    if not rows:
        print("(no completed checkpoint rows)")
        return
    current_heldout = None
    current_basis = None
    for row in rows:
        if row["heldout"] != current_heldout:
            current_heldout = row["heldout"]
            current_basis = None
            print(f"\n### {current_heldout}")
        if row["basis"] != current_basis:
            current_basis = row["basis"]
            print(f" basis={current_basis}")
        print(
            "  N={N:<4s} {line:12s} n={n:<3d} tf={tf:>5s} pf={pf:>5s} ff={ff:>5s} "
            "reg_med={reg:>8s} viol_med={viol:>8s} "
            "pool_tf={pool:>5s} sel_tf={sel:>5s} wall_med={wall:>7s}s".format(
                N=str(row.get("N", "")),
                line=str(row["line"]),
                n=int(row["n"] or 0),
                tf=fmt(row["true_feasible_rate"], 2),
                pf=fmt(row["posterior_feasible_rate"], 2),
                ff=fmt(row["false_feasible_rate"], 2),
                reg=fmt(row["regret_median"]),
                viol=fmt(row["violation_median"]),
                pool=fmt(row["pool_true_feasible"], 2),
                sel=fmt(row["selected_true_feasible"], 2),
                wall=fmt(row["wall_time_median_sec"], 4),
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="+")
    parser.add_argument("--expected-total", type=int, default=0)
    parser.add_argument("--out-prefix", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows, bad = load_rows(args.checkpoint)
    summary = summarize(rows) if rows else {}
    compact = compact_rows(summary)
    completed = len(rows)
    expected = int(args.expected_total or 0)
    print(json.dumps(json_safe({
        "completed_rows": completed,
        "bad_lines": bad,
        "expected_total": expected or None,
        "progress": (completed / expected if expected else None),
        "n_variants": len(summary),
    }), indent=2))
    print_compact(compact)

    if args.out_prefix:
        prefix = Path(args.out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        (prefix.with_suffix(".json")).write_text(
            json.dumps(json_safe({
                "rows": rows,
                "summary": summary,
                "compact": compact,
            }), indent=2),
            encoding="utf-8",
        )
        write_csv(prefix.with_name(prefix.name + "_summary.csv"), [
            flatten_summary(item) for item in summary.values()
        ])
        write_csv(prefix.with_name(prefix.name + "_compact.csv"), compact)
    if args.json:
        print(json.dumps(json_safe({"summary": summary, "compact": compact}), indent=2))


if __name__ == "__main__":
    main()
