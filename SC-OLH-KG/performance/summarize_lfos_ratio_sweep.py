"""Summarize LF-OS ratio sweep shards.

The paper-facing question is whether regret/feasibility remains stable as
``d / N`` grows.  This helper merges shard summaries and emits a compact
comparison table:

* one row per raw summary entry in ``*_merged_summary.csv``;
* one row per ``(problem, d, N, acquisition, variant)`` in
  ``*_best_by_ratio.csv`` comparing the best LF-OS configuration against the
  synthetic state-coupled baseline.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_quality import write_csv  # noqa: E402


def _to_float(value, default=float("nan")):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0):
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _score(row):
    """Sort key where lower is better.

    Feasibility dominates regret: a false-feasible or infeasible method should
    not win only because its objective looks good.
    """
    true_feasible = _to_float(row.get("true_feasible_rate"), 0.0)
    false_feasible = _to_float(row.get("false_feasible_rate"), 1.0)
    regret = _to_float(row.get("feasible_simple_regret_median"))
    if regret != regret:
        regret = _to_float(row.get("simple_regret_median"), 1e9)
    violation = max(_to_float(row.get("constraint_violation_mean"), 0.0), 0.0)
    wall = _to_float(row.get("wall_time_sec_mean"), 0.0)
    return (
        -true_feasible,
        false_feasible,
        violation,
        regret,
        wall,
    )


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


_RATIO_NAME_RE = re.compile(
    r"(?P<prefix>.+)_d(?P<d>\d+)_N(?P<N>\d+)_"
    r"(?P<encoder>synthetic|lf_os)_lf(?P<lf>\d+)_a(?P<active>\d+)_"
    r"floor(?P<floor>[-+0-9p.]+)_summary\.csv$"
)


def _metadata_from_path(path):
    match = _RATIO_NAME_RE.search(Path(path).name)
    if not match:
        return {}
    problem = match.group("prefix").split("_")[-1]
    d = int(match.group("d"))
    N = int(match.group("N"))
    floor = float(match.group("floor").replace("p", "."))
    return {
        "problem": problem,
        "d": d,
        "N": N,
        "d_over_N": float(d) / max(float(N), 1.0),
        "evals_per_dim": float(N) / max(float(d), 1.0),
        "encoder_kind": match.group("encoder"),
        "lf_os_low_frequency_components": int(match.group("lf")),
        "lf_os_max_active": int(match.group("active")),
        "lf_os_residual_floor_scale": floor,
    }


def _load_rows(patterns):
    rows = []
    seen_paths = set()
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        for path in matches:
            resolved = str(path.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            if path.name.endswith("_merged_summary.csv"):
                continue
            path_meta = _metadata_from_path(path)
            for row in _read_csv(path):
                row = dict(row)
                for key, value in path_meta.items():
                    row.setdefault(key, value)
                # Shard summaries add these fields.  Per-run summaries written
                # by benchmark_quality may share the suffix but do not carry
                # ratio metadata.  For partial runs, recover it from the
                # per-config filename.
                if not (
                    row.get("problem")
                    and row.get("encoder_kind")
                    and row.get("d")
                    and row.get("d_over_N")
                ):
                    continue
                row["source_summary_csv"] = resolved
                rows.append(row)
    unique = {}
    for row in rows:
        key = (
            row.get("problem"),
            str(row.get("d")),
            str(row.get("N")),
            row.get("encoder_kind"),
            str(row.get("lf_os_low_frequency_components")),
            str(row.get("lf_os_max_active")),
            str(row.get("lf_os_residual_floor_scale")),
            row.get("variant"),
            row.get("acquisition_mode"),
            row.get("true_feasible_rate"),
            row.get("false_feasible_rate"),
            row.get("feasible_simple_regret_median"),
            row.get("simple_regret_median"),
        )
        unique.setdefault(key, row)
    return list(unique.values())


def _group_key(row):
    return (
        row.get("problem", ""),
        _to_int(row.get("d")),
        _to_int(row.get("N")),
        row.get("acquisition_mode", ""),
        row.get("variant", ""),
    )


def _compact_metrics(prefix, row):
    if row is None:
        return {
            f"{prefix}_encoder_kind": "",
            f"{prefix}_lf_os_low_frequency_components": "",
            f"{prefix}_lf_os_max_active": "",
            f"{prefix}_lf_os_residual_floor_scale": "",
            f"{prefix}_true_feasible_rate": "",
            f"{prefix}_false_feasible_rate": "",
            f"{prefix}_feasible_regret_median": "",
            f"{prefix}_simple_regret_median": "",
            f"{prefix}_constraint_violation_mean": "",
            f"{prefix}_wall_time_sec_mean": "",
        }
    return {
        f"{prefix}_encoder_kind": row.get("encoder_kind", ""),
        f"{prefix}_lf_os_low_frequency_components": row.get(
            "lf_os_low_frequency_components", ""),
        f"{prefix}_lf_os_max_active": row.get("lf_os_max_active", ""),
        f"{prefix}_lf_os_residual_floor_scale": row.get(
            "lf_os_residual_floor_scale", ""),
        f"{prefix}_true_feasible_rate": row.get("true_feasible_rate", ""),
        f"{prefix}_false_feasible_rate": row.get("false_feasible_rate", ""),
        f"{prefix}_feasible_regret_median": row.get(
            "feasible_simple_regret_median", ""),
        f"{prefix}_simple_regret_median": row.get("simple_regret_median", ""),
        f"{prefix}_constraint_violation_mean": row.get(
            "constraint_violation_mean", ""),
        f"{prefix}_wall_time_sec_mean": row.get("wall_time_sec_mean", ""),
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    best_rows = []
    for key, group_rows in sorted(groups.items()):
        problem, d, N, acquisition, variant = key
        synthetic = [
            row for row in group_rows
            if row.get("encoder_kind") == "synthetic"
        ]
        lfos = [
            row for row in group_rows
            if row.get("encoder_kind") == "lf_os"
        ]
        best_synthetic = min(synthetic, key=_score) if synthetic else None
        best_lfos = min(lfos, key=_score) if lfos else None
        best_any = min(group_rows, key=_score) if group_rows else None
        d_over_N = float(d) / max(float(N), 1.0)
        row = {
            "problem": problem,
            "d": d,
            "N": N,
            "d_over_N": d_over_N,
            "evals_per_dim": float(N) / max(float(d), 1.0),
            "acquisition_mode": acquisition,
            "variant": variant,
            "n_configs": len(group_rows),
            "n_lfos_configs": len(lfos),
            "best_encoder_kind": best_any.get("encoder_kind", "")
            if best_any else "",
        }
        row.update(_compact_metrics("synthetic", best_synthetic))
        row.update(_compact_metrics("lf_os", best_lfos))
        lfos_regret = _to_float(
            row.get("lf_os_feasible_regret_median"))
        syn_regret = _to_float(
            row.get("synthetic_feasible_regret_median"))
        row["lf_os_minus_synthetic_feasible_regret_median"] = (
            lfos_regret - syn_regret
            if lfos_regret == lfos_regret and syn_regret == syn_regret
            else ""
        )
        row["lf_os_minus_synthetic_true_feasible_rate"] = (
            _to_float(row.get("lf_os_true_feasible_rate"), 0.0)
            - _to_float(row.get("synthetic_true_feasible_rate"), 0.0)
        )
        row["lf_os_minus_synthetic_false_feasible_rate"] = (
            _to_float(row.get("lf_os_false_feasible_rate"), 0.0)
            - _to_float(row.get("synthetic_false_feasible_rate"), 0.0)
        )
        best_rows.append(row)

    best_rows.sort(key=lambda r: (
        r["problem"],
        -float(r["d_over_N"]),
        int(r["N"]),
        r["acquisition_mode"],
        r["variant"],
    ))
    return best_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["SC-OLH-KG/profiles/lfos_ratio_stage1_20260707*_summary.csv"],
    )
    parser.add_argument(
        "--out-prefix",
        default="SC-OLH-KG/profiles/lfos_ratio_stage1_20260707",
    )
    args = parser.parse_args()

    rows = _load_rows(args.inputs)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    merged_path = out_prefix.with_name(out_prefix.name + "_merged_summary.csv")
    best_path = out_prefix.with_name(out_prefix.name + "_best_by_ratio.csv")
    write_csv(merged_path, rows)
    best_rows = summarize(rows)
    write_csv(best_path, best_rows)
    print({
        "n_rows": len(rows),
        "n_best_rows": len(best_rows),
        "merged_summary_csv": str(merged_path),
        "best_by_ratio_csv": str(best_path),
    })


if __name__ == "__main__":
    main()
