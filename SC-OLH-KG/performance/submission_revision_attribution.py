#!/usr/bin/env python3
"""Frozen first-center/metric attribution and new-family confirmation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core.profile_atlas import AtlasSelection, ProfileAtlasConfig, SourceScoredProfileAtlas, farthest_first_indices, profile_cosine_coordinate
from problems.randomized_profiles import PROFILE_STRESS_REGIMES, RandomizedOrderedProfileProblem, generate_structural_profile_library, source_profile_records
from performance import benchmark_profile_stress_suite as benchmark
from performance.submission_revision_controls import compact_row, summarize, source_greedy_portfolio

PROTOCOL = ROOT / "performance/manifests/submission_revision_20260930.json"
OUTPUT = ROOT / "paper_artifacts/submission_revision_20260930"


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def selections(records):
    """All interventions use identical library values and identical source ranks."""
    all_members = SourceScoredProfileAtlas(ProfileAtlasConfig(n0=64 if len({r.profile_id for r in records}) == 64 else len({r.profile_id for r in records}))).fit(records).selected()
    members = sorted(all_members.members, key=lambda m: m.profile_id)
    coordinates = np.vstack([profile_cosine_coordinate(m.profile, nodes=m.nodes, max_frequency=8, frequency_penalty=.25, include_diagonal_quadratic=True) for m in members])
    scale = coordinates.std(axis=0)
    z = (coordinates-coordinates.mean(axis=0))/np.where(scale > 1e-10, scale, 1)
    safety = np.array([m.safety_rank for m in members])
    objective = np.array([m.objective_rank for m in members])
    best = min(range(len(members)), key=lambda i: (.5*safety[i]+.5*objective[i], safety[i], objective[i], members[i].profile_id))
    distance = np.linalg.norm(z[:, None]-z[None, :], axis=2)
    medoid = int(np.argmin(distance.mean(axis=1)))
    augmented = np.column_stack([z, safety, objective])
    nodes = np.array(members[0].nodes)
    edges = np.r_[0., (nodes[:-1]+nodes[1:])/2, 1.]
    raw = np.vstack([m.profile for m in members])*np.sqrt(np.diff(edges))
    specifications = {
        "source_atlas": (best, augmented),
        "standardized_generic_dct_maximin": (medoid, z),
        "source_best_z": (best, z),
        "medoid_augmented": (medoid, augmented),
        "source_best_raw_l2": (best, raw),
    }
    result = {}
    for arm, (first, metric) in specifications.items():
        chosen = farthest_first_indices(metric, 10, initial_index=first)
        result[arm] = AtlasSelection(tuple(replace(members[i], selected_order=j+1) for j, i in enumerate(chosen)), {
            "first_center": members[first].profile_id,
            "first_center_rule": "fixed_structural_medoid" if arm in {"standardized_generic_dct_maximin", "medoid_augmented"} else "minimum_weighted_source_rank",
            "coordinate_dimension": int(metric.shape[1]),
            "source_outcomes_used": arm != "standardized_generic_dct_maximin",
            "target_outcomes_used": False, "target_oracle_used": False,
        })
    result["source_greedy_portfolio"] = source_greedy_portfolio(records, 10)
    return result


class FixedSelection:
    def __init__(self, selection):
        self.selection = selection
    def fit(self, records, *, target_descriptor=None):
        return self
    def selected(self):
        return self.selection


def run_group(group):
    started = time.perf_counter()
    family, regime, size = group["family_seed"], group["regime"], group["library_size"]
    library = generate_structural_profile_library(size, dimension=128, seed=family+991, maximum_frequency=40)
    source_family = family + benchmark._stable_seed(regime)
    sources = [RandomizedOrderedProfileProblem(regime=regime, role="source", task_seed=10000+i, family_seed=source_family, d=256) for i in range(2)]
    records = source_profile_records(sources, library, replications=3, seed=family+1237)
    designs = selections(records)
    original = {}
    if group["study"] == "attribution":
        references = json.loads((ROOT / "paper_artifacts/submission_revision_20260907/rows.json").read_text())["rows"]
        original = {(r["target_seed"], r["arm"]): r for r in references if r["regime"] == regime and r["nominal_dimension"] == group["dimension"] and r["arm"] in {"source_atlas", "standardized_generic_dct_maximin"}}
    rows = []
    actual_evaluations = 0
    for task in group["tasks"]:
        for arm in group["arms"]:
            selected = designs[arm]
            ids = [m.profile_id for m in selected.members]
            reusable = next((r for (seed, _), r in original.items() if seed == task["target_seed"] and r["design_seed"] == task["design_seed"] and r["selected_profile_ids"] == ids), None)
            if reusable is not None:
                row = dict(reusable)
                row["evaluation_reused_from"] = reusable["arm"]
            else:
                kwargs = dict(regime=regime, target_seed=task["target_seed"], design_seed=task["design_seed"], dimension=group["dimension"], family_seed=family, library_size=size, N=10)
                with patch.object(benchmark, "source_profile_records", return_value=records):
                    if arm == "standardized_generic_dct_maximin":
                        indices = tuple(next(i for i, p in enumerate(library) if p.profile_id == pid) for pid in ids)
                        with patch.object(benchmark, "generic_dct_maximin", return_value=(indices, selected.diagnostics)):
                            result = benchmark.run_task(arm="generic_dct_maximin", **kwargs)
                    else:
                        with patch.object(benchmark, "SourceScoredProfileAtlas", lambda config: FixedSelection(selected)):
                            result = benchmark.run_task(arm="source_atlas", **kwargs)
                row = compact_row(result, arm=arm, provenance="new_evaluation_under_frozen_20260930_rules")
                actual_evaluations += 1
            row.update(arm=arm, family_seed=family, library_size=size, study=group["study"], frontend_diagnostics=selected.diagnostics)
            row["source_calls"] = 0 if arm == "standardized_generic_dct_maximin" else 2*size*3
            row["all_in_calls_unamortized"] = row["source_calls"]+row["target_search_calls"]+row["verification_calls"]
            row["all_in_calls_amortized"] = row["source_calls"]/20+row["target_search_calls"]+row["verification_calls"]
            rows.append(row)
    payload = {"rows": rows, "actual_target_evaluations": actual_evaluations, "source_archive_calls_once": 2*size*3, "wall_time_sec": time.perf_counter()-started}
    write(OUTPUT / "profile_groups" / (group["name"]+".json"), payload)
    return payload


def groups(protocol):
    result = []
    references = json.loads((ROOT / "paper_artifacts/submission_revision_20260907/rows.json").read_text())["rows"]
    for dimension in protocol["attribution"]["dimensions"]:
        for regime in PROFILE_STRESS_REGIMES:
            tasks = [dict(target_seed=r["target_seed"], design_seed=r["design_seed"]) for r in references if r["arm"] == "source_atlas" and r["regime"] == regime and r["nominal_dimension"] == dimension]
            result.append(dict(study="attribution", name=f"attribution_d{dimension}_{regime}", family_seed=20260808, library_size=64, dimension=dimension, regime=regime, tasks=tasks, arms=protocol["attribution"]["arms"]))
    confirm = protocol["confirmation"]
    for fi, family in enumerate(confirm["family_seeds"]):
        for ri, regime in enumerate(PROFILE_STRESS_REGIMES):
            tasks = [dict(target_seed=700000000+10000*fi+100*ri+i, design_seed=800000000+10000*fi+100*ri+i) for i in range(10)]
            result.append(dict(study="confirmation", name=f"confirmation_f{family}_{regime}", family_seed=family, library_size=64, dimension=1000, regime=regime, tasks=tasks, arms=confirm["arms"]))
            if fi == 0:
                for size in confirm["library_size_sensitivity"]["sizes"]:
                    result.append(dict(study="library_size", name=f"library_size_{size}_{regime}", family_seed=family, library_size=size, dimension=1000, regime=regime, tasks=tasks, arms=confirm["library_size_sensitivity"]["arms"]))
    return result


def paired(rows, control):
    key = lambda r: (r["family_seed"], r["regime"], r["target_seed"], r["design_seed"])
    source = {key(r): r for r in rows if r["arm"] == "source_atlas"}
    comparison = {key(r): r for r in rows if r["arm"] == control}
    if source.keys() != comparison.keys():
        raise ValueError("paired contrasts require the same complete family-task-design keys")
    fields = ("certified_true_feasible", "initial_design_contains_true_feasible", "penalized_loss")
    strata = {}
    for k in sorted(source):
        strata.setdefault(k[:2], []).append([float(source[k][f])-float(comparison[k][f]) for f in fields])
    rng = np.random.default_rng(20260930)
    boot = np.zeros((10000, len(fields)))
    for values in strata.values():
        a = np.array(values)
        boot += a[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)/len(strata)
    return {"comparator": control, "paired_task_count": len(source), "fixed_strata": len(strata), "endpoints": {f: {"source_minus_control": float(np.mean([float(source[k][f])-float(comparison[k][f]) for k in source])), "paired_bootstrap_95ci": np.quantile(boot[:,i], [.025,.975]).tolist()} for i,f in enumerate(fields)}}


def analyze(rows):
    output = {"status": "complete", "row_count": len(rows), "studies": []}
    for study in ("attribution", "confirmation", "library_size"):
        subset = [r for r in rows if r["study"] == study]
        for dimension, size in sorted({(r["nominal_dimension"], r["library_size"]) for r in subset}):
            selected = [r for r in subset if r["nominal_dimension"] == dimension and r["library_size"] == size]
            arms = sorted({r["arm"] for r in selected})
            summaries = [summarize([r for r in selected if r["arm"] == arm]) for arm in arms]
            families = [{"family_seed": family, "summaries": [summarize([r for r in selected if r["arm"] == arm and r["family_seed"] == family]) for arm in arms], "paired": [paired([r for r in selected if r["family_seed"] == family], arm) for arm in arms if arm != "source_atlas"]} for family in sorted({r["family_seed"] for r in selected})]
            regimes = [{"regime": regime, "summaries": [summarize([r for r in selected if r["arm"] == arm and r["regime"] == regime]) for arm in arms]} for regime in PROFILE_STRESS_REGIMES]
            cost = []
            for arm in arms:
                arm_rows = [r for r in selected if r["arm"] == arm]
                archive_keys = {(r["family_seed"], r["regime"]) for r in arm_rows}
                source_once = sum(next(r["source_calls"] for r in arm_rows if (r["family_seed"], r["regime"]) == key) for key in archive_keys)
                cumulative = source_once + sum(r["target_search_calls"]+r["verification_calls"] for r in arm_rows)
                certified = sum(r["certified_true_feasible"] for r in arm_rows)
                cost.append(dict(arm=arm, target_task_count=len(arm_rows), archive_count=len(archive_keys) if source_once else 0, source_calls_once=source_once, cumulative_calls=cumulative, certified_true_feasible_count=certified, calls_per_true_certificate=cumulative/certified if certified else None))
            output["studies"].append(dict(study=study, dimension=dimension, library_size=size, summaries=summaries, paired=[paired(selected, arm) for arm in arms if arm != "source_atlas"], by_family=families, by_regime=regimes, actual_archive_reuse_cost=cost))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.analyze_only:
        payload = json.loads((OUTPUT / "profile_rows.json").read_text())
    else:
        jobs = groups(protocol)
        payloads = []
        pending = []
        for group in jobs:
            path = OUTPUT / "profile_groups" / (group["name"]+".json")
            if path.exists():
                payloads.append(json.loads(path.read_text()))
            else:
                pending.append(group)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_group, group) for group in pending]
            for future in as_completed(futures):
                payloads.append(future.result())
                print(f"PROFILE_GROUPS_COMPLETED {len(payloads)}/{len(jobs)}", flush=True)
        rows = [r for p in payloads for r in p["rows"]]
        payload = {"protocol": str(PROTOCOL.relative_to(ROOT)), "actual_target_evaluations": sum(p["actual_target_evaluations"] for p in payloads), "rows": rows}
        if len(rows) != 4160:
            raise ValueError(f"expected 4160 paired rows, got {len(rows)}")
        write(OUTPUT / "profile_rows.json", payload)
    result = analyze(payload["rows"])
    write(OUTPUT / "profile_analysis.json", result)
    write(OUTPUT / "first_center_truth_diagnostic.json", first_center_diagnostic(payload["rows"]))
    for study in result["studies"]:
        print(json.dumps({k:study[k] for k in ("study", "dimension", "library_size", "paired")}), flush=True)


def first_center_diagnostic(rows):
    rows = [r for r in rows if r["study"] == "confirmation" and r["arm"] in {"source_atlas", "standardized_generic_dct_maximin"}]
    libraries = {family: {p.profile_id: p for p in generate_structural_profile_library(64, dimension=128, seed=family+991, maximum_frequency=40)} for family in {r["family_seed"] for r in rows}}
    truth = []
    for row in rows:
        target = RandomizedOrderedProfileProblem(regime=row["regime"], role="target", task_seed=row["target_seed"], family_seed=row["family_seed"]+benchmark._stable_seed(row["regime"]), d=1000)
        profile = libraries[row["family_seed"]][row["selected_profile_ids"][0]]
        point = target.point_from_structural_profile(profile)
        truth.append({k:row[k] for k in ("arm", "family_seed", "regime", "target_seed")} | dict(first_profile_id=profile.profile_id, first_true_feasible=target.is_truly_feasible(point), first_chance_margin=target.true_chance_margin(point), first_true_objective=target.true_objective(point)))
    summaries = []
    for regime in PROFILE_STRESS_REGIMES:
        for arm in ("source_atlas", "standardized_generic_dct_maximin"):
            selected = [r for r in truth if r["regime"] == regime and r["arm"] == arm]
            summaries.append(dict(regime=regime, arm=arm, task_count=len(selected), first_true_feasible_count=sum(r["first_true_feasible"] for r in selected), mean_chance_margin=float(np.mean([r["first_chance_margin"] for r in selected]))))
    return dict(role="postdecision diagnostic only; target truth was not used for selection", rows=truth, summaries=summaries)


if __name__ == "__main__":
    main()
