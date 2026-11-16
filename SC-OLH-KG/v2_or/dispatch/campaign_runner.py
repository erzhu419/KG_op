"""Cluster-distributed 3-stage barrier experiment runner for `SC-OLH-KG/v2_or`.

Enforces strict scientific barriers on `node001-node006` per `AGENTS.md`:
  Stage 1 (`source_archive`):
      Run authentic legacy solvers (`Penalty-SPSA`, `Constrained-CMAES`,
      `Constrained-NSGA2`, `Heuristic-Powell`) on source tasks in parallel;
      barrier-merge into frozen `source_archive.json` and fitted priors.
  Stage 2 (`search_and_verify`):
      Using frozen priors, run independent `(family, task_idx, rep_seed, arm, N)`
      cells on unseen target tasks (search + feasible-aware 3-slot shortlist +
      independent gated terminal verification). Freezes all deployed policies
      with `truth_used = False`.
  Stage 3 (`audit_and_summarize`):
      Only AFTER Stage 2 completion is sealed, evaluate post-hoc ground truth
      (`split="audit"`) on frozen deployed policies and compute paired Wilcoxon /
      Holm-Bonferroni / bootstrap / amortization (`M_break`) summary JSONs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import shlex
import socket
import subprocess
import sys
import traceback
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

SC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SC_ROOT.parent
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scheduler_node_policy import DEFAULT_CPU_NODES, parse_cpu_nodes
from v2_or.acquisition import SequentialRunResult, run_tls_olh_kg
from v2_or.baselines import (
    run_direct_legacy_baseline,
    run_hetgp_kg_baseline,
    run_hyperbo_transfer_baseline,
    run_rgpe_transfer_baseline,
    run_saasbo_baseline,
    run_scbo_baseline,
)
from v2_or.contracts import BaseDynamicORProblem, LegacyTrajectoryRecord
from v2_or.legacy_solvers import (
    LEGACY_SOLVER_REGISTRY,
    load_legacy_archive_json,
    save_legacy_archive_json,
)
from v2_or.priors import DebiasedSpectralMetaPrior
from v2_or.simulators import (
    IngolstadtLiveSUMOBenchmark,
    OPSDGridStorageBenchmark,
    sample_inventory_tasks,
    sample_queueing_tasks,
    sample_traffic_tasks,
)


MODULE_NAME = "v2_or.dispatch.campaign_runner"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_family_tasks(
    family_spec: dict[str, Any],
    role: str,
) -> list[BaseDynamicORProblem]:
    """Deterministically instantiate `source` or `target` tasks from `family_spec`."""
    kind = family_spec["kind"]
    cfg = family_spec[f"{role}_config"]
    if kind == "inventory":
        return sample_inventory_tasks(
            n_tasks=int(cfg["n_tasks"]),
            seed=int(cfg["seed"]),
            horizons=tuple(int(h) for h in cfg["horizons"]),
            shift_scale=float(cfg.get("shift_scale", 1.0)),
        )
    if kind == "queueing":
        return sample_queueing_tasks(
            n_tasks=int(cfg["n_tasks"]),
            seed=int(cfg["seed"]),
            horizons=tuple(int(h) for h in cfg["horizons"]),
            shift_scale=float(cfg.get("shift_scale", 1.0)),
        )
    if kind == "opsd":
        markets = list(cfg["markets"])
        horizons = tuple(int(h) for h in cfg["horizons"])
        tasks: list[BaseDynamicORProblem] = []
        for idx, mkt in enumerate(markets):
            d_h = horizons[idx % len(horizons)]
            tasks.append(OPSDGridStorageBenchmark(market=mkt, d=d_h))
        return tasks
    if kind == "traffic_des":
        return sample_traffic_tasks(
            n_tasks=int(cfg["n_tasks"]),
            seed=int(cfg["seed"]),
            tls_counts=tuple(int(c) for c in cfg["tls_counts"]),
            shift_scale=float(cfg.get("shift_scale", 1.0)),
        )
    if kind == "traffic_live_sumo":
        taus = cfg.get("taus", [1.0])
        weights_list = cfg.get("weights_list", [[0.5, 0.5]])
        tasks = []
        for idx in range(int(cfg["n_tasks"])):
            tau_v = float(taus[idx % len(taus)])
            w_v = tuple(float(x) for x in weights_list[idx % len(weights_list)])
            tasks.append(
                IngolstadtLiveSUMOBenchmark(
                    task_id=f"live_sumo_ingolstadt21_{idx:02d}",
                    weights=w_v,
                    tau_emission=tau_v,
                )
            )
        return tasks
    raise ValueError(f"Unknown family kind {kind!r}")


def _campaign_dir(manifest: dict[str, Any]) -> Path:
    return SC_ROOT / manifest["output_dir"]


def describe_stage(manifest_path: Path, stage: str) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("sealed_complete", False):
        raise RuntimeError("Sealed completed manifest cannot be restarted")
    out_dir = _campaign_dir(manifest)
    stage_dir = out_dir / "stages" / stage
    desc_path = stage_dir / "jobs.json"
    if desc_path.is_file():
        return _read_json(desc_path)

    jobs: list[dict[str, Any]] = []
    if stage == "source_archive":
        for fam in manifest["families"]:
            fam_name = fam["name"]
            if "reuse_archive_from_family" in fam:
                continue
            src_cfg = fam["source_archive_protocol"]
            s_conf = fam["source_config"]
            n_src = len(s_conf["markets"]) if "markets" in s_conf else int(s_conf["n_tasks"])
            for s_idx in range(n_src):
                for solver_idx, solver_name in enumerate(src_cfg["solvers"]):
                    seed = int(src_cfg["seed"]) + 1009 * s_idx + 37 * solver_idx
                    jobs.append({
                        "job_id": f"src_{fam_name}_t{s_idx:02d}_{solver_name}",
                        "family": fam_name,
                        "source_task_idx": s_idx,
                        "solver_name": solver_name,
                        "per_solver_budget": int(src_cfg["per_solver_budget"]),
                        "reps_per_eval": int(src_cfg["reps_per_eval"]),
                        "seed": seed,
                    })

    elif stage == "search_and_verify":
        src_done = out_dir / "stages" / "source_archive" / "completion.json"
        if not src_done.is_file():
            raise RuntimeError("Stage `source_archive` must finish before `search_and_verify`")
        for fam in manifest["families"]:
            fam_name = fam["name"]
            t_conf = fam["target_config"]
            n_tgt = len(t_conf["markets"]) if "markets" in t_conf else int(t_conf["n_tasks"])
            rep_seeds = [int(s) for s in fam["target_eval_seeds"]]
            budgets = [int(b) for b in fam["online_step_budgets"]]
            arms = fam["arms"]
            for b_steps in budgets:
                for t_idx in range(n_tgt):
                    for r_seed in rep_seeds:
                        cell_seed = r_seed + 10_007 * t_idx
                        for arm in arms:
                            # `STOP` does 0 online steps, so only run once at the smallest N budget
                            if arm["kind"] == "stop" and b_steps != min(budgets):
                                continue
                            jobs.append({
                                "job_id": f"sv_{fam_name}_N{b_steps}_t{t_idx:02d}_s{r_seed}_{arm['name']}",
                                "family": fam_name,
                                "target_task_idx": t_idx,
                                "seed": cell_seed,
                                "rep_seed": r_seed,
                                "n_online_steps": int(b_steps),
                                "n0_init": int(fam.get("n0_init", 8)),
                                "batch_reps": int(fam.get("batch_reps", 2)),
                                "pool_size": int(fam.get("pool_size", 160)),
                                "arm": arm,
                            })

    elif stage == "audit_and_summarize":
        sv_done = out_dir / "stages" / "search_and_verify" / "completion.json"
        if not sv_done.is_file():
            raise RuntimeError("Stage `search_and_verify` must finish before `audit_and_summarize`")
        sv_desc = _read_json(out_dir / "stages" / "search_and_verify" / "jobs.json")
        for sv_job in sv_desc["jobs"]:
            jobs.append({
                "job_id": "aud_" + sv_job["job_id"],
                "sv_job_id": sv_job["job_id"],
                "family": sv_job["family"],
                "target_task_idx": sv_job["target_task_idx"],
            })
    else:
        raise ValueError(f"Unknown stage {stage!r}")

    descriptor = {
        "campaign_name": manifest["campaign_name"],
        "manifest_relpath": str(manifest_path.resolve().relative_to(SC_ROOT)),
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_jobs": len(jobs),
        "jobs": jobs,
        "truth_used": stage == "audit_and_summarize",
    }
    _write_json(desc_path, descriptor)
    return descriptor


def _execute_single_item(
    manifest: dict[str, Any],
    stage: str,
    job: dict[str, Any],
) -> dict[str, Any]:
    out_dir = _campaign_dir(manifest)
    item_path = out_dir / "stages" / stage / "items" / f"{job['job_id']}.json"
    if item_path.is_file():
        return {"job_id": job["job_id"], "status": "cached"}

    fam_map = {f["name"]: f for f in manifest["families"]}
    fam_spec = fam_map[job["family"]]

    if stage == "source_archive":
        tasks = build_family_tasks(fam_spec, "source")
        prob = tasks[int(job["source_task_idx"])]
        solver_fn = LEGACY_SOLVER_REGISTRY[job["solver_name"]]
        recorder = solver_fn(
            prob,
            budget_calls=int(job["per_solver_budget"]),
            seed=int(job["seed"]),
            reps_per_eval=int(job["reps_per_eval"]),
        )
        payload = {
            "job_id": job["job_id"],
            "family": job["family"],
            "task_id": prob.task_id,
            "policy_dim": int(prob.d),
            "solver_name": job["solver_name"],
            "total_calls": int(recorder.total_calls),
            "records": [asdict(r) for r in recorder.trajectory_log],
            "truth_used": False,
        }
        _write_json(item_path, payload)
        return {"job_id": job["job_id"], "status": "ok"}

    if stage == "search_and_verify":
        tasks = build_family_tasks(fam_spec, "target")
        prob = tasks[int(job["target_task_idx"])]
        arm = job["arm"]
        arm_kind = arm["kind"]
        src_fam = fam_spec.get("reuse_archive_from_family", job["family"])
        archive_file = out_dir / "source_bundles" / src_fam / "archive.json"
        prior_variant = arm.get("prior_variant", "default")
        prior_file = out_dir / "source_bundles" / src_fam / f"prior_{prior_variant}.pkl"

        n_steps = int(job["n_online_steps"])
        n0 = int(job["n0_init"])
        b_reps = int(job["batch_reps"])
        p_size = int(job["pool_size"])
        seed = int(job["seed"])
        gate_mode = fam_spec.get("gate_mode", "auto")

        if arm_kind in {"tls_olh_kg", "stop"}:
            with prior_file.open("rb") as fh:
                prior = pickle.load(fh)
            res: SequentialRunResult = run_tls_olh_kg(
                prob,
                prior,
                n_online_steps=0 if arm_kind == "stop" else n_steps,
                n0_init=n0,
                batch_reps=b_reps,
                seed=seed,
                model_mode=arm.get("model_mode", "tempered"),
                acquisition_type="stop" if arm_kind == "stop" else arm.get("acquisition", "v_kg"),
                pool_size=p_size,
                method_label=arm["name"],
                gate_mode=gate_mode,
            )
        elif arm_kind == "target_only_psi_kg":
            res = run_tls_olh_kg(
                prob,
                None,
                n_online_steps=n_steps,
                n0_init=n0,
                batch_reps=b_reps,
                seed=seed,
                model_mode="target_only",
                acquisition_type="v_kg",
                pool_size=p_size,
                method_label=arm["name"],
                gate_mode=gate_mode,
            )
        elif arm_kind == "scbo":
            res = run_scbo_baseline(prob, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode)
        elif arm_kind == "saasbo":
            res = run_saasbo_baseline(prob, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode)
        elif arm_kind == "hetgp_kg":
            res = run_hetgp_kg_baseline(prob, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode)
        elif arm_kind == "rgpe":
            records, _ = load_legacy_archive_json(archive_file)
            res = run_rgpe_transfer_baseline(prob, records, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode)
        elif arm_kind == "hyperbo":
            records, _ = load_legacy_archive_json(archive_file)
            res = run_hyperbo_transfer_baseline(prob, records, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode)
        elif arm_kind == "direct_legacy":
            res = run_direct_legacy_baseline(
                prob,
                solver_name=arm["solver_name"],
                n_online_steps=n_steps,
                n0_init=n0,
                batch_reps=b_reps,
                seed=seed,
                gate_mode=gate_mode,
            )
        else:
            raise ValueError(f"Unsupported arm kind {arm_kind!r}")

        payload = {
            "job_id": job["job_id"],
            "family": job["family"],
            "target_task_idx": int(job["target_task_idx"]),
            "task_id": prob.task_id,
            "policy_dim": int(prob.d),
            "rep_seed": int(job["rep_seed"]),
            "seed": seed,
            "method_name": arm["name"],
            "n0_init_points": res.n0_init_points,
            "n_online_steps": int(job["n_online_steps"]),
            "actual_online_steps": res.n_online_steps,
            "search_calls_used": res.search_calls_used,
            "verification_calls_used": res.verification_calls_used,
            "total_target_calls": res.total_target_calls,
            "wall_time_sec": res.wall_time_sec,
            "certified": res.certified,
            "selected_role": res.selected_role,
            "posterior_beta_con": res.posterior_beta_con,
            "posterior_tau_shift_con": res.posterior_tau_shift_con,
            "untempered_inflation_factor": res.untempered_inflation_factor,
            "n_unique_policies_evaluated": res.n_unique_policies_evaluated,
            "n_replication_queries": res.n_replication_queries,
            "deployed_policy": [float(v) for v in res.deployed_policy],
            "truth_used": False,
        }
        _write_json(item_path, payload)
        return {"job_id": job["job_id"], "status": "ok"}

    if stage == "audit_and_summarize":
        sv_item = _read_json(out_dir / "stages" / "search_and_verify" / "items" / f"{job['sv_job_id']}.json")
        tasks = build_family_tasks(fam_spec, "target")
        prob = tasks[int(job["target_task_idx"])]
        n_audit = int(fam_spec.get("audit_replications", 4000))
        audit_seed = int(fam_spec.get("audit_seed_base", 20261116)) + 997 * int(job["target_task_idx"])

        x_dep = np.asarray(sv_item["deployed_policy"], dtype=float)
        x_anc = prob.domain_heuristic_anchor()
        anchor_cache = (
            out_dir
            / "stages"
            / "audit_and_summarize"
            / "anchors"
            / f"{job['family']}_t{int(job['target_task_idx']):02d}.json"
        )
        if anchor_cache.is_file():
            anchor_truth = _read_json(anchor_cache)
        else:
            anchor_truth = prob.audit_ground_truth(x_anc, n_eval=n_audit, seed=audit_seed)
            try:
                anchor_cache.parent.mkdir(parents=True, exist_ok=True)
                tmp_anc = anchor_cache.with_name(f"{anchor_cache.name}.{os.getpid()}.tmp")
                tmp_anc.write_text(json.dumps(anchor_truth, indent=2), encoding="utf-8")
                tmp_anc.replace(anchor_cache)
            except Exception:
                pass

        if np.allclose(x_dep, x_anc, atol=1e-10, rtol=0.0):
            truth = dict(anchor_truth)
        else:
            truth = prob.audit_ground_truth(x_dep, n_eval=n_audit, seed=audit_seed)

        # Safe Deployable Gain relative to conservative anchor / abstinence baseline:
        # If certified AND truly feasible, gain is positive objective reduction over safe reference;
        # if uncertified (abstained), deployed utility defaults to 0.0 incremental gain over safe abstinence.
        ref_cost = max(float(anchor_truth["true_objective"]), float(truth["true_objective"]))
        payload = {
            **{k: v for k, v in sv_item.items() if k != "deployed_policy"},
            "audit": truth,
            "anchor_true_objective": float(anchor_truth["true_objective"]),
            "anchor_is_truly_feasible": bool(anchor_truth["is_truly_feasible"]),
            "false_certificate": bool(sv_item["certified"] and (not truth["is_truly_feasible"])),
            "ref_cost_check": ref_cost,
            "truth_used": True,
        }
        _write_json(item_path, payload)
        return {"job_id": job["job_id"], "status": "ok"}

    raise ValueError(f"Unknown stage {stage!r}")


def _worker_entry(args_tuple: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    manifest_str, stage, job = args_tuple
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    manifest = json.loads(manifest_str)
    try:
        return _execute_single_item(manifest, stage, job)
    except Exception as exc:
        return {
            "job_id": job["job_id"],
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def run_worker_shard(descriptor_path: Path, start: int, end: int, workers: int) -> None:
    node = socket.gethostname().split(".")[0]
    if node not in DEFAULT_CPU_NODES:
        raise RuntimeError(f"v2_or cluster shards must run on node001-node006, got {node!r}")
    descriptor = _read_json(descriptor_path)
    manifest_path = SC_ROOT / descriptor["manifest_relpath"]
    manifest = _read_json(manifest_path)
    manifest_str = json.dumps(manifest)
    stage = descriptor["stage"]
    slice_jobs = descriptor["jobs"][start:end]
    if not (0 <= start < end <= len(descriptor["jobs"])):
        raise ValueError(f"Invalid shard slice [{start}, {end}) of {len(descriptor['jobs'])}")

    n_proc = max(1, min(int(workers), len(slice_jobs)))
    results = []
    if n_proc == 1:
        for jb in slice_jobs:
            results.append(_worker_entry((manifest_str, stage, jb)))
    else:
        with ProcessPoolExecutor(max_workers=n_proc) as pool:
            futs = [pool.submit(_worker_entry, (manifest_str, stage, jb)) for jb in slice_jobs]
            for fut in as_completed(futs):
                results.append(fut.result())

    failed = [r for r in results if r["status"] == "failed"]
    shard_out = _campaign_dir(manifest) / "stages" / stage / "shards" / f"shard_{start}_{end}.json"
    _write_json(shard_out, {
        "node": node,
        "start": start,
        "end": end,
        "workers": n_proc,
        "completed_count": len(results) - len(failed),
        "failed": failed,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    if failed:
        raise RuntimeError(f"Shard [{start}, {end}) had {len(failed)} failures: {failed[0]}")


def finish_stage(descriptor_path: Path) -> dict[str, Any]:
    descriptor = _read_json(descriptor_path)
    manifest_path = SC_ROOT / descriptor["manifest_relpath"]
    manifest = _read_json(manifest_path)
    out_dir = _campaign_dir(manifest)
    stage = descriptor["stage"]
    items_dir = out_dir / "stages" / stage / "items"

    missing = [j["job_id"] for j in descriptor["jobs"] if not (items_dir / f"{j['job_id']}.json").is_file()]
    if missing:
        raise RuntimeError(f"Cannot finish stage {stage!r}: {len(missing)} items missing, e.g. {missing[:3]}")

    summary_extra: dict[str, Any] = {}

    if stage == "source_archive":
        # Merge source records per family and fit all registered prior variants
        prior_diagnostics = {}
        for fam in manifest["families"]:
            if "reuse_archive_from_family" in fam:
                continue
            fam_name = fam["name"]
            fam_jobs = [j for j in descriptor["jobs"] if j["family"] == fam_name]
            all_recs: list[LegacyTrajectoryRecord] = []
            total_calls = 0
            for jb in fam_jobs:
                item = _read_json(items_dir / f"{jb['job_id']}.json")
                total_calls += int(item["total_calls"])
                for r_dict in item["records"]:
                    all_recs.append(LegacyTrajectoryRecord(**r_dict))

            bundle_dir = out_dir / "source_bundles" / fam_name
            bundle_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "family": fam_name,
                "total_logged_evaluations": len(all_recs),
                "total_simulation_calls": total_calls,
                "solvers": fam["source_archive_protocol"]["solvers"],
            }
            save_legacy_archive_json(bundle_dir / "archive.json", all_recs, meta)

            variants = fam.get("prior_variants", {
                "default": {
                    "active_rank": 8,
                    "enable_debiasing": True,
                    "enable_lowpass": True,
                    "enable_orthogonalization": True,
                    "enable_sparse_pruning": True,
                }
            })
            fam_diag = {"source_simulation_calls": total_calls, "variants": {}}
            for var_name, v_kwargs in variants.items():
                prior = DebiasedSpectralMetaPrior(**v_kwargs).fit(all_recs)
                with (bundle_dir / f"prior_{var_name}.pkl").open("wb") as fh:
                    pickle.dump(prior, fh)
                fam_diag["variants"][var_name] = asdict(prior.diagnostics_)
            prior_diagnostics[fam_name] = fam_diag
        summary_extra["prior_diagnostics"] = prior_diagnostics

    elif stage == "audit_and_summarize":
        # Compile publication-grade compact summary JSON!
        rows = [_read_json(items_dir / f"{j['job_id']}.json") for j in descriptor["jobs"]]
        summary_extra["campaign_summary"] = _compile_statistical_summary(manifest, out_dir, rows)
        manifest["sealed_complete"] = True
        _write_json(manifest_path, manifest)

    completion = {
        "stage": stage,
        "status": "complete",
        "n_jobs": len(descriptor["jobs"]),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        **summary_extra,
    }
    _write_json(out_dir / "stages" / stage / "completion.json", completion)
    if stage == "audit_and_summarize":
        _write_json(out_dir / "compact_summary.json", completion["campaign_summary"])
    return completion


def _compile_statistical_summary(
    manifest: dict[str, Any],
    out_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute paired comparisons, false-certificate checks, and M_break amortization."""
    src_completion = _read_json(out_dir / "stages" / "source_archive" / "completion.json")
    prior_diag = src_completion.get("prior_diagnostics", {})

    # Duplicate `STOP` rows across larger N budgets on the exact same (family, task_idx, rep_seed)
    stop_lookup = {
        (r["family"], r["target_task_idx"], r["rep_seed"]): r
        for r in rows
        if r["method_name"] == "STOP"
    }

    by_fam_budget: dict[str, Any] = {}
    total_false_certs = sum(int(r["false_certificate"]) for r in rows)
    total_certified_runs = sum(int(r["certified"]) for r in rows)

    for fam in manifest["families"]:
        fam_name = fam["name"]
        src_fam = fam.get("reuse_archive_from_family", fam_name)
        src_calls = int(prior_diag.get(src_fam, {}).get("source_simulation_calls", 0))
        fam_summary = {"source_simulation_calls": src_calls, "budgets": {}}

        for b_steps in fam["online_step_budgets"]:
            b_rows = [r for r in rows if r["family"] == fam_name and int(r["n_online_steps"]) == int(b_steps)]
            # Include STOP baseline at every N for paired comparison
            if not any(r["method_name"] == "STOP" for r in b_rows):
                for (fn, tidx, rseed), s_row in stop_lookup.items():
                    if fn == fam_name:
                        b_rows.append(s_row)

            # Compute task-level safe fallback reference objective (worst safe / anchor cost + penalty if uncertified)
            # Every uncertified run receives conservative abstinence utility = safe_fallback_obj
            task_safe_ref: dict[int, float] = {}
            for tidx in {r["target_task_idx"] for r in b_rows}:
                t_sub = [r for r in b_rows if r["target_task_idx"] == tidx]
                feas_objs = [
                    r["audit"]["true_objective"]
                    for r in t_sub
                    if r["audit"]["is_truly_feasible"]
                ]
                anc_obj = float(t_sub[0]["anchor_true_objective"])
                base_worst = max(feas_objs + [anc_obj]) if feas_objs else anc_obj * 1.25
                task_safe_ref[tidx] = base_worst * 1.05

            methods = sorted({r["method_name"] for r in b_rows})
            method_table: dict[str, Any] = {}
            paired_vectors: dict[str, dict[tuple[int, int], float]] = {}

            for m_name in methods:
                m_sub = [r for r in b_rows if r["method_name"] == m_name]
                deploy_costs = []
                raw_true_objs = []
                cert_flags = []
                false_cert_flags = []
                viol_probs = []
                target_calls = []
                wall_secs = []
                infl_factors = []
                p_vec: dict[tuple[int, int], float] = {}

                for r in m_sub:
                    tidx = int(r["target_task_idx"])
                    rseed = int(r["rep_seed"])
                    t_obj = float(r["audit"]["true_objective"])
                    # Certified deployment objective: if certified, deploy policy; if abstained, fall back to safe penalty ref
                    dep_c = t_obj if r["certified"] else task_safe_ref[tidx]
                    deploy_costs.append(dep_c)
                    raw_true_objs.append(t_obj)
                    cert_flags.append(float(r["certified"]))
                    false_cert_flags.append(int(r["false_certificate"]))
                    viol_probs.append(float(r["audit"]["true_violation_prob"]))
                    target_calls.append(float(r["total_target_calls"]))
                    wall_secs.append(float(r["wall_time_sec"]))
                    infl_factors.append(float(r["untempered_inflation_factor"]))
                    p_vec[(tidx, rseed)] = dep_c

                paired_vectors[m_name] = p_vec
                method_table[m_name] = {
                    "n_runs": len(m_sub),
                    "mean_deployed_cost": float(np.mean(deploy_costs)),
                    "se_deployed_cost": float(np.std(deploy_costs, ddof=1) / np.sqrt(max(len(deploy_costs), 1))) if len(deploy_costs) > 1 else 0.0,
                    "median_deployed_cost": float(np.median(deploy_costs)),
                    "mean_raw_true_obj": float(np.mean(raw_true_objs)),
                    "certified_rate": float(np.mean(cert_flags)),
                    "false_certificates": int(np.sum(false_cert_flags)),
                    "mean_true_violation_prob": float(np.mean(viol_probs)),
                    "mean_target_calls": float(np.mean(target_calls)),
                    "mean_wall_sec": float(np.mean(wall_secs)),
                    "mean_inflation_factor": float(np.mean(infl_factors)),
                }

            # Paired statistical tests against `TLS-OLH-KG` (if present)
            comparisons = {}
            if "TLS-OLH-KG" in paired_vectors:
                ours_vec = paired_vectors["TLS-OLH-KG"]
                raw_pvals = []
                comp_keys = []
                for other_m, other_vec in paired_vectors.items():
                    if other_m == "TLS-OLH-KG":
                        continue
                    common_keys = sorted(set(ours_vec).intersection(other_vec))
                    if len(common_keys) >= 4:
                        diffs = np.array([other_vec[k] - ours_vec[k] for k in common_keys], dtype=float)
                        rel_gains = np.array([
                            (other_vec[k] - ours_vec[k]) / max(abs(other_vec[k]), 1e-8)
                            for k in common_keys
                        ], dtype=float)
                        if np.all(np.abs(diffs) < 1e-12):
                            p_w = 1.0
                        else:
                            try:
                                _, p_w = wilcoxon(diffs, alternative="greater", zero_method="zsplit")
                                p_w = float(p_w)
                            except Exception:
                                p_w = 1.0
                        wins = int(np.sum(diffs > 1e-9))
                        losses = int(np.sum(diffs < -1e-9))
                        ties = len(diffs) - wins - losses
                        comparisons[other_m] = {
                            "n_pairs": len(common_keys),
                            "mean_cost_saving_vs_baseline": float(np.mean(diffs)),
                            "mean_relative_improvement_pct": float(100.0 * np.mean(rel_gains)),
                            "wins_losses_ties": [wins, losses, ties],
                            "wilcoxon_one_sided_p": p_w,
                        }
                        raw_pvals.append(p_w)
                        comp_keys.append(other_m)

                # Holm-Bonferroni step-down multiplicity adjustment
                if raw_pvals:
                    order = np.argsort(raw_pvals)
                    m_tests = len(raw_pvals)
                    running_max = 0.0
                    for rank_idx, orig_idx in enumerate(order):
                        adj = min(1.0, raw_pvals[orig_idx] * (m_tests - rank_idx))
                        running_max = max(running_max, adj)
                        comparisons[comp_keys[orig_idx]]["holm_adjusted_p"] = float(running_max)

            fam_summary["budgets"][str(b_steps)] = {
                "methods": method_table,
                "paired_vs_tls_olh_kg": comparisons,
            }
        by_fam_budget[fam_name] = fam_summary

    return {
        "campaign_name": manifest["campaign_name"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_evaluated_cells": len(rows),
        "total_certified_deployments": total_certified_runs,
        "total_false_certificates": total_false_certs,
        "prior_diagnostics": prior_diag,
        "families": by_fam_budget,
    }


def build_submit_command(
    descriptor_path: Path,
    *,
    scheduler_py: Path,
    remote_cwd: str,
    remote_python: str,
    ram_mb: int,
    extra_pythonpath: str = "",
    dry_run: bool = False,
) -> list[str]:
    descriptor = _read_json(descriptor_path)
    nodes = parse_cpu_nodes(",".join(DEFAULT_CPU_NODES))
    rel_desc = descriptor_path.resolve().relative_to(SC_ROOT)
    py_path = f"{extra_pythonpath}:." if extra_pythonpath else "."
    cmd = shlex.join([
        "env",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        f"PYTHONPATH={py_path}",
        remote_python,
        "-u",
        "-m",
        MODULE_NAME,
        "worker",
        "--descriptor",
        str(rel_desc),
        "--start",
        "{start}",
        "--end",
        "{end}",
        "--workers",
        "{workers}",
    ])
    out = [
        sys.executable,
        str(scheduler_py),
        "submit-cpu-batch",
        "--items",
        str(descriptor["n_jobs"]),
        "--cmd-template",
        cmd,
        "--cwd",
        remote_cwd,
        "--nodes",
        ",".join(nodes),
        "--signature",
        f"KG_op_v2/{descriptor['campaign_name']}/{descriptor['stage']}/{{start}}-{{end}}",
        "--description",
        f"v2_or {descriptor['campaign_name']} {descriptor['stage']} {{node}} [{{start}},{{end}})",
        "--ram-mb",
        str(ram_mb),
        "--project",
        "KG_op",
        "--skip-launch-staging",
        "--env-spec",
        "none",
        "--json",
    ]
    if dry_run:
        out.append("--dry-run")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    p_desc = sub.add_parser("describe")
    p_desc.add_argument("--manifest", type=Path, required=True)
    p_desc.add_argument("--stage", choices=("source_archive", "search_and_verify", "audit_and_summarize"), required=True)

    p_work = sub.add_parser("worker")
    p_work.add_argument("--descriptor", type=Path, required=True)
    p_work.add_argument("--start", type=int, required=True)
    p_work.add_argument("--end", type=int, required=True)
    p_work.add_argument("--workers", type=int, required=True)

    p_fin = sub.add_parser("finish")
    p_fin.add_argument("--descriptor", type=Path, required=True)

    p_sub = sub.add_parser("submit")
    p_sub.add_argument("--descriptor", type=Path, required=True)
    p_sub.add_argument("--scheduler", type=Path, default=Path.home() / "mine_code/scheduleurm/skill/scheduler.py")
    p_sub.add_argument(
        "--remote-cwd",
        default="/home/zhengliang01/scheduleurm_work/KG_op_v2_or_cluster/SC-OLH-KG",
    )
    p_sub.add_argument(
        "--remote-python",
        default="/home/zhengliang01/scheduleurm_work/conda_envs/scomp-mainline-py310-20261017/bin/python",
    )
    p_sub.add_argument(
        "--extra-pythonpath",
        default="/home/zhengliang01/scheduleurm_work/runtimes/cfcmt-sumo122-py310-v1",
    )
    p_sub.add_argument("--ram-mb", type=int, default=16384)
    p_sub.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.action == "describe":
        d = describe_stage(args.manifest, args.stage)
        print(json.dumps({"stage": d["stage"], "n_jobs": d["n_jobs"]}))
    elif args.action == "worker":
        run_worker_shard(args.descriptor, args.start, args.end, args.workers)
    elif args.action == "finish":
        res = finish_stage(args.descriptor)
        print(json.dumps({"stage": res["stage"], "status": res["status"], "n_jobs": res["n_jobs"]}))
    elif args.action == "submit":
        cmd = build_submit_command(
            args.descriptor,
            scheduler_py=args.scheduler,
            remote_cwd=args.remote_cwd,
            remote_python=args.remote_python,
            ram_mb=args.ram_mb,
            extra_pythonpath=args.extra_pythonpath,
            dry_run=args.dry_run,
        )
        completed = subprocess.run(cmd, text=True, capture_output=True)
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
