"""Cluster batch diagnostic for early (step 0) & late (step 59) EC-KG query states.

Evaluates all 24 unseen Queueing_CrossHorizon tasks and all 24 unseen
Traffic_Ingolstadt tasks (96 states total: step 0 and step 59) under the
frozen U3 source priors, recording query disagreement rates, revisit vs
admission choices, dual VOI decomposition (`VOI_refine + VOI_admit`), and
per-step wall-clock latency (ms).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time
from typing import Any

import numpy as np

SC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SC_ROOT.parent
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scheduler_node_policy import DEFAULT_CPU_NODES
from v2_or.acquisition.verification_kg import (
    build_balanced_warmstart_points,
    generate_dynamic_candidate_pool,
    score_chance_feasibility_kg,
    score_eligible_cascade_kg,
    score_gate_power_vkg,
)
from v2_or.contracts import SufficientStatsRecord
from v2_or.dispatch.campaign_runner import build_family_tasks, compute_v2_or_code_sha256
from v2_or.models import TemperedHierarchicalHVDModel


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _snapshot_state(
    prob: Any,
    model: TemperedHierarchicalHVDModel,
    prior: Any,
    rng_pool: np.random.Generator,
    sq_est_obj: float,
    *,
    pool_size: int,
    batch_reps: int,
) -> dict[str, Any]:
    enable_tpl = bool(not model.is_source_released_)
    cands, exps = generate_dynamic_candidate_pool(
        prob,
        model,
        prior,
        rng_pool,
        pool_size=pool_size,
        enable_source_templates=enable_tpl,
    )
    pred = model.predict_joint(cands, exps)
    sq_ref = 0.5 * float(sq_est_obj) + 0.5 * float(pred.mean_obj[0])

    t0 = time.perf_counter()
    s_eckg, diag = score_eligible_cascade_kg(
        pred,
        model,
        cands,
        exps,
        alpha=prob.alpha,
        batch_reps=batch_reps,
        status_quo_obj=sq_ref,
        top_k_contenders=24,
        return_diagnostics=True,
    )
    ms_eckg = 1000.0 * (time.perf_counter() - t0)

    t1 = time.perf_counter()
    s_single = score_gate_power_vkg(pred, alpha=prob.alpha, batch_reps=batch_reps, status_quo_obj=sq_ref)
    ms_single = 1000.0 * (time.perf_counter() - t1)

    s_feas = score_chance_feasibility_kg(pred, alpha=prob.alpha, batch_reps=batch_reps, status_quo_obj=sq_ref)

    q_eckg = int(diag["chosen_query"])
    q_single = int(np.argmax(s_single))
    q_feas = int(np.argmax(s_feas))

    obs_keys = {model._policy_key(r.x_norm) for r in model.unique_records}
    return {
        "n_unique_evaluated": len(model.unique_records),
        "sure_lambda_obj": float(model.sure_lambda_obj_),
        "sure_lambda_con": float(model.sure_lambda_con_),
        "source_released": bool(model.is_source_released_),
        "q_eckg": q_eckg,
        "q_single": q_single,
        "q_feas": q_feas,
        "disagreed_single": bool(q_eckg != q_single),
        "disagreed_feas": bool(q_eckg != q_feas),
        "eckg_is_revisit": bool(model._policy_key(cands[q_eckg]) in obs_keys),
        "single_is_revisit": bool(model._policy_key(cands[q_single]) in obs_keys),
        "feas_is_revisit": bool(model._policy_key(cands[q_feas]) in obs_keys),
        "eckg_voi_total": float(diag["chosen_voi_total"]),
        "eckg_voi_refine": float(diag["chosen_voi_refine"]),
        "eckg_voi_admit": float(diag["chosen_voi_admit"]),
        "single_voi_under_eckg": float(s_eckg[q_single]) if s_eckg.size > q_single else 0.0,
        "feas_voi_under_eckg": float(s_eckg[q_feas]) if s_eckg.size > q_feas else 0.0,
        "step_wall_ms_eckg": float(ms_eckg),
        "step_wall_ms_single": float(ms_single),
        "contenders_evaluated": int(diag["contenders_evaluated"]),
        "rosters_integrated": int(diag["rosters_integrated"]),
    }


def _run_one_job(manifest: dict[str, Any], out_dir: Path, jb: dict[str, Any]) -> dict[str, Any]:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    ipath = out_dir / "items" / f"{jb['job_id']}.json"
    if ipath.is_file():
        return {"job_id": jb["job_id"], "status": "cached"}

    fam_name = jb["family"]
    t_idx = int(jb["target_task_idx"])
    fam_spec = next(f for f in manifest["families"] if f["name"] == fam_name)
    tasks = build_family_tasks(fam_spec, "target")
    prob = tasks[t_idx]

    src_campaign = SC_ROOT / "paper_artifacts" / manifest["reuse_source_campaign"]
    src_fam = fam_spec.get("reuse_archive_from_family", fam_name)
    with (src_campaign / "source_bundles" / src_fam / "prior_default.pkl").open("rb") as fh:
        prior = pickle.load(fh)

    rep_seed = int(fam_spec["target_eval_seeds"][0])
    seed = rep_seed + 1009 * t_idx
    ss = np.random.SeedSequence(seed)
    c_init, c_search, _ = ss.spawn(3)
    rng_init = np.random.default_rng(c_init)
    rng_search = np.random.default_rng(c_search)

    n0 = int(fam_spec.get("n0_init", 8))
    b_reps = int(fam_spec.get("batch_reps", 2))
    p_size = int(fam_spec.get("pool_size", 160))

    model = TemperedHierarchicalHVDModel(prior=prior, mode="tempered_sure_release", n0_freeze=n0)
    init_pts = build_balanced_warmstart_points(prob, prior, n0, rng_init)
    init_recs: list[SufficientStatsRecord] = []
    for x0 in init_pts:
        exp0 = prob.nominal_exposure(x0)
        samp0 = prob.simulate_replications(x0, b_reps, rng_init, split="search")
        rec0 = SufficientStatsRecord.from_samples(x0, samp0, exp0)
        init_recs.append(rec0)
        model.observe(rec0)

    sq_est_obj = float(init_recs[0].mean_obj)

    # Fork RNG state to inspect step 0 without disturbing closed-loop trajectory
    rng_snap0 = np.random.default_rng(np.random.SeedSequence([seed, 9000]))
    snap_step0 = _snapshot_state(prob, model, prior, rng_snap0, sq_est_obj, pool_size=p_size, batch_reps=b_reps)

    # Advance closed-loop GP-FTA-ECKG through steps 0..58 to reach exact step-59 state
    for _step in range(59):
        enable_tpl = bool(not model.is_source_released_)
        cands, exps = generate_dynamic_candidate_pool(
            prob, model, prior, rng_search, pool_size=p_size, enable_source_templates=enable_tpl
        )
        pred = model.predict_joint(cands, exps)
        sq_ref = 0.5 * float(sq_est_obj) + 0.5 * float(pred.mean_obj[0])
        _, step_diag = score_eligible_cascade_kg(
            pred,
            model,
            cands,
            exps,
            alpha=prob.alpha,
            batch_reps=b_reps,
            status_quo_obj=sq_ref,
            return_diagnostics=True,
        )
        x_next = cands[int(step_diag["chosen_query"])]
        samp = prob.simulate_replications(x_next, b_reps, rng_search, split="search")
        model.observe(SufficientStatsRecord.from_samples(x_next, samp, exps[int(step_diag["chosen_query"])]))

    rng_snap59 = np.random.default_rng(np.random.SeedSequence([seed, 9059]))
    snap_step59 = _snapshot_state(prob, model, prior, rng_snap59, sq_est_obj, pool_size=p_size, batch_reps=b_reps)

    res = {
        "job_id": jb["job_id"],
        "family": fam_name,
        "target_task_idx": t_idx,
        "task_id": prob.task_id,
        "seed": seed,
        "step_0": snap_step0,
        "step_59": snap_step59,
        "execution_code_sha256": compute_v2_or_code_sha256(),
    }
    _write_json(ipath, res)
    return {"job_id": jb["job_id"], "status": "ok"}


def build_jobs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for fam in manifest["families"]:
        if fam["name"] not in {"Queueing_CrossHorizon", "Traffic_Ingolstadt"}:
            continue
        n_tgt = int(fam["target_config"]["n_tasks"])
        for t_idx in range(n_tgt):
            jobs.append({
                "job_id": f"prefix_{fam['name']}_t{t_idx:02d}",
                "family": fam["name"],
                "target_task_idx": t_idx,
            })
    return jobs


def summarize_items(items: list[dict[str, Any]], code_sha: str) -> dict[str, Any]:
    by_fam_stage: dict[str, Any] = {}
    for fam in sorted({it["family"] for it in items}):
        f_items = [it for it in items if it["family"] == fam]
        by_fam_stage[fam] = {}
        for st_key in ("step_0", "step_59"):
            snaps = [it[st_key] for it in f_items]
            n = len(snaps)
            by_fam_stage[fam][st_key] = {
                "n_states": n,
                "mean_unique_evaluated": float(np.mean([s["n_unique_evaluated"] for s in snaps])),
                "disagreed_with_single_slot_count": int(sum(1 for s in snaps if s["disagreed_single"])),
                "disagreed_with_single_slot_rate": float(np.mean([1.0 if s["disagreed_single"] else 0.0 for s in snaps])),
                "disagreed_with_feasible_kg_count": int(sum(1 for s in snaps if s["disagreed_feas"])),
                "disagreed_with_feasible_kg_rate": float(np.mean([1.0 if s["disagreed_feas"] else 0.0 for s in snaps])),
                "eckg_revisit_count": int(sum(1 for s in snaps if s["eckg_is_revisit"])),
                "eckg_revisit_rate": float(np.mean([1.0 if s["eckg_is_revisit"] else 0.0 for s in snaps])),
                "single_revisit_rate": float(np.mean([1.0 if s["single_is_revisit"] else 0.0 for s in snaps])),
                "feas_revisit_rate": float(np.mean([1.0 if s["feas_is_revisit"] else 0.0 for s in snaps])),
                "mean_eckg_voi_total": float(np.mean([s["eckg_voi_total"] for s in snaps])),
                "mean_eckg_voi_refine": float(np.mean([s["eckg_voi_refine"] for s in snaps])),
                "mean_eckg_voi_admit": float(np.mean([s["eckg_voi_admit"] for s in snaps])),
                "mean_single_voi_under_eckg": float(np.mean([s["single_voi_under_eckg"] for s in snaps])),
                "mean_feas_voi_under_eckg": float(np.mean([s["feas_voi_under_eckg"] for s in snaps])),
                "mean_sure_lambda_con": float(np.mean([s["sure_lambda_con"] for s in snaps])),
                "source_released_rate": float(np.mean([1.0 if s["source_released"] else 0.0 for s in snaps])),
                "mean_step_wall_ms_eckg": float(np.mean([s["step_wall_ms_eckg"] for s in snaps])),
                "p95_step_wall_ms_eckg": float(np.percentile([s["step_wall_ms_eckg"] for s in snaps], 95)),
                "mean_step_wall_ms_single": float(np.mean([s["step_wall_ms_single"] for s in snaps])),
            }
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "execution_code_sha256": code_sha,
        "n_tasks_total": len(items),
        "n_states_total": 2 * len(items),
        "families": by_fam_stage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for act in ("describe", "finish"):
        p = sub.add_parser(act)
        p.add_argument("--manifest", type=Path, default=SC_ROOT / "v2_or/manifests/v2_or_eckg_sure_confirm_v1.json")
        p.add_argument("--out-dir", type=Path, default=SC_ROOT / "paper_artifacts/v2_or_eckg_sure_confirm_v1/prefix_audit")
    pw = sub.add_parser("worker")
    pw.add_argument("--manifest", type=Path, default=SC_ROOT / "v2_or/manifests/v2_or_eckg_sure_confirm_v1.json")
    pw.add_argument("--out-dir", type=Path, default=SC_ROOT / "paper_artifacts/v2_or_eckg_sure_confirm_v1/prefix_audit")
    pw.add_argument("--start", type=int, required=True)
    pw.add_argument("--end", type=int, required=True)
    pw.add_argument("--workers", type=int, required=True)

    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    out_dir = args.out_dir.resolve()
    jobs = build_jobs(manifest)

    if args.action == "describe":
        _write_json(out_dir / "jobs.json", {"n_jobs": len(jobs), "jobs": jobs})
        print(json.dumps({"n_jobs": len(jobs)}))
    elif args.action == "worker":
        node = socket.gethostname().split(".")[0]
        portable_ok = os.environ.get("GPFTA_PORTABLE_REVIEW") == "1" or (REPO_ROOT / ".gpfta_portable_review").is_file()
        if not portable_ok and node not in DEFAULT_CPU_NODES:
            raise RuntimeError(f"Cluster shards must run on node001-node006, got {node!r}")
        sl = jobs[args.start : args.end]
        n_proc = max(1, min(int(args.workers), len(sl)))
        if n_proc == 1:
            for jb in sl:
                _run_one_job(manifest, out_dir, jb)
        else:
            with ProcessPoolExecutor(max_workers=n_proc) as pool:
                futs = [pool.submit(_run_one_job, manifest, out_dir, jb) for jb in sl]
                for f in as_completed(futs):
                    f.result()
    elif args.action == "finish":
        items = [json.loads((out_dir / "items" / f"{jb['job_id']}.json").read_text(encoding="utf-8")) for jb in jobs]
        summary = summarize_items(items, compute_v2_or_code_sha256())
        _write_json(out_dir.parent / "eckg_prefix_audit_readout.json", summary)
        print(json.dumps({"status": "complete", "n_states_total": summary["n_states_total"]}))


if __name__ == "__main__":
    main()
