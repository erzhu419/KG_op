"""Cluster-distributed 3-stage barrier experiment runner with roster-invariant ledgers.

Enforces strict scientific barriers on `node001-node006` per `AGENTS.md`:
  Stage 1 (`source_archive`):
      Run adaptive legacy solvers on source tasks in parallel; barrier-merge into
      frozen `archive.json` + fitted priors, metering offline source calls & fit time.
  Stage 2 (`search_and_verify`):
      Using frozen priors, run independent `(family, task_idx, rep_seed, arm, N)` cells
      on unseen target tasks. Freezes `primary_policy`, all `shortlist_policies`,
      per-slot `GateCertificate` entries, and strict `decision_state` (`"certified_deployment"`
      with `deployed_policy`, or `"abstention"` with `deployed_policy = None`), plus
      runtime `execution_code_sha256` and `manifest_sha256`.
  Stage 3 (`audit_and_summarize`):
      Only AFTER Stage 2 completion is sealed, run post-hoc Monte Carlo auditing on
      frozen `deployed_policy`, frozen `primary_policy`, and status-quo `anchor`,
      emitting `per_run_ledger.csv`, `per_slot_certificates.csv`, and roster-invariant
      task-level (`M_tgt` independent tasks) and run-level summaries.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

SC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SC_ROOT.parent
if str(SC_ROOT) not in sys.path:
    sys.path.insert(0, str(SC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scheduler_node_policy import DEFAULT_CPU_NODES, parse_cpu_nodes
from v2_or.acquisition import SequentialRunResult, run_tls_olh_kg
from v2_or.baselines import (
    pretrain_hyperbo_eb_params,
    pretrain_rgpe_psi_source_ensemble,
    pretrain_rgpe_source_ensemble,
    run_direct_legacy_baseline,
    run_hetgp_kg_baseline,
    run_hyperbo_transfer_baseline,
    run_rgpe_on_psi_baseline,
    run_rgpe_transfer_baseline,
    run_saasbo_baseline,
    run_scbo_baseline,
    run_scbo_on_psi_baseline,
    run_warmstart_pattern_search_baseline,
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
from v2_or.verification.gated_verifier_v2 import DEFAULT_SLOT_BUDGETS


MODULE_NAME = "v2_or.dispatch.campaign_runner"
FRESH_GATE_CALL_CAP = sum(DEFAULT_SLOT_BUDGETS)
STAGES = ("source_archive", "search_and_verify", "audit_anchors", "audit_and_summarize")


def _read_json(path: Path) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            last_err = exc
            time.sleep(0.05 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    tmp = path.with_suffix(path.suffix + f".{host}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def compute_v2_or_code_sha256() -> str:
    """Compute deterministic SHA-256 across all `.py` files in `SC-OLH-KG/v2_or`."""
    h = hashlib.sha256()
    v2_dir = SC_ROOT / "v2_or"
    for p in sorted(v2_dir.rglob("*.py")):
        rel = p.relative_to(SC_ROOT).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        buf_h = int(cfg.get("buffer_hours", 24))
        clamp_lim = cfg.get("clamp_limit", 0.20)
        tasks: list[BaseDynamicORProblem] = []
        for idx, mkt in enumerate(markets):
            d_h = horizons[idx % len(horizons)]
            tasks.append(
                OPSDGridStorageBenchmark(
                    market=mkt,
                    d=d_h,
                    buffer_hours=buf_h,
                    clamp_limit=None if clamp_lim is None else float(clamp_lim),
                )
            )
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
        bl_override = cfg.get("baseline_override")
        tasks = []
        for idx in range(int(cfg["n_tasks"])):
            tau_v = float(taus[idx % len(taus)])
            w_v = tuple(float(x) for x in weights_list[idx % len(weights_list)])
            tasks.append(
                IngolstadtLiveSUMOBenchmark(
                    task_id=f"live_sumo_ingolstadt21_{idx:02d}",
                    weights=w_v,
                    tau_emission=tau_v,
                    baseline_override=bl_override,
                )
            )
        return tasks
    raise ValueError(f"Unknown family kind {kind!r}")


def _campaign_dir(manifest: dict[str, Any]) -> Path:
    return SC_ROOT / manifest["output_dir"]


def _source_campaign_dir(manifest: dict[str, Any]) -> Path:
    reused = manifest.get("reuse_source_campaign")
    return SC_ROOT / "paper_artifacts" / reused if reused else _campaign_dir(manifest)


def _target_call_limits(family: dict[str, Any], n_steps: int, n0: int, reps: int) -> dict[str, int]:
    search = (n_steps + n0) * reps
    if n0 < 1 or reps < 1 or n_steps < 0 or search + FRESH_GATE_CALL_CAP != int(family["target_call_cap"]):
        raise ValueError(f"Frozen target call cap disagrees with search and terminal slots: {family['name']}")
    return {"search": search, "verification": FRESH_GATE_CALL_CAP}


def _audit_anchor_id(family: str, target_task_idx: int) -> str:
    return f"anchor_{family}_t{target_task_idx:02d}"


def _validate_fresh_descriptor(descriptor: dict[str, Any], manifest_bytes: bytes, code_sha: str) -> None:
    if descriptor["manifest_sha256"] != _sha256_bytes(manifest_bytes) or descriptor["execution_code_sha256"] != code_sha:
        raise RuntimeError("Fresh campaign code or manifest differs from the frozen stage descriptor")


def describe_stage(manifest_path: Path, stage: str) -> dict[str, Any]:
    raw_bytes = manifest_path.read_bytes()
    manifest = json.loads(raw_bytes.decode("utf-8"))
    if manifest.get("sealed_complete", False):
        raise RuntimeError("Sealed completed manifest cannot be restarted")
    out_dir = _campaign_dir(manifest)
    fresh = bool(manifest.get("reuse_source_campaign"))
    stage_dir = out_dir / "stages" / stage
    desc_path = stage_dir / "jobs.json"
    if desc_path.is_file():
        current = _read_json(desc_path)
        if fresh:
            _validate_fresh_descriptor(current, raw_bytes, compute_v2_or_code_sha256())
        return current

    code_sha = compute_v2_or_code_sha256()
    man_sha = _sha256_bytes(raw_bytes)
    jobs: list[dict[str, Any]] = []

    if stage == "source_archive":
        for fam in manifest["families"]:
            fam_name = fam["name"]
            if fresh or "reuse_archive_from_family" in fam:
                continue
            src_cfg = fam["source_archive_protocol"]
            s_conf = fam["source_config"]
            n_src = len(s_conf["markets"]) if "markets" in s_conf else int(s_conf["n_tasks"])
            per_task_budgets = src_cfg.get("per_task_budgets")
            solver_subsets = src_cfg.get("solver_subsets_per_task")
            mixed_reps = src_cfg.get("mixed_reps_choices")
            alpha_overrides = src_cfg.get("source_alpha_overrides")
            for s_idx in range(n_src):
                task_solvers = (
                    solver_subsets[s_idx % len(solver_subsets)]
                    if solver_subsets
                    else src_cfg["solvers"]
                )
                task_budget = (
                    int(per_task_budgets[s_idx % len(per_task_budgets)])
                    if per_task_budgets
                    else int(src_cfg["per_solver_budget"])
                )
                task_alpha = (
                    float(alpha_overrides[s_idx % len(alpha_overrides)])
                    if alpha_overrides
                    else None
                )
                for solver_idx, solver_name in enumerate(task_solvers):
                    seed = int(src_cfg["seed"]) + 1009 * s_idx + 37 * solver_idx
                    jb_dict: dict[str, Any] = {
                        "job_id": f"src_{fam_name}_t{s_idx:02d}_{solver_name}",
                        "family": fam_name,
                        "source_task_idx": s_idx,
                        "solver_name": solver_name,
                        "per_solver_budget": task_budget,
                        "reps_per_eval": int(src_cfg.get("reps_per_eval", 3)),
                        "seed": seed,
                    }
                    if mixed_reps is not None:
                        jb_dict["reps_schedule"] = [int(r) for r in mixed_reps]
                    if task_alpha is not None:
                        jb_dict["solver_alpha"] = task_alpha
                    jobs.append(jb_dict)

    elif stage == "search_and_verify":
        src_done = out_dir / "stages" / "source_archive" / "completion.json"
        if not src_done.is_file():
            raise RuntimeError("Stage `source_archive` must finish before `search_and_verify`")
        for fam in manifest["families"]:
            fam_name = fam["name"]
            if not fam.get("online_step_budgets") or not fam.get("arms"):
                continue
            t_conf = fam["target_config"]
            n_tgt = len(t_conf["markets"]) if "markets" in t_conf else int(t_conf["n_tasks"])
            rep_seeds = [int(s) for s in fam["target_eval_seeds"]]
            budgets = [int(b) for b in fam["online_step_budgets"]]
            arms = fam["arms"]
            for b_steps in budgets:
                if fresh:
                    _target_call_limits(fam, b_steps, int(fam["n0_init"]), int(fam["batch_reps"]))
                for t_idx in range(n_tgt):
                    for r_seed in rep_seeds:
                        cell_seed = r_seed + 10_007 * t_idx
                        for arm in arms:
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

    elif stage in {"audit_anchors", "audit_and_summarize"}:
        sv_done = out_dir / "stages" / "search_and_verify" / "completion.json"
        if not sv_done.is_file():
            raise RuntimeError("Stage `search_and_verify` must finish before `audit_and_summarize`")
        sv_desc = _read_json(out_dir / "stages" / "search_and_verify" / "jobs.json")
        if stage == "audit_anchors":
            if not fresh:
                raise ValueError("This campaign uses the original audit workflow")
            task_keys = sorted({(job["family"], job["target_task_idx"]) for job in sv_desc["jobs"]})
            jobs = [{"job_id": _audit_anchor_id(family, idx), "family": family, "target_task_idx": idx}
                    for family, idx in task_keys]
        else:
            if fresh and not (out_dir / "stages/audit_anchors/completion.json").is_file():
                raise RuntimeError("Stage `audit_anchors` must finish before `audit_and_summarize`")
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
        "manifest_sha256": man_sha,
        "execution_code_sha256": code_sha,
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_jobs": len(jobs),
        "jobs": jobs,
        "truth_used": stage in {"audit_anchors", "audit_and_summarize"},
    }
    _write_json(desc_path, descriptor)
    return descriptor


def _execute_single_item(
    manifest: dict[str, Any],
    manifest_sha: str,
    code_sha: str,
    stage: str,
    job: dict[str, Any],
    attempt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = _campaign_dir(manifest)
    fresh = bool(manifest.get("reuse_source_campaign"))
    item_path = out_dir / "stages" / stage / "items" / f"{job['job_id']}.json"
    if item_path.is_file():
        if fresh:
            stored = _read_json(item_path)
            if (stored["manifest_sha256"], stored["execution_code_sha256"]) != (manifest_sha, code_sha):
                raise RuntimeError(f"Cached item differs from the frozen campaign: {job['job_id']}")
        return {"job_id": job["job_id"], "status": "cached"}

    fam_map = {f["name"]: f for f in manifest["families"]}
    fam_spec = fam_map[job["family"]]

    if stage == "source_archive":
        tasks = build_family_tasks(fam_spec, "source")
        prob = tasks[int(job["source_task_idx"])]
        solver_fn = LEGACY_SOLVER_REGISTRY[job["solver_name"]]
        t0 = time.perf_counter()
        solver_kwargs: dict[str, Any] = {
            "budget_calls": int(job["per_solver_budget"]),
            "seed": int(job["seed"]),
            "reps_per_eval": int(job["reps_per_eval"]),
        }
        if "reps_schedule" in job and job["reps_schedule"] is not None:
            solver_kwargs["reps_schedule"] = [int(v) for v in job["reps_schedule"]]
        if "solver_alpha" in job and job["solver_alpha"] is not None:
            solver_kwargs["solver_alpha"] = float(job["solver_alpha"])
        recorder = solver_fn(prob, **solver_kwargs)
        wall_s = float(time.perf_counter() - t0)
        payload = {
            "job_id": job["job_id"],
            "family": job["family"],
            "task_id": prob.task_id,
            "policy_dim": int(prob.d),
            "num_control_channels": int(prob.num_control_channels),
            "solver_name": job["solver_name"],
            "seed": int(job["seed"]),
            "total_calls": int(recorder.total_calls),
            "wall_time_sec": wall_s,
            "records": [asdict(r) for r in recorder.trajectory_log],
            "execution_code_sha256": code_sha,
            "manifest_sha256": manifest_sha,
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
        bundle_dir = _source_campaign_dir(manifest) / "source_bundles" / src_fam
        archive_file = bundle_dir / "archive.json"
        prior_variant = arm.get("prior_variant", "default")
        prior_file = bundle_dir / f"prior_{prior_variant}.pkl"

        n_steps = int(job["n_online_steps"])
        n0 = int(job["n0_init"])
        b_reps = int(job["batch_reps"])
        p_size = int(job["pool_size"])
        seed = int(job["seed"])
        gate_mode = fam_spec.get("gate_mode", "auto")
        src_pretrain_sec = 0.0
        if fresh:
            if arm_kind not in {"tls_olh_kg", "target_only_psi"}:
                raise ValueError(f"Unfrozen arm kind for reused-source campaign: {arm_kind}")
            prob.call_limits = _target_call_limits(fam_spec, n_steps, n0, b_reps)
            if attempt is not None:
                attempt["problem"] = prob
            if arm_kind == "tls_olh_kg" and (not arm.get("load_prior", True) or not prior_file.is_file()):
                raise RuntimeError(f"The frozen fitted prior is required: {prior_file}")

        if arm_kind in {"tls_olh_kg", "stop"}:
            prior = None
            if prior_file.is_file() and arm.get("load_prior", True):
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
                acquisition_type="stop" if arm_kind == "stop" else arm.get("acquisition", "gate_power_ts"),
                init_mode=arm.get("init_mode", "auto"),
                proposal_mode=arm.get("proposal_mode", "auto"),
                use_source_basis=arm.get("use_source_basis"),
                use_source_coef_prior=arm.get("use_source_coef_prior"),
                use_source_hvd_prior=arm.get("use_source_hvd_prior"),
                variance_estimator=arm.get("variance_estimator", "within_ssq"),
                use_empirical_r_perp=arm.get("use_empirical_r_perp", True),
                nomination_mode=arm.get("nomination_mode", "binomial_gate_power"),
                pool_size=p_size,
                method_label=arm["name"],
                gate_mode=gate_mode,
                coefficient_prior=arm.get("coefficient_prior"),
            )
        elif arm_kind == "target_only_psi":
            res = run_tls_olh_kg(
                prob,
                None,
                n_online_steps=n_steps,
                n0_init=n0,
                batch_reps=b_reps,
                seed=seed,
                model_mode="target_only",
                acquisition_type=arm.get("acquisition", "gate_power_ts"),
                init_mode="cold_dct",
                proposal_mode="local_dct_only",
                use_source_basis=False,
                use_source_coef_prior=False,
                use_source_hvd_prior=False,
                nomination_mode=arm.get("nomination_mode", "binomial_gate_power"),
                pool_size=p_size,
                method_label=arm["name"],
                gate_mode=gate_mode,
            )
        elif arm_kind == "warmstart_pattern_search":
            with prior_file.open("rb") as fh:
                prior = pickle.load(fh)
            res = run_warmstart_pattern_search_baseline(
                prob, prior, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, gate_mode=gate_mode
            )
            res = SequentialRunResult(**{**asdict(res), "method_name": arm["name"], "receipt": res.receipt, "deployed_policy": res.deployed_policy})
        elif arm_kind == "scbo":
            res = run_scbo_baseline(prob, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode)
        elif arm_kind == "scbo_on_psi":
            res = run_scbo_on_psi_baseline(prob, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode)
            res = SequentialRunResult(**{**asdict(res), "method_name": arm["name"], "receipt": res.receipt, "deployed_policy": res.deployed_policy})
        elif arm_kind in {"saasbo", "map_saas_cbo"}:
            res = run_saasbo_baseline(prob, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode)
        elif arm_kind == "hetgp_kg":
            res = run_hetgp_kg_baseline(prob, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode)
        elif arm_kind == "rgpe_on_psi":
            records, _ = load_legacy_archive_json(archive_file)
            rgpe_psi_cache = out_dir / "source_bundles" / src_fam / "rgpe_psi_pretrain.pkl"
            psi_bundle = None
            if rgpe_psi_cache.is_file():
                try:
                    with rgpe_psi_cache.open("rb") as fh:
                        psi_bundle = pickle.load(fh)
                except Exception:
                    psi_bundle = None
            if psi_bundle is None:
                psi_bundle = pretrain_rgpe_psi_source_ensemble(records)
                try:
                    tmp_p = rgpe_psi_cache.with_suffix(
                        f".{socket.gethostname().split('.')[0]}.{os.getpid()}.{time.time_ns()}.tmp"
                    )
                    with tmp_p.open("wb") as fh:
                        pickle.dump(psi_bundle, fh)
                    tmp_p.replace(rgpe_psi_cache)
                except Exception:
                    pass
            src_pretrain_sec = float(psi_bundle[6])
            res = run_rgpe_on_psi_baseline(
                prob,
                records,
                n_online_steps=n_steps,
                n0_init=n0,
                batch_reps=b_reps,
                seed=seed,
                pool_size=p_size,
                gate_mode=gate_mode,
                pretrained_bundle=psi_bundle,
            )
            res = SequentialRunResult(**{**asdict(res), "method_name": arm["name"], "receipt": res.receipt, "deployed_policy": res.deployed_policy})
        elif arm_kind == "rgpe":
            records, _ = load_legacy_archive_json(archive_file)
            rgpe_cache = out_dir / "source_bundles" / src_fam / f"rgpe_pretrain_d{int(prob.d)}.pkl"
            bundle = None
            if rgpe_cache.is_file():
                try:
                    with rgpe_cache.open("rb") as fh:
                        bundle = pickle.load(fh)
                except Exception:
                    bundle = None
            if bundle is None:
                bundle = pretrain_rgpe_source_ensemble(prob, records)
                try:
                    tmp_p = rgpe_cache.with_suffix(f".{socket.gethostname().split('.')[0]}.{os.getpid()}.tmp")
                    with tmp_p.open("wb") as fh:
                        pickle.dump(bundle, fh)
                    tmp_p.replace(rgpe_cache)
                except Exception:
                    pass
            src_pretrain_sec = float(bundle[3])
            res = run_rgpe_transfer_baseline(
                prob, records, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode, pretrained_bundle=bundle
            )
        elif arm_kind in {"hyperbo", "hyperbo_eb"}:
            records, _ = load_legacy_archive_json(archive_file)
            hyp_cache = out_dir / "source_bundles" / src_fam / f"hyperbo_pretrain_d{int(prob.d)}.pkl"
            eb_bundle = None
            if hyp_cache.is_file():
                try:
                    with hyp_cache.open("rb") as fh:
                        eb_bundle = pickle.load(fh)
                except Exception:
                    eb_bundle = None
            if eb_bundle is None:
                eb_bundle = pretrain_hyperbo_eb_params(prob, records)
                try:
                    tmp_p = hyp_cache.with_suffix(f".{socket.gethostname().split('.')[0]}.{os.getpid()}.tmp")
                    with tmp_p.open("wb") as fh:
                        pickle.dump(eb_bundle, fh)
                    tmp_p.replace(hyp_cache)
                except Exception:
                    pass
            src_pretrain_sec = float(eb_bundle[1])
            res = run_hyperbo_transfer_baseline(
                prob, records, n_online_steps=n_steps, n0_init=n0, batch_reps=b_reps, seed=seed, pool_size=p_size, gate_mode=gate_mode, pretrained_params=eb_bundle
            )
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

        rcpt = res.receipt
        if fresh and (res.search_calls_used != prob.call_limits["search"]
                      or res.verification_calls_used != prob.verification_calls
                      or res.total_target_calls != prob.search_calls + prob.verification_calls):
            raise RuntimeError(f"Target call receipt disagrees with the simulation counters: {job['job_id']}")
        payload = {
            "job_id": job["job_id"],
            "family": job["family"],
            "target_task_idx": int(job["target_task_idx"]),
            "task_id": prob.task_id,
            "policy_dim": int(prob.d),
            "num_control_channels": int(prob.num_control_channels),
            "rep_seed": int(job["rep_seed"]),
            "seed": seed,
            "search_seed": seed + 100_003,
            "verif_seed": seed + 200_007,
            "method_name": arm["name"],
            "n0_init_points": res.n0_init_points,
            "n_online_steps": int(job["n_online_steps"]),
            "actual_online_steps": res.n_online_steps,
            "search_calls_used": res.search_calls_used,
            "verification_calls_used": res.verification_calls_used,
            "total_target_calls": res.total_target_calls,
            "online_wall_time_sec": res.wall_time_sec,
            "source_pretrain_wall_sec": src_pretrain_sec,
            "decision_state": rcpt.decision_state,
            "certified": res.certified,
            "selected_slot": rcpt.selected_slot,
            "selected_role": res.selected_role,
            "posterior_beta_con": res.posterior_beta_con,
            "posterior_tau_shift_con": res.posterior_tau_shift_con,
            "empirical_var_ratio_vs_within": res.untempered_inflation_factor,
            "n_unique_policies_evaluated": res.n_unique_policies_evaluated,
            "n_replication_queries": res.n_replication_queries,
            "deployed_policy": None if res.deployed_policy is None else [float(v) for v in res.deployed_policy],
            "selected_uncertified_candidate": (
                None if rcpt.selected_uncertified_candidate is None else [float(v) for v in rcpt.selected_uncertified_candidate]
            ),
            "primary_policy": [float(v) for v in rcpt.primary_policy],
            "shortlist_roles": list(rcpt.shortlist_roles),
            "shortlist_policies": [[float(v) for v in sp] for sp in rcpt.shortlist_policies],
            "slot_budgets_configured": list(rcpt.slot_budgets_configured),
            "slot_deltas_configured": list(rcpt.slot_deltas_configured),
            "slot_certificates": [asdict(c) for c in rcpt.certificates],
            "execution_code_sha256": code_sha,
            "manifest_sha256": manifest_sha,
            "truth_used": False,
        }
        if fresh:
            payload["target_call_cap"] = sum(prob.call_limits.values())
        if res.mechanism_receipt is not None:
            payload["mechanism_receipt"] = res.mechanism_receipt
        _write_json(item_path, payload)
        return {"job_id": job["job_id"], "status": "ok"}

    if stage in {"audit_anchors", "audit_and_summarize"}:
        tasks = build_family_tasks(fam_spec, "target")
        prob = tasks[int(job["target_task_idx"])]
        n_audit = int(fam_spec.get("audit_replications", 4000))
        audit_seed = int(fam_spec.get("audit_seed_base", 20261116)) + 997 * int(job["target_task_idx"])

        x_anc = prob.domain_heuristic_anchor()
        if fresh:
            prob.call_limits = {"audit": n_audit if stage == "audit_anchors" else 2 * n_audit,
                                "verification": n_audit if fam_spec["kind"] == "opsd" else 0}
            if attempt is not None:
                attempt["problem"] = prob
        if stage == "audit_anchors":
            anchor = prob.audit_ground_truth(x_anc, n_eval=n_audit, seed=audit_seed, split="audit")
            _write_json(item_path, {
                "job_id": job["job_id"], "anchor_audit": anchor, "audit_seed": audit_seed,
                "audit_calls_used": prob.audit_calls, "truth_used": True,
                "manifest_sha256": manifest_sha, "execution_code_sha256": code_sha,
            })
            return {"job_id": job["job_id"], "status": "ok"}

        sv_item = _read_json(out_dir / "stages" / "search_and_verify" / "items" / f"{job['sv_job_id']}.json")
        anchor_cache = out_dir / "stages" / "audit_and_summarize" / "anchors" / f"{job['family']}_t{int(job['target_task_idx']):02d}.json"
        anchor_audit = None
        if fresh:
            anchor_path = out_dir / "stages/audit_anchors/items" / f"{_audit_anchor_id(job['family'], job['target_task_idx'])}.json"
            anchor_audit = _read_json(anchor_path)["anchor_audit"]
        elif anchor_cache.is_file():
            try:
                anchor_audit = _read_json(anchor_cache)
            except Exception:
                anchor_audit = None
        if anchor_audit is None:
            anchor_audit = prob.audit_ground_truth(x_anc, n_eval=n_audit, seed=audit_seed, split="audit")
            try:
                _write_json(anchor_cache, anchor_audit)
            except Exception:
                pass

        x_prim = np.asarray(sv_item["primary_policy"], dtype=float)
        if np.allclose(x_prim, x_anc, atol=1e-10, rtol=0.0):
            primary_audit = dict(anchor_audit)
        else:
            primary_audit = prob.audit_ground_truth(x_prim, n_eval=n_audit, seed=audit_seed, split="audit")

        cert = bool(sv_item["certified"])
        deployed_audit: dict[str, Any] | None = None
        verif_law_audit: dict[str, Any] | None = None
        if cert and sv_item["deployed_policy"] is not None:
            x_dep = np.asarray(sv_item["deployed_policy"], dtype=float)
            if np.allclose(x_dep, x_prim, atol=1e-10, rtol=0.0):
                deployed_audit = dict(primary_audit)
            elif np.allclose(x_dep, x_anc, atol=1e-10, rtol=0.0):
                deployed_audit = dict(anchor_audit)
            else:
                deployed_audit = prob.audit_ground_truth(x_dep, n_eval=n_audit, seed=audit_seed, split="audit")

            if fam_spec["kind"] == "opsd":
                verif_law_audit = prob.audit_ground_truth(
                    x_dep, n_eval=n_audit, seed=audit_seed + 500_009, split="verification"
                )

        anc_obj = float(anchor_audit["mc_objective_mean"])
        dep_obj = float(deployed_audit["mc_objective_mean"]) if (cert and deployed_audit is not None) else None
        status_quo_loss = dep_obj if dep_obj is not None else anc_obj
        fixed_reject_loss_10pct = dep_obj if dep_obj is not None else (1.10 * anc_obj)

        strip_keys = {"deployed_policy", "selected_uncertified_candidate", "primary_policy", "shortlist_policies"}
        payload = {
            **{k: v for k, v in sv_item.items() if k not in strip_keys},
            "audit_seed": audit_seed,
            "audit_calls_used": int(prob.audit_calls),
            "deployed_audit": deployed_audit,
            "verif_law_audit": verif_law_audit,
            "primary_audit": primary_audit,
            "anchor_audit": anchor_audit,
            "status_quo_loss": status_quo_loss,
            "fixed_reject_loss_10pct": fixed_reject_loss_10pct,
            "audit_point_exceeds_alpha": bool(
                cert and deployed_audit is not None and (not deployed_audit["is_truly_feasible"])
            ),
            "safety_audit_class": (
                deployed_audit["safety_audit_class"]
                if (cert and deployed_audit is not None)
                else "abstained_not_deployed"
            ),
            "truth_used": True,
        }
        if fresh:
            payload["verification_law_audit_calls_used"] = prob.verification_calls
        if "mechanism_receipt" in payload:
            variances = payload["mechanism_receipt"]["variance"]
            for policy, audited in (("primary", primary_audit), ("deployed", deployed_audit)):
                if audited is None:
                    continue
                audit_variance = {
                    "obj_var": audited["mc_objective_se"] ** 2 * audited["n_audit_samples"],
                    "con_var": audited["mc_constraint_std"] ** 2,
                    "n_samples": audited["n_audit_samples"],
                }
                variances[policy]["audit"] = audit_variance
                variances[policy]["squared_error_vs_audit"] = {
                    estimator: {f"{head}_var": (variances[policy][estimator][f"{head}_var"] - audit_variance[f"{head}_var"]) ** 2
                                for head in ("obj", "con")}
                    for estimator in ("fixed_batch", "pooled") if variances[policy][estimator]["dof"] > 0
                }
        _write_json(item_path, payload)
        return {"job_id": job["job_id"], "status": "ok"}

    raise ValueError(f"Unknown stage {stage!r}")


def _worker_entry(args_tuple: tuple[str, str, str, str, dict[str, Any]]) -> dict[str, Any]:
    manifest_str, manifest_sha, code_sha, stage, job = args_tuple
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    manifest = json.loads(manifest_str)
    job_id = job["job_id"]
    stage_dir = _campaign_dir(manifest) / "stages" / stage
    attempt_path = stage_dir / "attempts" / f"{job_id}.json"
    attempt: dict[str, Any] = {}
    t_start, cpu_start = time.perf_counter(), time.process_time()
    try:
        if manifest.get("reuse_source_campaign") and not (stage_dir / "items" / f"{job_id}.json").is_file():
            attempt_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with attempt_path.open("x", encoding="utf-8") as stream:
                    json.dump({"job_id": job_id, "status": "started", "node": socket.gethostname(),
                               "started_utc": datetime.now(timezone.utc).isoformat(),
                               "manifest_sha256": manifest_sha, "execution_code_sha256": code_sha}, stream)
            except FileExistsError:
                return {"job_id": job_id, "status": "failed", "error": "Previous paid or interrupted attempt is retained"}
            attempt["started"] = True
        result = _execute_single_item(manifest, manifest_sha, code_sha, stage, job, attempt)
    except Exception as exc:
        result = {
            "job_id": job_id,
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    if attempt.get("started"):
        prob = attempt.get("problem")
        _write_json(attempt_path, {
            **_read_json(attempt_path), **result,
            "attempted_calls": prob.attempted_calls if prob is not None else {},
            "cpu_time_sec": time.process_time() - cpu_start,
            "wall_time_sec": time.perf_counter() - t_start,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
    return result


def run_worker_shard(descriptor_path: Path, start: int, end: int, workers: int) -> None:
    node = socket.gethostname().split(".")[0]
    if node not in DEFAULT_CPU_NODES:
        raise RuntimeError(f"v2_or cluster shards must run on node001-node006, got {node!r}")
    descriptor = _read_json(descriptor_path)
    manifest_path = SC_ROOT / descriptor["manifest_relpath"]
    raw_bytes = manifest_path.read_bytes()
    manifest_str = raw_bytes.decode("utf-8")
    manifest_sha = descriptor.get("manifest_sha256") or _sha256_bytes(raw_bytes)
    code_sha = compute_v2_or_code_sha256()
    if json.loads(manifest_str).get("reuse_source_campaign"):
        _validate_fresh_descriptor(descriptor, raw_bytes, code_sha)
    stage = descriptor["stage"]
    slice_jobs = descriptor["jobs"][start:end]
    if not (0 <= start < end <= len(descriptor["jobs"])):
        raise ValueError(f"Invalid shard slice [{start}, {end}) of {len(descriptor['jobs'])}")

    n_proc = max(1, min(int(workers), len(slice_jobs)))
    results = []
    if n_proc == 1:
        for jb in slice_jobs:
            results.append(_worker_entry((manifest_str, manifest_sha, code_sha, stage, jb)))
    else:
        with ProcessPoolExecutor(max_workers=n_proc) as pool:
            futs = [pool.submit(_worker_entry, (manifest_str, manifest_sha, code_sha, stage, jb)) for jb in slice_jobs]
            for fut in as_completed(futs):
                results.append(fut.result())

    failed = [r for r in results if r["status"] == "failed"]
    shard_out = _campaign_dir(json.loads(manifest_str)) / "stages" / stage / "shards" / f"shard_{start}_{end}.json"
    _write_json(shard_out, {
        "node": node,
        "start": start,
        "end": end,
        "workers": n_proc,
        "execution_code_sha256": code_sha,
        "completed_count": len(results) - len(failed),
        "failed": failed,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    if failed:
        raise RuntimeError(f"Shard [{start}, {end}) had {len(failed)} failures: {failed[0]}")


def _holm_adjust(p_map: dict[str, float]) -> dict[str, float]:
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running_max = 0.0
    for idx, (k, p_raw) in enumerate(items):
        adj = min(1.0, float(p_raw) * float(m - idx))
        running_max = max(running_max, adj)
        out[k] = running_max
    return out


def _one_sided_wilcoxon(diffs_pos_better: np.ndarray) -> float:
    nz = diffs_pos_better[np.abs(diffs_pos_better) > 1e-12]
    if nz.size == 0:
        return 1.0
    try:
        return float(wilcoxon(nz, alternative="greater", zero_method="wilcox", method="auto").pvalue)
    except Exception:
        return 1.0


def _one_sided_paired_t(diffs_pos_better: np.ndarray) -> float:
    if diffs_pos_better.size < 2 or float(np.std(diffs_pos_better, ddof=1)) < 1e-14:
        return 1.0
    return float(ttest_1samp(diffs_pos_better, popmean=0.0, alternative="greater").pvalue)


def _export_ledgers(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    has_mechanism = any("mechanism_receipt" in r for r in rows)
    ledger_rows = []
    cert_rows = []
    for r in rows:
        dep_aud = r.get("deployed_audit") or {}
        prim_aud = r.get("primary_audit") or {}
        anc_aud = r.get("anchor_audit") or {}
        ver_aud = r.get("verif_law_audit") or {}
        ledger_rows.append({
            "job_id": r["job_id"],
            "family": r["family"],
            "target_task_idx": r["target_task_idx"],
            "task_id": r["task_id"],
            "policy_dim": r["policy_dim"],
            "rep_seed": r["rep_seed"],
            "seed": r["seed"],
            "search_seed": r["search_seed"],
            "verif_seed": r["verif_seed"],
            "audit_seed": r["audit_seed"],
            "method_name": r["method_name"],
            "n_online_steps": r["n_online_steps"],
            "search_calls_used": r["search_calls_used"],
            "verification_calls_used": r["verification_calls_used"],
            "total_target_calls": r["total_target_calls"],
            **({k: r[k] for k in ("target_call_cap", "audit_calls_used", "verification_law_audit_calls_used")}
               if "target_call_cap" in r else {}),
            "online_wall_time_sec": r["online_wall_time_sec"],
            "source_pretrain_wall_sec": r.get("source_pretrain_wall_sec", 0.0),
            "decision_state": r["decision_state"],
            "certified": r["certified"],
            "selected_slot": r["selected_slot"],
            "selected_role": r["selected_role"],
            "certified_deployed_mc_obj": dep_aud.get("mc_objective_mean"),
            "primary_slot0_mc_obj": prim_aud.get("mc_objective_mean"),
            "anchor_mc_obj": anc_aud.get("mc_objective_mean"),
            "status_quo_loss": r["status_quo_loss"],
            "fixed_reject_loss_10pct": r["fixed_reject_loss_10pct"],
            "deployed_mc_viol_prob": dep_aud.get("mc_violation_prob"),
            "deployed_mc_viol_ci95_lo": dep_aud.get("mc_viol_ci95_lo"),
            "deployed_mc_viol_ci95_hi": dep_aud.get("mc_viol_ci95_hi"),
            "verif_law_mc_viol_prob": ver_aud.get("mc_violation_prob"),
            "safety_audit_class": r["safety_audit_class"],
            "execution_code_sha256": r["execution_code_sha256"],
            "manifest_sha256": r["manifest_sha256"],
            **({"n_replication_queries": r["n_replication_queries"],
                "mechanism_receipt_json": json.dumps(r.get("mechanism_receipt"), separators=(",", ":"), sort_keys=True)}
               if has_mechanism else {}),
        })
        for c in r.get("slot_certificates", []):
            cert_rows.append({
                "job_id": r["job_id"],
                "family": r["family"],
                "target_task_idx": r["target_task_idx"],
                "rep_seed": r["rep_seed"],
                "method_name": r["method_name"],
                "n_online_steps": r["n_online_steps"],
                **c,
            })

    ledger_csv = out_dir / "per_run_ledger.csv"
    if ledger_rows:
        with ledger_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ledger_rows[0].keys()))
            w.writeheader()
            w.writerows(ledger_rows)

    cert_csv = out_dir / "per_slot_certificates.csv"
    if cert_rows:
        with cert_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cert_rows[0].keys()))
            w.writeheader()
            w.writerows(cert_rows)

    return {
        "per_run_ledger_csv_sha256": _sha256_bytes(ledger_csv.read_bytes()) if ledger_csv.is_file() else "",
        "per_slot_certificates_csv_sha256": _sha256_bytes(cert_csv.read_bytes()) if cert_csv.is_file() else "",
    }


def _reused_source_diagnostics(manifest: dict[str, Any]) -> dict[str, Any]:
    historical_dir = _source_campaign_dir(manifest)
    if historical_dir.resolve() == _campaign_dir(manifest).resolve():
        raise RuntimeError("The reused source campaign must differ from the new campaign")
    previous = _read_json(historical_dir / "stages/source_archive/completion.json")
    if previous["status"] != "complete" or previous["execution_code_sha256"] != manifest["reuse_source_execution_code_sha256"]:
        raise RuntimeError("The reused source completion differs from the frozen U3 source identity")
    prior_diagnostics = {}
    for fam in manifest["families"]:
        source = fam.get("reuse_archive_from_family", fam["name"])
        info = previous["prior_diagnostics"][source]
        bundle = historical_dir / "source_bundles" / source
        archive = bundle / "archive.json"
        if _sha256_bytes(archive.read_bytes()) != info["archive_sha256"]:
            raise RuntimeError(f"Frozen source archive differs from the U3 receipt: {source}")
        for arm in fam["arms"]:
            if arm["kind"] == "tls_olh_kg":
                variant = arm.get("prior_variant", "default")
                prior = bundle / f"prior_{variant}.pkl"
                if variant not in info["variants"] or not prior.is_file():
                    raise FileNotFoundError(f"Missing frozen fitted source: {prior}")
        prior_diagnostics[source] = info
    return {"prior_diagnostics": prior_diagnostics,
            "reused_source_campaign": manifest["reuse_source_campaign"],
            "reused_source_execution_code_sha256": previous["execution_code_sha256"],
            "source_simulation_calls_new": 0, "source_refits_new": 0}


def _campaign_fees(manifest: dict[str, Any]) -> dict[str, Any]:
    stage_root = _campaign_dir(manifest) / "stages"
    summary = {}
    for stage in STAGES:
        attempts = [_read_json(path) for path in stage_root.glob(f"{stage}/attempts/*.json")]
        calls: Counter[str] = Counter()
        for attempt in attempts:
            calls.update(attempt.get("attempted_calls", {}))
        summary[stage] = {
            "attempt_status": dict(Counter(a["status"] for a in attempts)),
            "attempted_calls_recorded": dict(calls),
            "cpu_time_sec_recorded": sum(a.get("cpu_time_sec", 0.0) for a in attempts),
            "wall_time_sec_recorded": sum(a.get("wall_time_sec", 0.0) for a in attempts),
            "unfinished_attempts": [a["job_id"] for a in attempts if a["status"] == "started"],
        }
    return summary


def finish_stage(descriptor_path: Path) -> dict[str, Any]:
    descriptor = _read_json(descriptor_path)
    manifest_path = SC_ROOT / descriptor["manifest_relpath"]
    raw_bytes = manifest_path.read_bytes()
    manifest = json.loads(raw_bytes)
    out_dir = _campaign_dir(manifest)
    stage = descriptor["stage"]
    items_dir = out_dir / "stages" / stage / "items"
    fresh = bool(manifest.get("reuse_source_campaign"))
    if fresh:
        completion_path = items_dir.parent / "completion.json"
        if completion_path.is_file() and manifest.get("sealed_complete"):
            return _read_json(completion_path)
        _validate_fresh_descriptor(descriptor, raw_bytes, compute_v2_or_code_sha256())

    missing = [j["job_id"] for j in descriptor["jobs"] if not (items_dir / f"{j['job_id']}.json").is_file()]
    if missing:
        raise RuntimeError(f"Cannot finish stage {stage!r}: {len(missing)} items missing, e.g. {missing[:3]}")

    summary_extra: dict[str, Any] = {}

    if stage == "source_archive" and fresh:
        summary_extra.update(_reused_source_diagnostics(manifest))
    elif stage == "source_archive":
        prior_diagnostics = {}
        for fam in manifest["families"]:
            if "reuse_archive_from_family" in fam:
                continue
            fam_name = fam["name"]
            fam_jobs = [j for j in descriptor["jobs"] if j["family"] == fam_name]
            all_recs: list[LegacyTrajectoryRecord] = []
            total_calls = 0
            total_wall = 0.0
            for jb in fam_jobs:
                item = _read_json(items_dir / f"{jb['job_id']}.json")
                total_calls += int(item["total_calls"])
                total_wall += float(item.get("wall_time_sec", 0.0))
                for r_dict in item["records"]:
                    all_recs.append(LegacyTrajectoryRecord(**r_dict))

            bundle_dir = out_dir / "source_bundles" / fam_name
            bundle_dir.mkdir(parents=True, exist_ok=True)
            n_single_rep = sum(1 for r in all_recs if r.n_reps <= 1)
            meta = {
                "family": fam_name,
                "n_solver_runs": len(fam_jobs),
                "total_logged_evaluations": len(all_recs),
                "n_single_rep_evaluations": int(n_single_rep),
                "single_rep_fraction": float(n_single_rep / max(len(all_recs), 1)),
                "total_simulation_calls": total_calls,
                "total_solver_wall_sec": total_wall,
                "solvers": fam["source_archive_protocol"].get("solvers", []),
            }
            save_legacy_archive_json(bundle_dir / "archive.json", all_recs, meta)
            archive_sha = _sha256_bytes((bundle_dir / "archive.json").read_bytes())

            variants = fam.get("prior_variants", {
                "default": {
                    "active_rank": 8,
                    "enable_debiasing": True,
                    "enable_lowpass": True,
                    "enable_orthogonalization": True,
                    "enable_sparse_pruning": True,
                }
            })
            fam_diag: dict[str, Any] = {
                "n_solver_trajectories": len(fam_jobs),
                "n_logged_evaluations": len(all_recs),
                "n_single_rep_evaluations": int(n_single_rep),
                "single_rep_fraction": float(n_single_rep / max(len(all_recs), 1)),
                "source_simulation_calls": total_calls,
                "source_solver_wall_sec": total_wall,
                "archive_sha256": archive_sha,
                "variants": {},
            }
            for var_name, v_kwargs in variants.items():
                t_fit0 = time.perf_counter()
                prior = DebiasedSpectralMetaPrior(**v_kwargs).fit(all_recs)
                fit_sec = float(time.perf_counter() - t_fit0)
                with (bundle_dir / f"prior_{var_name}.pkl").open("wb") as fh:
                    pickle.dump(prior, fh)
                fam_diag["variants"][var_name] = {
                    **asdict(prior.diagnostics_),
                    "prior_fit_wall_sec": fit_sec,
                }
            prior_diagnostics[fam_name] = fam_diag
        summary_extra["prior_diagnostics"] = prior_diagnostics

    elif stage == "audit_anchors":
        summary_extra["anchor_audit_calls_used"] = sum(
            _read_json(items_dir / f"{job['job_id']}.json")["audit_calls_used"] for job in descriptor["jobs"]
        )
    elif stage == "audit_and_summarize":
        rows = [_read_json(items_dir / f"{j['job_id']}.json") for j in descriptor["jobs"]]
        ledger_hashes = _export_ledgers(out_dir, rows)
        summary_extra["campaign_summary"] = _compile_statistical_summary(manifest, descriptor, out_dir, rows, ledger_hashes)
        if fresh:
            result = summary_extra["campaign_summary"]
            result["stage_attempt_fees"] = _campaign_fees(manifest)
            anchor_calls = _read_json(out_dir / "stages/audit_anchors/completion.json")["anchor_audit_calls_used"]
            result["call_accounting"] = {
                "source_simulation_calls_new": 0,
                "source_refits_new": 0,
                "search_calls_used": sum(r["search_calls_used"] for r in rows),
                "terminal_verification_calls_used": sum(r["verification_calls_used"] for r in rows),
                "target_call_cap_total": sum(r["target_call_cap"] for r in rows),
                "anchor_audit_calls_used": anchor_calls,
                "policy_audit_calls_used": sum(r["audit_calls_used"] for r in rows),
                "verification_law_audit_calls_used": sum(r["verification_law_audit_calls_used"] for r in rows),
            }
        manifest["sealed_complete"] = True
        manifest["sealed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["sealed_execution_code_sha256"] = descriptor.get("execution_code_sha256", compute_v2_or_code_sha256())
        if not fresh:
            _write_json(manifest_path, manifest)

    completion = {
        "stage": stage,
        "status": "complete",
        "n_jobs": len(descriptor["jobs"]),
        "execution_code_sha256": descriptor.get("execution_code_sha256", compute_v2_or_code_sha256()),
        "manifest_sha256": descriptor.get("manifest_sha256", ""),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        **summary_extra,
    }
    _write_json(out_dir / "stages" / stage / "completion.json", completion)
    if stage == "audit_and_summarize":
        _write_json(out_dir / "compact_summary.json", completion["campaign_summary"])
        if fresh:
            _write_json(manifest_path, manifest)
    return completion


def _compile_statistical_summary(
    manifest: dict[str, Any],
    descriptor: dict[str, Any],
    out_dir: Path,
    rows: list[dict[str, Any]],
    ledger_hashes: dict[str, str],
) -> dict[str, Any]:
    """Compile strictly roster-invariant task-level and run-level summaries."""
    src_completion = _read_json(out_dir / "stages" / "source_archive" / "completion.json")
    prior_diag = src_completion.get("prior_diagnostics", {})

    stop_lookup = {
        (r["family"], r["target_task_idx"], r["rep_seed"]): r
        for r in rows
        if r["method_name"] == "STOP"
    }

    by_fam_budget: dict[str, Any] = {}
    ref_arm_global = manifest.get("reference_arm", "TLS-OLH-KG")

    for fam in manifest["families"]:
        fam_name = fam["name"]
        if not fam.get("online_step_budgets"):
            continue
        ref_arm = fam.get("reference_arm", ref_arm_global)
        src_fam = fam.get("reuse_archive_from_family", fam_name)
        src_info = prior_diag.get(src_fam, {})
        src_calls = int(src_info.get("source_simulation_calls", 0))
        fam_summary: dict[str, Any] = {
            "reference_arm": ref_arm,
            "source_simulation_calls": src_calls,
            "n_solver_trajectories": int(src_info.get("n_solver_trajectories", 0)),
            "n_logged_evaluations": int(src_info.get("n_logged_evaluations", 0)),
            "budgets": {},
        }

        for b_steps in fam["online_step_budgets"]:
            b_rows = [r for r in rows if r["family"] == fam_name and int(r["n_online_steps"]) == int(b_steps)]
            if not any(r["method_name"] == "STOP" for r in b_rows):
                for (fn, _tidx, _rseed), s_row in stop_lookup.items():
                    if fn == fam_name:
                        b_rows.append(s_row)

            methods = sorted({r["method_name"] for r in b_rows})
            method_table: dict[str, Any] = {}
            task_sq_vecs: dict[str, dict[int, float]] = {}
            run_sq_vecs: dict[str, dict[tuple[int, int], float]] = {}

            for m_name in methods:
                m_sub = [r for r in b_rows if r["method_name"] == m_name]
                cert_objs = [
                    float(r["deployed_audit"]["mc_objective_mean"])
                    for r in m_sub
                    if r["certified"] and r.get("deployed_audit") is not None
                ]
                prim_objs = [float(r["primary_audit"]["mc_objective_mean"]) for r in m_sub if r.get("primary_audit")]
                sq_losses = [float(r["status_quo_loss"]) for r in m_sub]
                fr_losses = [float(r["fixed_reject_loss_10pct"]) for r in m_sub]
                cert_flags = [1.0 if r["certified"] else 0.0 for r in m_sub]

                by_task_sq: dict[int, list[float]] = {}
                by_task_fr: dict[int, list[float]] = {}
                by_task_cert: dict[int, list[float]] = {}
                by_task_prim: dict[int, list[float]] = {}
                for r in m_sub:
                    ti = int(r["target_task_idx"])
                    by_task_sq.setdefault(ti, []).append(float(r["status_quo_loss"]))
                    by_task_fr.setdefault(ti, []).append(float(r["fixed_reject_loss_10pct"]))
                    by_task_cert.setdefault(ti, []).append(1.0 if r["certified"] else 0.0)
                    if r.get("primary_audit"):
                        by_task_prim.setdefault(ti, []).append(float(r["primary_audit"]["mc_objective_mean"]))

                t_sq_map = {ti: float(np.mean(vs)) for ti, vs in sorted(by_task_sq.items())}
                t_sq_arr = np.asarray(list(t_sq_map.values()), dtype=float)
                t_fr_arr = np.asarray([float(np.mean(vs)) for _, vs in sorted(by_task_fr.items())], dtype=float)
                t_cert_arr = np.asarray([float(np.mean(vs)) for _, vs in sorted(by_task_cert.items())], dtype=float)

                task_sq_vecs[m_name] = t_sq_map
                run_sq_vecs[m_name] = {
                    (int(r["target_task_idx"]), int(r["rep_seed"])): float(r["status_quo_loss"]) for r in m_sub
                }

                ver_viols = [
                    float(r["verif_law_audit"]["mc_violation_prob"])
                    for r in m_sub
                    if r["certified"] and r.get("verif_law_audit") is not None
                ]
                dep_viols = [
                    float(r["deployed_audit"]["mc_violation_prob"])
                    for r in m_sub
                    if r["certified"] and r.get("deployed_audit") is not None
                ]

                method_table[m_name] = {
                    "n_runs": len(m_sub),
                    "n_independent_tasks": len(t_sq_map),
                    "certified_rate": float(np.mean(cert_flags)),
                    "abstention_rate": float(1.0 - np.mean(cert_flags)),
                    "task_certified_rate_mean": float(np.mean(t_cert_arr)),
                    "conditional_certified_obj_mean": float(np.mean(cert_objs)) if cert_objs else None,
                    "conditional_certified_obj_se": (
                        float(np.std(cert_objs, ddof=1) / np.sqrt(len(cert_objs))) if len(cert_objs) > 1 else None
                    ),
                    "primary_slot0_obj_mean": float(np.mean(prim_objs)) if prim_objs else None,
                    "status_quo_loss_task_mean": float(np.mean(t_sq_arr)),
                    "status_quo_loss_task_se": float(np.std(t_sq_arr, ddof=1) / np.sqrt(len(t_sq_arr))) if len(t_sq_arr) > 1 else 0.0,
                    "status_quo_loss_run_mean": float(np.mean(sq_losses)),
                    "status_quo_loss_run_se": float(np.std(sq_losses, ddof=1) / np.sqrt(len(sq_losses))) if len(sq_losses) > 1 else 0.0,
                    "fixed_reject_loss_10pct_task_mean": float(np.mean(t_fr_arr)),
                    "deployed_mc_viol_prob_mean": float(np.mean(dep_viols)) if dep_viols else None,
                    "verif_law_mc_viol_prob_mean": float(np.mean(ver_viols)) if ver_viols else None,
                    "audit_point_fc_count": int(sum(1 for r in m_sub if r["audit_point_exceeds_alpha"])),
                    "safety_audit_breakdown": dict(Counter(r["safety_audit_class"] for r in m_sub)),
                    "selected_role_breakdown": dict(Counter(r["selected_role"] for r in m_sub)),
                    "mean_search_calls": float(np.mean([r["search_calls_used"] for r in m_sub])),
                    "mean_verification_calls": float(np.mean([r["verification_calls_used"] for r in m_sub])),
                    "mean_total_target_calls": float(np.mean([r["total_target_calls"] for r in m_sub])),
                    "mean_online_wall_sec": float(np.mean([r["online_wall_time_sec"] for r in m_sub])),
                    "mean_source_pretrain_wall_sec": float(np.mean([r.get("source_pretrain_wall_sec", 0.0) for r in m_sub])),
                }

            paired_task: dict[str, Any] = {}
            paired_run: dict[str, Any] = {}
            if ref_arm in methods:
                ref_t = task_sq_vecs[ref_arm]
                ref_r = run_sq_vecs[ref_arm]
                p_map_t: dict[str, float] = {}
                p_map_r: dict[str, float] = {}
                for other_m in methods:
                    if other_m == ref_arm:
                        continue
                    tk = sorted(set(ref_t.keys()) & set(task_sq_vecs[other_m].keys()))
                    dt = np.array([task_sq_vecs[other_m][k] - ref_t[k] for k in tk], dtype=float)
                    bt = np.array([task_sq_vecs[other_m][k] for k in tk], dtype=float)
                    pw_t = _one_sided_wilcoxon(dt)
                    pt_t = _one_sided_paired_t(dt)
                    p_map_t[other_m] = pw_t
                    paired_task[other_m] = {
                        "n_tasks": len(tk),
                        "mean_saving_status_quo_loss": float(np.mean(dt)),
                        "se_saving_status_quo_loss": float(np.std(dt, ddof=1) / np.sqrt(len(dt))) if len(dt) > 1 else 0.0,
                        "mean_rel_improvement_pct": float(100.0 * np.mean(dt / np.maximum(np.abs(bt), 1e-8))),
                        "task_wins_losses_ties": [
                            int(np.sum(dt > 1e-9)),
                            int(np.sum(dt < -1e-9)),
                            int(np.sum(np.abs(dt) <= 1e-9)),
                        ],
                        "wilcoxon_one_sided_p": pw_t,
                        "paired_t_one_sided_p": pt_t,
                    }

                    rk = sorted(set(ref_r.keys()) & set(run_sq_vecs[other_m].keys()))
                    dr = np.array([run_sq_vecs[other_m][k] - ref_r[k] for k in rk], dtype=float)
                    br = np.array([run_sq_vecs[other_m][k] for k in rk], dtype=float)
                    pw_r = _one_sided_wilcoxon(dr)
                    p_map_r[other_m] = pw_r
                    paired_run[other_m] = {
                        "n_pairs": len(rk),
                        "mean_saving_status_quo_loss": float(np.mean(dr)),
                        "mean_rel_improvement_pct": float(100.0 * np.mean(dr / np.maximum(np.abs(br), 1e-8))),
                        "run_wins_losses_ties": [
                            int(np.sum(dr > 1e-9)),
                            int(np.sum(dr < -1e-9)),
                            int(np.sum(np.abs(dr) <= 1e-9)),
                        ],
                        "wilcoxon_one_sided_p": pw_r,
                    }

                for k, hp in _holm_adjust(p_map_t).items():
                    paired_task[k]["holm_adjusted_p"] = hp
                for k, hp in _holm_adjust(p_map_r).items():
                    paired_run[k]["holm_adjusted_p"] = hp

            fam_summary["budgets"][str(b_steps)] = {
                "methods": method_table,
                "paired_task_level_vs_ref": paired_task,
                "paired_run_level_vs_ref": paired_run,
            }
        by_fam_budget[fam_name] = fam_summary

    return {
        "campaign_name": manifest["campaign_name"],
        "contract_version": "v2.1_roster_invariant",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "execution_code_sha256": descriptor.get("execution_code_sha256", compute_v2_or_code_sha256()),
        "manifest_sha256": descriptor.get("manifest_sha256", ""),
        **ledger_hashes,
        "total_evaluated_cells": len(rows),
        "total_certified_deployments": int(sum(1 for r in rows if r["certified"])),
        "total_abstentions": int(sum(1 for r in rows if not r["certified"])),
        "safety_audit_summary": dict(Counter(r["safety_audit_class"] for r in rows)),
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
    p_desc.add_argument("--stage", choices=STAGES, required=True)

    p_fees = sub.add_parser("fees", help="Summarize retained item attempts and recorded fees")
    p_fees.add_argument("--manifest", type=Path, required=True)

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
    elif args.action == "fees":
        print(json.dumps(_campaign_fees(_read_json(args.manifest)), indent=2))
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
