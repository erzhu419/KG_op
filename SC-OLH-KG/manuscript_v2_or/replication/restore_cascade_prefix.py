"""Restore only the frozen first-query model state for the cascade diagnostic."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pickle
import time

import numpy as np

from v2_or.acquisition.verification_kg import (
    build_cold_dct_initial_points,
    generate_dynamic_candidate_pool,
    score_gate_power_vkg,
)
from v2_or.contracts import SufficientStatsRecord
from v2_or.dispatch.campaign_runner import (
    _read_json,
    _write_json,
    build_family_tasks,
    compute_v2_or_code_sha256,
)
from v2_or.models.tempered_hvd_posterior import TemperedHierarchicalHVDModel

COHORT = Path("paper_artifacts/v2_or_gpfta_referee_ablation_v1")
MANIFEST = Path("v2_or/manifests/v2_or_gpfta_referee_ablation_v1.json")
INDEXES = (0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21)
ARM = "HistoryFull-GateKG"


def restore(job: tuple[dict, int, Path]) -> str:
    family, index, output = job
    name = family["name"]
    rep_seed = family["target_eval_seeds"][0]
    job_id = f"{name}_t{index:02d}"
    receipt = output / f"{job_id}.json"
    # A started receipt protects paid work even if this process fails.
    with receipt.open("x") as stream:
        stream.write('{"status":"started"}\n')
    start = time.process_time()
    wall_start = time.perf_counter()
    problem = None
    status = {"job_id": job_id, "family": name, "target_task_idx": index, "rep_seed": rep_seed}
    try:
        original = _read_json(COHORT / "stages/search_and_verify/items" /
                              f"sv_{name}_N60_t{index:02d}_s{rep_seed}_{ARM}.json")
        source_path = Path("paper_artifacts/v2_or_u3_confirm_v1/source_bundles") / name / "prior_default.pkl"
        with source_path.open("rb") as stream:
            prior = pickle.load(stream)
        problem = build_family_tasks(family, "target")[index]
        n0 = family["n0_init"]
        reps = family["batch_reps"]
        problem.call_limits = {"search": n0 * reps, "verification": 0, "audit": 0}
        seed = original["seed"]
        if problem.task_id != original["task_id"] or seed != rep_seed + 10_007 * index:
            raise ValueError(f"Task or seed differs from the frozen run: {job_id}")
        arm = next(item for item in family["arms"] if item["name"] == ARM)
        model = TemperedHierarchicalHVDModel(
            prior,
            mode=arm["model_mode"],
            use_source_basis=arm["use_source_basis"],
            use_source_coef_prior=arm.get("use_source_coef_prior"),
            use_source_hvd_prior=arm["use_source_hvd_prior"],
            variance_estimator=arm["variance_estimator"],
            use_empirical_r_perp=arm["use_empirical_r_perp"],
            n0_freeze=n0,
            coefficient_prior=arm["coefficient_prior"],
        )
        rng_init = np.random.default_rng(seed)
        points = build_cold_dct_initial_points(problem, n0, rng_init)
        records = []
        for point in points:
            exposure = problem.nominal_exposure(point)
            samples = problem.simulate_replications(point, reps, rng_init, split="search")
            rec = SufficientStatsRecord.from_samples(point, samples, exposure)
            records.append(rec)
            model.observe(rec)
        model.freeze_matched_scales(records)
        candidates, exposures = generate_dynamic_candidate_pool(
            problem, model, prior, np.random.default_rng(original["search_seed"]),
            pool_size=family["pool_size"], enable_source_templates=False,
        )
        prediction = model.predict_joint(candidates, exposures)
        reference_obj = 0.5 * float(records[0].mean_obj) + 0.5 * float(prediction.mean_obj[0])
        scores = score_gate_power_vkg(prediction, alpha=problem.alpha, batch_reps=reps,
                                     status_quo_obj=reference_obj, common_target_screen=True)
        saved_query = original["mechanism_receipt"]["first_query_index"]
        if int(np.argmax(scores)) != saved_query:
            raise ValueError(f"Restored first query disagrees with the frozen run: {job_id}")
        initial_pred = model.predict_joint(
            [r.x_norm for r in model.unique_records], [r.exposure for r in model.unique_records]
        )
        state = {
            "family": family, "task_index": index, "records": records, "initial_pred": initial_pred,
            "candidates": candidates, "exposures": exposures, "prediction": prediction,
            "reference_obj": reference_obj, "alpha": problem.alpha, "saved_query": saved_query,
            "implemented_scores": scores,
        }
        state_path = output / f"{job_id}.pkl"
        tmp = state_path.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            pickle.dump(state, stream)
        tmp.replace(state_path)
        status.update(status="ok", saved_query=int(saved_query), n_candidates=len(candidates))
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status.update(
            worker_cpu_seconds=time.process_time() - start, wall_seconds=time.perf_counter() - wall_start,
            search_calls=problem.search_calls if problem else 0,
            verification_calls=problem.verification_calls if problem else 0,
            audit_calls=problem.audit_calls if problem else 0,
        )
        _write_json(receipt, status)
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = _read_json(MANIFEST)
    summary = _read_json(COHORT / "compact_summary.json")
    if compute_v2_or_code_sha256() != summary["execution_code_sha256"]:
        raise ValueError("Working runtime differs from the frozen cohort")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(f, i, args.output) for f in manifest["families"] for i in INDEXES]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for job_id in executor.map(restore, jobs[args.start:args.end]):
            print(job_id, flush=True)


if __name__ == "__main__":
    main()
