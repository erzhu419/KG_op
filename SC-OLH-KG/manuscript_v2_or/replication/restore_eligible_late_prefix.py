"""Restore four frozen 59-query states by extending their saved initial batches."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

import numpy as np

from v2_or.acquisition.verification_kg import (
    _initial_release_evidence,
    _observed_variances,
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
from v2_or.verification.gated_verifier_v2 import build_feasible_aware_shortlist


COHORT = Path("paper_artifacts/v2_or_gpfta_referee_ablation_v1")
MANIFEST = Path("v2_or/manifests/v2_or_gpfta_referee_ablation_v1.json")
ARM = "HistoryFull-GateKG"
TASK_INDICES = (0, 12)
ONLINE_STEPS = 60


def same_summary(actual, saved):
    if isinstance(saved, dict):
        return actual.keys() == saved.keys() and all(same_summary(actual[key], saved[key]) for key in saved)
    if isinstance(saved, float):
        return math.isclose(actual, saved, rel_tol=1e-9, abs_tol=1e-12)
    return actual == saved


def original_search_view(problem, model, prior, rng, family, anchor_mean):
    candidates, exposures = generate_dynamic_candidate_pool(
        problem, model, prior, rng, pool_size=family["pool_size"], enable_source_templates=False,
    )
    prediction = model.predict_joint(candidates, exposures)
    reference = 0.5 * anchor_mean + 0.5 * float(prediction.mean_obj[0])
    scores = score_gate_power_vkg(
        prediction, alpha=problem.alpha, batch_reps=family["batch_reps"],
        status_quo_obj=reference, common_target_screen=True,
    )
    return candidates, exposures, prediction, reference, scores


def validate_original_endpoint(problem, model, history, original, nomination_mode, revisits):
    records = model.unique_records
    policies = [r.x_norm for r in records]
    prediction = model.predict_joint(policies, [r.exposure for r in records])
    shortlist = build_feasible_aware_shortlist(
        candidate_pool=policies, mean_obj=prediction.mean_obj, mean_con=prediction.mean_con,
        std_con=prediction.std_con, ucb_aleat_var_con=prediction.ucb_aleatoric_var_con,
        r_perp_con=prediction.r_perp_con, initial_records=records, alpha=problem.alpha,
        nomination_mode=nomination_mode,
        reference_obj=0.5 * history[0].mean_obj + 0.5 * float(prediction.mean_obj[0]),
    )
    roles, nominees = zip(*shortlist)
    if list(roles) != original["shortlist_roles"] or not np.array_equal(nominees, original["shortlist_policies"]):
        raise ValueError("Replayed endpoint nominees differ from the original receipt")
    if revisits != original["n_replication_queries"] or len(records) != original["n_unique_policies_evaluated"]:
        raise ValueError("Replayed revisit or unique-policy count differs from the original receipt")
    terminal_receipt = SimpleNamespace(
        primary_policy=nominees[0], deployed_policy=original["deployed_policy"],
    )
    variances = _observed_variances(history, model, terminal_receipt)
    if not same_summary(variances, original["mechanism_receipt"]["variance"]):
        raise ValueError("Replayed search variance summaries differ from the original receipt")


def restore(job: tuple[dict, int, Path, Path]) -> str:
    family, index, cached_prefix, output = job
    name = family["name"]
    rep_seed = family["target_eval_seeds"][0]
    n0 = family["n0_init"]
    reps = family["batch_reps"]
    job_id = f"{name}_t{index:02d}"
    receipt_path = output / f"{job_id}.json"
    status = {
        "status": "started", "job_id": job_id, "family": name, "target_task_idx": index,
        "rep_seed": rep_seed, "search_calls_attempted": 0, "search_calls": 0,
        "verification_calls": 0, "audit_calls": 0, "initialization_calls_reused": 0,
    }
    with receipt_path.open("x") as stream:
        json.dump(status, stream)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    problem = None
    try:
        original = _read_json(COHORT / "stages/search_and_verify/items" /
                              f"sv_{name}_N60_t{index:02d}_s{rep_seed}_{ARM}.json")
        with (cached_prefix / f"{job_id}.pkl").open("rb") as stream:
            cached = pickle.load(stream)
        prior_path = Path("paper_artifacts/v2_or_u3_confirm_v1/source_bundles") / name / "prior_default.pkl"
        with prior_path.open("rb") as stream:
            prior = pickle.load(stream)
        problem = build_family_tasks(family, "target")[index]
        problem.call_limits = {"search": ONLINE_STEPS * reps, "verification": 0, "audit": 0}
        seed = rep_seed + 10_007 * index
        if (problem.task_id, seed, seed + 100_003, n0, ONLINE_STEPS, (n0 + ONLINE_STEPS) * reps) != (
            original["task_id"], original["seed"], original["search_seed"], original["n0_init_points"],
            original["actual_online_steps"], original["search_calls_used"],
        ):
            raise ValueError("Selected task differs from the frozen search contract")
        arm = next(item for item in family["arms"] if item["name"] == ARM)
        if (arm["model_mode"], arm["init_mode"], arm["proposal_mode"], arm["acquisition"], arm["coefficient_prior"]) != (
            "matched_coefficients", "cold_dct", "local_dct_only", "gate_power_vkg", "historical_full",
        ):
            raise ValueError("Late restoration requires the frozen default matched full-history arm")
        model = TemperedHierarchicalHVDModel(
            prior, mode=arm["model_mode"], use_source_basis=arm["use_source_basis"],
            use_source_coef_prior=arm.get("use_source_coef_prior"), use_source_hvd_prior=arm["use_source_hvd_prior"],
            variance_estimator=arm["variance_estimator"], use_empirical_r_perp=arm["use_empirical_r_perp"],
            n0_freeze=n0, coefficient_prior=arm["coefficient_prior"],
        )
        records = list(cached["records"])
        if cached["family"]["name"] != name or cached["task_index"] != index or len(records) != n0 or any(
            r.n_reps != reps for r in records
        ):
            raise ValueError("Initialization records differ from the frozen task and replication contract")
        evidence = _initial_release_evidence(prior, records)
        evidence["applied_weak"] = False
        if not same_summary(evidence, original["mechanism_receipt"]["early_release"]):
            raise ValueError("Restored initial evidence differs from the original receipt")
        for record in records:
            model.observe(record)
        model.freeze_matched_scales(records)
        history = list(records)
        status["initialization_calls_reused"] = n0 * reps
        rng = np.random.default_rng(original["search_seed"])
        revisits = 0
        for step in range(ONLINE_STEPS):
            candidates, exposures, prediction, reference, scores = original_search_view(
                problem, model, prior, rng, family, records[0].mean_obj,
            )
            chosen = int(np.argmax(scores))
            if step == 0:
                if chosen != original["mechanism_receipt"]["first_query_index"] or not np.array_equal(
                    candidates, cached["candidates"]
                ):
                    raise ValueError("Restored first pool or query differs from the archived first state")
                np.testing.assert_allclose(scores, cached["implemented_scores"], rtol=1e-9, atol=1e-12)
            elif step == ONLINE_STEPS - 1:
                unique_records = model.unique_records
                unique_policies = [r.x_norm for r in unique_records]
                initial_pred = model.predict_joint(unique_policies, [r.exposure for r in unique_records])
                state = {
                    "family": family, "task_index": index, "records": records, "batch_history": history,
                    "initial_pred": initial_pred, "model": model, "unique_records": unique_records,
                    "candidates": candidates, "exposures": exposures, "prediction": prediction,
                    "reference_obj": reference, "alpha": problem.alpha, "saved_query": chosen,
                    "implemented_scores": scores, "completed_online_steps": step,
                }
                with (output / f"{job_id}.pending").open("xb") as stream:
                    pickle.dump(state, stream)
                status.update(
                    completed_online_steps=step, native_next_query=chosen, n_candidates=len(candidates),
                    n_unique_evaluated_at_snapshot=len(unique_records), search_calls_at_snapshot=problem.search_calls,
                )
            policy = candidates[chosen]
            revisits += int(model._policy_key(policy) in model._records_by_key)
            status.update(search_calls_attempted=(step + 1) * reps, search_calls=problem.search_calls)
            _write_json(receipt_path, status)
            samples = problem.simulate_replications(policy, reps, rng, split="search")
            record = SufficientStatsRecord.from_samples(policy, samples, exposures[chosen])
            model.observe(record)
            history.append(record)

        validate_original_endpoint(problem, model, history, original, arm["nomination_mode"], revisits)
        (output / f"{job_id}.pending").replace(output / f"{job_id}.pkl")
        status.update(status="ok", verified_original_endpoint=True, endpoint_roles=original["shortlist_roles"],
                      endpoint_revisits=revisits, endpoint_unique_policies=len(model.unique_records))
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if problem:
            status.update(search_calls_attempted=problem.attempted_calls.get("search", 0),
                          search_calls=problem.search_calls, verification_calls=problem.verification_calls,
                          audit_calls=problem.audit_calls)
        status.update(worker_cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.perf_counter() - wall_start)
        _write_json(receipt_path, status)
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--cached-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if compute_v2_or_code_sha256() != _read_json(COHORT / "compact_summary.json")["execution_code_sha256"]:
        raise ValueError("Working runtime differs from the frozen cohort")
    manifest = _read_json(MANIFEST)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(family, index, args.cached_prefix, args.output) for family in manifest["families"] for index in TASK_INDICES]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for job_id in executor.map(restore, jobs[args.start:args.end]):
            print(job_id, flush=True)


if __name__ == "__main__":
    main()
