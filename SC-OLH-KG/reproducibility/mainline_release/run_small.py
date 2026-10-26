"""One registered d1000 AFTER case, using the frozen production implementation.

Run with ordinary Python and the release's case JSON. Frozen source models and
ten paid initialization rows are reused; three online calls and a fresh old
terminal are executed. This is release acceptance, separate from sealed studies.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import platform
import resource
import signal
import sys
import time
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[2]
ARM = "legacy_role_after_init_point_only"


def read(path):
    return json.loads(Path(path).read_text())


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def save(path, value):
    Path(path).write_text(json.dumps(value, default=_plain, indent=2, allow_nan=False) + "\n")


def history(engine):
    return [{"point": list(point), "raw": _plain(raw) if hasattr(raw, "tolist") else list(raw)}
            for point, raw in engine.history]


class _ConfigReady(Exception):
    def __init__(self, config):
        self.config = config


def build_engine(case, directory, receipt):
    """Use the original config constructor with an already resolved public task.

    Production build_native resolves this task through a scheduler submission
    script. Exporting the resolved task removes that deployment dependency; the
    source model, config constructor, native frontend and AFTER class are shared.
    """
    from performance import benchmark_lodo_meta_prior as benchmark
    from performance.mainline_role_backend import RolePosteriorAlgorithm
    from representation.meta_prior import MetaPriorProblemAdapter

    task = read(directory / case["resolved_task_file"])
    task = deepcopy(task)
    if task["heldout"] != case["domain"] or int(task["seed"]) != case["geometry_task_seed"]:
        raise ValueError("Resolved task differs from the registered release case")
    args = task["args"]
    if (args["d"], args["N"], args["n0"]) != (1000, 13, 10):
        raise ValueError("Release case must retain d1000 and the 10+3 search budget")
    # Evaluation logging reads oracle truth but is not used by any decisions.
    # The frozen scientific computation is unchanged when this audit is omitted.
    receipt["audit_only_config_changes"] = {
        key: {"original": args.get(key), "release": value}
        for key, value in (("evaluate_interval", 0), ("truth_pool_diagnostics", False),
                           ("progress_logging", False), ("checkpoint_path", ""),
                           ("runtime_checkpoint_dir", ""), ("runtime_checkpoint_resume", False))
        if args.get(key) != value}
    args.update({key: row["release"] for key, row in receipt["audit_only_config_changes"].items()})
    with (directory / case["native_source_prior_file"]).open("rb") as stream:
        receipt["source_model_restorations"] += 1
        prior = pickle.load(stream)
    with (directory / case["source_models_file"]).open("rb") as stream:
        receipt["source_model_restorations"] += 1
        bundle = pickle.load(stream)
    benchmark._source_informed_archive_contract(args, prior.diagnostics())
    target = benchmark.build_scalarized_problem(
        case["domain"], args["d"], args["L"], args["sigma"], args["alpha"],
        benchmark.parse_weights(args["weights"]), problem_kwargs=case["problem_kwargs"])
    problem = MetaPriorProblemAdapter(target, prior,
        proposal_pool_size=args["meta_proposal_pool_size"],
        refinement_count=args["meta_refinement_count"])

    def capture_config(_problem, config):
        raise _ConfigReady(config)

    with patch.object(benchmark, "build_target_problem", return_value=(problem, prior.diagnostics())), \
            patch.object(benchmark, "SingleOLHKGAlgorithm", capture_config):
        try:
            benchmark.run_one(task)
        except _ConfigReady as captured:
            config = captured.config
    config.observation_seed = int(case["observation_seed"])
    if config.decision_backend != "constrained_ts" or config.llm_prior_enabled:
        raise ValueError("Release task must use frozen constrained TS without external services")
    initial = read(directory / case["initial_rows_file"])
    if isinstance(initial, dict):
        initial = initial["history"]
    if len(initial) != 10 or [row["point"] for row in initial] != [list(p) for p in config.initial_design_points]:
        raise ValueError("Frozen initialization must match the ten registered policies in order")
    engine = RolePosteriorAlgorithm(problem, config, initial_rows=initial, defer_terminal=True)
    engine.legacy_task, engine.source_prior = task, prior
    engine.backend_arm, engine.backend_bundle = ARM, bundle
    receipt["source_identities"] = bundle["identities"]
    return engine


def terminal_input(case, engine, search):
    """Apply the same distinct-challenger transform as the sealed studies."""
    from performance.run_mainline_same_policy import transform

    frozen = search["terminal"]
    if frozen["candidate_verification_budgets"] != [80, 128, 128] or frozen["max_verification_budget"] != 352:
        raise ValueError("The old terminal budget changed")
    if (frozen["objective_comparison_budget_per_policy"] != 8
            or frozen["familywise_delta"] != .05
            or frozen["objective_comparison"]["delta"] != .05/3):
        raise ValueError("The registered old terminal error allocation changed")
    rows = frozen["frozen_shortlist"]
    observed = {tuple(row["point"]) for row in search["history"]}
    if len(rows) != 3 or any(tuple(row["point"]) not in observed for row in rows):
        raise ValueError("The terminal shortlist must contain three frozen observed policies")
    return transform(dict(seed=case["geometry_task_seed"], arm=ARM,
        domain=case["domain"], verification_seed=case["verification_seed"],
        frozen_shortlist=rows, primary=rows[0]["point"],
        candidate_budgets=[80, 128, 128], search_evaluation_count=13,
        familywise_delta=frozen["familywise_delta"], method="normal_quantile_tolerance",
        mean_delta_fraction=frozen["mean_delta_fraction"], shortlist_mode=frozen["shortlist_mode"],
        objective_incumbent_position=frozen["objective_incumbent_position"],
        objective_comparison_budget=frozen["objective_comparison_budget_per_policy"],
        objective_comparison_delta=frozen["objective_comparison"]["delta"],
        tau=float(engine.problem.tau), truth_used=False))


def run(case_path, output):
    """Execute once; every failed attempt keeps its raw rows and cost receipt."""
    case_path, output = Path(case_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, wall = time.process_time(), time.perf_counter()
    receipt = dict(schema="mainline-small-receipt-v1", kind="release_acceptance_only",
        status="failed", started_utc=datetime.now(timezone.utc).isoformat(),
        automatic_retries=0, scientific_cohort_overwritten=False, truth_used=False,
        new_source_calls=0, new_source_fits=0, reused_initial_calls=0,
        new_target_search_calls=0, new_verification_calls=0,
        source_model_restorations=0, new_truth_function_calls=0,
        versions={"python": platform.python_version()}, stage="load_case")
    engine = None
    try:
        case = read(case_path)
        if case["schema"] != "mainline-small-v1" or case["kind"] != "release_acceptance_only":
            raise ValueError("A preselected release acceptance case is required")
        receipt.update(domain=case["domain"], arm=ARM,
            geometry_task_seed=case["geometry_task_seed"], observation_seed=case["observation_seed"],
            verification_seed=case["verification_seed"], source_commit=case["source_commit"])
        for name in ("numpy", "scipy", "requests"):
            receipt["versions"][name] = version(name)
        receipt["stage"] = "source_restore_and_initialization"
        engine = build_engine(case, case_path.parent, receipt)
        from performance.mainline_shared_neighbors import shared_neighbor_candidates
        from performance.run_mainline_terminal_equal import run_verifier
        neighbors = read(case_path.parent / case["shared_neighbors_file"])
        points = neighbors["points"]
        if len(points) != int(neighbors["metadata"]["count"]):
            raise ValueError("Frozen neighbor metadata does not match its physical policy bank")
        phase = "search"
        original_simulate = engine.problem.simulate

        def simulate(point, rng=None):
            key = "new_target_search_calls" if phase == "search" else "new_verification_calls"
            receipt[key] += 1  # An attempted call is charged even if it raises.
            row = dict(phase=phase, call=receipt[key], point=list(map(int, point)), status="failed")
            try:
                value = original_simulate(point, rng)
                row.update(raw=value, status="complete")
                return value
            finally:
                with (output / "raw_calls.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, default=_plain, allow_nan=False) + "\n")

        def no_truth(*_args, **_kwargs):
            raise AssertionError("Release acceptance may not read oracle target outcomes")

        with ExitStack() as stack:
            stack.enter_context(patch.object(engine.problem, "simulate", simulate))
            stack.enter_context(patch.object(engine, "_evaluate_recommendation", no_truth))
            stack.enter_context(patch.object(engine, "_true_chance_margin", no_truth))
            receipt["stage"] = "online_search"
            with shared_neighbor_candidates(engine, points) as audit:
                engine.run_search()
            receipt["reused_initial_calls"] = engine.replayed_initial_calls
            rows = history(engine)
            if len(rows) != 13 or receipt["new_target_search_calls"] != 3:
                raise ValueError("Actual search calls differ from the frozen 10+3 budget")
            if [row["paid_prefix"] for row in audit["stages"]] != [10, 11, 12]:
                raise ValueError("All three production TS decisions must precede their observations")
            search = dict(**engine.frozen_terminal, history=rows, logical_search_calls=13,
                geometry=case["problem_kwargs"], truth_used=False)
            save(output / "search.json", search)
            decisions = []
            for stage, observed in zip(audit["stages"], rows[10:], strict=True):
                selected = list(audit["points"][stage["selected_id"]])
                if selected != observed["point"]:
                    raise ValueError("The paid observation differs from its preceding TS selection")
                score, index = stage["score"], stage["selected_index"]
                decisions.append(dict(paid_prefix=stage["paid_prefix"], selected=selected,
                    native_candidate_count=stage["native_candidate_count"],
                    added_candidate_count=stage["added_candidate_count"],
                    scored_candidate_count=len(stage["scored_ids"]),
                    selected_route=stage["action_sources"][index],
                    selected_score=float(score["total"][index]),
                    sampled_feasibility_mode=score["sampled_feasibility_mode"],
                    sampled_feasible_count=score["sampled_feasible_count"]))
            save(output / "query_audit.json", dict(decisions=decisions,
                shared_neighbors=len(points), truth_used=False))
            item = terminal_input(case, engine, search)
            save(output / "frozen_terminal_input.json", item)
            receipt["stage"] = "independent_terminal_verification"
            phase = "verification"
            deployed, verified = run_verifier(engine.problem, item)
            if history(engine) != rows:
                raise ValueError("Independent verification changed the search history")
            if receipt["new_verification_calls"] != verified["verification_budget"]:
                raise ValueError("Verifier accounting differs from the actual simulator calls")
            if verified["max_verification_budget"] != 352 or receipt["new_verification_calls"] > 352:
                raise ValueError("Verification exceeded the frozen terminal cap")
            save(output / "verification.json", dict(deployed=list(deployed), verification=verified,
                truth_used=False))
            receipt.update(status="complete", stage="complete", logical_search_calls=13,
                logical_total_calls=13 + receipt["new_verification_calls"],
                new_physical_target_calls=3 + receipt["new_verification_calls"],
                certified=verified["certified"], independent_terminal_history_unchanged=True)
        return receipt
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        if engine is not None:
            receipt["reused_initial_calls"] = engine.replayed_initial_calls
            save(output / "returned_search_history.json", history(engine))
        receipt["new_physical_target_calls"] = (
            receipt["new_target_search_calls"] + receipt["new_verification_calls"])
        receipt["logical_total_calls"] = (
            receipt["reused_initial_calls"] + receipt["new_physical_target_calls"])
        usage = resource.getrusage(resource.RUSAGE_SELF)
        receipt.update(cpu_seconds=time.process_time()-started, wall_seconds=time.perf_counter()-wall,
            peak_rss_bytes=int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
            finished_utc=datetime.now(timezone.utc).isoformat())
        save(output / "receipt.json", receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory; an existing attempt is never retried or overwritten")
    args = parser.parse_args(argv)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(PROJECT))

    def terminate(signum, _frame):
        raise KeyboardInterrupt(f"External termination signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    receipt = run(args.case, args.output)
    print(json.dumps(receipt, default=_plain, allow_nan=False))


if __name__ == "__main__":
    main()
