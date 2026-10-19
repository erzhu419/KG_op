"""Three-family native-backend bridge: freeze all search, verify all, then score.

Each case is an independent process/file unit. Completed stages are reused;
source preparation and every attempted stage retain their actual CPU/call cost.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import pickle
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from performance.benchmark_quality import json_safe
from performance.mainline_legacy_native import materialize_v69_task, restore_source_prior
from representation.meta_prior import AdmissibleProblemAdapter, TaskExpertProblemView

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "performance/manifests/mainline_legacy_backend_bridge_v1.json"
OUTPUT = ROOT / "paper_artifacts/mainline_legacy_backend_bridge_v1"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2) + "\n")
    temporary.replace(path)


def _empty_problem_view(cls):
    # Native delegation otherwise recurses when pickle probes __setstate__
    # before base/problem has been restored. Keep normal native methods intact.
    value = cls.__new__(cls)
    if issubclass(cls, AdmissibleProblemAdapter):
        value.base = None
    else:
        value.problem = None
    return value


class _NativePickler(pickle.Pickler):
    def reducer_override(self, value):
        if isinstance(value, (AdmissibleProblemAdapter, TaskExpertProblemView)):
            # Use the importable module name even when this CLI runs with -m.
            from performance.run_mainline_legacy_backend_bridge import _empty_problem_view as restore
            return (restore, (type(value),), value.__dict__)
        return NotImplemented


def save_pickle(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        _NativePickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    temporary.replace(path)


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def save_engine(path, engine):
    bundle = getattr(engine, "backend_bundle", None)
    cache = getattr(engine, "_moment_cache", None)
    if bundle is not None:
        engine.backend_bundle = None
    if cache is not None:
        engine._moment_cache = {}
    try:
        save_pickle(path, engine)
    finally:
        if bundle is not None:
            engine.backend_bundle = bundle
        if cache is not None:
            engine._moment_cache = cache


def load_engine(output, family, arm):
    engine = load_pickle(output / family / arm / "search_engine.pkl")
    if arm in {"same_source_gp", "current_structural_mixture"}:
        engine.backend_bundle = load_pickle(output / "source_models" / family / "source_models.pkl")
    return engine


def cases(plan):
    return [(family, arm["name"]) for family in plan["domains"] for arm in plan["arms"]]


@contextmanager
def attempt(directory, stage, family=None, arm=None):
    directory.mkdir(parents=True, exist_ok=True)
    started, wall = time.process_time(), time.perf_counter()
    row = {"stage": stage, "family": family, "arm": arm,
           "started_utc": datetime.now(timezone.utc).isoformat(), "status": "failed",
           "new_source_calls": 0, "new_target_search_calls": 0,
           "new_verification_calls": 0, "source_model_restorations": 0}
    try:
        yield row
        row["status"] = "completed"
    except BaseException as error:
        row["error"] = repr(error)
        raise
    finally:
        row.update(cpu_seconds=time.process_time()-started,
                   wall_seconds=time.perf_counter()-wall,
                   finished_utc=datetime.now(timezone.utc).isoformat())
        with (directory / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(json_safe(row)) + "\n")


def require_all(output, plan, filename):
    missing = [str(output/family/arm/filename) for family, arm in cases(plan)
               if not (output/family/arm/filename).is_file()]
    if missing:
        raise RuntimeError(f"All 12 preceding stages must finish first; missing {filename}: {missing}")


def prepare_native(output, plan, family):
    directory = output / "source_models" / family
    path = directory / "native_source_prior.pkl"
    with attempt(directory, "prepare-native", family) as cost:
        if path.is_file():
            cost["reused"] = True
            return {"path": str(path), "reused": True}
        task = materialize_v69_task(family, plan["target_seed"], plan["anchor"]["search_calls"],
            factor_shock_scale=plan["problem_kwargs_by_domain"].get(family, {}).get("shared_shock_scale", 1.0))
        save(directory / "native_resolved_task.json", task)
        cost["source_model_restorations"] = 1
        prior = restore_source_prior(task)
        save_pickle(path, prior)
        save(directory / "native_source_diagnostics.json", prior.diagnostics())
        return {"path": str(path), "reused": False}


@contextmanager
def count_simulations(engine, cost, field, observed):
    original = engine.problem.simulate

    def simulate(point, rng=None):
        value = original(point, rng)
        cost[field] += 1
        observed.append({"point": list(map(int, point)), "raw": np.asarray(value).tolist()})
        return value

    with patch.object(engine.problem, "simulate", simulate):
        yield


def search_case(output, plan, family, arm):
    from performance.mainline_legacy_backend import build_backend

    directory = output / family / arm
    path = directory / "search.json"
    observations = []
    with attempt(directory, "search", family, arm) as cost:
        if path.is_file():
            cost["reused"] = True
            return read(path)
        source_dir = output / "source_models" / family
        prior = load_pickle(source_dir / "native_source_prior.pkl")
        bundle = (load_pickle(source_dir / "source_models.pkl")
                  if arm in {"same_source_gp", "current_structural_mixture"} else None)
        N = plan["anchor"]["n0"] if arm == "initialization_only" else plan["anchor"]["search_calls"]
        engine = build_backend(family, arm, bundle, source_prior=prior, N=N, seed=plan["target_seed"])
        try:
            with count_simulations(engine, cost, "new_target_search_calls", observations):
                engine.run_search()
        except BaseException:
            save(directory / f"failed_search_raw_{time.time_ns()}.json", observations)
            raise
        if cost["new_target_search_calls"] != N-plan["anchor"]["n0"]:
            raise RuntimeError("search physical calls differ from the frozen budget")
        expected = next(row for row in read(ROOT / plan["inputs_file"])["domains"] if row["domain"] == family)
        np.testing.assert_array_equal([value for _, value in engine.history[:10]], expected["initial_observations"])
        result = {"family": family, "arm": arm, "seed": plan["target_seed"],
            "logical_search_calls": len(engine.history), "reused_initial_calls": engine.replayed_initial_calls,
            "new_search_calls": cost["new_target_search_calls"],
            "source_archive_calls_reused": 384, **engine.frozen_terminal,
            "history": [{"point": list(point), "raw": value} for point, value in engine.history],
            "iterations": engine.iteration_log,
            "backend_trace": getattr(engine, "backend_trace", []), "truth_used": False,
            "frozen_utc": datetime.now(timezone.utc).isoformat()}
        save_engine(directory / "search_engine.pkl", engine)
        save(path, result)
        return result


def verify_case(output, plan, family, arm):
    directory = output / family / arm
    path = directory / "verification.json"
    observations = []
    with attempt(directory, "verify", family, arm) as cost:
        if path.is_file():
            cost["reused"] = True
            return read(path)
        require_all(output, plan, "search.json")
        search = read(directory / "search.json")
        engine = load_engine(output, family, arm)
        engine.defer_terminal = False
        try:
            with count_simulations(engine, cost, "new_verification_calls", observations):
                deployed, verification = engine._terminal_verification_protocol(tuple(search["primary"]))
        except BaseException:
            save(directory / f"failed_verification_raw_{time.time_ns()}.json", observations)
            raise
        old_shortlist = [row["point"] for row in search["terminal"]["frozen_shortlist"]]
        new_shortlist = [row["point"] for row in verification["frozen_shortlist"]]
        if old_shortlist != new_shortlist:
            raise RuntimeError("native terminal shortlist changed after search freeze")
        if cost["new_verification_calls"] != verification["verification_budget"]:
            raise RuntimeError("terminal physical counter disagrees with native verifier")
        result = {"family": family, "arm": arm, "primary": search["primary"],
            "deployed": list(map(int, deployed)), "verification": verification,
            "new_verification_calls": cost["new_verification_calls"], "truth_used": False,
            "completed_utc": datetime.now(timezone.utc).isoformat()}
        save(path, result)
        return result


def score_all(output, plan):
    with attempt(output, "score") as cost:
        if (output / "results.json").is_file():
            cost["reused"] = True
            return read(output / "results.json")
        require_all(output, plan, "verification.json")
        rows = []
        for family, arm in cases(plan):
            directory = output / family / arm
            search, verified = read(directory / "search.json"), read(directory / "verification.json")
            truth_path = directory / "truth.json"
            if truth_path.is_file():
                truth = read(truth_path)
            else:
                engine = load_pickle(directory / "search_engine.pkl")
                problem = engine.problem
                reference = float(np.dot(problem.weights, problem.ref_point))
                truth = {}
                for label in ("primary", "deployed"):
                    point = verified[label]
                    f = float(problem.true_objective(point))
                    g = float(problem.true_constraint_mean(point))-problem.tau
                    sigma = float(problem.true_sigma(point)[1])
                    margin = g + norm.ppf(1-problem.alpha)*sigma
                    truth[label] = {"objective": f, "constraint_mean_centered": g,
                        "constraint_sigma": sigma, "chance_margin": float(margin),
                        "true_feasible": bool(margin <= 0), "quality": float(np.clip(1-f/reference, 0, 1))}
                save(truth_path, truth)
            certified = bool(verified["verification"]["certified"])
            deployed = truth["deployed"]
            rows.append({"family": family, "arm": arm, "primary": search["primary"],
                "deployed": verified["deployed"], "primary_truth": truth["primary"],
                "deployed_truth": deployed, "certified": certified,
                "false_certificate": bool(certified and not deployed["true_feasible"]),
                "actual_safe_reward": float(deployed["quality"] if certified and deployed["true_feasible"] else 0),
                "new_search_calls": search["new_search_calls"],
                "new_verification_calls": verified["new_verification_calls"],
                "logical_search_calls": search["logical_search_calls"]})
        result = {"protocol": plan, "rows": rows,
            "completed_physical_calls": {"source": 0,
                "target_search": sum(r["new_search_calls"] for r in rows),
                "verification": sum(r["new_verification_calls"] for r in rows)},
            "failed_attempt_costs": "Retained separately in per-case/source attempts.jsonl; completed counts exclude failed attempts."}
        save(output / "results.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare-native", "search", "verify", "score"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--family")
    parser.add_argument("--arm")
    args = parser.parse_args(argv)
    plan = read(args.manifest)
    if args.stage != "score" and args.family not in plan["domains"]:
        parser.error("--family must be one of the three registered domains")
    if args.stage in {"search", "verify"} and args.arm not in [a["name"] for a in plan["arms"]]:
        parser.error("--arm must be one of the four registered arms")
    if args.stage == "prepare-native":
        result = prepare_native(args.output, plan, args.family)
    elif args.stage == "search":
        result = search_case(args.output, plan, args.family, args.arm)
    elif args.stage == "verify":
        result = verify_case(args.output, plan, args.family, args.arm)
    else:
        result = score_all(args.output, plan)
    print(json.dumps({"stage": args.stage, "family": args.family, "arm": args.arm,
                      "completed": True, "output": str(args.output)}), flush=True)
    return result


if __name__ == "__main__":
    main()
