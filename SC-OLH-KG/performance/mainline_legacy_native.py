"""Restore the complete V69 search engine from its existing frozen source data.

The scheduler and manifest materializer remain the configuration authority. Only
source acquisition and cached target initialization are replayed; native source
fitting, expert proposals, GPR/HVD feedback and the original run loop are reused.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shlex
from pathlib import Path
from unittest.mock import patch

import numpy as np

from algorithms.single_olhkg import SingleOLHKGAlgorithm
from baselines.transfer_archive import FrozenTransferArchive
from performance import benchmark_lodo_meta_prior as benchmark
from performance.run_lodo_manifest_shard import build_run_one_task
from representation.meta_prior import LearnedMetaPrior, MetaPriorProblemAdapter, SourceRecord
from representation.observable_exposure import get_observable_state_exposure

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "paper_artifacts/legacy_joint_bridge_v1/inputs"


class _Captured(Exception):
    def __init__(self, value):
        self.value = value


def materialize_v69_task(domain, seed=80, N=13, *, factor_shock_scale=1.0):
    """Resolve V69 with its fixed seed80 design and requested target stream.

    Scale 1 matches the cached legacy_joint_bridge target. Historical V69's
    scheduler defaults to scale 0, so this task difference is explicit.
    New target seeds reuse those ten source-only policies; only observation,
    proposal, acquisition and verification random streams receive ``seed``.
    """
    path = ROOT.parent / "scripts/submit_scolhkg_v53_constrained_certificate_gate_scheduler.py"
    spec = importlib.util.spec_from_file_location("legacy_v69_scheduler", path)
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    parse_args = argparse.ArgumentParser.parse_args

    def capture(parser, *args, **kwargs):
        options = parse_args(parser, [
            "--N", str(N), "--n0", "10", "--seed-start", "80",
            "--n-seeds", "1", "--variants", scheduler.V69,
            "--scenario-domains", domain, "--factor-shock-scale", str(factor_shock_scale),
            "--deploy", str(ROOT.parent), "--manifest", str(ROOT / "performance/manifests/v18b_exactkg_mcdiag.json"),
        ])
        raise _Captured(options)

    with patch.object(argparse.ArgumentParser, "parse_args", capture):
        try:
            scheduler.main()
        except _Captured as captured:
            options = captured.value
    commands = scheduler.build_specs(options)
    if len(commands) != 1:
        raise ValueError("expected one original V69 command")
    tokens = shlex.split(commands[0]["cmd"])
    start = tokens.index("performance/run_lodo_manifest_shard.py") + 1
    argv = tokens[start:tokens.index("&&", start)]
    for flag, value in {
        "--manifest": ROOT / "performance/manifests/v18b_exactkg_mcdiag.json",
        "--initial-design-file": INPUTS / domain / "design.json",
        "--out": ROOT / "paper_artifacts/mainline_legacy_native_restore_v1/unused.json",
    }.items():
        argv[argv.index(flag) + 1] = str(value)
    task = build_run_one_task(argv)
    task["seed"] = int(seed)
    task["initial_design_seed"] = 80
    # Post-decision oracle audits and progress printing are not search inputs.
    task["args"].update(truth_pool_diagnostics=False, progress_logging=False)
    return task


class FrozenArchiveMetaPrior(LearnedMetaPrior):
    def __init__(self, *args, archive, **kwargs):
        self.frozen_archive = archive
        super().__init__(*args, **kwargs)

    def _record_source_data(self, source_problems, n_records_per_domain, rng,
                            n_records_by_problem=None):
        del rng
        tasks = {task.name: task for task in self.frozen_archive.tasks}
        records = []
        counts = n_records_by_problem or [n_records_per_domain] * len(source_problems)
        for (name, problem), count in zip(source_problems, counts):
            task = tasks[name]
            if len(task.X) != count or task.X.shape[1] != problem.d:
                raise ValueError("frozen source rows do not match original config")
            for i, point in enumerate(task.X_integer):
                x = tuple(map(int, point))
                records.append(SourceRecord(
                    domain=name, x=x, y=task.Y_mean[i].copy(),
                    descriptor=self.descriptor(problem, x), profile=task.X[i].copy(),
                    tau=task.tau, alpha=task.alpha, sigma_level=problem.sigma_level,
                    constraint_sigma=float(task.constraint_sigma[i]),
                    observable_state_descriptor=self.observable_state_descriptor(problem, x),
                    observable_state_invariant_descriptor=self.observable_state_descriptor(problem, x, mode="set_invariant"),
                    observable_state_exposure=get_observable_state_exposure(problem, x),
                    provider_risk_descriptor=self.provider_risk_descriptor(problem, x),
                    provider_risk_coordinate=self.provider_risk_coordinate(problem, x),
                    origin=task.origins[i], replicate_count=len(task.Y_replicates[i]),
                    replicates=task.Y_replicates[i].copy(),
                ))
        return records


def restore_source_prior(task):
    archive = FrozenTransferArchive.load(INPUTS / task["heldout"] / "archive.json")

    def factory(*args, **kwargs):
        return FrozenArchiveMetaPrior(*args, archive=archive, **kwargs)

    with patch.object(benchmark, "LearnedMetaPrior", factory):
        prior = benchmark.train_meta_prior(task["args"], task["heldout"], task["seed"], teacher=False)
    benchmark._source_informed_archive_contract(task["args"], prior.diagnostics())
    return prior


def cached_initial_rows(domain, seed=80):
    data = json.loads((ROOT / "paper_artifacts/legacy_joint_bridge_v1/results.json").read_text())
    if int(data["protocol"]["target_seed"]) != int(seed):
        raise ValueError("legacy bridge initialization seed differs")
    row = next(row for row in data["rows"] if row["domain"] == domain and row["arm"] == "legacy_proposal_only")
    task = materialize_v69_task(domain, seed)
    return [{"point": point, "raw": raw} for point, raw in zip(
        task["args"]["initial_design_points"], row["observations"])]


class SearchFrozen(Exception):
    """The complete native search and native terminal shortlist are frozen."""


class LegacyNativeAlgorithm(SingleOLHKGAlgorithm):
    def __init__(self, problem, config, *, initial_rows=(), defer_terminal=True):
        self.initial_rows = copy.deepcopy(list(initial_rows))
        self.defer_terminal = bool(defer_terminal)
        self.replayed_initial_calls = 0
        self.actual_search_calls = 0
        self.frozen_terminal = None
        super().__init__(problem, config)

    def _simulate_and_store(self, x):
        position = len(self.history)
        if position < len(self.initial_rows):
            row = self.initial_rows[position]
            point = tuple(map(int, x))
            if point != tuple(map(int, row["point"])):
                raise ValueError("cached initialization policy/order differs")
            value = np.asarray(row["raw"], dtype=float).copy()
            self.observations.setdefault(point, []).append(value)
            self.history.append((point, value))
            self.replayed_initial_calls += 1
            return value
        value = super()._simulate_and_store(x)
        self.actual_search_calls += 1
        return value

    def _terminal_verification_protocol(self, primary):
        if not self.defer_terminal:
            return super()._terminal_verification_protocol(primary)
        budget = self.config.terminal_verification_budget
        self.config.terminal_verification_budget = 0
        try:
            _, terminal = super()._terminal_verification_protocol(primary)
        finally:
            self.config.terminal_verification_budget = budget
        terminal["status"] = "frozen_pending_independent_verification"
        terminal["verification_budget_per_candidate"] = int(budget)
        terminal["candidate_verification_budgets"][0] = int(budget)
        terminal["max_verification_budget"] = sum(terminal["candidate_verification_budgets"]) + 2 * self.config.terminal_objective_comparison_budget
        self.frozen_terminal = {"primary": list(map(int, primary)), "terminal": terminal}
        raise SearchFrozen()

    def run_search(self):
        try:
            self.run(verbose=False)
        except SearchFrozen:
            return self
        raise RuntimeError("search did not stop at the deferred terminal boundary")


def build_native(domain, seed=80, N=13, *, initial_rows=None, source_prior=None,
                 algorithm_class=LegacyNativeAlgorithm, defer_terminal=True,
                 factor_shock_scale=1.0, problem_kwargs=None):
    task = materialize_v69_task(domain, seed, N, factor_shock_scale=factor_shock_scale)
    prior = restore_source_prior(task) if source_prior is None else source_prior
    args = task["args"]
    benchmark._source_informed_archive_contract(args, prior.diagnostics())
    kwargs = {"shared_shock_scale": factor_shock_scale} if domain == "FactorShockStatePolicyRZDT1" else {}
    kwargs.update(problem_kwargs or {})
    target = benchmark.build_scalarized_problem(domain, args["d"], args["L"], args["sigma"], args["alpha"], benchmark.parse_weights(args["weights"]), problem_kwargs=kwargs)
    problem = MetaPriorProblemAdapter(target, prior, proposal_pool_size=args["meta_proposal_pool_size"], refinement_count=args["meta_refinement_count"])

    def capture_config(problem, config):
        raise _Captured(config)

    # Reuse run_one's complete config constructor without entering its run/audits.
    with patch.object(benchmark, "build_target_problem", return_value=(problem, prior.diagnostics())), patch.object(benchmark, "SingleOLHKGAlgorithm", capture_config):
        try:
            benchmark.run_one(task)
        except _Captured as captured:
            config = captured.value
    engine = algorithm_class(problem, config, initial_rows=initial_rows or (), defer_terminal=defer_terminal)
    engine.legacy_task = task
    engine.source_prior = prior
    return engine
