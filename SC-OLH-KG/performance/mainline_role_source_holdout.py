"""Training-only native role law and source-policy projection for held-out rows.

The deployed role operator is retained, including its independent new-task
projection. Familiar source families are evaluation domains here; this does
not reinterpret the empirical native law as a same-task Bayesian posterior.
"""

from types import SimpleNamespace
from unittest.mock import patch

from algorithms.single_olhkg import SingleOLHKGAlgorithm
from core.gpr import ParametricGPR
from performance import benchmark_lodo_meta_prior as benchmark
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_legacy_native import (
    FrozenArchiveMetaPrior, materialize_v69_task,
)
from representation.meta_prior import ObservableConstraintMeanBasis


def fit_native(training_archive, heldout_family="FactorShockStatePolicyRZDT1"):
    """Refit the frozen native recipe on supplied whole-policy training rows.

    The caller records this fit's cost. No saved full-archive law, held-out
    response, simulator, target pilot or source-outcome function is used.
    """
    task = materialize_v69_task(heldout_family, seed=80, N=13)
    args = dict(task["args"])
    counts = [len(t.X) for t in training_archive.tasks]
    if len(counts) != 2 or counts[0] != counts[1]:
        raise ValueError("Native holdout fitting requires two equally sized source families")
    args["source_records_per_domain"] = counts[0]
    build_problem = benchmark.build_scalarized_problem

    def frozen_factory(*positional, **keywords):
        return FrozenArchiveMetaPrior(
            *positional, archive=training_archive, **keywords)

    def source_provider(*positional, **keywords):
        return no_outcomes(build_problem(*positional, **keywords))

    with patch.object(benchmark, "LearnedMetaPrior", frozen_factory), patch.object(
            benchmark, "build_scalarized_problem", source_provider):
        prior = benchmark.train_meta_prior(args, task["heldout"], task["seed"], teacher=False)
    if prior.n_records != sum(counts):
        raise ValueError("Native source fit did not consume the supplied training rows")
    return prior


def build_source_engine(prior, family, d=50, *,
                        heldout_family="FactorShockStatePolicyRZDT1"):
    """Provide only the native fields consumed by unchanged role_inputs.

    No optimizer is started. Its two source-coefficient methods and the actual
    GPR basis implementation are reused, with configuration materialized from
    the frozen V69 task rather than a target engine or target observations.
    """
    args = materialize_v69_task(heldout_family, seed=80, N=13)["args"]
    problem = no_outcomes(benchmark.build_scalarized_problem(
        family, d, args["L"], args["sigma"], args["alpha"],
        benchmark.parse_weights(args["weights"])))
    basis = ObservableConstraintMeanBasis(prior, problem)
    model = ParametricGPR(d=d, basis_map=basis)
    engine = object.__new__(SingleOLHKGAlgorithm)
    engine.problem = problem
    engine.config = SimpleNamespace(**args)
    engine.gpr = [None, model]
    engine.source_prior = prior
    return engine
