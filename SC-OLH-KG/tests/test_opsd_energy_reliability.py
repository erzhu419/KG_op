import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import canonical_risk_descriptor  # noqa: E402
from data.opsd import load_opsd_market, preprocess_opsd  # noqa: E402
from problems.energy_reliability import (  # noqa: E402
    OPSDStorageReliabilityProblem,
)
from performance.audit_opsd_energy_certifiability import (  # noqa: E402
    audit_problem,
)
from performance.analyze_external_energy_gate import analyze  # noqa: E402
from performance.analyze_external_energy_confirmation import (  # noqa: E402
    analyze as analyze_confirmation,
)
from performance.benchmark_external_energy_confirmation import (  # noqa: E402
    CONFIRMATORY_CONTRACT_ID,
)
from performance.benchmark_external_energy_gate import (  # noqa: E402
    _binomial_lower,
    low_frequency_constant_design,
    run_gate,
)
from core.designs import integer_design_fingerprint  # noqa: E402
from performance.paper_method_contract import FRONTEND_CONTRACT_ID  # noqa: E402
from performance.task_descriptor_retrieval import (  # noqa: E402
    DESCRIPTOR_NEAREST,
    ENERGY_TARGET,
    source_selection_contract,
)
from performance.screen_opsd_energy_formulation import (  # noqa: E402
    select_configuration,
)
from representation.meta_prior import LearnedMetaPrior  # noqa: E402
from representation.meta_prior import SourceRecord  # noqa: E402
from representation.observable_exposure import (  # noqa: E402
    canonical_observable_state_descriptor,
)


def _write_market_archive(path, *, actual_shift=0.0):
    start = np.datetime64("2017-01-01T00", "h").astype(np.int64)
    hours = start + np.arange(3 * 365 * 24, dtype=np.int64)
    phase = 2.0 * np.pi * np.arange(len(hours)) / 24.0
    seasonal = 2.0 * np.pi * np.arange(len(hours)) / (365.0 * 24.0)
    forecast = 1000.0 + 120.0 * np.sin(phase - 0.5) + 80.0 * np.sin(seasonal)
    shock = 60.0 * np.maximum(np.sin(phase), 0.0) + 12.0 * np.sin(3.0 * phase)
    actual = forecast + shock + float(actual_shift)
    solar = np.maximum(80.0 * np.sin(phase - 1.0), 0.0)
    wind = 120.0 + 30.0 * np.sin(phase / 3.0 + seasonal)
    price = 45.0 + 10.0 * np.sin(phase - 0.8)
    metadata = {
        "schema_version": 1,
        "dataset": "test fixture",
        "version": "fixture",
        "markets": ["DK_2"],
        "years": [2017, 2018, 2019],
    }
    prefix = "DK_2__"
    with Path(path).open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(metadata), dtype="U"),
            **{
                prefix + "timestamp_hour": hours,
                prefix + "load_actual": actual,
                prefix + "load_forecast": forecast,
                prefix + "price": price,
                prefix + "solar": solar,
                prefix + "wind": wind,
            },
        )


def _problem(path, **kwargs):
    return OPSDStorageReliabilityProblem(
        path,
        market="DK_2",
        year=2018,
        d=24,
        minimum_windows=16,
        **kwargs,
    )


def test_local_preprocessor_records_provenance_and_loads_without_pickle(tmp_path):
    source = tmp_path / "tiny.csv"
    rows = [
        "utc_timestamp,DK_2_load_actual_entsoe_transparency,"
        "DK_2_load_forecast_entsoe_transparency,DK_2_price_day_ahead,"
        "DK_2_solar_generation_actual,DK_2_wind_generation_actual",
    ]
    for hour in range(8):
        rows.append(
            f"2018-01-01 {hour:02d}:00:00+00:00,100,99,40,5,10")
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    output = tmp_path / "compact.npz"
    metadata = preprocess_opsd(
        output,
        source=source,
        markets=("DK_2",),
        years=(2018,),
    )
    assert metadata["source"]["sha256"]
    assert metadata["output"]["sha256"]
    assert metadata["outcome_dependent_filtering"] is False
    loaded = load_opsd_market(output, "DK_2")
    assert len(loaded.timestamp_hour) == 8
    assert np.all(loaded.load_actual == 100.0)


def test_chronological_splits_are_disjoint_and_hourly(tmp_path):
    archive = tmp_path / "energy.npz"
    _write_market_archive(archive)
    problem = _problem(archive)
    starts = problem._starts
    assert all(len(values) > 100 for values in starts.values())
    search_last = starts["search"][-1] + problem.d - 1
    audit_first = starts["audit"][0]
    audit_last = starts["audit"][-1] + problem.d - 1
    verification_first = starts["verification"][0]
    assert search_last < audit_first
    assert audit_last < verification_first
    assert problem.information_contract()[
        "actual_target_error_used_by_observable_coordinate"] is False


def test_observable_and_risk_coordinates_do_not_read_actual_outcomes(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_market_archive(first, actual_shift=0.0)
    _write_market_archive(second, actual_shift=500.0)
    p1 = _problem(first)
    p2 = _problem(second)
    x = tuple(np.linspace(10, 90, p1.d).round().astype(int))
    risk1 = canonical_risk_descriptor(p1.risk_exposures(x))
    risk2 = canonical_risk_descriptor(p2.risk_exposures(x))
    state1 = canonical_observable_state_descriptor(
        p1.observable_state_exposure(x))
    state2 = canonical_observable_state_descriptor(
        p2.observable_state_exposure(x))
    np.testing.assert_allclose(risk1, risk2)
    np.testing.assert_allclose(state1, state2)
    y1 = p1.simulate(x, np.random.default_rng(7))
    y2 = p2.simulate(x, np.random.default_rng(7))
    assert not np.allclose(y1, y2)


def test_outcome_disabled_problem_cannot_evaluate_target_actual(tmp_path):
    archive = tmp_path / "energy.npz"
    _write_market_archive(archive)
    problem = OPSDStorageReliabilityProblem(
        archive,
        market="DK_2",
        year=2018,
        d=24,
        minimum_windows=16,
        outcome_access=False,
    )
    assert problem.series.load_actual is None
    assert problem.information_contract()["outcome_access_enabled"] is False
    x = tuple([50] * problem.d)
    assert np.all(np.isfinite(problem.cumulative_risk_features(x)))
    try:
        problem.simulate(x, np.random.default_rng(7))
    except RuntimeError as exc:
        assert "outcome access is disabled" in str(exc)
    else:  # pragma: no cover - explicit leakage guard
        raise AssertionError("outcome-disabled problem executed a simulator")


def test_high_reserve_reduces_empirical_shortfall_and_features_are_finite(tmp_path):
    archive = tmp_path / "energy.npz"
    _write_market_archive(archive)
    problem = _problem(archive)
    low = tuple([0] * problem.d)
    high = tuple([100] * problem.d)
    low_values = problem.split_population(low, "audit", maximum_windows=128)
    high_values = problem.split_population(high, "audit", maximum_windows=128)
    assert float(np.mean(high_values[:, 1])) < float(np.mean(low_values[:, 1]))
    features = problem.cumulative_risk_features(high)
    assert features.shape == (14,)
    assert np.all(np.isfinite(features))
    assert problem.cumulative_risk_parameters() is None
    assert problem.cumulative_risk_provider_status()["target_outcomes_used"] is False
    assert problem.initial_samples(5) == []
    assert problem.structured_candidates(5) == []


def test_split_sampling_is_reproducible_and_split_specific(tmp_path):
    archive = tmp_path / "energy.npz"
    _write_market_archive(archive)
    problem = _problem(archive)
    x = tuple([60] * problem.d)
    first = problem.simulate_from_split(x, "search", np.random.default_rng(91))
    second = problem.simulate_from_split(x, "search", np.random.default_rng(91))
    np.testing.assert_allclose(first, second)
    verification = problem.simulate_from_split(
        x, "verification", np.random.default_rng(91))
    assert first.shape == (2,)
    assert verification.shape == (2,)


def test_registered_certifiability_gate_is_nonvacuous_on_fixture(tmp_path):
    archive = tmp_path / "energy.npz"
    _write_market_archive(archive)
    result = audit_problem(_problem(archive), maximum_windows=128)
    assert result["status"] == "pass"
    assert result["feasible_policy_count"] > 0
    assert result["clearly_infeasible_policy_count"] > 0


def test_energy_descriptor_retrieval_uses_registered_roles_only():
    selection = source_selection_contract(
        DESCRIPTOR_NEAREST, target_domain=ENERGY_TARGET)
    assert selection.target_domain == ENERGY_TARGET
    assert len(selection.source_domains) == 2
    assert selection.source_domains[0] == "InventorySupplyChain"
    assert selection.heldout_task_family_identifier_used is True
    assert selection.as_dict()["target_outcomes_used"] is False


def test_formulation_screen_uses_registered_lexicographic_selection():
    rows = [
        {
            "status": "pass",
            "kappa": 0.13,
            "energy_capacity": 0.20,
            "power_capacity": 0.20,
        },
        {
            "status": "pass",
            "kappa": 0.11,
            "energy_capacity": 0.40,
            "power_capacity": 0.50,
        },
        {
            "status": "pass",
            "kappa": 0.11,
            "energy_capacity": 0.40,
            "power_capacity": 0.40,
        },
        {
            "status": "fail",
            "kappa": 0.07,
            "energy_capacity": 0.20,
            "power_capacity": 0.20,
        },
    ]
    selected = select_configuration(rows)
    assert selected["kappa"] == 0.11
    assert selected["energy_capacity"] == 0.40
    assert selected["power_capacity"] == 0.40


def test_shared_low_frequency_source_design_replays_same_profiles(tmp_path):
    archive = tmp_path / "energy.npz"
    _write_market_archive(archive)
    first = _problem(archive)
    second = _problem(archive)
    prior = LearnedMetaPrior(
        source_observation_mode="replicated",
        source_observation_replicates=2,
        source_design_mode="shared_low_frequency",
        source_universal_fraction=1.0,
        source_consensus_template_count=2,
        seed=17,
    )
    rows1 = list(prior._source_design_candidates(
        first, 6, np.random.default_rng(1)))
    rows2 = list(prior._source_design_candidates(
        second, 6, np.random.default_rng(999)))
    assert rows1 == rows2
    assert {origin for _, origin in rows1} == {
        "universal_shared_low_frequency"}


def test_source_only_problem_can_require_only_search_split(tmp_path):
    archive = tmp_path / "energy.npz"
    _write_market_archive(archive)
    problem = OPSDStorageReliabilityProblem(
        archive,
        market="DK_2",
        year=2018,
        d=24,
        minimum_windows=16,
        required_splits=("search",),
    )
    assert problem.information_contract()["required_splits"] == ["search"]


def test_source_monotone_envelope_uses_only_agreed_source_direction(tmp_path):
    archive = tmp_path / "energy.npz"
    _write_market_archive(archive)
    problem = _problem(archive)
    prior = LearnedMetaPrior()
    records = []
    for domain in ("source_a", "source_b"):
        for level, margin in ((0.2, 0.4), (0.5, 0.0), (0.8, -0.4)):
            profile = np.full(problem.d, level, dtype=float)
            records.append(SourceRecord(
                domain=domain,
                x=problem.continuous_to_int(profile),
                y=np.asarray([level, margin]),
                descriptor=np.asarray([level]),
                profile=profile,
                tau=0.0,
                alpha=0.05,
                sigma_level=1e-8,
                constraint_sigma=1e-8,
            ))
    prior.source_records_ = records
    candidate = prior.source_monotone_envelope_candidate(problem)
    assert candidate == tuple([problem.L] * problem.d)
    diagnostics = prior.source_monotone_envelope_diagnostics
    assert diagnostics["status"] == "admitted"
    assert diagnostics["target_data_used"] is False


def test_exact_binomial_certificate_reaches_point95_only_with_enough_successes():
    delta = 0.05 / 3.0
    assert _binomial_lower(80, 80, delta) >= 0.95
    assert _binomial_lower(79, 80, delta) < 0.95


def test_low_frequency_constant_design_is_bound_only_and_dimension_equivariant():
    class Problem:
        d = 7
        L = 100

    points = low_frequency_constant_design(Problem(), 10)
    assert len(points) == 10
    assert len(set(points)) == 10
    assert all(len(point) == Problem.d for point in points)
    assert all(len(set(point)) == 1 for point in points)
    assert points[0] == (0,) * Problem.d
    assert points[-1] == (100,) * Problem.d


def test_stochastic_gate_keeps_search_and_verification_separate(tmp_path):
    archive = tmp_path / "energy.npz"
    _write_market_archive(archive)
    d = 24
    points = [tuple([level] * d) for level in range(10, 101, 10)]
    design = tmp_path / "design.json"
    design.write_text(json.dumps({
        "schema_version": 1,
        "design_kind": "frozen_source_informed_risk_objective_atlas",
        "proposal_mode": "risk_objective_atlas",
        "structural_prior_profile": "low_frequency_only",
        "source_archive_oracle_aided": False,
        "target_labels_used": False,
        "target_oracle_used": False,
        "heldout_target_domain": "OPSDStorageReliability:DK_2:2018",
        "dimension": d,
        "source_dimension": 12,
        "n0": 10,
        "source_archive_fingerprint": "fixture-source",
        "source_archive_simulator_calls": 384,
        "paper_frontend_contract_id": FRONTEND_CONTRACT_ID,
        "designs": {
            "80": {
                "points": [list(point) for point in points],
                "fingerprint": integer_design_fingerprint(points),
            }
        },
    }), encoding="utf-8")
    result = run_gate(
        data_path=archive,
        seed=80,
        arm="frozen_proposal",
        market="DK_2",
        year=2018,
        dimension=d,
        n0=10,
        N=13,
        design_path=design,
        verification_budgets=(80, 128, 128),
    )
    assert result["status"] == "ok"
    assert result["target_search_calls"] == 13
    assert result["source_calls"] == 384
    assert result["verification"]["search_samples_reused"] is False
    assert result["verification"]["posterior_updated_from_verification"] is False
    assert result["target_oracle_used_during_search"] is False


def test_energy_gate_analysis_requires_paired_five_seed_evidence(tmp_path):
    paths = []
    for arm in ("frozen_proposal", "common_sobol"):
        for seed in range(80, 85):
            path = tmp_path / f"{arm}_{seed}.json"
            path.write_text(json.dumps({
                "status": "ok",
                "arm": arm,
                "seed": seed,
                "independently_certified": arm == "frozen_proposal",
                "false_certificate": False,
                "verification_calls": 80,
                "deployment_truth_audit": (
                    {
                        "truly_chance_feasible": True,
                        "true_objective_mean": 0.02,
                    }
                    if arm == "frozen_proposal" else None
                ),
            }), encoding="utf-8")
            paths.append(path)
    result = analyze(paths)
    assert result["status"] == "pass"
    assert result["next_action"] == "freeze_and_open_gb_gbn_confirmatory_target"


def test_energy_confirmation_requires_safe_paired_advantage(tmp_path):
    paths = []
    for arm in ("frozen_proposal", "common_sobol"):
        for seed in range(100, 120):
            path = tmp_path / f"{arm}_{seed}.json"
            objective = 0.02 if arm == "frozen_proposal" else 0.08
            path.write_text(json.dumps({
                "status": "ok",
                "contract_id": CONFIRMATORY_CONTRACT_ID,
                "market": "GB_GBN",
                "year": 2018,
                "arm": arm,
                "seed": seed,
                "independently_certified": True,
                "false_certificate": False,
                "verification_calls": 80,
                "deployment_truth_audit": {
                    "truly_chance_feasible": True,
                    "true_objective_mean": objective,
                },
            }), encoding="utf-8")
            paths.append(path)
    result = analyze_confirmation(paths)
    assert result["status"] == "pass"
    assert result["paired_primary_endpoint"]["frozen_wins"] == 20
    assert result["checks"]["frozen_has_zero_false_certificates"] is True


def test_energy_confirmation_fails_incomplete_pairing(tmp_path):
    paths = []
    for arm in ("frozen_proposal", "common_sobol"):
        for seed in range(100, 119):
            path = tmp_path / f"{arm}_{seed}.json"
            path.write_text(json.dumps({
                "status": "ok",
                "contract_id": CONFIRMATORY_CONTRACT_ID,
                "market": "GB_GBN",
                "year": 2018,
                "arm": arm,
                "seed": seed,
                "independently_certified": True,
                "false_certificate": False,
                "verification_calls": 80,
                "deployment_truth_audit": {
                    "truly_chance_feasible": True,
                    "true_objective_mean": (
                        0.02 if arm == "frozen_proposal" else 0.08),
                },
            }), encoding="utf-8")
            paths.append(path)
    result = analyze_confirmation(paths)
    assert result["status"] == "fail"
    assert result["checks"]["complete_twenty_seed_pairing"] is False
