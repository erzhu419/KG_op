import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.single_olhkg import (  # noqa: E402
    SingleOLHKGAlgorithm,
    SingleOLHKGConfig,
)
from core.designs import (  # noqa: E402
    integer_design_fingerprint,
    load_frozen_source_informed_design,
)
from problems.rzdt import FactorShockStatePolicyRZDT1  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from performance import benchmark_lodo_meta_prior as lodo  # noqa: E402


def _payload(points, fingerprint, archive_fingerprint="archive-123"):
    return {
        "schema_version": 1,
        "design_kind": "frozen_source_informed_rank_spanning",
        "heldout_target_domain": "FactorShockStatePolicyRZDT1",
        "dimension": 4,
        "n0": 3,
        "source_archive_fingerprint": archive_fingerprint,
        "source_archive_oracle_aided": False,
        "target_labels_used": False,
        "target_oracle_used": False,
        "designs": {
            "7": {
                "points": [list(point) for point in points],
                "fingerprint": fingerprint,
            }
        },
    }


def test_sc_consumes_exact_frozen_source_informed_points(tmp_path):
    points = ((5, 10, 15, 20), (25, 30, 35, 40), (45, 50, 55, 60))
    fingerprint = integer_design_fingerprint(points)
    path = tmp_path / "source_initial_designs.json"
    path.write_text(json.dumps(_payload(points, fingerprint)), encoding="utf-8")
    loaded, contract = load_frozen_source_informed_design(
        path,
        heldout="FactorShockStatePolicyRZDT1",
        seed=7,
        n0=3,
        dimension=4,
    )
    algorithm = SingleOLHKGAlgorithm(
        ScalarizedProblem(FactorShockStatePolicyRZDT1(d=4, L=100)),
        SingleOLHKGConfig(
            N=3,
            n0=3,
            seed=7,
            initial_design="source_informed",
            initial_design_points=loaded,
            initial_design_fingerprint=contract["fingerprint"],
            initial_design_source_archive_fingerprint=contract[
                "source_archive_fingerprint"],
        ),
    )

    assert algorithm._initial_samples() == list(points)
    info = algorithm._task_initial_design_info
    assert info["fingerprint"] == fingerprint
    assert info["source_archive_fingerprint"] == "archive-123"
    assert info["target_oracle_used"] is False
    assert info["problem_specific_hook_used"] is False


def test_source_informed_contract_rejects_tampering(tmp_path):
    points = ((5, 10, 15, 20), (25, 30, 35, 40), (45, 50, 55, 60))
    path = tmp_path / "source_initial_designs.json"
    path.write_text(
        json.dumps(_payload(points, "wrong-fingerprint")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_frozen_source_informed_design(
            path,
            heldout="FactorShockStatePolicyRZDT1",
            seed=7,
            n0=3,
            dimension=4,
        )


def test_source_informed_contract_accepts_dimension_equivariant_atlas(tmp_path):
    points = ((5, 10, 15, 20), (25, 30, 35, 40), (45, 50, 55, 60))
    payload = _payload(points, integer_design_fingerprint(points))
    payload.update({
        "design_kind": "frozen_source_informed_risk_coordinate_atlas",
        "proposal_mode": "risk_coordinate_atlas",
        "structural_prior_profile": "full",
        "source_dimension": 12,
    })
    path = tmp_path / "atlas.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, contract = load_frozen_source_informed_design(
        path,
        heldout="FactorShockStatePolicyRZDT1",
        seed=7,
        n0=3,
        dimension=4,
    )

    assert loaded == points
    assert contract["proposal_mode"] == "risk_coordinate_atlas"
    assert contract["structural_prior_profile"] == "full"
    assert contract["source_dimension"] == 12


def test_source_informed_contract_accepts_risk_objective_atlas(tmp_path):
    points = ((5, 10, 15, 20), (25, 30, 35, 40), (45, 50, 55, 60))
    payload = _payload(points, integer_design_fingerprint(points))
    payload.update({
        "design_kind": "frozen_source_informed_risk_objective_atlas",
        "proposal_mode": "risk_objective_atlas",
        "structural_prior_profile": "orthogonality_only",
        "source_dimension": 12,
    })
    path = tmp_path / "risk_objective_atlas.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, contract = load_frozen_source_informed_design(
        path,
        heldout="FactorShockStatePolicyRZDT1",
        seed=7,
        n0=3,
        dimension=4,
    )

    assert loaded == points
    assert contract["proposal_mode"] == "risk_objective_atlas"
    assert contract["structural_prior_profile"] == "orthogonality_only"
    assert contract["proposal_component_mode"] == "combined"
    assert contract["uses_source_archive"] is True


def test_frozen_universal_design_has_no_source_archive_contract(tmp_path):
    points = ((5, 5, 5, 5), (25, 25, 25, 25), (45, 45, 45, 45))
    payload = _payload(
        points,
        integer_design_fingerprint(points),
        archive_fingerprint=None,
    )
    payload.update({
        "design_kind": "frozen_source_informed_risk_objective_atlas",
        "proposal_mode": "risk_objective_atlas",
        "proposal_component_mode": "universal_only",
        "source_dimension": 4,
    })
    path = tmp_path / "universal_atlas.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, contract = load_frozen_source_informed_design(
        path,
        heldout="FactorShockStatePolicyRZDT1",
        seed=7,
        n0=3,
        dimension=4,
    )

    assert loaded == points
    assert contract["proposal_component_mode"] == "universal_only"
    assert contract["uses_source_archive"] is False
    assert contract["source_archive_fingerprint"] is None


def test_lodo_rejects_source_archive_mismatch_before_target_calls(monkeypatch):
    monkeypatch.setattr(
        lodo,
        "build_target_problem",
        lambda *_args, **_kwargs: (
            object(),
            {"training": {"source_archive_fingerprint": "actual-archive"}},
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        lodo.run_one({
            "args": {
                "initial_design": "source_informed",
                "initial_design_source_archive_fingerprint": "wrong-archive",
                "use_state_basis": True,
            },
            "heldout": "FactorShockStatePolicyRZDT1",
            "line": "lodo",
            "seed": 7,
        })


def test_paired_frozen_control_audits_archive_intervention():
    contract = lodo._source_informed_archive_contract(
        {
            "initial_design": "source_informed",
            "initial_design_source_archive_fingerprint": "proposal-archive",
            "initial_design_archive_match_mode": "paired_frozen_control",
            "meta_source_budget_mode": "per_base_domain",
        },
        {"training": {
            "source_archive_fingerprint": "posterior-archive",
            "source_episode_target_data_used": False,
            "source_episode_target_oracle_used": False,
        }},
    )
    assert contract["mode"] == "paired_frozen_control"
    assert contract["matches"] is False
    assert contract["proposal_frozen_across_arms"] is True
    assert contract["target_data_used"] is False
    assert contract["target_oracle_used"] is False


def test_paired_frozen_control_requires_cost_matched_source_budget():
    with pytest.raises(ValueError, match="cost-matched"):
        lodo._source_informed_archive_contract(
            {
                "initial_design": "source_informed",
                "initial_design_source_archive_fingerprint": "proposal-archive",
                "initial_design_archive_match_mode": "paired_frozen_control",
                "meta_source_budget_mode": "per_episode",
            },
            {"training": {
                "source_archive_fingerprint": "posterior-archive",
            }},
        )
