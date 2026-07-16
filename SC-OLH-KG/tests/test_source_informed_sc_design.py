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
