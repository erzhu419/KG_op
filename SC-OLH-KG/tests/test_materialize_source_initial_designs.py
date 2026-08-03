from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance import materialize_source_initial_designs as MATERIALIZE  # noqa: E402
from performance.paper_method_contract import (  # noqa: E402
    FRONTEND_LOWER_ENVELOPE_CHALLENGER_ID,
    FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID,
)


class _Problem:
    d = 4


class _Archive:
    fingerprint = "archive-fingerprint"
    source_seed = 17
    tasks = [SimpleNamespace(
        X=np.zeros((1, 4), dtype=float),
        Y_replicates=[[1.0, 1.1, 0.9]],
    )]

    def validate(self, **_kwargs):
        return None


class _Prior:
    def __init__(self):
        self.lower_envelope_flags = []
        self.monotone_envelope_flags = []
        self.risk_objective_proposal_diagnostics = {}

    def risk_objective_initial_candidates(
        self,
        problem,
        n,
        rng,
        *,
        protect_lower_envelope_sentinel=False,
        protect_source_monotone_envelope=False,
    ):
        del rng
        self.lower_envelope_flags.append(
            bool(protect_lower_envelope_sentinel))
        self.monotone_envelope_flags.append(
            bool(protect_source_monotone_envelope))
        self.risk_objective_proposal_diagnostics = {
            "universal_lower_envelope_sentinel": bool(
                protect_lower_envelope_sentinel),
            "source_monotone_envelope": {
                "status": (
                    "admitted"
                    if protect_source_monotone_envelope
                    else "disabled"
                ),
            },
        }
        return [tuple([index] * problem.d) for index in range(int(n))]


def test_materializer_freezes_distinct_lower_envelope_contract(
    tmp_path,
    monkeypatch,
):
    archive = _Archive()
    prior = _Prior()
    monkeypatch.setattr(
        MATERIALIZE,
        "oracle_free_lodo_config",
        lambda _manifest: {
            "d": 4,
            "L": 100,
            "sigma": 0.1,
            "alpha": 0.05,
            "weights": "0.5,0.5",
        },
    )
    monkeypatch.setattr(
        MATERIALIZE,
        "build_scalarized_problem",
        lambda *_args, **_kwargs: _Problem(),
    )
    monkeypatch.setattr(
        MATERIALIZE.FrozenTransferArchive,
        "load",
        lambda _path: archive,
    )
    monkeypatch.setattr(
        MATERIALIZE,
        "train_meta_prior",
        lambda *_args, **_kwargs: prior,
    )
    monkeypatch.setattr(
        MATERIALIZE,
        "frozen_archive_from_meta_prior",
        lambda *_args, **_kwargs: SimpleNamespace(
            fingerprint=archive.fingerprint),
    )
    monkeypatch.setattr(
        MATERIALIZE,
        "apply_structural_prior_profile",
        lambda config, profile: config.update({
            "structural_prior_active_components": [profile],
        }),
    )

    payload = MATERIALIZE.materialize_source_designs(
        "manifest.json",
        "FactorShockStatePolicyRZDT1",
        "archive.json",
        tmp_path / "source_initial_designs.json",
        dimension=4,
        source_dimension=4,
        n0=3,
        seed_start=80,
        n_seeds=2,
        structural_prior_profile="low_frequency_only",
        proposal_mode="risk_objective_atlas",
        proposal_component_mode="combined",
        source_design_mode="universal_mixture",
        protect_lower_envelope_sentinel=True,
    )

    assert prior.lower_envelope_flags == [True, True]
    assert payload["universal_lower_envelope_sentinel"] is True
    assert payload["paper_frontend_contract_id"] == (
        FRONTEND_LOWER_ENVELOPE_CHALLENGER_ID)
    assert payload["paper_frontend_contract_audit"]["validated"] is True


def test_materializer_freezes_distinct_source_monotone_contract(
    tmp_path,
    monkeypatch,
):
    archive = _Archive()
    prior = _Prior()
    monkeypatch.setattr(
        MATERIALIZE,
        "oracle_free_lodo_config",
        lambda _manifest: {
            "d": 4,
            "L": 100,
            "sigma": 0.1,
            "alpha": 0.05,
            "weights": "0.5,0.5",
        },
    )
    monkeypatch.setattr(
        MATERIALIZE,
        "build_scalarized_problem",
        lambda *_args, **_kwargs: _Problem(),
    )
    monkeypatch.setattr(
        MATERIALIZE.FrozenTransferArchive,
        "load",
        lambda _path: archive,
    )
    monkeypatch.setattr(
        MATERIALIZE,
        "train_meta_prior",
        lambda *_args, **_kwargs: prior,
    )
    monkeypatch.setattr(
        MATERIALIZE,
        "frozen_archive_from_meta_prior",
        lambda *_args, **_kwargs: SimpleNamespace(
            fingerprint=archive.fingerprint),
    )
    monkeypatch.setattr(
        MATERIALIZE,
        "apply_structural_prior_profile",
        lambda config, profile: config.update({
            "structural_prior_active_components": [profile],
        }),
    )

    payload = MATERIALIZE.materialize_source_designs(
        "manifest.json",
        "FactorShockStatePolicyRZDT1",
        "archive.json",
        tmp_path / "source_initial_designs.json",
        dimension=4,
        source_dimension=4,
        n0=3,
        seed_start=80,
        n_seeds=2,
        structural_prior_profile="low_frequency_only",
        proposal_mode="risk_objective_atlas",
        proposal_component_mode="combined",
        source_design_mode="universal_mixture",
        protect_source_monotone_envelope=True,
    )

    assert prior.lower_envelope_flags == [False, False]
    assert prior.monotone_envelope_flags == [True, True]
    assert payload["source_monotone_envelope"] is True
    assert payload["paper_frontend_contract_id"] == (
        FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID)
    assert payload["paper_frontend_contract_audit"]["validated"] is True
