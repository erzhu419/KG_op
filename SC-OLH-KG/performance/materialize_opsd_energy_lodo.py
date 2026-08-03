#!/usr/bin/env python3
"""Freeze the AT/DK_1 source archive and DK_2 LODO proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.transfer_archive import (  # noqa: E402
    FrozenTransferArchive,
    frozen_archive_from_meta_prior,
)
from core.designs import integer_design_fingerprint  # noqa: E402
from performance.execution_provenance import attach_execution_provenance  # noqa: E402
from performance.paper_method_contract import (  # noqa: E402
    ALLOWED_TARGET_DESCRIPTORS,
    FORBIDDEN_TARGET_INFORMATION,
    FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID,
    validate_frozen_proposal_payload,
)
from problems.energy_reliability import OPSDStorageReliabilityProblem  # noqa: E402
from representation.meta_prior import LearnedMetaPrior  # noqa: E402


SOURCE_MARKETS = ("AT", "DK_1")
SOURCE_RECORDS_PER_MARKET = 64
SOURCE_REPLICATES = 3
SOURCE_SEED = 20260803


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def train_energy_lodo_prior(
    data_path,
    *,
    source_markets=SOURCE_MARKETS,
    year=2018,
    dimension=1000,
    source_seed=SOURCE_SEED,
):
    """Fit the frozen prior from ordinary source-market evaluations only."""

    source_problems = [
        (
            f"OPSDStorageReliability:{market}:search_{int(year) - 1}",
            OPSDStorageReliabilityProblem(
                data_path,
                market=market,
                year=year,
                d=dimension,
                required_splits=("search",),
            ),
        )
        for market in source_markets
    ]
    prior = LearnedMetaPrior(
        source_observation_mode="replicated",
        source_observation_replicates=SOURCE_REPLICATES,
        source_design_mode="shared_low_frequency",
        source_universal_fraction=1.0,
        source_consensus_template_count=32,
        component_stage="legacy_all",
        seed=int(source_seed),
    )
    prior.fit_from_source_problems(
        source_problems,
        n_records_per_domain=SOURCE_RECORDS_PER_MARKET,
        rng=np.random.default_rng(int(source_seed) + 1009),
    )
    return prior


def materialize_energy_lodo(
    data_path,
    archive_path,
    output,
    *,
    market="DK_2",
    year=2018,
    dimension=1000,
    n0=10,
    seed_start=80,
    n_seeds=5,
    overwrite_archive=False,
):
    """Create one immutable source archive and source-only target designs."""

    prior = train_energy_lodo_prior(
        data_path,
        year=year,
        dimension=dimension,
    )
    reconstructed = frozen_archive_from_meta_prior(
        prior, source_seed=SOURCE_SEED)
    archive_path = Path(archive_path)
    if archive_path.is_file() and not overwrite_archive:
        archive = FrozenTransferArchive.load(archive_path)
        if archive.fingerprint != reconstructed.fingerprint:
            raise ValueError(
                "existing OPSD source archive does not match deterministic "
                "source reconstruction")
    else:
        reconstructed.save(archive_path)
        archive = reconstructed
    expected_domains = tuple(
        f"OPSDStorageReliability:{name}:search_{int(year) - 1}"
        for name in SOURCE_MARKETS
    )
    archive.validate(
        expected_domains=expected_domains,
        expected_dimension=int(dimension),
    )
    if int(archive.simulator_calls) != 384:
        raise ValueError("registered OPSD source archive must cost 384 calls")

    problem = OPSDStorageReliabilityProblem(
        data_path,
        market=market,
        year=year,
        d=dimension,
        outcome_access=False,
    )
    designs = {}
    for seed in range(int(seed_start), int(seed_start) + int(n_seeds)):
        points = prior.risk_objective_initial_candidates(
            problem,
            n=int(n0),
            rng=np.random.default_rng(seed),
            protect_source_monotone_envelope=True,
        )
        points = [tuple(map(int, point)) for point in points]
        if len(points) != int(n0) or len(set(points)) != int(n0):
            raise RuntimeError("energy LODO proposal did not produce n0 points")
        designs[str(seed)] = {
            "points": [list(point) for point in points],
            "fingerprint": integer_design_fingerprint(points),
        }

    payload = {
        "schema_version": 1,
        "design_kind": "frozen_source_informed_risk_objective_atlas",
        "proposal_mode": "risk_objective_atlas",
        "source_monotone_envelope": True,
        "structural_prior_profile": "low_frequency_only",
        "structural_prior_active_components": ["low_frequency"],
        "heldout_target_domain": (
            f"OPSDStorageReliability:{market}:{int(year)}"),
        "target_family": "OPSDStorageReliability",
        "target_market": str(market),
        "target_year": int(year),
        "dimension": int(problem.d),
        "source_dimension": int(dimension),
        "dimension_holdout": False,
        "n0": int(n0),
        "seed_start": int(seed_start),
        "n_seeds": int(n_seeds),
        "source_archive_fingerprint": archive.fingerprint,
        "source_archive_simulator_calls": int(archive.simulator_calls),
        "source_archive_oracle_aided": False,
        "source_domains": list(archive.source_domains),
        "source_selection_mode": "registered_energy_lodo",
        "source_split_rule": f"AT_and_DK1_to_heldout_{market}",
        "target_labels_used": False,
        "target_oracle_used": False,
        "target_actual_error_used_during_materialization": False,
        "target_simulator_calls_during_materialization": 0,
        "paper_frontend_contract_id": (
            FRONTEND_MONOTONE_ENVELOPE_CHALLENGER_ID),
        "source_archive_information_contract": archive.information_contract(),
        "data_contract": {
            "compact_archive_sha256": _sha256(data_path),
            "problem_information": problem.information_contract(),
            "actual_target_error_read": False,
        },
        "target_descriptor_contract": {
            "track": "within_family_market_lodo",
            "allowed": list(ALLOWED_TARGET_DESCRIPTORS),
            "forbidden": list(FORBIDDEN_TARGET_INFORMATION),
            "observed": [
                "policy dimension and integer bounds",
                "battery schema and frozen physical constants",
                "day-ahead load forecast and price descriptor",
            ],
            "heldout_task_family_identifier_used": True,
        },
        "proposal_diagnostics": dict(
            prior.risk_objective_proposal_diagnostics),
        "training_diagnostics": {
            "source_only": True,
            "source_observation_mode": "replicated",
            "source_observation_replicates": SOURCE_REPLICATES,
            "source_records_per_market": SOURCE_RECORDS_PER_MARKET,
            "source_design_mode": "shared_low_frequency",
            "source_consensus": dict(prior.source_consensus_diagnostics),
        },
        "designs": designs,
    }
    payload["paper_frontend_contract_audit"] = (
        validate_frozen_proposal_payload(payload, expected_n0=n0)
    )
    attach_execution_provenance(payload)
    _atomic_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--market", default="DK_2")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--overwrite-archive", action="store_true")
    args = parser.parse_args()
    payload = materialize_energy_lodo(
        args.data,
        args.archive,
        args.out,
        market=args.market,
        year=args.year,
        dimension=args.d,
        n0=args.n0,
        seed_start=args.seed_start,
        n_seeds=args.n_seeds,
        overwrite_archive=args.overwrite_archive,
    )
    print(json.dumps({
        "status": "ok",
        "out": str(args.out),
        "archive": str(args.archive),
        "source_calls": payload["source_archive_simulator_calls"],
        "source_domains": payload["source_domains"],
        "target_calls": 0,
        "n_designs": len(payload["designs"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
