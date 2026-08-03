#!/usr/bin/env python3
"""Freeze target-label-free designs for the OPSD storage holdout."""

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
from performance.benchmark_lodo_meta_prior import train_meta_prior  # noqa: E402
from performance.benchmark_sota_fairness import oracle_free_lodo_config  # noqa: E402
from performance.execution_provenance import attach_execution_provenance  # noqa: E402
from performance.paper_method_contract import (  # noqa: E402
    ALLOWED_TARGET_DESCRIPTORS,
    FORBIDDEN_TARGET_INFORMATION,
    FRONTEND_CONTRACT_ID,
    validate_frozen_proposal_payload,
)
from performance.structural_ablation import apply_structural_prior_profile  # noqa: E402
from performance.task_descriptor_retrieval import (  # noqa: E402
    DESCRIPTOR_NEAREST,
    ENERGY_TARGET,
    SOURCE_SELECTION_MODES,
    source_selection_contract,
)
from problems.energy_reliability import OPSDStorageReliabilityProblem  # noqa: E402


def target_task_id(market, year):
    return f"{ENERGY_TARGET}:{str(market)}:{int(year)}"


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


def materialize_external_energy_design(
    manifest,
    archive_path,
    data_path,
    output,
    *,
    market="DK_2",
    year=2018,
    dimension=1000,
    source_dimension=50,
    n0=10,
    seed_start=80,
    n_seeds=5,
    source_selection_mode=DESCRIPTOR_NEAREST,
):
    """Map one frozen cross-family source atlas without target outcomes."""

    selection = source_selection_contract(
        source_selection_mode,
        target_domain=ENERGY_TARGET,
    )
    config = oracle_free_lodo_config(manifest)
    config["d"] = int(source_dimension)
    config["meta_source_dimension"] = int(source_dimension)
    config["meta_source_design_mode"] = "universal_mixture"
    apply_structural_prior_profile(config, "low_frequency_only")

    archive = FrozenTransferArchive.load(archive_path)
    archive.validate(
        expected_domains=selection.source_domains,
        expected_dimension=int(source_dimension),
    )
    prior = train_meta_prior(
        config,
        selection.source_split_heldout,
        0,
        teacher=False,
    )
    reconstructed = frozen_archive_from_meta_prior(
        prior, source_seed=int(archive.source_seed))
    if reconstructed.fingerprint != archive.fingerprint:
        raise ValueError(
            "external energy proposal and frozen source archive disagree")

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
        )
        points = [tuple(map(int, point)) for point in points]
        if len(points) != int(n0) or len(set(points)) != int(n0):
            raise RuntimeError(
                "external energy proposal did not produce n0 unique points")
        designs[str(seed)] = {
            "points": [list(point) for point in points],
            "fingerprint": integer_design_fingerprint(points),
        }

    task_id = target_task_id(market, year)
    payload = {
        "schema_version": 1,
        "design_kind": "frozen_source_informed_risk_objective_atlas",
        "proposal_mode": "risk_objective_atlas",
        "structural_prior_profile": "low_frequency_only",
        "structural_prior_active_components": list(config.get(
            "structural_prior_active_components", [])),
        "heldout_target_domain": task_id,
        "target_family": ENERGY_TARGET,
        "target_market": str(market),
        "target_year": int(year),
        "dimension": int(problem.d),
        "source_dimension": int(source_dimension),
        "dimension_holdout": bool(int(problem.d) != int(source_dimension)),
        "n0": int(n0),
        "seed_start": int(seed_start),
        "n_seeds": int(n_seeds),
        "source_archive_fingerprint": archive.fingerprint,
        "source_archive_simulator_calls": int(archive.simulator_calls),
        "source_archive_oracle_aided": False,
        "source_domains": list(archive.source_domains),
        "source_selection_mode": selection.mode,
        "source_selection_contract": selection.as_dict(),
        "source_split_rule": selection.track,
        "source_split_heldout": selection.source_split_heldout,
        "target_labels_used": False,
        "target_oracle_used": False,
        "target_actual_error_used_during_materialization": False,
        "target_simulator_calls_during_materialization": 0,
        "paper_frontend_contract_id": FRONTEND_CONTRACT_ID,
        "data_contract": {
            "compact_archive_sha256": _sha256(data_path),
            "problem_information": problem.information_contract(),
            "actual_target_error_read": False,
        },
        "target_descriptor_contract": {
            "track": selection.track,
            "allowed": list(ALLOWED_TARGET_DESCRIPTORS),
            "forbidden": list(FORBIDDEN_TARGET_INFORMATION),
            "observed": [
                "policy dimension and integer bounds",
                "battery input/output schema and fixed physics",
                "day-ahead load forecast and price descriptor",
                *selection.as_dict()["target_observable_roles"],
            ],
            "heldout_task_family_identifier_used": (
                selection.heldout_task_family_identifier_used),
        },
        "proposal_diagnostics": dict(
            prior.risk_objective_proposal_diagnostics),
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--market", default="DK_2")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--d", type=int, default=1000)
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--source-selection-mode",
        choices=SOURCE_SELECTION_MODES,
        default=DESCRIPTOR_NEAREST,
    )
    args = parser.parse_args()
    payload = materialize_external_energy_design(
        args.manifest,
        args.archive,
        args.data,
        args.out,
        market=args.market,
        year=args.year,
        dimension=args.d,
        source_dimension=args.source_d,
        n0=args.n0,
        seed_start=args.seed_start,
        n_seeds=args.n_seeds,
        source_selection_mode=args.source_selection_mode,
    )
    print(json.dumps({
        "status": "ok",
        "out": str(args.out),
        "target": payload["heldout_target_domain"],
        "source_domains": payload["source_domains"],
        "source_calls": payload["source_archive_simulator_calls"],
        "target_calls": 0,
        "n_designs": len(payload["designs"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
