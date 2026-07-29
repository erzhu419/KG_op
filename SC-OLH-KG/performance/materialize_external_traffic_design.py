#!/usr/bin/env python3
"""Freeze the paper front end for a no-history SUMO holdout."""

from __future__ import annotations

import argparse
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
from performance.paper_method_contract import (  # noqa: E402
    ALLOWED_TARGET_DESCRIPTORS,
    FORBIDDEN_TARGET_INFORMATION,
    FRONTEND_CONTRACT_ID,
    validate_frozen_proposal_payload,
)
from performance.execution_provenance import (  # noqa: E402
    attach_execution_provenance,
)
from performance.structural_ablation import apply_structural_prior_profile  # noqa: E402
from performance.task_descriptor_retrieval import (  # noqa: E402
    DOMAIN_BLIND_CONTROL,
    SOURCE_SELECTION_MODES,
    source_selection_contract,
)
from problems.traffic_ingolstadt21 import (  # noqa: E402
    Ingolstadt21ScalarizedTrafficProblem,
)


TARGET_DOMAIN = "Ingolstadt21Traffic"


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_external_traffic_design(
    manifest,
    archive_path,
    output,
    *,
    source_dimension=50,
    n0=10,
    seed_start=80,
    n_seeds=5,
    source_selection_mode=DOMAIN_BLIND_CONTROL,
):
    """Create target-label-free traffic designs from one frozen source archive."""

    selection = source_selection_contract(
        source_selection_mode,
        target_domain=TARGET_DOMAIN,
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
        prior,
        source_seed=int(archive.source_seed),
    )
    if reconstructed.fingerprint != archive.fingerprint:
        raise ValueError(
            "external traffic proposal and frozen source archive disagree")

    problem = Ingolstadt21ScalarizedTrafficProblem(
        seed=0,
        historical_anchor_policy="strict_none",
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
                "external traffic proposal did not produce n0 unique points")
        designs[str(seed)] = {
            "points": [list(point) for point in points],
            "fingerprint": integer_design_fingerprint(points),
        }

    payload = {
        "schema_version": 1,
        "design_kind": "frozen_source_informed_risk_objective_atlas",
        "proposal_mode": "risk_objective_atlas",
        "structural_prior_profile": "low_frequency_only",
        "structural_prior_active_components": list(config.get(
            "structural_prior_active_components", [])),
        "heldout_target_domain": TARGET_DOMAIN,
        "dimension": int(problem.d),
        "source_dimension": int(source_dimension),
        "source_design_mode": "universal_mixture",
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
        "sumo_imported_for_bounds_only": True,
        "sumo_simulator_calls_during_materialization": 0,
        "paper_frontend_contract_id": FRONTEND_CONTRACT_ID,
        "target_descriptor_contract": {
            "track": selection.track,
            "allowed": list(ALLOWED_TARGET_DESCRIPTORS),
            "forbidden": list(FORBIDDEN_TARGET_INFORMATION),
            "observed": [
                "policy dimension and integer bounds",
                "simulator input and output schema",
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--source-selection-mode",
        choices=SOURCE_SELECTION_MODES,
        default=DOMAIN_BLIND_CONTROL,
    )
    args = parser.parse_args()
    payload = materialize_external_traffic_design(
        args.manifest,
        args.archive,
        args.out,
        source_dimension=args.source_d,
        n0=args.n0,
        seed_start=args.seed_start,
        n_seeds=args.n_seeds,
        source_selection_mode=args.source_selection_mode,
    )
    print(json.dumps({
        "status": "ok",
        "out": str(args.out),
        "source_domains": payload["source_domains"],
        "source_calls": payload["source_archive_simulator_calls"],
        "source_selection_mode": payload["source_selection_mode"],
        "target_sumo_calls": 0,
        "n_designs": len(payload["designs"]),
    }, indent=2))


if __name__ == "__main__":
    main()
