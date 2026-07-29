#!/usr/bin/env python3
"""Freeze source-only proposals for the legacy RZDT family bridge."""

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
from performance.benchmark_lodo_meta_prior import (  # noqa: E402
    build_scalarized_problem,
    train_meta_prior,
)
from performance.benchmark_quality import parse_weights  # noqa: E402
from performance.benchmark_sota_fairness import oracle_free_lodo_config  # noqa: E402
from performance.paper_method_contract import (  # noqa: E402
    ALLOWED_TARGET_DESCRIPTORS,
    FORBIDDEN_TARGET_INFORMATION,
    FRONTEND_CONTRACT_ID,
    validate_frozen_proposal_payload,
)
from performance.structural_ablation import apply_structural_prior_profile  # noqa: E402


SOURCE_SPLIT_HELDOUT = "QueueResourceControl"
SOURCE_DOMAINS = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
)
PAPER_TARGETS = ("PaperRZDT1", "PaperRZDT2", "PaperRZDT5_RR")


def _parse_csv(value):
    return tuple(
        item.strip() for item in str(value).split(",") if item.strip())


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_bridge_designs(
    manifest,
    archive_path,
    output_dir,
    *,
    targets=PAPER_TARGETS,
    source_dimension=50,
    target_dimension=5,
    n0=10,
    seed_start=80,
    n_seeds=20,
):
    config = oracle_free_lodo_config(manifest)
    config["d"] = int(source_dimension)
    config["meta_source_dimension"] = int(source_dimension)
    config["meta_source_design_mode"] = "universal_mixture"
    apply_structural_prior_profile(config, "low_frequency_only")
    archive = FrozenTransferArchive.load(archive_path)
    archive.validate(
        expected_domains=SOURCE_DOMAINS,
        expected_dimension=int(source_dimension),
    )
    prior = train_meta_prior(
        config,
        SOURCE_SPLIT_HELDOUT,
        0,
        teacher=False,
    )
    reconstructed = frozen_archive_from_meta_prior(
        prior,
        source_seed=int(archive.source_seed),
    )
    if reconstructed.fingerprint != archive.fingerprint:
        raise ValueError(
            "paper bridge proposal and frozen source archive disagree")

    output_dir = Path(output_dir)
    paths = {}
    for target in targets:
        problem = build_scalarized_problem(
            target,
            int(target_dimension),
            int(config["L"]),
            float(config["sigma"]),
            float(config["alpha"]),
            parse_weights(config["weights"]),
        )
        designs = {}
        for seed in range(
            int(seed_start),
            int(seed_start) + int(n_seeds),
        ):
            points = prior.risk_objective_initial_candidates(
                problem,
                n=int(n0),
                rng=np.random.default_rng(seed),
            )
            points = [tuple(map(int, point)) for point in points]
            if len(points) != int(n0) or len(set(points)) != int(n0):
                raise RuntimeError(
                    f"{target} bridge proposal lacks n0 unique points")
            designs[str(seed)] = {
                "points": [list(point) for point in points],
                "fingerprint": integer_design_fingerprint(points),
            }
        payload = {
            "schema_version": 1,
            "design_kind": (
                "frozen_source_informed_risk_objective_atlas"),
            "proposal_mode": "risk_objective_atlas",
            "structural_prior_profile": "low_frequency_only",
            "structural_prior_active_components": list(config.get(
                "structural_prior_active_components", [])),
            "heldout_target_domain": str(target),
            "dimension": int(problem.d),
            "source_dimension": int(source_dimension),
            "source_design_mode": "universal_mixture",
            "dimension_holdout": bool(
                int(problem.d) != int(source_dimension)),
            "n0": int(n0),
            "seed_start": int(seed_start),
            "n_seeds": int(n_seeds),
            "source_archive_fingerprint": archive.fingerprint,
            "source_archive_simulator_calls": int(
                archive.simulator_calls),
            "source_archive_oracle_aided": False,
            "source_domains": list(archive.source_domains),
            "source_split_heldout_analogue": SOURCE_SPLIT_HELDOUT,
            "target_labels_used": False,
            "target_oracle_used": False,
            "paper_frontend_contract_id": FRONTEND_CONTRACT_ID,
            "target_descriptor_contract": {
                "track": "descriptor_conditional_related_family_bridge",
                "allowed": list(ALLOWED_TARGET_DESCRIPTORS),
                "forbidden": list(FORBIDDEN_TARGET_INFORMATION),
                "observed": [
                    "heldout task-family identifier",
                    "policy dimension and integer bounds",
                ],
            },
            "metric_contract": {
                "new_method": (
                    "scalarized chance-constrained objective and regret"),
                "legacy_manuscript": "bi-objective HV, IGD, and CVR",
                "direct_metric_equality_claim_allowed": False,
            },
            "proposal_diagnostics": dict(
                prior.risk_objective_proposal_diagnostics),
            "designs": designs,
        }
        payload["paper_frontend_contract_audit"] = (
            validate_frozen_proposal_payload(payload, expected_n0=n0)
        )
        path = output_dir / target / "source_initial_designs.json"
        _atomic_json(path, payload)
        paths[target] = str(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--targets", default=",".join(PAPER_TARGETS))
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument("--target-d", type=int, default=5)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=20)
    args = parser.parse_args()
    paths = materialize_bridge_designs(
        args.manifest,
        args.archive,
        args.out_dir,
        targets=_parse_csv(args.targets),
        source_dimension=args.source_d,
        target_dimension=args.target_d,
        n0=args.n0,
        seed_start=args.seed_start,
        n_seeds=args.n_seeds,
    )
    print(json.dumps({
        "status": "ok",
        "source_calls": 384,
        "target_outcomes_used": False,
        "paths": paths,
    }, indent=2))


if __name__ == "__main__":
    main()
