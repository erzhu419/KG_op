#!/usr/bin/env python3
"""Materialize the pre-existing universal traffic library for posthoc audit."""

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
from performance.execution_provenance import (  # noqa: E402
    attach_execution_provenance,
)
from performance.structural_ablation import apply_structural_prior_profile  # noqa: E402
from performance.task_descriptor_retrieval import (  # noqa: E402
    DESCRIPTOR_NEAREST,
    source_selection_contract,
)
from problems.traffic_ingolstadt21 import (  # noqa: E402
    Ingolstadt21ScalarizedTrafficProblem,
)


METHOD = "Universal-Library-Posthoc"
PARTITION = "universal_shape_v1"
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


def materialize(
    manifest,
    archive_path,
    output_dir,
    *,
    source_dimension=50,
    source_selection_mode=DESCRIPTOR_NEAREST,
):
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
        raise ValueError("posthoc library and frozen source archive disagree")

    problem = Ingolstadt21ScalarizedTrafficProblem(
        seed=0,
        historical_anchor_policy="strict_none",
    )
    points = prior.universal_shape_candidates(
        problem,
        n=10000,
        rng=np.random.default_rng(0),
        force=True,
    )
    points = [tuple(map(int, point)) for point in points]
    if not points or len(points) != len(set(points)):
        raise RuntimeError("universal traffic diagnostic library is invalid")

    payload = {
        "schema_version": 1,
        "method": METHOD,
        "partition_method": PARTITION,
        "seed": 0,
        "final_pareto_set": [list(point) for point in points],
        "library_fingerprint": integer_design_fingerprint(points),
        "diagnostic_contract": {
            "evidence_phase": "posthoc_certifiability_diagnostic",
            "library_frozen_before_target_outcomes": True,
            "library_definition": (
                "universal_shape_candidates existing before the traffic "
                "development result"
            ),
            "target_labels_used_to_construct_library": False,
            "target_oracle_used": False,
            "historical_target_anchor_used": False,
            "admissible_for_method_selection": False,
            "admissible_for_confirmatory_claim": False,
            "source_selection_mode": selection.mode,
            "source_domains": list(selection.source_domains),
            "source_archive_fingerprint": archive.fingerprint,
            "source_archive_simulator_calls": int(archive.simulator_calls),
            "target_policy_dimension": int(problem.d),
            "library_size": len(points),
        },
    }
    attach_execution_provenance(payload)
    output_dir = Path(output_dir)
    _atomic_json(output_dir / "summary.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-d", type=int, default=50)
    parser.add_argument(
        "--source-selection-mode",
        default=DESCRIPTOR_NEAREST,
    )
    args = parser.parse_args()
    payload = materialize(
        args.manifest,
        args.archive,
        args.output_dir,
        source_dimension=args.source_d,
        source_selection_mode=args.source_selection_mode,
    )
    print(json.dumps({
        "status": "ok",
        "library_size": payload["diagnostic_contract"]["library_size"],
        "library_fingerprint": payload["library_fingerprint"],
        "out": str(Path(args.output_dir) / "summary.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
