#!/usr/bin/env python3
"""Audit exact V1/V3 front-end equivalence before target replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_design_equivalence(manifest, baseline, challenger):
    """Prove replay equivalence when the frozen point sets are identical.

    The downstream proposal-only runner is a deterministic function of the
    frozen points and target seed. If both front ends emit the same ordered
    points for every registered seed, target simulation cannot distinguish
    them; running duplicate target calls would add no evidence.
    """

    contract = manifest["source_only_equivalence_gate"]
    failures = []
    static_fields = (
        "heldout_target_domain",
        "dimension",
        "source_dimension",
        "n0",
        "seed_start",
        "n_seeds",
        "source_archive_fingerprint",
        "offline_source_calls",
        "source_design_mode",
        "proposal_mode",
        "proposal_component_mode",
        "structural_prior_profile",
    )
    for field in static_fields:
        if baseline.get(field) != challenger.get(field):
            failures.append(f"paired design field differs: {field}")

    expected = {
        "heldout_target_domain": str(contract["domain"]),
        "dimension": int(contract["target_dimension"]),
        "source_dimension": int(contract["source_dimension"]),
        "n0": int(contract["n0"]),
        "seed_start": int(contract["seed_start"]),
        "n_seeds": int(contract["n_seeds"]),
        "offline_source_calls": int(contract["source_calls"]),
    }
    for field, value in expected.items():
        if baseline.get(field) != value:
            failures.append(
                f"baseline {field}={baseline.get(field)!r}, expected {value!r}")

    if baseline.get("paper_frontend_contract_id") != manifest[
        "baseline_frontend"
    ]:
        failures.append("baseline front-end contract drifted")
    if challenger.get("paper_frontend_contract_id") != manifest[
        "challenger_frontend"
    ]:
        failures.append("challenger front-end contract drifted")
    if baseline.get("source_monotone_envelope") is not False:
        failures.append("baseline unexpectedly enables the V3 envelope")
    if challenger.get("source_monotone_envelope") is not True:
        failures.append("challenger does not enable the V3 envelope")
    for label, payload in (("baseline", baseline), ("challenger", challenger)):
        for field in (
            "source_archive_oracle_aided",
            "target_labels_used",
            "target_oracle_used",
        ):
            if payload.get(field) is not False:
                failures.append(f"{label} information contract drifted: {field}")

    expected_seeds = {
        str(seed)
        for seed in range(
            int(contract["seed_start"]),
            int(contract["seed_start"]) + int(contract["n_seeds"]),
        )
    }
    baseline_seeds = set(baseline.get("designs", {}))
    challenger_seeds = set(challenger.get("designs", {}))
    if baseline_seeds != expected_seeds:
        failures.append("baseline seed set differs from registration")
    if challenger_seeds != expected_seeds:
        failures.append("challenger seed set differs from registration")

    unequal_seeds = []
    for seed in sorted(expected_seeds, key=int):
        left = baseline.get("designs", {}).get(seed)
        right = challenger.get("designs", {}).get(seed)
        if left is None or right is None:
            continue
        if (
            left.get("fingerprint") != right.get("fingerprint")
            or left.get("points") != right.get("points")
        ):
            unequal_seeds.append(int(seed))

    diagnostic = challenger.get("proposal_diagnostics", {}).get(
        "source_monotone_envelope", {})
    fail_closed = diagnostic.get("status") == "rejected"
    if not fail_closed:
        failures.append(
            "source-monotone intervention activated; paired target replay is required")
    exact_equivalence = bool(not unequal_seeds and not failures)
    return {
        "domain": str(contract["domain"]),
        "status": "pass" if exact_equivalence else "target_replay_required",
        "source_archive_fingerprint": baseline.get(
            "source_archive_fingerprint"),
        "baseline_design_count": len(baseline.get("designs", {})),
        "challenger_design_count": len(challenger.get("designs", {})),
        "source_monotone_diagnostic": diagnostic,
        "source_intervention_failed_closed": bool(fail_closed),
        "ordered_point_sets_identical": not unequal_seeds,
        "unequal_seeds": unequal_seeds,
        "contract_failures": failures,
        "target_replay_required": not exact_equivalence,
        "target_simulator_calls_used_for_gate": 0,
    }


def evaluate_gate(manifest, pairs):
    domains = tuple(map(str, manifest["domains"]))
    rows = {}
    for domain in domains:
        baseline, challenger = pairs[domain]
        domain_manifest = dict(manifest)
        domain_manifest["source_only_equivalence_gate"] = {
            **manifest["source_only_equivalence_gate"],
            "domain": domain,
        }
        rows[domain] = evaluate_design_equivalence(
            domain_manifest, baseline, challenger)
    passed = all(row["status"] == "pass" for row in rows.values())
    return {
        "schema_version": 1,
        "gate_id": manifest["gate_id"],
        "status": "pass" if passed else "target_replay_required",
        "decision": (
            "v3_no_regression_by_exact_frozen_design_equivalence"
            if passed
            else "run_preregistered_paired_target_replay_for_nonidentical_domains"
        ),
        "domains": rows,
        "target_simulator_calls_used_for_gate": 0,
        "saas_used": False,
        "gpu_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--challenger-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = _read_json(args.manifest)
    pairs = {}
    input_hashes = {}
    for domain in manifest["domains"]:
        filename = f"heldout_{domain}.json"
        baseline_path = Path(args.baseline_dir) / filename
        challenger_path = Path(args.challenger_dir) / filename
        pairs[str(domain)] = (
            _read_json(baseline_path),
            _read_json(challenger_path),
        )
        input_hashes[str(domain)] = {
            "baseline_sha256": _sha256(baseline_path),
            "challenger_sha256": _sha256(challenger_path),
        }
    payload = evaluate_gate(manifest, pairs)
    payload["input_artifact_sha256"] = input_hashes
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
