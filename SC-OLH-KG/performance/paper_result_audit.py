#!/usr/bin/env python3
"""Build compact, budget-explicit audits from paper experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def _first(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _nested(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _int_or_none(value):
    return None if value is None else int(value)


def _float_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _saas_identity(result):
    fidelity = str(result.get("algorithm_fidelity") or "")
    schedule = result.get("saas_nuts_schedule") or {}
    refit = str(schedule.get(
        "hyperposterior_refit_schedule") or "")
    if (
        fidelity == "saas_fully_bayesian_nuts_constrained_qlogei"
        and refit == "every_iteration"
        and bool(schedule.get("posterior_conditions_on_every_observation"))
    ):
        return "canonical_saasbo_every_iteration"
    if "periodic" in fidelity or refit not in {"", "every_iteration"}:
        return "saasbo_periodic_capped"
    return f"saasbo_unclassified:{fidelity or 'missing_fidelity'}"


def _method_identity(payload, result):
    method = str(_first(
        result.get("method"),
        payload.get("method"),
        "unknown",
    ))
    if method == "botorch_saasbo":
        identity = _saas_identity(result)
        head_mode = str(_first(
            _nested(payload, "information_contract", "aleatoric_head_mode"),
            result.get("aleatoric_head_mode"),
            "nominal",
        ))
        if head_mode != "nominal":
            identity += f"+hvd:{head_mode}"
        return identity
    if method == "botorch_turbo":
        return "botorch_turbo:canonical_turbo1_ts"
    if method == "botorch_scbo":
        return "botorch_scbo:canonical_scbo_constrained_ts"
    implementation_contract = result.get("implementation_contract_id")
    if method == "unknown" and implementation_contract:
        return f"scolh:{implementation_contract}"
    fidelity = str(_first(
        result.get("implementation_fidelity"),
        result.get("algorithm_fidelity"),
        payload.get("implementation"),
        "",
    ))
    return method if not fidelity else f"{method}:{fidelity}"


def _verifier_signature(verification):
    if not isinstance(verification, dict) or not verification:
        return None
    return json.dumps({
        "method": verification.get("method"),
        "protocol": verification.get("protocol"),
        "familywise_delta": verification.get("familywise_delta"),
        "candidate_budgets": verification.get(
            "candidate_verification_budgets"),
        "shortlist_mode": verification.get("shortlist_mode"),
        "updates_optimizer": verification.get(
            "posterior_updated_from_verification"),
        "search_samples_reused": verification.get(
            "search_samples_reused"),
    }, sort_keys=True, separators=(",", ":"))


def _scalarization_weights(value):
    if value is None:
        return [0.5, 0.5]
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = list(value)
    return [float(item) for item in values]


def _problem_contract(payload, result, *, domain, dimension):
    """Return the immutable target problem contract used for pairing.

    Older baseline result schemas did not serialize the nominal problem
    parameters. Their runners use the paper defaults recorded here. Newer SC
    results serialize those fields in ``config`` or the result row, which lets
    this audit catch mismatched target scenarios such as shared-shock scale.
    """

    config = payload.get("config") or {}
    info = payload.get("information_contract") or {}
    factor_domain = str(domain) == "FactorShockStatePolicyRZDT1"
    shared_shock_scale = None
    if factor_domain:
        shared_shock_scale = float(_first(
            result.get("target_shared_shock_scale"),
            config.get("target_shared_shock_scale"),
            result.get("shared_shock_scale"),
            info.get("target_shared_shock_scale"),
            1.0,
        ))
    contract = {
        "domain": str(domain),
        "dimension": _int_or_none(dimension),
        "L": int(_first(config.get("L"), info.get("target_L"), 100)),
        "sigma": float(_first(
            config.get("sigma"), info.get("target_sigma"), 0.04)),
        "alpha": float(_first(
            config.get("alpha"), info.get("target_alpha"), 0.05)),
        "scalarization_weights": _scalarization_weights(_first(
            config.get("weights"),
            info.get("scalarization_weights"),
            (0.5, 0.5),
        )),
        "heteroscedastic": True,
        "tau": 0.0,
        "shared_shock_scale": shared_shock_scale,
        "task_geometry": "nominal",
    }
    serialized = json.dumps(
        contract, sort_keys=True, separators=(",", ":"))
    return contract, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def extract_result_record(path, *, track_id):
    """Extract one compact row without copying histories or policy vectors."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result")
    if not isinstance(result, dict):
        row_payload = payload.get("rows")
        if isinstance(row_payload, list) and len(row_payload) == 1:
            result = row_payload[0]
        else:
            result = payload
    info = payload.get("information_contract") or {}
    comparison = _first(
        payload.get("comparison_contract"),
        result.get("comparison_contract"),
        {},
    )
    source_info = result.get("source_information_contract") or {}
    adaptation_info = result.get("source_target_adaptation_contract") or {}
    target_info = result.get("target_information_contract") or {}
    verification = result.get("terminal_verification") or {}
    aleatoric_audit = result.get("post_run_aleatoric_audit") or {}
    execution = _first(
        payload.get("execution_provenance"),
        result.get("execution_provenance"),
        {},
    )
    status = str(
        payload.get("status")
        or result.get("status")
        or ("ok" if result is not payload else "unknown")
    )

    source_calls = _first(
        result.get("source_calls"),
        info.get("offline_source_calls"),
        comparison.get("source_simulator_calls"),
        source_info.get("source_simulator_calls"),
        adaptation_info.get("source_simulator_calls"),
        0,
    )
    search_calls = _first(
        result.get("n_search_simulations"),
        info.get("target_search_calls"),
        comparison.get("target_search_calls"),
        target_info.get("target_search_calls"),
    )
    verification_calls = _first(
        result.get("n_verification_simulations"),
        info.get("target_verification_calls"),
        comparison.get("target_verification_calls"),
        target_info.get("target_verification_calls"),
        0,
    )
    target_total = _first(
        result.get("n_target_simulations_total"),
        info.get("target_total_calls"),
        comparison.get("target_total_calls"),
        target_info.get("target_total_calls"),
    )
    if target_total is None and search_calls is not None:
        target_total = int(search_calls) + int(verification_calls or 0)
    source_plus_target = (
        None
        if target_total is None or source_calls is None
        else int(source_calls) + int(target_total)
    )
    optimization_calls = (
        None
        if source_calls is None or search_calls is None
        else int(source_calls) + int(search_calls)
    )
    initial_fingerprint = _first(
        payload.get("initial_points_fingerprint"),
        payload.get("initial_design_fingerprint"),
        result.get("initial_points_fingerprint"),
        result.get("initial_design_fingerprint"),
        adaptation_info.get("target_initial_design_fingerprint"),
        comparison.get("target_initial_design_fingerprint"),
        target_info.get("initial_design_fingerprint"),
        result.get("target_design_fingerprint"),
    )
    archive_fingerprint = _first(
        payload.get("source_archive_fingerprint"),
        result.get("source_archive_fingerprint"),
        comparison.get("source_archive_fingerprint"),
        source_info.get("archive_fingerprint"),
        adaptation_info.get("source_archive_fingerprint"),
    )
    domain = str(_first(
        payload.get("heldout"),
        payload.get("heldout_target_domain"),
        result.get("heldout_target_domain"),
        result.get("problem"),
        result.get("heldout"),
        "unknown",
    ))
    target_dimension = _int_or_none(_first(
        info.get("target_dimension"),
        comparison.get("target_dimension"),
        target_info.get("dimension"),
        result.get("proposal_target_dimension"),
        result.get("d"),
        payload.get("d"),
        _nested(payload, "config", "d"),
    ))
    problem_contract, problem_fingerprint = _problem_contract(
        payload,
        result,
        domain=domain,
        dimension=target_dimension,
    )
    true_feasible = result.get("true_feasible")
    certified = verification.get("certified")
    return {
        "track_id": str(track_id),
        "path": str(path),
        "result_sha256": _sha256(path),
        "content_verified_at_extraction": True,
        "status": status,
        "domain": domain,
        "seed": _int_or_none(_first(
            payload.get("seed"), result.get("seed"))),
        "target_dimension": target_dimension,
        "problem_contract": problem_contract,
        "problem_contract_fingerprint": problem_fingerprint,
        "method": str(_first(
            result.get("method"), payload.get("method"), "unknown")),
        "method_identity": _method_identity(payload, result),
        "implementation": _first(
            payload.get("implementation"),
            result.get("implementation"),
        ),
        "execution_provenance_status": execution.get("status"),
        "execution_repository_commit": execution.get(
            "repository_commit"),
        "execution_scolhkg_tree": execution.get("scolhkg_tree"),
        "execution_proof_tree": execution.get("proof_tree"),
        "execution_scripts_tree": execution.get("scripts_tree"),
        "execution_method_contract_id": execution.get(
            "method_contract_id"),
        "execution_theory_contract_id": execution.get(
            "theory_contract_id"),
        "execution_snapshot_root": execution.get("snapshot_root"),
        "source_calls": _int_or_none(source_calls),
        "target_search_calls": _int_or_none(search_calls),
        "target_verification_calls": _int_or_none(verification_calls),
        "target_total_calls": _int_or_none(target_total),
        "optimization_calls_excluding_verification": _int_or_none(
            optimization_calls),
        "source_plus_target_total_calls": _int_or_none(
            source_plus_target),
        "initial_design_fingerprint": initial_fingerprint,
        "source_archive_fingerprint": archive_fingerprint,
        "verifier_signature": _verifier_signature(verification),
        "terminal_certified": (
            None if certified is None else bool(certified)),
        "true_feasible": (
            None if true_feasible is None else bool(true_feasible)),
        "false_certificate": bool(
            certified is True and true_feasible is False),
        "feasible_regret": _float_or_none(
            _first(
                result.get("feasible_regret"),
                result.get("feasible_simple_regret"),
                result.get("recommendation_true_best_feasible_regret"),
            )),
        "constraint_violation": _float_or_none(
            result.get("constraint_violation")),
        "aleatoric_log_variance_rmse": _float_or_none(
            aleatoric_audit.get("log_variance_rmse")),
        "aleatoric_variance_rmse": _float_or_none(
            aleatoric_audit.get("variance_rmse")),
        "aleatoric_upper_coverage": _float_or_none(
            aleatoric_audit.get("upper_coverage")),
        "aleatoric_variance_shape_correlation": _float_or_none(
            aleatoric_audit.get("variance_shape_correlation")),
        "aleatoric_audit_size": _int_or_none(
            aleatoric_audit.get("audit_size")),
        "aleatoric_audit_post_run_only": bool(
            aleatoric_audit
            and aleatoric_audit.get("target_oracle_used_post_run_only")
            is True
            and aleatoric_audit.get("used_for_search_or_selection")
            is False
        ),
        "failure_type": payload.get("failure_type"),
        "failure_message": payload.get("failure_message"),
    }


def _result_sources(specification):
    sources = specification.get("result_sources")
    if sources is not None:
        if not isinstance(sources, list) or not sources:
            raise ValueError("result_sources must be a nonempty list")
        return [dict(source) for source in sources]
    roots = specification.get("result_roots")
    if roots is None:
        roots = [specification["result_root"]]
    return [
        {
            "result_root": root,
            "glob": specification.get("glob", "**/result.json"),
        }
        for root in roots
    ]


def _source_accepts(row, source):
    include_methods = set(map(
        str, source.get("include_method_identities", ())))
    exclude_methods = set(map(
        str, source.get("exclude_method_identities", ())))
    include_domains = set(map(str, source.get("include_domains", ())))
    include_dimensions = set(map(
        int, source.get("include_dimensions", ())))
    include_seeds = set(map(int, source.get("include_seeds", ())))
    if include_methods and row["method_identity"] not in include_methods:
        return False
    if row["method_identity"] in exclude_methods:
        return False
    if include_domains and row["domain"] not in include_domains:
        return False
    if (
        include_dimensions
        and row["target_dimension"] not in include_dimensions
    ):
        return False
    if include_seeds and row["seed"] not in include_seeds:
        return False
    return True


def extract_registry_records(registry, *, root, origin=None):
    root = Path(root)
    records = []
    source_receipts = []
    for specification in registry["tracks"]:
        track_id = str(specification["track_id"])
        excluded_methods = set(map(str, specification.get(
            "exclude_methods", ())))
        for source in _result_sources(specification):
            result_root = root / str(source["result_root"])
            glob = str(source.get(
                "glob",
                specification.get("glob", "**/result.json"),
            ))
            paths = sorted(set(result_root.glob(glob)))
            accepted = 0
            for path in paths:
                row = extract_result_record(path, track_id=track_id)
                if row["method"] in excluded_methods:
                    continue
                if not _source_accepts(row, source):
                    continue
                if origin is not None:
                    row["extraction_origin"] = str(origin)
                records.append(row)
                accepted += 1
            source_receipts.append({
                "track_id": track_id,
                "result_root": str(source["result_root"]),
                "glob": glob,
                "matched_file_count": len(paths),
                "accepted_record_count": accepted,
                "origin": None if origin is None else str(origin),
            })
    return records, source_receipts


def _median(values):
    values = [value for value in values if value is not None]
    return None if not values else float(statistics.median(values))


def _mean(values):
    values = [value for value in values if value is not None]
    return None if not values else float(statistics.fmean(values))


def summarize_records(records):
    groups = {}
    for row in records:
        key = (row["track_id"], row["method_identity"], row["domain"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (track_id, method, domain), rows in sorted(groups.items()):
        ok = [row for row in rows if row["status"] == "ok"]
        regret = [
            row["feasible_regret"] for row in ok
            if row["feasible_regret"] is not None
        ]
        summaries.append({
            "track_id": track_id,
            "method_identity": method,
            "domain": domain,
            "submitted_rows": len(rows),
            "successful_rows": len(ok),
            "failed_rows": len(rows) - len(ok),
            "true_feasible_count": sum(
                row["true_feasible"] is True for row in ok),
            "certified_count": sum(
                row["terminal_certified"] is True for row in ok),
            "false_certificate_count": sum(
                row["false_certificate"] for row in ok),
            "median_feasible_regret": _median(regret),
            "mean_feasible_regret": _mean(regret),
            "mean_source_calls": _mean(
                [row["source_calls"] for row in ok]),
            "mean_target_search_calls": _mean(
                [row["target_search_calls"] for row in ok]),
            "mean_target_verification_calls": _mean(
                [row["target_verification_calls"] for row in ok]),
            "mean_target_total_calls": _mean(
                [row["target_total_calls"] for row in ok]),
            "mean_optimization_calls_excluding_verification": _mean([
                row["optimization_calls_excluding_verification"]
                for row in ok
            ]),
            "mean_source_plus_target_total_calls": _mean([
                row["source_plus_target_total_calls"] for row in ok
            ]),
            "mean_aleatoric_log_variance_rmse": _mean([
                row["aleatoric_log_variance_rmse"] for row in ok
            ]),
            "mean_aleatoric_variance_rmse": _mean([
                row["aleatoric_variance_rmse"] for row in ok
            ]),
            "mean_aleatoric_upper_coverage": _mean([
                row["aleatoric_upper_coverage"] for row in ok
            ]),
            "mean_aleatoric_variance_shape_correlation": _mean([
                row["aleatoric_variance_shape_correlation"] for row in ok
            ]),
            "post_run_aleatoric_audit_count": sum(
                row["aleatoric_audit_post_run_only"] for row in ok),
        })
    return summaries


def audit_track(records, specification):
    track_id = str(specification["track_id"])
    rows = [row for row in records if row["track_id"] == track_id]
    expected_methods = set(map(str, specification.get(
        "expected_method_identities", ())))
    observed_methods = {row["method_identity"] for row in rows}
    expected_domains = set(map(str, specification.get(
        "expected_domains", ())))
    observed_domains = {row["domain"] for row in rows}
    expected_seeds = set(map(int, specification.get(
        "expected_seeds", ())))
    expected_dimensions = set(map(int, specification.get(
        "expected_dimensions", ())))
    observed_dimensions = {
        row["target_dimension"] for row in rows
        if row["target_dimension"] is not None
    }
    failures = []
    if expected_methods and observed_methods != expected_methods:
        failures.append({
            "kind": "method_identity_mismatch",
            "expected": sorted(expected_methods),
            "observed": sorted(observed_methods),
        })
    if expected_domains and observed_domains != expected_domains:
        failures.append({
            "kind": "domain_mismatch",
            "expected": sorted(expected_domains),
            "observed": sorted(observed_domains),
        })
    if expected_dimensions and observed_dimensions != expected_dimensions:
        failures.append({
            "kind": "target_dimension_mismatch",
            "expected": sorted(expected_dimensions),
            "observed": sorted(observed_dimensions),
        })
    cell_map = {}
    for row in rows:
        key = (
            row["method_identity"],
            row["domain"],
            row["target_dimension"],
            row["seed"],
        )
        cell_map.setdefault(key, []).append(row)
    duplicates = [key for key, group in cell_map.items() if len(group) != 1]
    if duplicates:
        failures.append({
            "kind": "duplicate_cells",
            "count": len(duplicates),
        })
    if expected_methods and expected_domains and expected_seeds:
        dimensions = expected_dimensions or observed_dimensions or {None}
        expected_cells = {
            (method, domain, dimension, seed)
            for method in expected_methods
            for domain in expected_domains
            for dimension in dimensions
            for seed in expected_seeds
        }
        missing = expected_cells - set(cell_map)
        if missing:
            failures.append({
                "kind": "missing_cells",
                "count": len(missing),
            })

    comparable = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        comparable.setdefault(
            (
                row["domain"],
                row["target_dimension"],
                row["seed"],
            ),
            [],
        ).append(row)
    equality_fields = list(specification.get(
        "paired_equality_fields", ()))
    if (
        len(expected_methods) > 1
        and "problem_contract_fingerprint" not in equality_fields
    ):
        equality_fields.append("problem_contract_fingerprint")
    paired_checks = {}
    for field in equality_fields:
        bad = []
        for key, group in comparable.items():
            values = {row.get(field) for row in group}
            if len(values) != 1 or None in values:
                bad.append(key)
        paired_checks[str(field)] = {
            "passed_cells": len(comparable) - len(bad),
            "total_cells": len(comparable),
            "failed_cells": len(bad),
        }
        if bad:
            failures.append({
                "kind": f"paired_{field}_mismatch",
                "count": len(bad),
            })
    required_source_calls = specification.get("required_source_calls")
    if required_source_calls is not None:
        bad = [
            row for row in rows
            if row["status"] == "ok"
            if row["source_calls"] != int(required_source_calls)
        ]
        if bad:
            failures.append({
                "kind": "source_budget_mismatch",
                "count": len(bad),
            })
    required_source_calls_by_method = {
        str(method): int(value)
        for method, value in specification.get(
            "required_source_calls_by_method", {}
        ).items()
    }
    if required_source_calls_by_method:
        missing_budget_contracts = (
            expected_methods - set(required_source_calls_by_method)
        )
        if missing_budget_contracts:
            failures.append({
                "kind": "missing_method_source_budget_contract",
                "methods": sorted(missing_budget_contracts),
            })
        bad = [
            row for row in rows
            if row["status"] == "ok"
            if row["method_identity"] in required_source_calls_by_method
            if row["source_calls"] != required_source_calls_by_method[
                row["method_identity"]
            ]
        ]
        if bad:
            failures.append({
                "kind": "method_source_budget_mismatch",
                "count": len(bad),
            })
    required_search_calls = specification.get("required_search_calls")
    if required_search_calls is not None:
        bad = [
            row for row in rows
            if row["status"] == "ok"
            if row["target_search_calls"] != int(required_search_calls)
        ]
        if bad:
            failures.append({
                "kind": "target_search_budget_mismatch",
                "count": len(bad),
            })
    required_optimization_calls = specification.get(
        "required_optimization_calls")
    if required_optimization_calls is not None:
        bad = [
            row for row in rows
            if row["status"] == "ok"
            if row["optimization_calls_excluding_verification"]
            != int(required_optimization_calls)
        ]
        if bad:
            failures.append({
                "kind": "optimization_budget_mismatch",
                "count": len(bad),
            })
    disallowed_identities = set(map(str, specification.get(
        "disallowed_method_identities", ())))
    contaminated = observed_methods & disallowed_identities
    if contaminated:
        failures.append({
            "kind": "disallowed_algorithm_identity",
            "observed": sorted(contaminated),
        })
    required_provenance = specification.get(
        "required_execution_provenance_status")
    if required_provenance is not None:
        bad = [
            row for row in rows
            if row["status"] == "ok"
            if row.get("execution_provenance_status")
            != str(required_provenance)
        ]
        if bad:
            failures.append({
                "kind": "execution_provenance_status_mismatch",
                "count": len(bad),
            })
    allowed_commits = set(map(str, specification.get(
        "allowed_execution_commits", ())))
    if allowed_commits:
        bad = [
            row for row in rows
            if row["status"] == "ok"
            if row.get("execution_repository_commit")
            not in allowed_commits
        ]
        if bad:
            failures.append({
                "kind": "execution_commit_mismatch",
                "count": len(bad),
                "allowed": sorted(allowed_commits),
            })
    for specification_field, row_field, failure_kind in (
        (
            "required_method_contract_id",
            "execution_method_contract_id",
            "execution_method_contract_mismatch",
        ),
        (
            "required_theory_contract_id",
            "execution_theory_contract_id",
            "execution_theory_contract_mismatch",
        ),
        (
            "required_scolhkg_tree",
            "execution_scolhkg_tree",
            "execution_scolhkg_tree_mismatch",
        ),
    ):
        expected = specification.get(specification_field)
        if expected is None:
            continue
        bad = [
            row for row in rows
            if row["status"] == "ok"
            if row.get(row_field) != str(expected)
        ]
        if bad:
            failures.append({
                "kind": failure_kind,
                "count": len(bad),
            })
    contract_by_method = {
        str(method): str(contract)
        for method, contract in specification.get(
            "required_method_contract_by_method", {}
        ).items()
    }
    if contract_by_method:
        missing_contracts = expected_methods - set(contract_by_method)
        if missing_contracts:
            failures.append({
                "kind": "missing_method_execution_contract",
                "methods": sorted(missing_contracts),
            })
        bad = [
            row for row in rows
            if row["status"] == "ok"
            if row["method_identity"] in contract_by_method
            if row.get("execution_method_contract_id")
            != contract_by_method[row["method_identity"]]
        ]
        if bad:
            failures.append({
                "kind": "method_execution_contract_mismatch",
                "count": len(bad),
            })
    failed_rows = [row for row in rows if row["status"] != "ok"]
    if failed_rows and not bool(specification.get("allow_failures", False)):
        failures.append({
            "kind": "failed_rows",
            "count": len(failed_rows),
        })
    return {
        "track_id": track_id,
        "status": "pass" if not failures else "fail",
        "row_count": len(rows),
        "successful_count": sum(
            row["status"] == "ok" for row in rows),
        "failed_count": sum(
            row["status"] != "ok" for row in rows),
        "observed_method_identities": sorted(observed_methods),
        "observed_domains": sorted(observed_domains),
        "observed_dimensions": sorted(observed_dimensions),
        "paired_checks": paired_checks,
        "failures": failures,
    }


def build_audit_from_records(
    registry,
    records,
    *,
    source_mode="local_result_files",
    record_shard_receipts=None,
    source_receipts=None,
):
    records = list(records)
    track_audits = []
    for specification in registry["tracks"]:
        track_audits.append(audit_track(records, specification))
    return {
        "schema_version": 1,
        "registry_id": registry.get("registry_id"),
        "source_mode": str(source_mode),
        "record_shard_receipts": list(record_shard_receipts or ()),
        "source_receipts": list(source_receipts or ()),
        "status": (
            "pass"
            if all(row["status"] == "pass" for row in track_audits)
            else "incomplete_or_failed"
        ),
        "record_count": len(records),
        "track_audits": track_audits,
        "summaries": summarize_records(records),
        "records": records,
    }


def build_audit(registry, *, root):
    records, receipts = extract_registry_records(registry, root=root)
    return build_audit_from_records(
        registry,
        records,
        source_mode="local_result_files",
        source_receipts=receipts,
    )


def _canonical_sha256(payload):
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_record_shards(paths, *, registry):
    registry_fingerprint = _canonical_sha256(registry)
    records = []
    receipts = []
    source_receipts = []
    for path in map(Path, paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported record shard schema: {path}")
        if payload.get("registry_id") != registry.get("registry_id"):
            raise ValueError(f"record shard registry identity mismatch: {path}")
        if payload.get("registry_sha256") != registry_fingerprint:
            raise ValueError(f"record shard registry hash mismatch: {path}")
        shard_records = list(payload.get("records", ()))
        expected = _canonical_sha256(shard_records)
        if payload.get("records_sha256") != expected:
            raise ValueError(f"record shard payload hash mismatch: {path}")
        if not all(
            row.get("content_verified_at_extraction") is True
            and isinstance(row.get("result_sha256"), str)
            and len(row["result_sha256"]) == 64
            for row in shard_records
        ):
            raise ValueError(f"record shard lacks extraction receipts: {path}")
        records.extend(shard_records)
        source_receipts.extend(payload.get("source_receipts", ()))
        receipts.append({
            "path": str(path),
            "sha256": _sha256(path),
            "origin": payload.get("origin"),
            "record_count": len(shard_records),
            "records_sha256": expected,
        })
    return records, receipts, source_receipts


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--root")
    parser.add_argument(
        "--record-shard",
        action="append",
        default=[],
        help="Compact record shard generated beside remote result files.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    if bool(args.root) == bool(args.record_shard):
        parser.error("provide exactly one of --root or --record-shard")
    if args.record_shard:
        records, receipts, source_receipts = load_record_shards(
            args.record_shard,
            registry=registry,
        )
        audit = build_audit_from_records(
            registry,
            records,
            source_mode="remote_compact_record_shards",
            record_shard_receipts=receipts,
            source_receipts=source_receipts,
        )
    else:
        audit = build_audit(registry, root=args.root)
    _atomic_json(args.out, audit)
    _write_csv(args.summary_csv, audit["summaries"])
    print(json.dumps({
        "status": audit["status"],
        "record_count": audit["record_count"],
        "track_count": len(audit["track_audits"]),
        "out": str(args.out),
        "summary_csv": str(args.summary_csv),
    }, indent=2))


if __name__ == "__main__":
    main()
