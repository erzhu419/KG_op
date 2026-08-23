"""Deterministic, offline falsification loop for the four structural priors.

The loop evaluates a finite graph of already registered ablation profiles.  It
does not execute experiments, call a model, or promote a paper method.  A
decision is always scoped to the supplied versioned contract and recorded
aggregate rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .structural_ablation import PRIOR_COMPONENTS, STRUCTURAL_PRIOR_PROFILES


SCHEMA_VERSION = "sc-olh-kg.structural-hypothesis-loop/1"
_GENESIS_HASH = "sha256:" + "0" * 64
_PROFILE_NAMES = (
    "none",
    "low_frequency_only",
    "orthogonality_only",
    "sparsity_only",
    "additivity_only",
    "full",
    "leave_out_low_frequency",
    "leave_out_orthogonality",
    "leave_out_sparsity",
    "leave_out_additivity",
)
_MANDATORY_NONCLAIMS = {
    "retrospective_aggregate_only",
    "no_causal_identification",
    "no_universal_truth",
    "no_paper_promotion",
    "no_online_execution",
    "no_archive_fingerprint_verification",
}
_FIXED_ROW_VALUE_KEYS = {
    "total_calls",
    "hvd_profile",
    "source_discrepancy_update",
    "recheck_top_k",
    "risk_penalty",
    "utility_weight",
    "adaptive_replication_voi",
    "posterior_dominance_enabled",
    "posterior_dominance_switch_count",
}
REQUIRED_EVIDENCE_FIELDS = frozenset({
    "track", "run_id", "variant", "method", "structural_prior_profile",
    "domain", "seed", "d", "N", "n0", "source_calls", "implementation",
    "initial_design", "decision_backend", "status", "true_feasible",
    "adaptive_loss", "feasible_regret", *_FIXED_ROW_VALUE_KEYS,
})


class ContractValidationError(ValueError):
    """Raised when a hypothesis-loop contract is not the frozen V1 shape."""


class Disposition(str, Enum):
    SUPPORTED_SCOPED = "SUPPORTED_SCOPED"
    REFUTED_SCOPED = "REFUTED_SCOPED"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unsupported_type": type(value).__name__, "repr": repr(value)}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _stable_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    kind: str
    component: str | None
    challenger_profile: str
    reference_profile: str
    generation: int
    parent_ids: tuple[str, ...]
    claim: str
    challenger_components: tuple[str, ...]
    reference_components: tuple[str, ...]
    proposal_operation: str
    trigger_disposition: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "kind": self.kind,
            "component": self.component,
            "challenger_profile": self.challenger_profile,
            "reference_profile": self.reference_profile,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "claim": self.claim,
            "challenger_components": list(self.challenger_components),
            "reference_components": list(self.reference_components),
            "proposal_operation": self.proposal_operation,
            "trigger_disposition": self.trigger_disposition,
        }


@dataclass(frozen=True)
class HypothesisDecision:
    hypothesis_id: str
    disposition: Disposition
    reason_codes: tuple[str, ...]
    metrics: Mapping[str, Any]
    missing_cells: tuple[Mapping[str, Any], ...]
    invalid_issues: tuple[str, ...]
    evidence_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "metrics": _stable_value(self.metrics),
            "missing_cells": [_stable_value(item) for item in self.missing_cells],
            "invalid_issues": list(self.invalid_issues),
            "evidence_digest": self.evidence_digest,
        }


def _report_body_payload(
    *,
    contract_id: str,
    contract_digest: str,
    evidence_digest: str,
    status: str,
    stop_reason: str,
    evidence_scope: Mapping[str, Any],
    gate: Mapping[str, Any],
    hypotheses: Sequence[HypothesisSpec],
    decisions: Sequence[HypothesisDecision],
    verdict_counts: Mapping[str, int],
    pending_evidence: Sequence[Mapping[str, Any]],
    synthesis: Mapping[str, Any],
    nonclaims: Sequence[str],
    input_artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": contract_id,
        "contract_digest": contract_digest,
        "evidence_digest": evidence_digest,
        "status": status,
        "stop_reason": stop_reason,
        "evidence_scope": _stable_value(evidence_scope),
        "gate": _stable_value(gate),
        "hypotheses": [item.to_dict() for item in hypotheses],
        "decisions": [item.to_dict() for item in decisions],
        "verdict_counts": dict(verdict_counts),
        "pending_evidence": [_stable_value(item) for item in pending_evidence],
        "synthesis": _stable_value(synthesis),
        "nonclaims": list(nonclaims),
        "input_artifacts": _stable_value(input_artifacts),
    }


@dataclass(frozen=True)
class LoopResult:
    contract_id: str
    contract_digest: str
    evidence_digest: str
    status: str
    stop_reason: str
    hypotheses: tuple[HypothesisSpec, ...]
    decisions: tuple[HypothesisDecision, ...]
    verdict_counts: Mapping[str, int]
    pending_evidence: tuple[Mapping[str, Any], ...]
    synthesis: Mapping[str, Any]
    audit_events: tuple[Mapping[str, Any], ...]
    audit_head: str
    evidence_scope: Mapping[str, Any]
    gate: Mapping[str, Any]
    nonclaims: tuple[str, ...]
    input_artifacts: Mapping[str, Any] | None
    report_body_digest: str

    def _body_dict(self) -> dict[str, Any]:
        return _report_body_payload(
            contract_id=self.contract_id,
            contract_digest=self.contract_digest,
            evidence_digest=self.evidence_digest,
            status=self.status,
            stop_reason=self.stop_reason,
            evidence_scope=self.evidence_scope,
            gate=self.gate,
            hypotheses=self.hypotheses,
            decisions=self.decisions,
            verdict_counts=self.verdict_counts,
            pending_evidence=self.pending_evidence,
            synthesis=self.synthesis,
            nonclaims=self.nonclaims,
            input_artifacts=self.input_artifacts,
        )

    def to_dict(self) -> dict[str, Any]:
        result = self._body_dict()
        result["audit"] = {
            "algorithm": "sha256-canonical-json-chain-v1",
            "genesis": _GENESIS_HASH,
            "head": self.audit_head,
            "report_body_digest": self.report_body_digest,
            "events": [_stable_value(item) for item in self.audit_events],
        }
        return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise ContractValidationError(
            f"{label} keys differ: missing={missing}, extra={extra}"
        )


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ContractValidationError(f"{label} must be an exact integer")
    return value


def validate_contract(contract: Mapping[str, Any]) -> None:
    if not isinstance(contract, Mapping):
        raise ContractValidationError("contract must be an object")
    _exact_keys(
        contract,
        {
            "schema_version", "contract_id", "components", "profiles",
            "evidence_scope", "gate", "nonclaims",
        },
        "contract",
    )
    if contract["schema_version"] != SCHEMA_VERSION:
        raise ContractValidationError("unsupported schema_version")
    if type(contract["contract_id"]) is not str or not contract["contract_id"]:
        raise ContractValidationError("contract_id must be non-empty text")
    components = contract["components"]
    if type(components) is not list or tuple(components) != tuple(PRIOR_COMPONENTS):
        raise ContractValidationError("components must equal PRIOR_COMPONENTS in order")

    profiles = contract["profiles"]
    if not isinstance(profiles, Mapping) or set(profiles) != set(_PROFILE_NAMES):
        raise ContractValidationError("profiles must contain the exact ten registered profiles")
    if not set(_PROFILE_NAMES).issubset(STRUCTURAL_PRIOR_PROFILES):
        raise ContractValidationError("an executable V1 profile is unavailable")
    for profile in _PROFILE_NAMES:
        value = profiles[profile]
        if type(value) is not list or any(type(item) is not str for item in value):
            raise ContractValidationError(f"profiles.{profile} must be a string list")
        active = set(STRUCTURAL_PRIOR_PROFILES[profile]["structural_prior_active_components"])
        expected = [component for component in PRIOR_COMPONENTS if component in active]
        if value != expected:
            raise ContractValidationError(f"profiles.{profile} does not match executable profile")

    scope = contract["evidence_scope"]
    if not isinstance(scope, Mapping):
        raise ContractValidationError("evidence_scope must be an object")
    _exact_keys(
        scope,
        {
            "track", "run_id", "domains", "seeds", "d", "N", "n0",
            "source_calls", "implementation", "initial_design",
            "decision_backend", "variant_template", "fixed_row_values",
        },
        "evidence_scope",
    )
    for key in (
        "track", "run_id", "implementation", "initial_design",
        "decision_backend", "variant_template",
    ):
        if type(scope[key]) is not str or not scope[key]:
            raise ContractValidationError(f"evidence_scope.{key} must be non-empty text")
    if scope["variant_template"] != "structural_backend/priors/{profile}":
        raise ContractValidationError("variant_template differs from the V1 template")
    domains = scope["domains"]
    if (
        type(domains) is not list or not domains
        or any(type(item) is not str or not item for item in domains)
        or len(set(domains)) != len(domains)
    ):
        raise ContractValidationError("domains must be unique non-empty strings")
    seeds = scope["seeds"]
    if (
        type(seeds) is not list or not seeds
        or any(type(item) is not int for item in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ContractValidationError("seeds must be unique exact integers")
    for key in ("d", "N", "n0", "source_calls"):
        if _exact_int(scope[key], f"evidence_scope.{key}") < 0:
            raise ContractValidationError(f"evidence_scope.{key} must be nonnegative")
    fixed_values = scope["fixed_row_values"]
    if not isinstance(fixed_values, Mapping):
        raise ContractValidationError("fixed_row_values must be an object")
    _exact_keys(fixed_values, _FIXED_ROW_VALUE_KEYS, "fixed_row_values")
    expected_fixed_types = {
        "total_calls": int,
        "hvd_profile": str,
        "source_discrepancy_update": bool,
        "recheck_top_k": int,
        "risk_penalty": float,
        "utility_weight": float,
        "adaptive_replication_voi": bool,
        "posterior_dominance_enabled": bool,
        "posterior_dominance_switch_count": int,
    }
    for key, expected_type in expected_fixed_types.items():
        if type(fixed_values[key]) is not expected_type:
            raise ContractValidationError(
                f"fixed_row_values.{key} must have exact type "
                f"{expected_type.__name__}"
            )
        if expected_type is float and not math.isfinite(fixed_values[key]):
            raise ContractValidationError(
                f"fixed_row_values.{key} must be finite"
            )
    if fixed_values["total_calls"] != scope["source_calls"] + scope["N"]:
        raise ContractValidationError("total_calls must equal source_calls + N")

    gate = contract["gate"]
    if not isinstance(gate, Mapping):
        raise ContractValidationError("gate must be an object")
    _exact_keys(
        gate,
        {
            "min_overall_feasible", "min_per_domain_feasible",
            "max_failed_rows", "max_adaptive_losses", "regret_tolerance",
        },
        "gate",
    )
    expected_rows = len(domains) * len(seeds)
    for key in (
        "min_overall_feasible", "min_per_domain_feasible",
        "max_failed_rows", "max_adaptive_losses",
    ):
        if _exact_int(gate[key], f"gate.{key}") < 0:
            raise ContractValidationError(f"gate.{key} must be nonnegative")
    if gate["min_overall_feasible"] > expected_rows:
        raise ContractValidationError("min_overall_feasible exceeds expected rows")
    if gate["min_per_domain_feasible"] > len(seeds):
        raise ContractValidationError("min_per_domain_feasible exceeds seeds per domain")
    if gate["max_failed_rows"] != 0:
        raise ContractValidationError("V1 requires max_failed_rows to be exactly zero")
    tolerance = gate["regret_tolerance"]
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ContractValidationError("regret_tolerance must be finite and nonnegative")

    nonclaims = contract["nonclaims"]
    if (
        type(nonclaims) is not list
        or any(type(item) is not str or not item for item in nonclaims)
        or len(set(nonclaims)) != len(nonclaims)
        or not _MANDATORY_NONCLAIMS.issubset(nonclaims)
    ):
        raise ContractValidationError("nonclaims are malformed or incomplete")


def _make_hypothesis(
    *, kind: str, component: str | None, challenger: str, reference: str,
    generation: int, parent_ids: tuple[str, ...], profiles: Mapping[str, list[str]],
    proposal_operation: str, trigger_disposition: str | None,
) -> HypothesisSpec:
    payload = {
        "kind": kind,
        "component": component,
        "challenger_profile": challenger,
        "reference_profile": reference,
        "generation": generation,
        "parent_ids": list(parent_ids),
    }
    hypothesis_id = "hypothesis:" + _digest(payload).split(":", 1)[1][:24]
    return HypothesisSpec(
        hypothesis_id=hypothesis_id,
        kind=kind,
        component=component,
        challenger_profile=challenger,
        reference_profile=reference,
        generation=generation,
        parent_ids=parent_ids,
        claim=f"{challenger} outperforms {reference} under the frozen recorded-evidence gate",
        challenger_components=tuple(profiles[challenger]),
        reference_components=tuple(profiles[reference]),
        proposal_operation=proposal_operation,
        trigger_disposition=trigger_disposition,
    )


def propose_initial_hypothesis(contract: Mapping[str, Any]) -> HypothesisSpec:
    validate_contract(contract)
    profiles = contract["profiles"]
    return _make_hypothesis(
        kind="COMPOSITE", component=None, challenger="full", reference="none",
        generation=0, parent_ids=(), profiles=profiles,
        proposal_operation="INITIAL_COMPOSITE_CHALLENGE",
        trigger_disposition=None,
    )


def revise_hypothesis(
    hypothesis: HypothesisSpec,
    decision: HypothesisDecision,
    contract: Mapping[str, Any],
) -> tuple[HypothesisSpec, ...]:
    """Propose the next bounded probes from an observed verdict.

    A supported composite is challenged for necessity before simplification.
    A refuted, incomplete, or invalid composite is diagnosed with standalone
    probes before interaction/necessity probes.  Child probes terminate the V1
    graph; a future executor adapter may feed their missing cells into another
    invocation without changing verifier semantics.
    """
    validate_contract(contract)
    if hypothesis.kind != "COMPOSITE":
        return ()
    if decision.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError("decision does not belong to hypothesis")
    profiles = contract["profiles"]
    supported = decision.disposition is Disposition.SUPPORTED_SCOPED
    kind_order = (
        ("NECESSITY", "STANDALONE")
        if supported else ("STANDALONE", "NECESSITY")
    )
    result: list[HypothesisSpec] = []
    for component in PRIOR_COMPONENTS:
        for kind in kind_order:
            if kind == "STANDALONE":
                challenger = f"{component}_only"
                reference = "none"
                operation = "SIMPLIFY_TO_SINGLE_COMPONENT"
            else:
                challenger = "full"
                reference = f"leave_out_{component}"
                operation = "ABLATE_COMPONENT_FROM_FULL"
            result.append(_make_hypothesis(
                kind=kind,
                component=component,
                challenger=challenger,
                reference=reference,
                generation=hypothesis.generation + 1,
                parent_ids=(hypothesis.hypothesis_id,),
                profiles=profiles,
                proposal_operation=operation,
                trigger_disposition=decision.disposition.value,
            ))
    return tuple(result)


def propose_hypotheses(
    contract: Mapping[str, Any],
    trigger_disposition: Disposition = Disposition.NEEDS_EVIDENCE,
) -> tuple[HypothesisSpec, ...]:
    """Return the V1 graph in the order induced by a root disposition."""
    root = propose_initial_hypothesis(contract)
    placeholder = HypothesisDecision(
        root.hypothesis_id,
        trigger_disposition,
        (),
        {},
        (),
        (),
        _digest({"placeholder": trigger_disposition.value}),
    )
    return (root, *revise_hypothesis(root, placeholder, contract))


def _parse_wire_int(value: Any) -> int:
    if type(value) is int:
        return value
    if type(value) is str and value and value == str(int(value)):
        return int(value)
    raise ValueError("not a canonical integer")


def _parse_wire_bool(value: Any) -> bool:
    if type(value) is bool:
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("not an exact boolean")


def _parse_finite(value: Any) -> float:
    if type(value) is bool:
        raise ValueError("boolean is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("number is not finite")
    return number


def _parse_fixed_value(value: Any, expected: Any) -> Any:
    if type(expected) is bool:
        return _parse_wire_bool(value)
    if type(expected) is int:
        return _parse_wire_int(value)
    if type(expected) is float:
        return _parse_finite(value)
    if type(expected) is str and type(value) is str:
        return value
    raise ValueError("fixed field has the wrong wire type")


def _cell(
    profile: str, domain: str, seed: int, scope: Mapping[str, Any]
) -> dict[str, Any]:
    result = {
        "track": scope["track"],
        "run_id": scope["run_id"],
        "variant": scope["variant_template"].format(profile=profile),
        "profile": profile,
        "domain": domain,
        "seed": seed,
        "d": scope["d"],
        "N": scope["N"],
        "n0": scope["n0"],
        "source_calls": scope["source_calls"],
        "implementation": scope["implementation"],
        "initial_design": scope["initial_design"],
        "decision_backend": scope["decision_backend"],
    }
    result.update(scope["fixed_row_values"])
    return result


def _index_rows(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, list[str]], int]:
    scope = contract["evidence_scope"]
    known_profiles = set(contract["profiles"])
    domains = set(scope["domains"])
    seeds = set(scope["seeds"])
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    issues: dict[str, list[str]] = {profile: [] for profile in known_profiles}
    ignored = 0

    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            ignored += 1
            continue
        if raw.get("track") != scope["track"] or raw.get("run_id") != scope["run_id"]:
            ignored += 1
            continue
        profile = raw.get("method")
        if profile not in known_profiles:
            for name in issues:
                issues[name].append(f"row[{ordinal}]: unexpected profile {profile!r}")
            continue
        try:
            domain = raw.get("domain")
            seed = _parse_wire_int(raw.get("seed"))
            if domain not in domains or seed not in seeds:
                raise ValueError("domain or seed outside frozen scope")
            expected_text = {
                "implementation": scope["implementation"],
                "initial_design": scope["initial_design"],
                "decision_backend": scope["decision_backend"],
                "variant": scope["variant_template"].format(profile=profile),
                "structural_prior_profile": profile,
            }
            for field, expected in expected_text.items():
                if raw.get(field) != expected:
                    raise ValueError(f"{field} differs from frozen scope")
            for field in ("d", "N", "n0", "source_calls"):
                if _parse_wire_int(raw.get(field)) != scope[field]:
                    raise ValueError(f"{field} differs from frozen scope")
            normalized_fixed = {}
            for field, expected in scope["fixed_row_values"].items():
                observed = _parse_fixed_value(raw.get(field), expected)
                if observed != expected:
                    raise ValueError(f"{field} differs from frozen scope")
                normalized_fixed[field] = observed
            status = raw.get("status")
            if type(status) is not str or not status:
                raise ValueError("status is not non-empty text")
            adaptive_loss = _parse_wire_bool(raw.get("adaptive_loss"))
            true_feasible = _parse_wire_bool(raw.get("true_feasible"))
            raw_regret = raw.get("feasible_regret")
            parsed_regret = None
            if raw_regret not in (None, ""):
                parsed_regret = _parse_finite(raw_regret)
                if parsed_regret < -float(contract["gate"]["regret_tolerance"]):
                    raise ValueError("feasible_regret is negative")
                if parsed_regret < 0.0:
                    parsed_regret = 0.0
            if status == "ok" and true_feasible and parsed_regret is None:
                raise ValueError("feasible row lacks finite feasible_regret")
            regret = parsed_regret if status == "ok" and true_feasible else None
            key = (profile, domain, seed)
            if key in index:
                raise ValueError("duplicate evidence cell")
            index[key] = {
                "profile": profile,
                "domain": domain,
                "seed": seed,
                "status": status,
                "adaptive_loss": adaptive_loss,
                "true_feasible": bool(status == "ok" and true_feasible),
                "feasible_regret": regret,
                "fixed_row_values": normalized_fixed,
            }
        except (TypeError, ValueError) as exc:
            issues[profile].append(f"row[{ordinal}]: {exc}")
    return index, issues, ignored


def _evaluate(
    spec: HypothesisSpec,
    index: Mapping[tuple[str, str, int], Mapping[str, Any]],
    issues: Mapping[str, list[str]],
    contract: Mapping[str, Any],
) -> HypothesisDecision:
    scope = contract["evidence_scope"]
    gate = contract["gate"]
    profiles = (spec.challenger_profile, spec.reference_profile)
    invalid = tuple(sorted({item for profile in profiles for item in issues[profile]}))
    missing = tuple(
        _cell(profile, domain, seed, scope)
        for profile in profiles
        for domain in scope["domains"]
        for seed in scope["seeds"]
        if (profile, domain, seed) not in index
    )
    evidence_projection = {
        "hypothesis": spec.to_dict(),
        "rows": [
            index[key]
            for key in sorted(index)
            if key[0] in profiles
        ],
        "missing": missing,
        "invalid": invalid,
    }
    evidence_digest = _digest(evidence_projection)
    if invalid:
        return HypothesisDecision(
            spec.hypothesis_id, Disposition.INVALID_EVIDENCE,
            ("evidence_integrity_failed",), {}, missing, invalid, evidence_digest,
        )
    if missing:
        return HypothesisDecision(
            spec.hypothesis_id, Disposition.NEEDS_EVIDENCE,
            ("comparison_cells_missing",), {}, missing, (), evidence_digest,
        )

    non_ok_cells = []
    for profile in profiles:
        for domain in scope["domains"]:
            for seed in scope["seeds"]:
                row = index[(profile, domain, seed)]
                if row["status"] != "ok":
                    cell = _cell(profile, domain, seed, scope)
                    cell["observed_status"] = row["status"]
                    non_ok_cells.append(cell)
    if non_ok_cells:
        return HypothesisDecision(
            spec.hypothesis_id,
            Disposition.NEEDS_EVIDENCE,
            ("comparison_execution_incomplete",),
            {"non_ok_row_count": len(non_ok_cells)},
            tuple(non_ok_cells),
            (),
            evidence_digest,
        )

    challenger_rows = [
        index[(spec.challenger_profile, domain, seed)]
        for domain in scope["domains"] for seed in scope["seeds"]
    ]
    failed_rows = sum(row["status"] != "ok" for row in challenger_rows)
    feasible_count = sum(row["true_feasible"] for row in challenger_rows)
    adaptive_losses = sum(row["adaptive_loss"] for row in challenger_rows)
    per_domain = {
        domain: sum(
            index[(spec.challenger_profile, domain, seed)]["true_feasible"]
            for seed in scope["seeds"]
        )
        for domain in scope["domains"]
    }
    safety_pass = bool(
        feasible_count >= gate["min_overall_feasible"]
        and all(value >= gate["min_per_domain_feasible"] for value in per_domain.values())
        and failed_rows <= gate["max_failed_rows"]
        and adaptive_losses <= gate["max_adaptive_losses"]
    )

    feasibility_wins = feasibility_losses = feasibility_ties = 0
    regret_wins = regret_losses = regret_ties = 0
    tolerance = float(gate["regret_tolerance"])
    for domain in scope["domains"]:
        for seed in scope["seeds"]:
            challenger = index[(spec.challenger_profile, domain, seed)]
            reference = index[(spec.reference_profile, domain, seed)]
            delta = int(challenger["true_feasible"]) - int(reference["true_feasible"])
            feasibility_wins += delta > 0
            feasibility_losses += delta < 0
            feasibility_ties += delta == 0
            if challenger["true_feasible"] and reference["true_feasible"]:
                change = challenger["feasible_regret"] - reference["feasible_regret"]
                regret_wins += change < -tolerance
                regret_losses += change > tolerance
                regret_ties += abs(change) <= tolerance
    feasibility_net = feasibility_wins - feasibility_losses
    paired_pass = bool(
        feasibility_net > 0
        or (feasibility_net == 0 and regret_wins > regret_losses)
    )
    metrics = {
        "expected_pairs": len(scope["domains"]) * len(scope["seeds"]),
        "challenger_safety": {
            "pass": safety_pass,
            "true_feasible": feasible_count,
            "per_domain_true_feasible": per_domain,
            "failed_rows": failed_rows,
            "adaptive_loss_count": adaptive_losses,
        },
        "paired_feasibility": {
            "pass": paired_pass,
            "wins": feasibility_wins,
            "losses": feasibility_losses,
            "ties": feasibility_ties,
            "net": feasibility_net,
        },
        "conditional_regret": {
            "wins": regret_wins,
            "losses": regret_losses,
            "ties": regret_ties,
            "used_only_after_feasibility_tie": True,
        },
        "source_archive_match_verified": False,
    }
    supported = safety_pass and paired_pass
    reasons: list[str] = []
    if not safety_pass:
        reasons.append("challenger_safety_gate_failed")
    if feasibility_net < 0:
        reasons.append("paired_feasibility_net_negative")
    elif feasibility_net == 0 and regret_wins <= regret_losses:
        reasons.append("paired_feasibility_tie_without_regret_win")
    if supported:
        reasons.extend(("challenger_safety_gate_passed", "paired_effect_gate_passed"))
    return HypothesisDecision(
        spec.hypothesis_id,
        Disposition.SUPPORTED_SCOPED if supported else Disposition.REFUTED_SCOPED,
        tuple(reasons), metrics, (), (), evidence_digest,
    )


def _audit_chain(
    hypotheses: Sequence[HypothesisSpec],
    decisions: Sequence[HypothesisDecision],
    report_body_digest: str,
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    events: list[dict[str, Any]] = []
    previous = _GENESIS_HASH
    for spec, decision in zip(hypotheses, decisions):
        proposal = {
            "seq": len(events),
            "event_type": "HYPOTHESIS_PROPOSED",
            "state_before": (
                "INIT" if spec.generation == 0 else "REVISING"
            ),
            "state_after": "PROPOSED",
            "hypothesis_id": spec.hypothesis_id,
            "parent_ids": list(spec.parent_ids),
            "generation": spec.generation,
            "proposal_operation": spec.proposal_operation,
            "trigger_disposition": spec.trigger_disposition,
            "hypothesis_digest": _digest(spec.to_dict()),
            "prev_hash": previous,
        }
        proposal_hash = _digest(proposal)
        proposal["event_hash"] = proposal_hash
        events.append(proposal)
        previous = proposal_hash
        event = {
            "seq": len(events),
            "event_type": "HYPOTHESIS_DECIDED",
            "state_before": "PROPOSED",
            "state_after": decision.disposition.value,
            "hypothesis_id": decision.hypothesis_id,
            "generation": spec.generation,
            "evidence_digest": decision.evidence_digest,
            "decision_digest": _digest(decision.to_dict()),
            "reason_codes": list(decision.reason_codes),
            "prev_hash": previous,
        }
        event_hash = _digest(event)
        event["event_hash"] = event_hash
        events.append(event)
        previous = event_hash
    final = {
        "seq": len(events),
        "event_type": "LOOP_COMPLETED",
        "state_before": "EVALUATING",
        "state_after": "FINITE_GRAPH_EXHAUSTED",
        "hypothesis_count": len(decisions),
        "report_body_digest": report_body_digest,
        "prev_hash": previous,
    }
    final_hash = _digest(final)
    final["event_hash"] = final_hash
    events.append(final)
    return tuple(events), final_hash


def verify_audit_chain(events: Sequence[Mapping[str, Any]], expected_head: str) -> bool:
    previous = _GENESIS_HASH
    for seq, observed in enumerate(events):
        if not isinstance(observed, Mapping) or observed.get("seq") != seq:
            return False
        if observed.get("prev_hash") != previous:
            return False
        event = dict(observed)
        event_hash = event.pop("event_hash", None)
        if event_hash != _digest(event):
            return False
        previous = event_hash
    return bool(events) and previous == expected_head


def verify_report_integrity(report: Mapping[str, Any]) -> bool:
    """Check the V1 report-body commitment and its internal event chain.

    This detects a report that changed without updating its recorded hashes. It
    is not an authority signature: a party able to rewrite the whole report can
    also recompute an unanchored hash chain.
    """
    if not isinstance(report, Mapping):
        return False
    audit = report.get("audit")
    if not isinstance(audit, Mapping):
        return False
    required_audit = {
        "algorithm", "genesis", "head", "report_body_digest", "events",
    }
    if set(audit) != required_audit:
        return False
    if (
        audit.get("algorithm") != "sha256-canonical-json-chain-v1"
        or audit.get("genesis") != _GENESIS_HASH
        or not isinstance(audit.get("events"), list)
    ):
        return False
    body = {key: report[key] for key in report if key != "audit"}
    if _digest(body) != audit.get("report_body_digest"):
        return False
    if not verify_audit_chain(audit["events"], audit.get("head")):
        return False
    final = audit["events"][-1] if audit["events"] else None
    if not isinstance(final, Mapping):
        return False
    if final.get("report_body_digest") != audit.get("report_body_digest"):
        return False
    hypotheses = report.get("hypotheses")
    decisions = report.get("decisions")
    if not isinstance(hypotheses, list) or not isinstance(decisions, list):
        return False
    proposal_events = [
        event for event in audit["events"]
        if event.get("event_type") == "HYPOTHESIS_PROPOSED"
    ]
    decision_events = [
        event for event in audit["events"]
        if event.get("event_type") == "HYPOTHESIS_DECIDED"
    ]
    if len(proposal_events) != len(hypotheses) or len(decision_events) != len(decisions):
        return False
    if any(
        event.get("hypothesis_digest") != _digest(item)
        for event, item in zip(proposal_events, hypotheses)
    ):
        return False
    if any(
        event.get("decision_digest") != _digest(item)
        for event, item in zip(decision_events, decisions)
    ):
        return False
    return True


def _component_interpretation(standalone: Disposition, necessity: Disposition) -> str:
    if Disposition.INVALID_EVIDENCE in (standalone, necessity):
        return "INVALID_EVIDENCE"
    if Disposition.NEEDS_EVIDENCE in (standalone, necessity):
        if standalone is Disposition.REFUTED_SCOPED:
            return "STANDALONE_REFUTED_NECESSITY_UNKNOWN"
        if standalone is Disposition.SUPPORTED_SCOPED:
            return "STANDALONE_SUPPORTED_NECESSITY_UNKNOWN"
        return "INCOMPLETE_EVIDENCE"
    if standalone is Disposition.SUPPORTED_SCOPED and necessity is Disposition.SUPPORTED_SCOPED:
        return "SUPPORTED_STANDALONE_AND_NECESSARY"
    if standalone is Disposition.SUPPORTED_SCOPED:
        return "USEFUL_ALONE_NOT_NECESSARY_IN_FULL"
    if necessity is Disposition.SUPPORTED_SCOPED:
        return "INTERACTION_DEPENDENT"
    return "UNSUPPORTED_IN_SCOPE"


def run_structural_hypothesis_loop(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    input_artifacts: Mapping[str, Any] | None = None,
) -> LoopResult:
    """Run the bounded verdict-driven V1 loop and return an audit report."""
    validate_contract(contract)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("rows must be a sequence of objects")
    if input_artifacts is not None and not isinstance(input_artifacts, Mapping):
        raise ValueError("input_artifacts must be an object or None")
    index, issues, ignored_rows = _index_rows(rows, contract)
    frontier = [propose_initial_hypothesis(contract)]
    hypothesis_items: list[HypothesisSpec] = []
    decision_items: list[HypothesisDecision] = []
    visited: set[str] = set()
    while frontier:
        hypothesis = frontier.pop(0)
        if hypothesis.hypothesis_id in visited:
            raise RuntimeError("hypothesis revision produced a cycle")
        visited.add(hypothesis.hypothesis_id)
        decision = _evaluate(hypothesis, index, issues, contract)
        hypothesis_items.append(hypothesis)
        decision_items.append(decision)
        frontier.extend(revise_hypothesis(hypothesis, decision, contract))
    hypotheses = tuple(hypothesis_items)
    decisions = tuple(decision_items)
    counts = {item.value: 0 for item in Disposition}
    for decision in decisions:
        counts[decision.disposition.value] += 1

    pending_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for decision in decisions:
        if decision.disposition is not Disposition.NEEDS_EVIDENCE:
            continue
        for cell in decision.missing_cells:
            key = (cell["profile"], cell["domain"], cell["seed"])
            pending_index[key] = dict(cell)
    scope = contract["evidence_scope"]
    order_profile = {name: index for index, name in enumerate(_PROFILE_NAMES)}
    order_domain = {name: index for index, name in enumerate(scope["domains"])}
    pending = tuple(
        pending_index[key]
        for key in sorted(
            pending_index,
            key=lambda item: (order_profile[item[0]], order_domain[item[1]], item[2]),
        )
    )

    decision_by_id = {item.hypothesis_id: item for item in decisions}
    spec_by_kind_component = {(item.kind, item.component): item for item in hypotheses}
    root = spec_by_kind_component[("COMPOSITE", None)]
    synthesis_components = {}
    for component in PRIOR_COMPONENTS:
        standalone = decision_by_id[
            spec_by_kind_component[("STANDALONE", component)].hypothesis_id
        ].disposition
        necessity = decision_by_id[
            spec_by_kind_component[("NECESSITY", component)].hypothesis_id
        ].disposition
        synthesis_components[component] = {
            "standalone": standalone.value,
            "necessity": necessity.value,
            "interpretation": _component_interpretation(standalone, necessity),
        }
    synthesis = {
        "composite": decision_by_id[root.hypothesis_id].disposition.value,
        "components": synthesis_components,
        "ignored_out_of_scope_rows": ignored_rows,
        "revision_policy": "VERDICT_DRIVEN_BOUNDED_V1",
    }
    if counts[Disposition.INVALID_EVIDENCE.value]:
        status = "COMPLETED_WITH_INVALID_EVIDENCE"
    elif counts[Disposition.NEEDS_EVIDENCE.value]:
        status = "COMPLETED_WITH_EVIDENCE_GAPS"
    else:
        status = "COMPLETED"
    contract_digest = _digest(contract)
    evidence_digest = _digest(rows)
    stop_reason = "FINITE_GRAPH_EXHAUSTED"
    evidence_scope = dict(contract["evidence_scope"])
    gate = dict(contract["gate"])
    nonclaims = tuple(contract["nonclaims"])
    artifact_snapshot = (
        None if input_artifacts is None else _stable_value(input_artifacts)
    )
    body = _report_body_payload(
        contract_id=contract["contract_id"],
        contract_digest=contract_digest,
        evidence_digest=evidence_digest,
        status=status,
        stop_reason=stop_reason,
        evidence_scope=evidence_scope,
        gate=gate,
        hypotheses=hypotheses,
        decisions=decisions,
        verdict_counts=counts,
        pending_evidence=pending,
        synthesis=synthesis,
        nonclaims=nonclaims,
        input_artifacts=artifact_snapshot,
    )
    report_body_digest = _digest(body)
    audit_events, audit_head = _audit_chain(
        hypotheses, decisions, report_body_digest
    )
    return LoopResult(
        contract_id=contract["contract_id"],
        contract_digest=contract_digest,
        evidence_digest=evidence_digest,
        status=status,
        stop_reason=stop_reason,
        hypotheses=hypotheses,
        decisions=decisions,
        verdict_counts=counts,
        pending_evidence=pending,
        synthesis=synthesis,
        audit_events=audit_events,
        audit_head=audit_head,
        evidence_scope=evidence_scope,
        gate=gate,
        nonclaims=nonclaims,
        input_artifacts=artifact_snapshot,
        report_body_digest=report_body_digest,
    )


__all__ = [
    "ContractValidationError",
    "Disposition",
    "HypothesisDecision",
    "HypothesisSpec",
    "LoopResult",
    "REQUIRED_EVIDENCE_FIELDS",
    "SCHEMA_VERSION",
    "canonical_json_bytes",
    "propose_initial_hypothesis",
    "propose_hypotheses",
    "revise_hypothesis",
    "run_structural_hypothesis_loop",
    "validate_contract",
    "verify_audit_chain",
    "verify_report_integrity",
]
