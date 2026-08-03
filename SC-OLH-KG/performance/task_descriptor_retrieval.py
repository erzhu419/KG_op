"""Target-label-free source retrieval from observable task descriptors.

The descriptor is intentionally coarse.  It records simulator schema and
state/exposure roles, never target objective values, constraint values, or
oracle labels.  The same weighted set distance is used for every target.
"""

from __future__ import annotations

from dataclasses import dataclass


SOURCE_DOMAIN_POOL = (
    "FactorShockStatePolicyRZDT1",
    "InventorySupplyChain",
    "QueueResourceControl",
)
TRAFFIC_TARGET = "Ingolstadt21Traffic"
ENERGY_TARGET = "OPSDStorageReliability"

DOMAIN_BLIND_CONTROL = "domain_blind_exclude_nearest"
DESCRIPTOR_NEAREST = "descriptor_nearest"
SOURCE_SELECTION_MODES = (
    DOMAIN_BLIND_CONTROL,
    DESCRIPTOR_NEAREST,
)

_ROLE_WEIGHTS = {
    "bounded_integer_policy": 0.5,
    "static_policy_response": 2.0,
    "event_driven_dynamics": 2.0,
    "network_flow_balance": 2.0,
    "queue_state": 2.0,
    "stock_state": 1.5,
    "shared_exogenous_shock": 1.5,
    "state_action_exposure": 1.5,
    "cumulative_safety_output": 1.0,
}

_OBSERVABLE_ROLES = {
    "FactorShockStatePolicyRZDT1": frozenset({
        "bounded_integer_policy",
        "static_policy_response",
        "shared_exogenous_shock",
        "state_action_exposure",
        "cumulative_safety_output",
    }),
    "InventorySupplyChain": frozenset({
        "bounded_integer_policy",
        "event_driven_dynamics",
        "network_flow_balance",
        "stock_state",
        "shared_exogenous_shock",
        "state_action_exposure",
        "cumulative_safety_output",
    }),
    "QueueResourceControl": frozenset({
        "bounded_integer_policy",
        "event_driven_dynamics",
        "network_flow_balance",
        "queue_state",
        "shared_exogenous_shock",
        "state_action_exposure",
        "cumulative_safety_output",
    }),
    "Ingolstadt21Traffic": frozenset({
        "bounded_integer_policy",
        "event_driven_dynamics",
        "network_flow_balance",
        "queue_state",
        "shared_exogenous_shock",
        "state_action_exposure",
        "cumulative_safety_output",
    }),
    "OPSDStorageReliability": frozenset({
        "bounded_integer_policy",
        "event_driven_dynamics",
        "network_flow_balance",
        "stock_state",
        "shared_exogenous_shock",
        "state_action_exposure",
        "cumulative_safety_output",
    }),
}


@dataclass(frozen=True)
class SourceSelection:
    mode: str
    track: str
    target_domain: str
    source_domains: tuple[str, ...]
    source_split_heldout: str
    heldout_task_family_identifier_used: bool
    ranking: tuple[dict, ...]

    def as_dict(self):
        return {
            "schema_version": 1,
            "mode": self.mode,
            "track": self.track,
            "target_domain": self.target_domain,
            "source_domains": list(self.source_domains),
            "source_split_heldout": self.source_split_heldout,
            "heldout_task_family_identifier_used": (
                self.heldout_task_family_identifier_used
            ),
            "target_observable_roles": sorted(
                _OBSERVABLE_ROLES[self.target_domain]),
            "role_weights": dict(sorted(_ROLE_WEIGHTS.items())),
            "ranking": [dict(row) for row in self.ranking],
            "target_outcomes_used": False,
            "target_oracle_used": False,
        }


def observable_roles(domain):
    try:
        return _OBSERVABLE_ROLES[str(domain)]
    except KeyError as exc:
        raise ValueError(f"no observable descriptor for {domain!r}") from exc


def weighted_role_distance(left, right):
    """Weighted Jaccard distance over declared, observable role sets."""

    left_roles = observable_roles(left)
    right_roles = observable_roles(right)
    union = left_roles | right_roles
    if not union:
        return 0.0
    numerator = sum(
        _ROLE_WEIGHTS[role] for role in left_roles ^ right_roles)
    denominator = sum(_ROLE_WEIGHTS[role] for role in union)
    return float(numerator / denominator)


def rank_source_domains(target_domain, source_pool=SOURCE_DOMAIN_POOL):
    rows = []
    for domain in source_pool:
        rows.append({
            "domain": str(domain),
            "distance": weighted_role_distance(target_domain, domain),
            "observable_roles": sorted(observable_roles(domain)),
        })
    return tuple(sorted(
        rows,
        key=lambda row: (float(row["distance"]), str(row["domain"])),
    ))


def source_selection_contract(
    mode,
    *,
    target_domain=TRAFFIC_TARGET,
    source_pool=SOURCE_DOMAIN_POOL,
    source_count=2,
):
    mode = str(mode)
    source_pool = tuple(map(str, source_pool))
    if set(source_pool) != set(SOURCE_DOMAIN_POOL):
        raise ValueError(
            "external-domain source pool must be the registered three domains")
    if int(source_count) != 2:
        raise ValueError("external-domain contract requires exactly two sources")

    if mode == DOMAIN_BLIND_CONTROL:
        source_domains = (
            "FactorShockStatePolicyRZDT1",
            "InventorySupplyChain",
        )
        heldout = "QueueResourceControl"
        ranking = ()
        track = "domain_blind_external_holdout"
        uses_family = False
    elif mode == DESCRIPTOR_NEAREST:
        ranking = rank_source_domains(target_domain, source_pool)
        source_domains = tuple(
            row["domain"] for row in ranking[: int(source_count)])
        excluded = sorted(set(source_pool) - set(source_domains))
        if len(excluded) != 1:
            raise RuntimeError("descriptor retrieval did not define one split")
        heldout = excluded[0]
        track = "descriptor_conditional_external_holdout"
        uses_family = True
    else:
        raise ValueError(
            f"unknown source selection mode {mode!r}; "
            f"expected one of {SOURCE_SELECTION_MODES}")

    return SourceSelection(
        mode=mode,
        track=track,
        target_domain=str(target_domain),
        source_domains=source_domains,
        source_split_heldout=heldout,
        heldout_task_family_identifier_used=uses_family,
        ranking=ranking,
    )


def traffic_method_label(mode):
    if str(mode) == DOMAIN_BLIND_CONTROL:
        return "PaperFinal-DomainBlindProposal-SAAS"
    if str(mode) == DESCRIPTOR_NEAREST:
        return "PaperFinal-DescriptorProposal-SAAS"
    raise ValueError(f"unknown source selection mode {mode!r}")
