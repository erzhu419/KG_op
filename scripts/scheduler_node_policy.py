"""Shared scheduler node policy for KG_op submission helpers."""

from __future__ import annotations


DEFAULT_CPU_NODES = (
    "node001",
    "node002",
    "node003",
    "node004",
    "node005",
    "node006",
)
FORBIDDEN_CPU_NODES = {"jtl110cpu", "jtl110cpu2"}


def parse_csv(text):
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def parse_cpu_nodes(text):
    nodes = parse_csv(text)
    if not nodes:
        nodes = list(DEFAULT_CPU_NODES)
    forbidden = sorted(FORBIDDEN_CPU_NODES.intersection(nodes))
    if forbidden:
        raise SystemExit(
            "Forbidden CPU scheduler nodes requested: "
            + ",".join(forbidden)
            + ". Use node001-node006 for KG/SUMO CPU tasks."
        )
    return nodes


def allowed_node_flags(nodes):
    flags = []
    for node in nodes:
        flags.extend(["--allowed-node", node])
    return flags
