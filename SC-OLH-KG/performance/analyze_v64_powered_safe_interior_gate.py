#!/usr/bin/env python3
"""Audit V64 powered safe-interior verification against V51."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "v63_analysis_engine_v64",
    HERE / "analyze_v63_safe_interior_gate.py",
)
V63 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V63)

CONTROL = V63.CONTROL
CHALLENGER = "v64_powered_safe_interior_verification"
IMPLEMENTATION_CONTRACT_ID = (
    "v64_powered_cumulative_risk_safe_interior_verification")
THEORY_CONTRACT_ID = (
    "v64_frozen_selector_powered_familywise_quantile_certificate_v1")
REPORT_SCOPE = "v64_powered_cumulative_risk_safe_interior_gate"
SEARCH_BUDGET = V63.SEARCH_BUDGET
PRIMARY_BUDGET = 80
FALLBACK_BUDGET = 96
SHORTLIST_SIZE = V63.SHORTLIST_SIZE
FAMILYWISE_DELTA = V63.FAMILYWISE_DELTA
PER_CANDIDATE_DELTA = V63.PER_CANDIDATE_DELTA
SHORTLIST_MODE = V63.SHORTLIST_MODE
SELECTION_CONTRACT = V63.SELECTION_CONTRACT
ATTEMPT_METHOD = V63.ATTEMPT_METHOD
PROTOCOL_METHOD = V63.PROTOCOL_METHOD
V56 = V63.V56


def _configure_engine():
    V63.CHALLENGER = CHALLENGER
    V63.IMPLEMENTATION_CONTRACT_ID = IMPLEMENTATION_CONTRACT_ID
    V63.THEORY_CONTRACT_ID = THEORY_CONTRACT_ID
    V63.REPORT_SCOPE = REPORT_SCOPE
    V63.PRIMARY_BUDGET = PRIMARY_BUDGET
    V63.FALLBACK_BUDGET = FALLBACK_BUDGET


def analyze(
    control_root,
    challenger_root,
    *,
    seed_start=60,
    expected_seeds=20,
):
    _configure_engine()
    return V63.analyze(
        control_root,
        challenger_root,
        seed_start=seed_start,
        expected_seeds=expected_seeds,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control_root", type=Path)
    parser.add_argument("challenger_root", type=Path)
    parser.add_argument("--seed-start", type=int, default=60)
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.control_root,
        args.challenger_root,
        seed_start=args.seed_start,
        expected_seeds=args.expected_seeds,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["formal_gate_passed"] else 1)


if __name__ == "__main__":
    main()
