from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO
    / "scripts/submit_scolhkg_v53_constrained_certificate_gate_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location("v53_submit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _args(tmp_path, variants):
    defaults = MODULE.closure._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(MODULE.CPU_NODES),
        "deploy": deploy,
        "python": defaults.REMOTE_PYTHON,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": MODULE.DEFAULT_SOURCE_RUN_ID,
        "remote_design_only": True,
        "run_id": "v53-test",
        "rank": 4,
        "source_d": 50,
        "source_records_per_domain": 64,
        "d": 1000,
        "N": 11,
        "n0": 10,
        "seed_start": 100,
        "n_seeds": 2,
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "variants": ",".join(variants),
        "exact_mc_samples": 8,
        "exact_sampling_mode": "antithetic_nested",
        "risk_eta": 0.01,
        "certificate_eta": 0.02,
        "cpu": 12,
        "ram_mb": 8192,
    })()


def test_v53_fidelity_gate_is_paired_mc8_mc32(tmp_path):
    variants = MODULE.FIDELITY
    specs = MODULE.build_specs(_args(tmp_path, variants))
    assert len(specs) == 2 * 3 * 2
    by_variant = {
        variant: [
            spec for spec in specs if f"/{variant}/" in spec["signature"]
        ]
        for variant in variants
    }
    assert all(len(rows) == 6 for rows in by_variant.values())
    assert all("--exact-mc-samples 8 " in spec["cmd"]
               for spec in by_variant["v53_mc8"])
    assert all("--exact-mc-samples 32 " in spec["cmd"]
               for spec in by_variant["v53_mc32"])
    assert all("--exact-sampling-mode antithetic_nested" in spec["cmd"]
               for spec in specs)
    assert all("--policy-improvement-mode certificate_constrained"
               in spec["cmd"] for spec in specs)
    assert all("--policy-improvement-certificate-mc-error-bound 0.02"
               in spec["cmd"] for spec in specs)
    assert all("--policy-improvement-rollout-max-arms 0"
               in spec["cmd"] for spec in specs)
    assert all("--runtime-checkpoint-interval 0" in spec["cmd"]
               for spec in specs)
    assert all(spec["allowed_nodes"] == list(MODULE.CPU_NODES)
               for spec in specs)
    assert all(spec["cpu"] == 12 and spec["vram"] == 0 for spec in specs)


def test_v53_sentinel_keeps_literal_v51_and_v52_controls(tmp_path):
    variants = (MODULE.CONTROL, MODULE.V52, MODULE.V53)
    args = _args(tmp_path, variants)
    args.N = 20
    args.seed_start = 0
    specs = MODULE.build_specs(args)
    assert len(specs) == 3 * 3 * 2
    by_variant = {
        variant: [
            spec for spec in specs if f"/{variant}/" in spec["signature"]
        ]
        for variant in variants
    }
    assert all("--policy-improvement-mode off" in spec["cmd"]
               for spec in by_variant[MODULE.CONTROL])
    assert all("--policy-improvement-mode action_superset" in spec["cmd"]
               for spec in by_variant[MODULE.V52])
    assert all("--policy-improvement-mode certificate_constrained"
               in spec["cmd"] for spec in by_variant[MODULE.V53])
    assert all(
        "--implementation-contract-id v53_constrained_certificate_deficit"
        in spec["cmd"] for spec in by_variant[MODULE.V53]
    )
    assert all(
        "--theory-contract-id v53_constrained_certificate_deficit_v1"
        in spec["cmd"] for spec in by_variant[MODULE.V53]
    )


def _write_frozen_design(path, seeds):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "designs": {str(seed): [] for seed in seeds},
    }), encoding="utf-8")


def test_v53_preflight_rejects_missing_frozen_design_seed(tmp_path):
    args = _args(tmp_path, ("v53_mc8",))
    specs = MODULE.build_specs(args)
    for spec in specs:
        remote = MODULE._command_option(spec["cmd"], "--initial-design-file")
        local = MODULE._local_deploy_path(args, remote)
        _write_frozen_design(local, (0, 1))

    with pytest.raises(ValueError, match="missing requested seeds.*100"):
        MODULE.validate_frozen_design_seed_coverage(args, specs)


def test_v53_preflight_accepts_covered_frozen_design_seeds(tmp_path):
    args = _args(tmp_path, ("v53_mc8",))
    args.seed_start = 0
    specs = MODULE.build_specs(args)
    for spec in specs:
        remote = MODULE._command_option(spec["cmd"], "--initial-design-file")
        local = MODULE._local_deploy_path(args, remote)
        _write_frozen_design(local, (0, 1))

    coverage = MODULE.validate_frozen_design_seed_coverage(args, specs)
    assert len(coverage) == 3
    assert all(seeds == [0, 1] for seeds in coverage.values())
