import importlib.util
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "SC-OLH-KG"))

from algorithms.single_olhkg import SingleOLHKGAlgorithm, SingleOLHKGConfig
from problems.rzdt import RZDT1
from problems.single_objective import ScalarizedProblem


ANALYZE_PATH = (
    REPO / "SC-OLH-KG/performance/analyze_mean_alignment_v40_gate.py")
sys.path.insert(0, str(ANALYZE_PATH.parent))
ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "mean_v40_analyze", ANALYZE_PATH)
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
ANALYZE_SPEC.loader.exec_module(analyze)

SUBMIT_PATH = (
    REPO / "scripts/submit_scolhkg_mean_alignment_v40_robust_terminal_gate_scheduler.py"
)
SUBMIT_SPEC = importlib.util.spec_from_file_location(
    "mean_v40_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
SUBMIT_SPEC.loader.exec_module(submit)


def _args(tmp_path):
    defaults = submit._root_module()
    deploy = tmp_path / "deploy"
    return type("Args", (), {
        "nodes": ",".join(submit.CPU_NODES),
        "variants": ",".join(submit.VARIANTS),
        "deploy": deploy,
        "manifest": deploy / "SC-OLH-KG/performance/manifests/base.json",
        "source_run_id": defaults.DEFAULT_SOURCE_RUN_ID,
        "run_id": "mean-v40",
        "rank": 4,
        "source_d": 50,
        "d": 1000,
        "N": 20,
        "n0": 10,
        "seed_start": 0,
        "n_seeds": 1,
        "pool_size": 512,
        "variance_audit_size": 512,
        "misspecification_prior_df": 4.0,
        "misspecification_ridge": 1.0,
        "misspecification_max_scale": 100.0,
        "misspecification_delta": 0.05,
        "contrast_scale": 1.0,
        "null_geometry_ridge": 1e-3,
        "python": defaults.REMOTE_PYTHON,
        "cpu": 12,
        "ram_mb": 8192,
    })()


def test_v40_submitter_closes_robust_terminal_contract(tmp_path):
    specs = submit.build_specs(_args(tmp_path))
    assert len(specs) == 4 * 3
    for variant, samples in (
        ("v40_robust_lex_mc2", 2),
        ("v40_robust_lex_mc4", 4),
    ):
        selected = [
            spec for spec in specs if f"/{variant}/" in spec["signature"]]
        assert len(selected) == 3
        for spec in selected:
            command = spec["cmd"]
            assert "--exact-terminal-mode certified_lexicographic" in command
            assert "--decision-contract-mode certified_lexicographic" in command
            assert "--finalist-terminal-value-mode certified_lexicographic" in command
            assert "--no-posterior-dominance-enabled" in command
            assert "--no-exact-clip-negative" in command
            assert f"--exact-mc-samples {samples}" in command


def test_v40_robust_contract_requires_shared_terminal_rule():
    row = {
        "gate_variant": "v40_robust_lex_mc2",
        "decision_contract_mode": "certified_lexicographic",
        "decision_backend_terminal_rule": "robust_certified_lexicographic",
        "posterior_dominance_enabled": False,
        "finalist_replication": {"mathematically_closed": True},
        "adaptive_replication_voi": {"target_oracle_used": False},
        "exact_kg_diagnostics": {
            "mc_samples": 2,
            "ranking_uses_signed_values": True,
        },
    }
    assert analyze._robust_contract([row], "v40_robust_lex_mc2", 2)
    row["decision_backend_terminal_rule"] = "posterior_bayes_risk"
    assert not analyze._robust_contract([row], "v40_robust_lex_mc2", 2)


def test_terminal_value_remains_lexicographic():
    values = np.asarray([
        [1.0, 0.20, -10.0],
        [1.0, 0.10, 100.0],
        [0.0, 0.00, 5.0],
    ])
    assert SingleOLHKGAlgorithm._terminal_value_index(values) == 2


def test_v40_exact_backend_uses_same_robust_terminal_rule_at_recommendation():
    problem = ScalarizedProblem(RZDT1(d=3, L=20, sigma=0.03))
    result = SingleOLHKGAlgorithm(
        problem,
        SingleOLHKGConfig(
            N=5,
            n0=4,
            K1=4,
            K2=0,
            decision_backend="sobol_exact_joint_voi",
            adaptive_replication_voi=True,
            replication_candidate_count=2,
            exact_kg_mc_samples=2,
            exact_kg_sampling_mode="antithetic",
            exact_kg_clip_negative=False,
            exact_kg_terminal_mode="certified_lexicographic",
            decision_contract_mode="certified_lexicographic",
            finalist_terminal_value_mode="certified_lexicographic",
            finalist_empirical_override="off",
            posterior_dominance_enabled=False,
            eval_pool_size=12,
            seed=409,
        ),
    ).run()
    assert result["decision_backend_terminal_rule"] == (
        "robust_certified_lexicographic")
    assert result["finalist_replication"]["mathematically_closed"]
