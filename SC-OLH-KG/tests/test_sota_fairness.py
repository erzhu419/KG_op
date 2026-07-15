from pathlib import Path
from types import SimpleNamespace

from performance.benchmark_sota_fairness import (
    oracle_free_lodo_config,
    run_one,
    source_archive_cost,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "performance/manifests/v18b_exactkg_mcdiag.json"


def _args(tmp_path, protocol):
    return SimpleNamespace(
        protocol=protocol,
        method="botorch_turbo",
        heldout="QueueResourceControl",
        seed=3,
        manifest=str(MANIFEST),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        target_budget=5,
        d=8,
        L=100,
        sigma=0.04,
        alpha=0.05,
        weights="0.5,0.5",
        n0=4,
        beta_g=2.0,
        ts_candidates=32,
        raw_samples=8,
        num_restarts=2,
        maxiter=10,
        candidate_timeout_sec=30.0,
        saas_warmup_steps=4,
        saas_num_samples=4,
        saas_thinning=1,
        saas_max_tree_depth=2,
        saas_mc_samples=16,
    )


def test_oracle_free_source_archive_cost_is_384():
    config = oracle_free_lodo_config(MANIFEST)
    assert source_archive_cost(config, "QueueResourceControl") == 384
    assert config["meta_source_observation_mode"] == "replicated"
    assert config["meta_source_observation_replicates"] == 3


def test_target_only_and_archive_shared_protocols_have_auditable_designs(tmp_path):
    target = run_one(_args(tmp_path, "target_n20"))
    assert target["status"] == "ok"
    assert target["information_contract"]["offline_source_calls"] == 0
    assert target["result"]["initial_design"] == "sobol"

    shared = run_one(_args(tmp_path, "shared_archive_n20"))
    assert shared["status"] == "ok"
    assert shared["information_contract"]["offline_source_calls"] == 384
    assert shared["information_contract"]["source_oracle_aided"] is False
    assert shared["source_archive_fingerprint"]
    assert shared["initial_points_fingerprint"]
    assert shared["result"]["initial_design"] == "shared_external"
