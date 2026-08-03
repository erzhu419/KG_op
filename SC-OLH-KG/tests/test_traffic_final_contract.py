import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SC-OLH-KG"))

from performance.analyze_traffic_final_contract import (  # noqa: E402
    analyze,
    audit_oos_payload,
    exact_binomial_lower,
)
from scripts.submit_scolhkg_final_traffic_gate_scheduler import (  # noqa: E402
    build_specs,
)
from scripts.submit_scolhkg_external_traffic_cpu_frontier import (  # noqa: E402
    build_specs as build_cpu_frontier_specs,
)
from performance.task_descriptor_retrieval import (  # noqa: E402
    DESCRIPTOR_NEAREST,
    DOMAIN_BLIND_CONTROL,
    source_selection_contract,
    weighted_role_distance,
)


def _candidate(index, successes, *, seed=80, trials=100):
    return {
        "method": "PaperFinal-SourceProposal-SAAS",
        "partition": f"paper_final_external_v1_seed{seed:04d}",
        "run_seed": seed,
        "source_index": index,
        "x": [index + 1] * 21,
        "validation": {
            "R": trials,
            "seeds": list(range(900000, 900000 + trials)),
            "feasible_count": successes,
            "feasible_probability": successes / trials,
            "mean": [1.0 + index, 2.0 + index, 0.8],
        },
    }


def test_exact_familywise_binomial_certificate_is_not_wilson_shortcut():
    lower = exact_binomial_lower(100, 100, 0.05 / 3.0)
    assert lower >= 0.95
    assert exact_binomial_lower(99, 100, 0.05 / 3.0) < 0.95
    assert math.isclose(lower, (0.05 / 3.0) ** (1.0 / 100.0))


def test_traffic_audit_deploys_first_certified_frozen_rank():
    payload = {
        "candidates": [
            _candidate(0, 99),
            _candidate(1, 100),
            _candidate(2, 100),
        ],
    }
    audit = audit_oos_payload(payload)
    assert audit["deployed_source_index"] == 1
    assert audit["deployed_certified"]
    assert audit["verification_calls"] == 300
    assert audit["verification_samples_used_to_reorder_shortlist"] is False


def test_traffic_submitter_separates_cuda_search_from_cpu_sumo(tmp_path):
    args = SimpleNamespace(
        deploy=tmp_path,
        run_id="unit",
        archive_run_id="archive",
        source_d=50,
        seed_start=80,
        n_seeds=2,
        N=13,
        n0=10,
        R=100,
        verification_seed_start=900000,
        cpu=12,
        ram_mb=24576,
        source_selection_mode=DESCRIPTOR_NEAREST,
    )
    specs = build_specs(args)
    assert len(specs) == 6
    expected_nodes = [
        "node001", "node002", "node003",
        "node004", "node005", "node006",
    ]
    expected_gpu_nodes = ["jtl110gpu", "jtl110gpu2", "node007"]
    assert all("jtl311linux" not in str(spec) for spec in specs)
    search = [spec for spec in specs if "/search/" in spec["signature"]]
    oos = [spec for spec in specs if "/oos/" in spec["signature"]]
    non_search = [spec for spec in specs if spec not in search]
    assert len(search) == 2
    assert len(oos) == 2
    assert all(
        spec["allowed_nodes"] == expected_gpu_nodes for spec in search)
    assert all(spec["vram"] == 2048 for spec in search)
    assert all(spec["project"] == "KG-SYNTH" for spec in search)
    assert all("--torch-device cuda" in spec["cmd"] for spec in search)
    assert all(
        "runners/run_traffic_gpu_python.sh" in spec["cmd"]
        for spec in search
    )
    assert all("--method-label PaperFinal-DescriptorProposal-SAAS" in (
        spec["cmd"]) for spec in search)
    assert all(
        spec["allowed_nodes"] == expected_nodes for spec in non_search)
    assert all(spec["vram"] == 0 for spec in non_search)
    assert all(
        "--historical-anchor" not in spec["cmd"] for spec in search)
    assert all("unit_seed008" in spec["cmd"] for spec in search)
    assert all("unit_seed008" in spec["cmd"] for spec in oos)
    assert all("--R 100" in spec["cmd"] for spec in oos)
    assert all("--source-indexes 0,1,2" in spec["cmd"] for spec in oos)
    assert all(
        "checkpoints" in spec["stage_excludes"] for spec in specs)


def test_external_traffic_frozen_snapshot_keeps_tracked_static_results(
    tmp_path,
):
    code_root = tmp_path / "snapshot"
    code_root.mkdir()
    marker = {
        "status": "frozen",
        "repository_commit": "a" * 40,
        "scolhkg_tree": "b" * 40,
        "proof_tree": "c" * 40,
        "scripts_tree": "d" * 40,
        "legacy_traffic_tree": "e" * 40,
        "traffic_decision_space_blob": "f" * 40,
        "traffic_baseline_blob": "1" * 40,
        "theory_contract_id": "theory-v1",
        "snapshot_root": str(code_root),
    }
    (code_root / ".scolhkg_execution_snapshot.json").write_text(
        json.dumps(marker), encoding="utf-8")
    args = SimpleNamespace(
        deploy=tmp_path / "deploy",
        code_root=code_root,
        require_frozen_snapshot=True,
        run_id="frozen_cpu_frontier",
        archive_run_id="archive",
        source_selection_modes=DESCRIPTOR_NEAREST,
        backend="botorch_scbo",
        budgets="13",
        source_d=50,
        seed_start=80,
        n_seeds=1,
        n0=10,
        R=100,
        verification_seed_start=900000,
        raw_samples=1024,
        num_restarts=10,
        maxiter=100,
        ts_candidates=2000,
        candidate_timeout_sec=3600.0,
        cpu=12,
        ram_mb=24576,
    )

    specs, _snapshot = build_cpu_frontier_specs(args)

    assert all("results" not in spec["stage_excludes"] for spec in specs)
    assert all("checkpoints" in spec["stage_excludes"] for spec in specs)
    assert all("profiles" in spec["stage_excludes"] for spec in specs)
    aggregate = next(
        spec for spec in specs if spec["signature"].endswith("/audit"))
    assert (
        "--source-domains "
        "QueueResourceControl,InventorySupplyChain"
        in aggregate["cmd"]
    )
    assert (
        "--source-split-heldout FactorShockStatePolicyRZDT1"
        in aggregate["cmd"]
    )
    assert "--heldout-task-family-identifier-used" in aggregate["cmd"]


def test_traffic_submitter_binds_sparse_execution_snapshot(tmp_path):
    code_root = tmp_path / "snapshot"
    code_root.mkdir()
    marker = {
        "status": "frozen",
        "snapshot_kind": "traffic_sparse",
        "repository_commit": "a" * 40,
        "scolhkg_tree": "b" * 40,
        "proof_tree": "c" * 40,
        "scripts_tree": "d" * 40,
        "legacy_traffic_tree": "e" * 40,
        "traffic_decision_space_blob": "f" * 40,
        "traffic_baseline_blob": "1" * 40,
        "theory_contract_id": (
            "source_target_geometric_atlas_coverage_v1"),
        "snapshot_root": str(code_root),
    }
    (code_root / ".scolhkg_execution_snapshot.json").write_text(
        json.dumps(marker),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        deploy=tmp_path / "deploy",
        code_root=code_root,
        require_frozen_snapshot=True,
        run_id="frozen",
        archive_run_id="archive",
        source_d=50,
        seed_start=80,
        n_seeds=1,
        N=13,
        n0=10,
        R=100,
        verification_seed_start=900000,
        cpu=12,
        ram_mb=24576,
        source_selection_mode=DESCRIPTOR_NEAREST,
    )

    specs = build_specs(args)

    assert len(specs) == 4
    assert all(str(code_root / "SC-OLH-KG") in spec["cwd"]
               for spec in specs)
    assert all(
        "SCOLHKG_EXECUTION_PROVENANCE_REQUIRED=1" in spec["cmd"]
        for spec in specs
    )
    assert all(
        f"SCOLHKG_LEGACY_TRAFFIC_TREE={'e' * 40}" in spec["cmd"]
        for spec in specs
    )
    oos = next(
        spec for spec in specs if "/oos/" in spec["signature"])
    assert "run_traffic_oos_explicit.py" in oos["cmd"]
    assert "--results-root" in oos["cmd"]


def test_traffic_aggregate_records_domain_blind_information_contract(
    tmp_path,
):
    paths = []
    for seed in (80, 81):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(
            json.dumps({
                "candidates": [
                    _candidate(0, 99, seed=seed),
                    _candidate(1, 100, seed=seed),
                    _candidate(2, 100, seed=seed),
                ]
            }),
            encoding="utf-8",
        )
        paths.append(path)
    payload = analyze(paths)
    assert payload["policy_vectors_exported"] is False
    assert all("deployed_x" not in row for row in payload["rows"])
    assert all(
        "x" not in candidate
        for row in payload["rows"]
        for candidate in row["candidate_rows"]
    )
    contract = payload["information_contract"]
    assert contract["track"] == "domain_blind_external_holdout"
    assert contract[
        "heldout_task_family_identifier_used_by_proposal"] is False
    assert contract["source_domains"] == [
        "FactorShockStatePolicyRZDT1",
        "InventorySupplyChain",
    ]
    assert contract["excluded_nearest_source_analogue"] == (
        "QueueResourceControl")


def test_observable_descriptor_retrieval_is_target_label_free_and_stable():
    selection = source_selection_contract(DESCRIPTOR_NEAREST)
    assert selection.source_domains == (
        "QueueResourceControl",
        "InventorySupplyChain",
    )
    assert selection.source_split_heldout == (
        "FactorShockStatePolicyRZDT1")
    assert selection.heldout_task_family_identifier_used
    assert all(
        "distance" in row and "observable_roles" in row
        for row in selection.ranking
    )
    assert selection.as_dict()["target_outcomes_used"] is False
    assert selection.as_dict()["target_oracle_used"] is False
    assert weighted_role_distance(
        "Ingolstadt21Traffic", "QueueResourceControl"
    ) < weighted_role_distance(
        "Ingolstadt21Traffic", "FactorShockStatePolicyRZDT1"
    )


def test_domain_blind_control_retains_registered_hard_split():
    selection = source_selection_contract(DOMAIN_BLIND_CONTROL)
    assert selection.source_domains == (
        "FactorShockStatePolicyRZDT1",
        "InventorySupplyChain",
    )
    assert selection.source_split_heldout == "QueueResourceControl"
    assert not selection.heldout_task_family_identifier_used


def test_external_traffic_frontier_is_cpu_only_and_budget_separated(tmp_path):
    args = SimpleNamespace(
        deploy=tmp_path,
        code_root=None,
        require_frozen_snapshot=False,
        run_id="cpu_frontier",
        archive_run_id="archive",
        source_selection_modes=(
            f"{DESCRIPTOR_NEAREST},{DOMAIN_BLIND_CONTROL}"),
        backend="botorch_scbo",
        budgets="13,40",
        source_d=50,
        seed_start=80,
        n_seeds=1,
        n0=10,
        R=100,
        verification_seed_start=900000,
        raw_samples=1024,
        num_restarts=10,
        maxiter=100,
        ts_candidates=2000,
        candidate_timeout_sec=3600.0,
        cpu=12,
        ram_mb=24576,
    )

    specs, snapshot = build_cpu_frontier_specs(args)

    assert snapshot is None
    assert len(specs) == 14
    assert all(spec["vram"] == 0 for spec in specs)
    assert all(spec["allowed_nodes"] == [
        "node001", "node002", "node003",
        "node004", "node005", "node006",
    ] for spec in specs)
    search = [spec for spec in specs if "/search/" in spec["signature"]]
    assert len(search) == 4
    assert all(spec["project"] == "KG-SUMO" for spec in search)
    assert all("--backend botorch_scbo" in spec["cmd"] for spec in search)
    assert all("--torch-device cpu" in spec["cmd"] for spec in search)
    assert all("--historical-anchor" not in spec["cmd"] for spec in search)
    assert {13, 40} == {
        int(spec["signature"].split("/N", 1)[1].split("/", 1)[0])
        for spec in search
    }
    assert all(
        "checkpoints" in spec["stage_excludes"] for spec in specs)
