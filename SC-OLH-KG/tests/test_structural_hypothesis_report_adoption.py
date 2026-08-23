import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import runners.run_structural_hypothesis_report_adoption as runner  # noqa: E402


RUNNER = ROOT / "runners/run_structural_hypothesis_report_adoption.py"
ADOPTION_CORE = ROOT / "performance/structural_hypothesis_report_adoption.py"
ADOPTION_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_report_adoption_v1.json"
)
PUBLISHER_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_reingestion_publisher_v1.json"
)
HYPOTHESIS_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_loop_v1.json"
)
EXECUTOR_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_executor_v1.json"
)
MATERIALIZER_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_task_materializer_v1.json"
)
RUNTIME_CONTRACT = (
    ROOT
    / "performance/manifests/structural_hypothesis_single_task_runtime_v1.json"
)
BASE_MANIFEST = ROOT / "performance/manifests/v18b_exactkg_mcdiag.json"
ASSET_ROOT = ROOT / "performance/task_inputs/structural_hypothesis_materializer_v1"

PROFILES = (
    "none",
    "low_frequency_only",
    "orthogonality_only",
    "sparsity_only",
    "additivity_only",
    "leave_out_low_frequency",
    "leave_out_orthogonality",
    "leave_out_sparsity",
    "leave_out_additivity",
)
BASE_CSV_FIELDS = (
    "run_id", "track", "variant", "method", "implementation",
    "initial_design", "domain", "seed", "d", "N", "n0",
    "source_calls", "total_calls", "d_over_target_calls",
    "d_over_total_calls", "status", "true_feasible", "feasible_regret",
    "true_objective", "constraint_violation", "initial_has_true_feasible",
    "initial_true_feasible_count", "initial_best_feasible_regret",
    "adaptive_rescue", "adaptive_loss", "adaptive_improves_initial_best",
    "adaptive_regret_change", "posterior_feasible",
    "posterior_certificate_vacuous", "posterior_certified_count",
    "false_certificate_count", "certificate_precision",
    "certificate_recall", "decision_backend", "structural_prior_profile",
    "hvd_profile", "source_discrepancy_update", "recheck_top_k",
    "risk_penalty", "utility_weight", "adaptive_replication_voi",
    "adaptive_replication_count", "adaptive_new_point_count",
    "posterior_dominance_enabled", "posterior_dominance_switch_count",
    "wall_time_sec", "result_path",
)


def _digest(character):
    return "sha256:" + character * 64


def _json_file(path, payload=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {}), encoding="utf-8")
    return path.resolve()


def _unit_case(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    publication = tmp_path / "publication"
    publication.mkdir()
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    assets = tmp_path / "assets"
    assets.mkdir()
    evidence = tmp_path / "base.csv"
    evidence.write_text("track,run_id\npriors,test\n", encoding="utf-8")
    return {
        "publication": publication.resolve(),
        "adoption_contract": _json_file(
            tmp_path / "adoption-contract.json"
        ),
        "adoption": (tmp_path / "adoption").resolve(),
        "adoption_id": "local-report-adoption-0001",
        "base_evidence": evidence.resolve(),
        "attempt": attempt.resolve(),
        "hypothesis_contract": _json_file(
            tmp_path / "hypothesis-contract.json"
        ),
        "executor_contract": _json_file(tmp_path / "executor-contract.json"),
        "runtime_contract": _json_file(tmp_path / "runtime-contract.json"),
        "publisher_contract": _json_file(
            tmp_path / "publisher-contract.json"
        ),
        "base_manifest": _json_file(tmp_path / "base-manifest.json"),
        "asset_root": assets.resolve(),
    }


def _chain_args(case):
    return [
        "--publication-root", str(case["publication"]),
        "--adoption-contract", str(case["adoption_contract"]),
        "--adoption-root", str(case["adoption"]),
        "--adoption-id", case["adoption_id"],
        "--base-evidence-csv", str(case["base_evidence"]),
        "--attempt-root", str(case["attempt"]),
        "--hypothesis-contract", str(case["hypothesis_contract"]),
        "--executor-contract", str(case["executor_contract"]),
        "--runtime-contract", str(case["runtime_contract"]),
        "--publisher-contract", str(case["publisher_contract"]),
        "--base-manifest", str(case["base_manifest"]),
        "--asset-root", str(case["asset_root"]),
        "--expected-source-evidence-digest", _digest("1"),
        "--expected-plan-digest", _digest("2"),
        "--expected-authorization-digest", _digest("3"),
        "--expected-execution-receipt-digest", _digest("4"),
        "--expected-execution-journal-head-digest", _digest("5"),
        "--expected-execution-attempt-digest", _digest("6"),
        "--expected-publication-digest", _digest("7"),
        "--expected-reingestion-digest", _digest("8"),
        "--expected-output-report-body-digest", _digest("9"),
        "--expected-output-audit-head", _digest("a"),
        "--expected-output-evidence-digest", _digest("b"),
        "--expected-publication-marker-raw-sha256", _digest("d"),
        "--expected-combined-rows-raw-sha256", _digest("e"),
        "--expected-output-report-raw-sha256", _digest("f"),
        "--expected-reingestion-receipt-raw-sha256", _digest("0"),
    ]


def _adopt_argv(case):
    return ["adopt", *_chain_args(case), "--confirm-local-report-adoption"]


def _verify_argv(case):
    return [
        "verify",
        *_chain_args(case),
        "--expected-adoption-digest", _digest("c"),
    ]


def _adopted_payload(case, *, verified=False):
    status = "ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED"
    return {
        "status": "VERIFIED_" + status if verified else status,
        "adoption_root": str(case["adoption"]),
        "adoption_digest": _digest("c"),
        "publication_digest": _digest("7"),
        "reingestion_digest": _digest("8"),
        "output_report_body_digest": _digest("9"),
        "output_audit_head": _digest("a"),
        "output_evidence_digest": _digest("b"),
        "planning_status": "NOT_PLANNED",
    }


def _expected_common_kwargs(case):
    return {
        "adoption_id": case["adoption_id"],
        "expected_source_evidence_digest": _digest("1"),
        "expected_plan_digest": _digest("2"),
        "expected_authorization_digest": _digest("3"),
        "expected_execution_receipt_digest": _digest("4"),
        "expected_execution_journal_head_digest": _digest("5"),
        "expected_execution_attempt_digest": _digest("6"),
        "expected_publication_digest": _digest("7"),
        "expected_reingestion_digest": _digest("8"),
        "expected_output_report_body_digest": _digest("9"),
        "expected_output_audit_head": _digest("a"),
        "expected_output_evidence_digest": _digest("b"),
        "expected_publication_marker_raw_sha256": _digest("d"),
        "expected_combined_rows_raw_sha256": _digest("e"),
        "expected_output_report_raw_sha256": _digest("f"),
        "expected_reingestion_receipt_raw_sha256": _digest("0"),
    }


def _write_self_contained_base_csv(path, hypothesis_contract):
    scope = hypothesis_contract["evidence_scope"]
    rows = []
    for profile in PROFILES:
        for domain in scope["domains"]:
            for seed in scope["seeds"]:
                row = {field: "" for field in BASE_CSV_FIELDS}
                row.update({
                    "run_id": scope["run_id"],
                    "track": scope["track"],
                    "variant": scope["variant_template"].format(
                        profile=profile
                    ),
                    "method": profile,
                    "implementation": scope["implementation"],
                    "initial_design": scope["initial_design"],
                    "domain": domain,
                    "seed": str(seed),
                    "d": str(scope["d"]),
                    "N": str(scope["N"]),
                    "n0": str(scope["n0"]),
                    "source_calls": str(scope["source_calls"]),
                    "d_over_target_calls": "0.0",
                    "d_over_total_calls": "0.0",
                    "status": "ok",
                    "true_feasible": "True" if seed < 9 else "False",
                    "feasible_regret": "1.0",
                    "true_objective": "0.0",
                    "constraint_violation": "0.0",
                    "initial_has_true_feasible": "True",
                    "initial_true_feasible_count": "1",
                    "initial_best_feasible_regret": "1.0",
                    "adaptive_rescue": "False",
                    "adaptive_loss": "False",
                    "adaptive_improves_initial_best": "False",
                    "adaptive_regret_change": "0.0",
                    "posterior_feasible": "True",
                    "posterior_certificate_vacuous": "False",
                    "posterior_certified_count": "1",
                    "false_certificate_count": "0",
                    "certificate_precision": "1.0",
                    "certificate_recall": "1.0",
                    "decision_backend": scope["decision_backend"],
                    "structural_prior_profile": profile,
                    "adaptive_replication_count": "0",
                    "adaptive_new_point_count": "0",
                    "wall_time_sec": "0.0",
                    "result_path": (
                        f"synthetic/{profile}/{domain}/seed{seed}"
                    ),
                })
                row.update({
                    key: str(value)
                    for key, value in scope["fixed_row_values"].items()
                })
                rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)
    return rows


def _fake_native_result(task):
    args = task["args"]
    return {
        "line": task["line"],
        "heldout": task["heldout"],
        "seed": task["seed"],
        "n_search_simulations": args["N"],
        "n_simulations": args["N"],
        "n0": args["n0"],
        "source_target_adaptation_contract": {
            "source_simulator_calls": 384,
        },
        "true_feasible": True,
        "feasible_simple_regret": 0.0,
        "adaptive_loss": False,
        "decision_backend": args["decision_backend"],
        "structural_prior_profile": args["structural_prior_profile"],
        "hvd_ablation_profile": args["hvd_ablation_profile"],
        "source_discrepancy_update": args["source_discrepancy_update"],
        "certification_recheck_top_k": args[
            "certification_recheck_top_k"
        ],
        "decision_risk_penalty": args["decision_risk_penalty"],
        "decision_source_utility_weight": args[
            "decision_source_utility_weight"
        ],
        "adaptive_replication_voi_enabled": args[
            "adaptive_replication_voi"
        ],
        "posterior_dominance_enabled": args[
            "posterior_dominance_enabled"
        ],
        "posterior_dominance_switch_count": 0,
    }


def _valid_fake_preflight(runtime_core, runtime_contract, task, prepared):
    binding = runtime_contract["runtime_binding"]
    body = {
        "schema_version": runtime_core.PREFLIGHT_SCHEMA_VERSION,
        "status": "PASSED_LOCAL_PREFLIGHT",
        "attempt_digest": prepared["attempt_digest"],
        "authorization_digest": prepared["authorization_digest"],
        "task_id": task["task_id"],
        "task_digest": task["task_digest"],
        "requirements": runtime_contract["preflight"],
        "observed": {
            "affinity_cpu_ids": list(range(12)),
            "affinity_cpu_count": 12,
            "memory_available_bytes": 12884901888,
            "checkpoint_free_bytes": 2147483648,
            "fork_probe_passed": True,
            "fork_probe_process_count": 12,
            "thread_pools": [{
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "version": "test-only",
                "threading_layer": "pthreads",
                "num_threads": 1,
            }],
            "executor_callable": {
                "module": binding["executor_module"],
                "callable": binding["executor_callable"],
                "source_file": str(
                    (ROOT / "performance/benchmark_lodo_meta_prior.py").resolve()
                ),
                "source_sha256": binding[
                    "executor_callable_source_sha256"
                ],
                "code_sha256": binding["executor_callable_code_sha256"],
                "firstlineno": binding[
                    "executor_callable_firstlineno"
                ],
            },
            "required_environment": dict(runtime_core._REQUIRED_ENVIRONMENT),
        },
        "nonclaims": [
            "local_preflight_is_not_external_authority",
            "local_preflight_does_not_guarantee_completion",
            "no_runtime_duration_claim",
            "no_peak_memory_claim",
        ],
    }
    return {
        **body,
        "integrity": {
            "algorithm": "sha256-canonical-json-v1",
            "preflight_digest": runtime_core._digest(body),
        },
    }


def _raw_digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def self_contained_publication(monkeypatch):
    from performance import structural_hypothesis_reingestion_publisher as publisher
    from performance import structural_hypothesis_single_task_runtime as runtime_core
    from performance.structural_hypothesis_loop import (
        run_structural_hypothesis_loop,
        verify_report_integrity,
    )
    from performance.structural_hypothesis_task_materializer import (
        materialize_task_bundle,
        verify_materialized_task_bundle,
    )
    from runners import run_structural_hypothesis_loop as loop_runner
    from runners import run_structural_hypothesis_single_task as single_runner

    state_home = Path(tempfile.mkdtemp(
        prefix="kgop-report-adoption-test.", dir="/tmp"
    ))
    state_home.chmod(0o700)
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    for key, value in single_runner.REQUIRED_EXECUTION_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)

    evidence = state_home / "self-contained-base-evidence.csv"
    hypothesis, contract_artifact = loop_runner._load_contract(
        HYPOTHESIS_CONTRACT
    )
    generated_rows = _write_self_contained_base_csv(evidence, hypothesis)
    base_rows, evidence_artifact = loop_runner._load_evidence(evidence)
    assert len(BASE_CSV_FIELDS) == 47
    assert len(generated_rows) == len(base_rows) == 270
    assert all(
        type(value) is str for row in base_rows for value in row.values()
    )
    report = run_structural_hypothesis_loop(
        base_rows,
        hypothesis,
        input_artifacts={
            "evidence_csv": evidence_artifact,
            "contract_json": contract_artifact,
        },
    ).to_dict()
    assert verify_report_integrity(report)
    assert report["status"] == "COMPLETED_WITH_EVIDENCE_GAPS"
    assert len(report["pending_evidence"]) == 30

    executor_contract = json.loads(EXECUTOR_CONTRACT.read_text(encoding="utf-8"))
    materializer_contract = json.loads(
        MATERIALIZER_CONTRACT.read_text(encoding="utf-8")
    )
    runtime_contract = json.loads(
        RUNTIME_CONTRACT.read_text(encoding="utf-8")
    )
    attempt = (
        state_home
        / "kg-op/structural-hypothesis-execution/v1/attempt-adoption-kat"
    )
    checkpoint_root = attempt / "checkpoints"
    bundle = materialize_task_bundle(
        report,
        hypothesis,
        executor_contract,
        materializer_contract,
        BASE_MANIFEST,
        ASSET_ROOT,
        checkpoint_root,
    )
    assert verify_materialized_task_bundle(
        bundle,
        report,
        hypothesis,
        executor_contract,
        materializer_contract,
        BASE_MANIFEST,
        ASSET_ROOT,
        checkpoint_root,
    )
    task = bundle["plan"]["tasks"][0]
    prepared = runtime_core.prepare_single_task_attempt(
        report,
        bundle,
        hypothesis,
        executor_contract,
        materializer_contract,
        runtime_contract,
        BASE_MANIFEST,
        ASSET_ROOT,
        attempt,
        task_id=task["task_id"],
        expected_bundle_digest=bundle["integrity"]["bundle_digest"],
        expected_plan_digest=bundle["plan"]["integrity"]["plan_digest"],
        authorization_id="adoption-kat-authorization-0001",
    )
    monkeypatch.setattr(
        runtime_core,
        "_load_real_executor",
        lambda _contract: _fake_native_result,
    )
    monkeypatch.setattr(
        runtime_core,
        "_run_preflight",
        lambda *args, **kwargs: _valid_fake_preflight(
            runtime_core, runtime_contract, task, prepared
        ),
    )
    executed = runtime_core.execute_single_task_attempt(
        attempt,
        runtime_contract,
        expected_authorization_digest=prepared["authorization_digest"],
    )

    publication_id = "publication-adoption-kat-0001"
    publication = (
        state_home
        / "kg-op/structural-hypothesis-reingestion/v1"
        / publication_id
    )
    published = publisher.publish_single_task_reingestion(
        evidence,
        attempt,
        HYPOTHESIS_CONTRACT,
        EXECUTOR_CONTRACT,
        RUNTIME_CONTRACT,
        PUBLISHER_CONTRACT,
        BASE_MANIFEST,
        ASSET_ROOT,
        publication,
        publication_id=publication_id,
        expected_source_evidence_digest=report["evidence_digest"],
        expected_plan_digest=bundle["plan"]["integrity"]["plan_digest"],
        expected_authorization_digest=prepared["authorization_digest"],
        expected_execution_receipt_digest=executed["receipt_digest"],
        expected_execution_journal_head_digest=executed["journal_head_digest"],
        expected_execution_attempt_digest=executed["attempt_digest"],
    )
    adoption_id = "adoption-kat-0001"
    adoption = (
        state_home
        / "kg-op/structural-hypothesis-report-adoption/v1"
        / adoption_id
    )
    context = {
        "state_home": state_home,
        "evidence": evidence,
        "attempt": attempt,
        "publication": publication,
        "adoption": adoption,
        "adoption_id": adoption_id,
        "source_evidence_digest": report["evidence_digest"],
        "plan_digest": bundle["plan"]["integrity"]["plan_digest"],
        "authorization_digest": prepared["authorization_digest"],
        "execution_receipt_digest": executed["receipt_digest"],
        "execution_journal_head_digest": executed["journal_head_digest"],
        "execution_attempt_digest": executed["attempt_digest"],
        "publication_digest": published["publication_digest"],
        "reingestion_digest": published["reingestion_digest"],
        "output_report_body_digest": published["output_report_body_digest"],
        "output_audit_head": published["output_audit_head"],
        "output_evidence_digest": published["output_evidence_digest"],
        "publication_marker_raw_sha256": _raw_digest(
            publication / "publication.json"
        ),
        "combined_rows_raw_sha256": _raw_digest(
            publication / "combined_rows.json"
        ),
        "output_report_raw_sha256": _raw_digest(
            publication / "output_report.json"
        ),
        "reingestion_receipt_raw_sha256": _raw_digest(
            publication / "reingestion_receipt.json"
        ),
    }
    try:
        yield context
    finally:
        shutil.rmtree(state_home, ignore_errors=True)


def test_runner_surface_is_offline_exact_and_stdlib_only():
    source = RUNNER.read_text(encoding="utf-8")
    assert "benchmark_lodo_meta_prior" not in source
    assert "benchmark_quality" not in source
    assert "import numpy" not in source
    assert "import subprocess" not in source
    assert "import socket" not in source
    assert "import requests" not in source
    assert "import scheduler" not in source.lower()
    assert "from scheduler" not in source.lower()
    choices = set(runner._parser()._subparsers._group_actions[0].choices)
    assert choices == {"adopt", "verify"}

    script = r'''
import builtins
import sys

blocked = {
    "numpy",
    "performance.benchmark_lodo_meta_prior",
    "performance.benchmark_quality",
}
original_import = builtins.__import__

def bomb_import(name, *args, **kwargs):
    if name in blocked:
        raise AssertionError("adoption startup crossed an execution boundary")
    return original_import(name, *args, **kwargs)

builtins.__import__ = bomb_import
from runners.run_structural_hypothesis_report_adoption import _parser
assert set(_parser()._subparsers._group_actions[0].choices) == {
    "adopt", "verify"
}
assert blocked.isdisjoint(sys.modules)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_adopt_wires_exact_full_chain_and_prints_canonical_anchor(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    calls = []

    def adopt(*args, **kwargs):
        calls.append((args, kwargs))
        return _adopted_payload(case)

    monkeypatch.setattr(
        runner,
        "_load_adoption_core",
        lambda: SimpleNamespace(adopt_structural_hypothesis_report=adopt),
    )
    assert runner.main(_adopt_argv(case)) == 0
    assert calls == [(
        (
            case["publication"],
            case["adoption_contract"],
            case["adoption"],
            case["base_evidence"],
            case["attempt"],
            case["hypothesis_contract"],
            case["executor_contract"],
            case["runtime_contract"],
            case["publisher_contract"],
            case["base_manifest"],
            case["asset_root"],
        ),
        _expected_common_kwargs(case),
    )]
    captured = capsys.readouterr()
    expected = _adopted_payload(case)
    assert captured.out == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert "action=adopt" in captured.err
    assert "status=ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED" in captured.err
    assert "planning_status=NOT_PLANNED" in captured.err


def test_verify_wires_live_full_chain_and_external_adoption_anchor(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    case["adoption"].mkdir()
    calls = []

    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return _adopted_payload(case, verified=True)

    monkeypatch.setattr(
        runner,
        "_load_adoption_core",
        lambda: SimpleNamespace(
            verify_structural_hypothesis_report_adoption=verify
        ),
    )
    assert runner.main(_verify_argv(case)) == 0
    expected_kwargs = {
        **_expected_common_kwargs(case),
        "expected_adoption_digest": _digest("c"),
    }
    assert calls == [(
        (
            case["publication"],
            case["adoption_contract"],
            case["adoption"],
            case["base_evidence"],
            case["attempt"],
            case["hypothesis_contract"],
            case["executor_contract"],
            case["runtime_contract"],
            case["publisher_contract"],
            case["base_manifest"],
            case["asset_root"],
        ),
        expected_kwargs,
    )]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "action=verify" in captured.err
    assert (
        "status=VERIFIED_ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED"
        in captured.err
    )


@pytest.mark.parametrize(
    "omitted",
    [
        "--publication-root",
        "--adoption-root",
        "--adoption-id",
        "--base-evidence-csv",
        "--attempt-root",
        "--expected-source-evidence-digest",
        "--expected-plan-digest",
        "--expected-authorization-digest",
        "--expected-execution-receipt-digest",
        "--expected-execution-journal-head-digest",
        "--expected-execution-attempt-digest",
        "--expected-publication-digest",
        "--expected-reingestion-digest",
        "--expected-output-report-body-digest",
        "--expected-output-audit-head",
        "--expected-output-evidence-digest",
        "--expected-publication-marker-raw-sha256",
        "--expected-combined-rows-raw-sha256",
        "--expected-output-report-raw-sha256",
        "--expected-reingestion-receipt-raw-sha256",
        "--confirm-local-report-adoption",
    ],
)
def test_adopt_requires_confirmation_and_every_full_chain_anchor(
    tmp_path, omitted
):
    argv = _adopt_argv(_unit_case(tmp_path))
    index = argv.index(omitted)
    size = 1 if omitted == "--confirm-local-report-adoption" else 2
    del argv[index:index + size]
    with pytest.raises(SystemExit) as exc:
        runner._parser().parse_args(argv)
    assert exc.value.code == 2


def test_verify_requires_an_independent_adoption_digest(tmp_path):
    case = _unit_case(tmp_path)
    case["adoption"].mkdir()
    argv = _verify_argv(case)
    index = argv.index("--expected-adoption-digest")
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as exc:
        runner._parser().parse_args(argv)
    assert exc.value.code == 2


def test_adoption_has_no_current_planning_execution_or_override_surface():
    parsers = runner._parser()._subparsers._group_actions[0].choices.values()
    forbidden = {
        "current",
        "current_pointer",
        "current-pointer",
        "plan",
        "planner",
        "materialize",
        "authorize",
        "execute",
        "executor",
        "run_one",
        "scheduler",
        "network",
        "dry_run",
        "dry-run",
        "overwrite",
        "replace",
        "replan",
    }
    assert all(
        forbidden.isdisjoint(action.dest for action in parser._actions)
        for parser in parsers
    )


def test_roots_are_absolute_source_exists_and_adoption_is_no_clobber(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        runner,
        "_load_adoption_core",
        lambda: (_ for _ in ()).throw(
            AssertionError("path gate must precede core import")
        ),
    )
    existing = _unit_case(tmp_path / "existing")
    existing["adoption"].mkdir()
    assert runner.main(_adopt_argv(existing)) == 2
    assert "refusing to overwrite" in capsys.readouterr().err

    relative = _unit_case(tmp_path / "relative")
    argv = _adopt_argv(relative)
    argv[argv.index("--adoption-root") + 1] = "relative-adoption"
    assert runner.main(argv) == 2
    assert "absolute path" in capsys.readouterr().err

    missing_publication = _unit_case(tmp_path / "missing-publication")
    missing_publication["publication"].rmdir()
    assert runner.main(_adopt_argv(missing_publication)) == 2
    assert "publication root must be an existing" in capsys.readouterr().err

    target = tmp_path / "publication-alias-target"
    target.mkdir()
    alias = _unit_case(tmp_path / "publication-alias")
    alias["publication"].rmdir()
    alias["publication"].symlink_to(target, target_is_directory=True)
    assert runner.main(_adopt_argv(alias)) == 2
    assert "existing non-symlink" in capsys.readouterr().err

    missing_adoption = _unit_case(tmp_path / "missing-adoption")
    assert runner.main(_verify_argv(missing_adoption)) == 2
    assert "adoption root must be an existing" in capsys.readouterr().err


def test_digest_adoption_id_and_core_rejection_are_fail_closed(
    tmp_path, monkeypatch, capsys
):
    digest_case = _unit_case(tmp_path / "digest")
    argv = _adopt_argv(digest_case)
    argv[argv.index("--expected-publication-digest") + 1] = "sha256:ABC"
    assert runner.main(argv) == 2
    assert "lowercase sha256" in capsys.readouterr().err

    id_case = _unit_case(tmp_path / "id")
    argv = _adopt_argv(id_case)
    argv[argv.index("--adoption-id") + 1] = "../../current"
    assert runner.main(argv) == 2
    assert "non-path local mechanics label" in capsys.readouterr().err

    rejection_case = _unit_case(tmp_path / "core-reject")

    def reject(*args, **kwargs):
        raise ValueError("synthetic adoption full-chain rejection")

    monkeypatch.setattr(
        runner,
        "_load_adoption_core",
        lambda: SimpleNamespace(adopt_structural_hypothesis_report=reject),
    )
    assert runner.main(_adopt_argv(rejection_case)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "synthetic adoption full-chain rejection" in captured.err
    assert "Traceback" not in captured.err


def _real_adopt_args(context, *, adoption=None, adoption_id=None):
    return [
        "adopt",
        "--publication-root", str(context["publication"]),
        "--adoption-contract", str(ADOPTION_CONTRACT),
        "--adoption-root", str(adoption or context["adoption"]),
        "--adoption-id", adoption_id or context["adoption_id"],
        "--base-evidence-csv", str(context["evidence"]),
        "--attempt-root", str(context["attempt"]),
        "--hypothesis-contract", str(HYPOTHESIS_CONTRACT),
        "--executor-contract", str(EXECUTOR_CONTRACT),
        "--runtime-contract", str(RUNTIME_CONTRACT),
        "--publisher-contract", str(PUBLISHER_CONTRACT),
        "--base-manifest", str(BASE_MANIFEST),
        "--asset-root", str(ASSET_ROOT),
        "--expected-source-evidence-digest",
        context["source_evidence_digest"],
        "--expected-plan-digest", context["plan_digest"],
        "--expected-authorization-digest", context["authorization_digest"],
        "--expected-execution-receipt-digest",
        context["execution_receipt_digest"],
        "--expected-execution-journal-head-digest",
        context["execution_journal_head_digest"],
        "--expected-execution-attempt-digest",
        context["execution_attempt_digest"],
        "--expected-publication-digest", context["publication_digest"],
        "--expected-reingestion-digest", context["reingestion_digest"],
        "--expected-output-report-body-digest",
        context["output_report_body_digest"],
        "--expected-output-audit-head", context["output_audit_head"],
        "--expected-output-evidence-digest",
        context["output_evidence_digest"],
        "--expected-publication-marker-raw-sha256",
        context["publication_marker_raw_sha256"],
        "--expected-combined-rows-raw-sha256",
        context["combined_rows_raw_sha256"],
        "--expected-output-report-raw-sha256",
        context["output_report_raw_sha256"],
        "--expected-reingestion-receipt-raw-sha256",
        context["reingestion_receipt_raw_sha256"],
        "--confirm-local-report-adoption",
    ]


def _real_verify_args(context, adopted):
    args = _real_adopt_args(context)
    args[0] = "verify"
    args.remove("--confirm-local-report-adoption")
    args.extend([
        "--expected-adoption-digest",
        adopted["adoption_digest"],
    ])
    return args


def _adoption_subprocess(script, args, context):
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
            "XDG_STATE_HOME": str(context["state_home"]),
        },
        text=True,
        capture_output=True,
        check=False,
    )


def _core_adopt(context, core, *, adoption=None, adoption_id=None):
    return core.adopt_structural_hypothesis_report(
        context["publication"],
        ADOPTION_CONTRACT,
        adoption or context["adoption"],
        context["evidence"],
        context["attempt"],
        HYPOTHESIS_CONTRACT,
        EXECUTOR_CONTRACT,
        RUNTIME_CONTRACT,
        PUBLISHER_CONTRACT,
        BASE_MANIFEST,
        ASSET_ROOT,
        adoption_id=adoption_id or context["adoption_id"],
        expected_source_evidence_digest=context["source_evidence_digest"],
        expected_plan_digest=context["plan_digest"],
        expected_authorization_digest=context["authorization_digest"],
        expected_execution_receipt_digest=(
            context["execution_receipt_digest"]
        ),
        expected_execution_journal_head_digest=(
            context["execution_journal_head_digest"]
        ),
        expected_execution_attempt_digest=(
            context["execution_attempt_digest"]
        ),
        expected_publication_digest=context["publication_digest"],
        expected_reingestion_digest=context["reingestion_digest"],
        expected_output_report_body_digest=(
            context["output_report_body_digest"]
        ),
        expected_output_audit_head=context["output_audit_head"],
        expected_output_evidence_digest=context["output_evidence_digest"],
        expected_publication_marker_raw_sha256=(
            context["publication_marker_raw_sha256"]
        ),
        expected_combined_rows_raw_sha256=(
            context["combined_rows_raw_sha256"]
        ),
        expected_output_report_raw_sha256=(
            context["output_report_raw_sha256"]
        ),
        expected_reingestion_receipt_raw_sha256=(
            context["reingestion_receipt_raw_sha256"]
        ),
    )


def _tree_observation(root):
    return {
        str(path.relative_to(root)): (
            "directory" if path.is_dir() else path.read_bytes(),
            path.stat().st_mode,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
    }


def test_clean_checkout_fake_only_adopt_verify_and_no_clobber(
    self_contained_publication,
):
    context = self_contained_publication
    script = r'''
import builtins
import sys

blocked = {
    "performance.benchmark_lodo_meta_prior",
    "performance.benchmark_quality",
}
original_import = builtins.__import__

def bomb_import(name, *args, **kwargs):
    if name in blocked:
        raise AssertionError("adoption imported the native benchmark")
    return original_import(name, *args, **kwargs)

builtins.__import__ = bomb_import
from runners.run_structural_hypothesis_report_adoption import main
raise SystemExit(main(sys.argv[1:]))
'''
    source_before = {
        str(path.relative_to(context["publication"])): path.read_bytes()
        for path in context["publication"].rglob("*")
        if path.is_file()
    }
    args = _real_adopt_args(context)
    made = _adoption_subprocess(script, args, context)
    assert made.returncode == 0, made.stderr
    summary = json.loads(made.stdout)
    assert set(summary) == {
        "status",
        "adoption_root",
        "adoption_digest",
        "publication_digest",
        "reingestion_digest",
        "output_report_body_digest",
        "output_audit_head",
        "output_evidence_digest",
        "planning_status",
    }
    assert summary["status"] == (
        "ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED"
    )
    assert summary["planning_status"] == "NOT_PLANNED"
    assert summary["adoption_root"] == str(context["adoption"])
    assert made.stdout == json.dumps(
        summary, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert "action=adopt" in made.stderr

    adoption = context["adoption"]
    contract = json.loads(ADOPTION_CONTRACT.read_text(encoding="utf-8"))
    directory_keys = {
        "publication_directory",
        "publication_execution_directory",
        "publication_execution_input_directory",
        "publication_execution_journal_directory",
    }
    snapshot_directory_keys = {
        "execution_directory",
        "execution_input_directory",
        "execution_journal_directory",
    }
    expected_directories = {
        value
        for key, value in contract["artifact_layout"].items()
        if key in directory_keys
    }
    expected_files = {
        contract["artifact_layout"]["adoption_contract"],
        contract["artifact_layout"]["adoption_commit"],
        *(
            "publication/" + value
            for key, value in contract["publication_snapshot_layout"].items()
            if key not in snapshot_directory_keys
        ),
    }
    observed_directories = {
        str(path.relative_to(adoption))
        for path in adoption.rglob("*")
        if path.is_dir()
    }
    observed_files = {
        str(path.relative_to(adoption))
        for path in adoption.rglob("*")
        if path.is_file()
    }
    assert observed_directories == expected_directories
    assert observed_files == expected_files
    assert oct(adoption.stat().st_mode & 0o777) == "0o700"
    assert all(
        oct(path.stat().st_mode & 0o777) == "0o700"
        for path in adoption.rglob("*")
        if path.is_dir()
    )
    assert all(
        oct(path.stat().st_mode & 0o777) == "0o600"
        for path in adoption.rglob("*")
        if path.is_file()
    )
    assert (adoption / "adoption.json").is_file()
    assert not (adoption / "current.json").exists()
    assert set(path.name for path in adoption.iterdir()) == {
        "adoption_contract.json",
        "publication",
        "adoption.json",
    }
    for key, relative in contract["publication_snapshot_layout"].items():
        if key not in snapshot_directory_keys:
            copied = adoption / "publication" / relative
            source = context["publication"] / relative
            assert copied.read_bytes() == source.read_bytes()
            assert (copied.stat().st_dev, copied.stat().st_ino) != (
                source.stat().st_dev,
                source.stat().st_ino,
            )
            assert copied.stat().st_nlink == 1
    assert source_before == {
        str(path.relative_to(context["publication"])): path.read_bytes()
        for path in context["publication"].rglob("*")
        if path.is_file()
    }

    before_repeat = _tree_observation(adoption)
    repeated = _adoption_subprocess(script, args, context)
    assert repeated.returncode == 2
    assert repeated.stdout == ""
    assert "refusing to overwrite" in repeated.stderr
    assert before_repeat == _tree_observation(adoption)

    before_verify = _tree_observation(adoption)
    verified = _adoption_subprocess(
        script, _real_verify_args(context, summary), context
    )
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout == ""
    assert "action=verify" in verified.stderr
    assert (
        "status=VERIFIED_ADOPTED_AS_LOCAL_REPORT_VERSION_NOT_PLANNED"
        in verified.stderr
    )
    assert before_verify == _tree_observation(adoption)

    marker = adoption / "adoption.json"
    marker.chmod(0o644)
    wrong_mode_before = _tree_observation(adoption)
    rejected_mode = _adoption_subprocess(
        script, _real_verify_args(context, summary), context
    )
    assert rejected_mode.returncode == 2
    assert rejected_mode.stdout == ""
    assert "mode" in rejected_mode.stderr
    assert wrong_mode_before == _tree_observation(adoption)
    marker.chmod(0o600)

    injected_current = adoption / "current.json"
    injected_current.write_text("{}\n", encoding="utf-8")
    injected_current.chmod(0o600)
    rejected_extra = _adoption_subprocess(
        script, _real_verify_args(context, summary), context
    )
    assert rejected_extra.returncode == 2
    assert rejected_extra.stdout == ""
    assert "unexpected artifacts" in rejected_extra.stderr
    injected_current.unlink()

    copied_report = adoption / "publication/output_report.json"
    copied_report.write_bytes(copied_report.read_bytes() + b"\n")
    rejected_tamper = _adoption_subprocess(
        script, _real_verify_args(context, summary), context
    )
    assert rejected_tamper.returncode == 2
    assert rejected_tamper.stdout == ""
    assert "Traceback" not in rejected_tamper.stderr


@pytest.mark.parametrize(
    "option",
    [
        "--expected-publication-marker-raw-sha256",
        "--expected-combined-rows-raw-sha256",
        "--expected-output-report-raw-sha256",
        "--expected-reingestion-receipt-raw-sha256",
    ],
)
def test_each_wrong_independent_raw_anchor_creates_no_adoption_root(
    self_contained_publication, option
):
    context = self_contained_publication
    suffix = option.removeprefix("--expected-").removesuffix("-raw-sha256")
    bad_id = "adoption-bad-raw-" + suffix
    bad_root = (
        context["state_home"]
        / "kg-op/structural-hypothesis-report-adoption/v1"
        / bad_id
    )
    args = _real_adopt_args(
        context, adoption=bad_root, adoption_id=bad_id
    )
    index = args.index(option) + 1
    args[index] = _digest("f")
    completed = subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
            "XDG_STATE_HOME": str(context["state_home"]),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr
    assert not bad_root.exists()


def test_nested_source_leaf_aliases_and_fifo_fail_fast_before_adoption(
    self_contained_publication,
):
    context = self_contained_publication
    leaf = context["publication"] / "combined_rows.json"
    for kind in ("hardlink", "symlink", "fifo"):
        backup = context["state_home"] / f"combined-rows-{kind}.json"
        leaf.rename(backup)
        if kind == "hardlink":
            os.link(backup, leaf)
        elif kind == "symlink":
            leaf.symlink_to(backup)
        else:
            os.mkfifo(leaf, mode=0o600)
        adoption_id = f"adoption-source-{kind}-0001"
        adoption = (
            context["state_home"]
            / "kg-op/structural-hypothesis-report-adoption/v1"
            / adoption_id
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    *_real_adopt_args(
                        context,
                        adoption=adoption,
                        adoption_id=adoption_id,
                    ),
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(ROOT),
                    "XDG_STATE_HOME": str(context["state_home"]),
                },
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
            )
            assert completed.returncode == 2
            assert completed.stdout == ""
            assert "Traceback" not in completed.stderr
            assert not adoption.exists()
        finally:
            if leaf.exists() or leaf.is_symlink():
                leaf.unlink()
            backup.rename(leaf)


def test_captured_invalid_generation_cannot_hide_behind_live_valid_path(
    self_contained_publication, monkeypatch
):
    from performance import structural_hypothesis_report_adoption as core

    context = self_contained_publication
    publication = context["publication"]
    valid_generation = context["state_home"] / "valid-publication-generation"
    bad_generation = context["state_home"] / "bad-publication-generation"
    publication.rename(valid_generation)
    shutil.copytree(valid_generation, publication)
    combined_path = publication / "combined_rows.json"
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    combined[-1]["seed"] = str(combined[-1]["seed"])
    combined_path.write_text(
        json.dumps(combined, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    original_capture = core._capture_publication
    swapped = []

    def capture_bad_then_restore_live_valid(path):
        captured = original_capture(path)
        if not swapped:
            publication.rename(bad_generation)
            valid_generation.rename(publication)
            swapped.append(True)
        return captured

    monkeypatch.setattr(
        core, "_capture_publication", capture_bad_then_restore_live_valid
    )
    with pytest.raises(
        core.StructuralHypothesisReportAdoptionError,
        match="captured publication semantic artifact differs",
    ):
        _core_adopt(context, core)
    assert swapped == [True]
    assert not context["adoption"].exists()


def test_precommit_extra_file_injection_never_publishes_adoption_marker(
    self_contained_publication, monkeypatch
):
    from performance import structural_hypothesis_report_adoption as core

    context = self_contained_publication
    original_write = core._write_new_bytes
    injected = []

    def write_then_inject(path, raw):
        original_write(path, raw)
        if path == context["adoption"] / "publication/publication.json":
            unexpected = context["adoption"] / "unexpected.json"
            unexpected.write_text("{}\n", encoding="utf-8")
            unexpected.chmod(0o600)
            injected.append(unexpected)

    monkeypatch.setattr(core, "_write_new_bytes", write_then_inject)
    with pytest.raises(
        core.StructuralHypothesisReportAdoptionError,
        match="missing or unexpected artifacts",
    ):
        _core_adopt(context, core)
    assert len(injected) == 1
    assert context["adoption"].is_dir()
    assert not (context["adoption"] / "adoption.json").exists()
