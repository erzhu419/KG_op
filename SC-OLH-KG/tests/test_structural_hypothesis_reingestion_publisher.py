import csv
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

import runners.run_structural_hypothesis_reingestion as runner  # noqa: E402


RUNNER = ROOT / "runners/run_structural_hypothesis_reingestion.py"
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
    evidence = tmp_path / "base.csv"
    evidence.write_text("track,run_id\npriors,test\n", encoding="utf-8")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    assets = tmp_path / "assets"
    assets.mkdir()
    return {
        "base_evidence": evidence.resolve(),
        "attempt": attempt.resolve(),
        "hypothesis_contract": _json_file(
            tmp_path / "hypothesis-contract.json"
        ),
        "executor_contract": _json_file(tmp_path / "executor-contract.json"),
        "publisher_contract": _json_file(
            tmp_path / "publisher-contract.json"
        ),
        "runtime_contract": _json_file(tmp_path / "runtime-contract.json"),
        "base_manifest": _json_file(tmp_path / "base-manifest.json"),
        "asset_root": assets.resolve(),
        "publication": (tmp_path / "publication").resolve(),
    }


def _publish_argv(case):
    return [
        "publish",
        "--base-evidence-csv", str(case["base_evidence"]),
        "--attempt-root", str(case["attempt"]),
        "--hypothesis-contract", str(case["hypothesis_contract"]),
        "--executor-contract", str(case["executor_contract"]),
        "--publisher-contract", str(case["publisher_contract"]),
        "--runtime-contract", str(case["runtime_contract"]),
        "--base-manifest", str(case["base_manifest"]),
        "--asset-root", str(case["asset_root"]),
        "--publication-root", str(case["publication"]),
        "--expected-source-evidence-digest", _digest("1"),
        "--expected-plan-digest", _digest("2"),
        "--expected-authorization-digest", _digest("3"),
        "--expected-execution-receipt-digest", _digest("4"),
        "--expected-execution-journal-head-digest", _digest("5"),
        "--expected-execution-attempt-digest", _digest("6"),
        "--publication-id", "local-reingestion-0001",
        "--confirm-local-reingestion",
    ]


def _verify_argv(case):
    return [
        "verify",
        "--base-evidence-csv", str(case["base_evidence"]),
        "--attempt-root", str(case["attempt"]),
        "--hypothesis-contract", str(case["hypothesis_contract"]),
        "--executor-contract", str(case["executor_contract"]),
        "--publisher-contract", str(case["publisher_contract"]),
        "--runtime-contract", str(case["runtime_contract"]),
        "--base-manifest", str(case["base_manifest"]),
        "--asset-root", str(case["asset_root"]),
        "--publication-root", str(case["publication"]),
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
    ]


def _published_payload():
    return {
        "status": "PUBLISHED_NOT_ADOPTED",
        "accepted_successful_rows": 1,
        "ignored_failed_attempts": 0,
        "publication_digest": _digest("7"),
        "reingestion_digest": _digest("8"),
        "output_report_body_digest": _digest("9"),
        "output_audit_head": _digest("a"),
        "output_evidence_digest": _digest("b"),
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
                    "result_path": f"synthetic/{profile}/{domain}/seed{seed}",
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


@pytest.fixture
def self_contained_completed_attempt(monkeypatch):
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
        prefix="kgop-reingestion-publisher-test.", dir="/tmp"
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
    assert all(type(value) is str for row in base_rows for value in row.values())
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
    runtime_contract = json.loads(RUNTIME_CONTRACT.read_text(encoding="utf-8"))
    attempt = (
        state_home
        / "kg-op/structural-hypothesis-execution/v1/attempt-publisher-kat"
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
        authorization_id="publisher-kat-authorization-0001",
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
    publication_id = "publication-publisher-kat-0001"
    publication = (
        state_home
        / "kg-op/structural-hypothesis-reingestion/v1"
        / publication_id
    )
    context = {
        "state_home": state_home,
        "evidence": evidence,
        "base_rows": base_rows,
        "report": report,
        "attempt": attempt,
        "publication": publication,
        "publication_id": publication_id,
        "source_evidence_digest": report["evidence_digest"],
        "plan_digest": bundle["plan"]["integrity"]["plan_digest"],
        "authorization_digest": prepared["authorization_digest"],
        "execution_receipt_digest": executed["receipt_digest"],
        "execution_journal_head_digest": executed["journal_head_digest"],
        "execution_attempt_digest": executed["attempt_digest"],
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
    assert choices == {"publish", "verify"}

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
        raise AssertionError("publisher startup crossed an execution boundary")
    return original_import(name, *args, **kwargs)

builtins.__import__ = bomb_import
from runners.run_structural_hypothesis_reingestion import _parser
assert set(_parser()._subparsers._group_actions[0].choices) == {
    "publish", "verify"
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


def test_publish_wires_exact_chain_and_prints_only_canonical_anchor(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    calls = []

    def publish(*args, **kwargs):
        calls.append((args, kwargs))
        return _published_payload()

    monkeypatch.setattr(
        runner,
        "_load_publisher_core",
        lambda: SimpleNamespace(publish_single_task_reingestion=publish),
    )
    assert runner.main(_publish_argv(case)) == 0
    assert len(calls) == 1
    positional, keyword = calls[0]
    assert positional == (
        case["base_evidence"],
        case["attempt"],
        case["hypothesis_contract"],
        case["executor_contract"],
        case["runtime_contract"],
        case["publisher_contract"],
        case["base_manifest"],
        case["asset_root"],
        case["publication"],
    )
    assert keyword == {
        "publication_id": "local-reingestion-0001",
        "expected_source_evidence_digest": _digest("1"),
        "expected_plan_digest": _digest("2"),
        "expected_authorization_digest": _digest("3"),
        "expected_execution_receipt_digest": _digest("4"),
        "expected_execution_journal_head_digest": _digest("5"),
        "expected_execution_attempt_digest": _digest("6"),
    }
    captured = capsys.readouterr()
    expected = {
        "accepted_successful_rows": 1,
        "authorization_digest": _digest("3"),
        "execution_receipt_digest": _digest("4"),
        "ignored_failed_attempts": 0,
        "output_audit_head": _digest("a"),
        "output_evidence_digest": _digest("b"),
        "output_report_body_digest": _digest("9"),
        "plan_digest": _digest("2"),
        "publication_digest": _digest("7"),
        "publication_root": str(case["publication"]),
        "reingestion_digest": _digest("8"),
        "status": "PUBLISHED_NOT_ADOPTED",
    }
    assert captured.out == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert "action=publish" in captured.err
    assert "status=PUBLISHED_NOT_ADOPTED" in captured.err


def test_verify_wires_live_attempt_and_all_external_anchors_without_stdout(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    case["publication"].mkdir()
    calls = []

    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "VERIFIED_PUBLISHED_NOT_ADOPTED"}

    monkeypatch.setattr(
        runner,
        "_load_publisher_core",
        lambda: SimpleNamespace(
            verify_single_task_reingestion_publication=verify
        ),
    )
    assert runner.main(_verify_argv(case)) == 0
    assert calls == [(
        (
            case["base_evidence"],
            case["attempt"],
            case["hypothesis_contract"],
            case["executor_contract"],
            case["runtime_contract"],
            case["publisher_contract"],
            case["base_manifest"],
            case["asset_root"],
            case["publication"],
        ),
        {
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
        },
    )]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "action=verify" in captured.err
    assert "status=VERIFIED_PUBLISHED_NOT_ADOPTED" in captured.err


@pytest.mark.parametrize(
    "omitted",
    [
        "--attempt-root",
        "--expected-source-evidence-digest",
        "--expected-plan-digest",
        "--expected-authorization-digest",
        "--expected-execution-receipt-digest",
        "--expected-execution-journal-head-digest",
        "--expected-execution-attempt-digest",
        "--publication-id",
        "--confirm-local-reingestion",
    ],
)
def test_publish_requires_explicit_transition_and_every_upstream_anchor(
    tmp_path, omitted
):
    argv = _publish_argv(_unit_case(tmp_path))
    index = argv.index(omitted)
    del argv[index:index + (1 if omitted == "--confirm-local-reingestion" else 2)]
    with pytest.raises(SystemExit) as exc:
        runner._parser().parse_args(argv)
    assert exc.value.code == 2


def test_publish_has_no_dry_run_override_adoption_or_replan_surface():
    publish = runner._parser()._subparsers._group_actions[0].choices["publish"]
    forbidden = {
        "dry_run",
        "dry-run",
        "executor",
        "adopt",
        "current",
        "replan",
        "scheduler",
    }
    assert forbidden.isdisjoint(action.dest for action in publish._actions)


def test_publication_root_is_absolute_fresh_and_never_replaced(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        runner,
        "_load_publisher_core",
        lambda: (_ for _ in ()).throw(
            AssertionError("path gate must precede core import")
        ),
    )
    case = _unit_case(tmp_path / "existing-case")
    case["publication"].mkdir()
    assert runner.main(_publish_argv(case)) == 2
    assert "refusing to overwrite" in capsys.readouterr().err

    relative = _unit_case(tmp_path / "relative-case")
    argv = _publish_argv(relative)
    argv[argv.index("--publication-root") + 1] = "relative-publication"
    assert runner.main(argv) == 2
    assert "absolute path" in capsys.readouterr().err

    alias_case = _unit_case(tmp_path / "alias-case")
    target = tmp_path / "alias-target"
    target.mkdir()
    alias_case["publication"].symlink_to(target, target_is_directory=True)
    assert runner.main(_publish_argv(alias_case)) == 2
    assert "refusing to overwrite" in capsys.readouterr().err

    missing_verify = _unit_case(tmp_path / "verify-case")
    assert runner.main(_verify_argv(missing_verify)) == 2
    assert "existing non-symlink" in capsys.readouterr().err


def test_contract_core_rejection_digest_and_publication_id_are_fail_closed(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)
    case["publisher_contract"].write_text(
        '{"schema_version":"a","schema_version":"b"}\n',
        encoding="utf-8",
    )
    def reject_contract(*args, **kwargs):
        raise ValueError("publisher contract JSON has duplicate key 'schema_version'")

    monkeypatch.setattr(
        runner,
        "_load_publisher_core",
        lambda: SimpleNamespace(
            publish_single_task_reingestion=reject_contract
        ),
    )
    assert runner.main(_publish_argv(case)) == 2
    stderr = capsys.readouterr().err
    assert "duplicate key 'schema_version'" in stderr
    assert "Traceback" not in stderr

    digest_case = _unit_case(tmp_path / "digest")
    argv = _publish_argv(digest_case)
    argv[argv.index("--expected-plan-digest") + 1] = "sha256:ABC"
    assert runner.main(argv) == 2
    assert "lowercase sha256" in capsys.readouterr().err

    label_case = _unit_case(tmp_path / "label")
    argv = _publish_argv(label_case)
    argv[argv.index("--publication-id") + 1] = "../../adopt-current"
    assert runner.main(argv) == 2
    assert "non-path local mechanics label" in capsys.readouterr().err


def test_core_rejection_is_exit_two_empty_stdout_and_no_traceback(
    tmp_path, monkeypatch, capsys
):
    case = _unit_case(tmp_path)

    def reject(*args, **kwargs):
        raise ValueError("synthetic full-chain rejection")

    monkeypatch.setattr(
        runner,
        "_load_publisher_core",
        lambda: SimpleNamespace(
            publish_single_task_reingestion=reject
        ),
    )
    assert runner.main(_publish_argv(case)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "synthetic full-chain rejection" in captured.err
    assert "Traceback" not in captured.err


def _real_publish_args(context, *, publication=None, publication_id=None):
    return [
        "publish",
        "--base-evidence-csv", str(context["evidence"]),
        "--attempt-root", str(context["attempt"]),
        "--hypothesis-contract", str(HYPOTHESIS_CONTRACT),
        "--executor-contract", str(EXECUTOR_CONTRACT),
        "--runtime-contract", str(RUNTIME_CONTRACT),
        "--publisher-contract", str(PUBLISHER_CONTRACT),
        "--base-manifest", str(BASE_MANIFEST),
        "--asset-root", str(ASSET_ROOT),
        "--publication-root", str(publication or context["publication"]),
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
        "--publication-id", publication_id or context["publication_id"],
        "--confirm-local-reingestion",
    ]


def _publisher_subprocess(script, args, context):
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


def test_clean_checkout_fake_only_publish_verify_and_no_clobber(
    self_contained_completed_attempt,
):
    context = self_contained_completed_attempt
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
        raise AssertionError("publisher imported the native benchmark")
    return original_import(name, *args, **kwargs)

builtins.__import__ = bomb_import
from runners.run_structural_hypothesis_reingestion import main
raise SystemExit(main(sys.argv[1:]))
'''
    publish_args = _real_publish_args(context)
    made = _publisher_subprocess(script, publish_args, context)
    assert made.returncode == 0, made.stderr
    summary = json.loads(made.stdout)
    assert set(summary) == {
        "accepted_successful_rows",
        "authorization_digest",
        "execution_receipt_digest",
        "ignored_failed_attempts",
        "output_audit_head",
        "output_evidence_digest",
        "output_report_body_digest",
        "plan_digest",
        "publication_digest",
        "publication_root",
        "reingestion_digest",
        "status",
    }
    assert summary["status"] == "PUBLISHED_NOT_ADOPTED"
    assert summary["accepted_successful_rows"] == 1
    assert summary["ignored_failed_attempts"] == 0
    assert summary["publication_root"] == str(context["publication"])
    assert made.stdout == json.dumps(
        summary, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert "action=publish" in made.stderr

    publication = context["publication"]
    contract = json.loads(PUBLISHER_CONTRACT.read_text(encoding="utf-8"))
    directory_keys = {
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
        value
        for key, value in contract["artifact_layout"].items()
        if key not in directory_keys
    }
    observed_files = {
        str(path.relative_to(publication))
        for path in publication.rglob("*")
        if path.is_file()
    }
    observed_directories = {
        str(path.relative_to(publication))
        for path in publication.rglob("*")
        if path.is_dir()
    }
    assert observed_files == expected_files
    assert observed_directories == expected_directories
    assert oct(publication.stat().st_mode & 0o777) == "0o700"
    assert all(
        oct(path.stat().st_mode & 0o777) == "0o700"
        for path in publication.rglob("*")
        if path.is_dir()
    )
    assert all(
        oct(path.stat().st_mode & 0o777) == "0o600"
        for path in publication.rglob("*")
        if path.is_file()
    )
    assert (publication / "publication.json").is_file()
    combined = json.loads(
        (publication / "combined_rows.json").read_text(encoding="utf-8")
    )
    assert len(combined) == 271
    assert len(combined[0]) == 47
    assert all(type(value) is str for value in combined[0].values())
    assert len(combined[-1]) == 27
    assert type(combined[-1]["seed"]) is int
    assert type(combined[-1]["true_feasible"]) is bool
    output_report = json.loads(
        (publication / "output_report.json").read_text(encoding="utf-8")
    )
    assert len(output_report["pending_evidence"]) == 29

    before = {
        str(path.relative_to(publication)): path.read_bytes()
        for path in publication.rglob("*")
        if path.is_file()
    }
    repeated = _publisher_subprocess(script, publish_args, context)
    assert repeated.returncode == 2
    assert repeated.stdout == ""
    assert "refusing to overwrite" in repeated.stderr
    assert before == {
        str(path.relative_to(publication)): path.read_bytes()
        for path in publication.rglob("*")
        if path.is_file()
    }

    verify_args = [
        "verify",
        "--base-evidence-csv", str(context["evidence"]),
        "--attempt-root", str(context["attempt"]),
        "--hypothesis-contract", str(HYPOTHESIS_CONTRACT),
        "--executor-contract", str(EXECUTOR_CONTRACT),
        "--runtime-contract", str(RUNTIME_CONTRACT),
        "--publisher-contract", str(PUBLISHER_CONTRACT),
        "--base-manifest", str(BASE_MANIFEST),
        "--asset-root", str(ASSET_ROOT),
        "--publication-root", str(publication),
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
        "--expected-publication-digest", summary["publication_digest"],
        "--expected-reingestion-digest", summary["reingestion_digest"],
        "--expected-output-report-body-digest",
        summary["output_report_body_digest"],
        "--expected-output-audit-head", summary["output_audit_head"],
        "--expected-output-evidence-digest",
        summary["output_evidence_digest"],
    ]
    verified = _publisher_subprocess(script, verify_args, context)
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout == ""
    assert "action=verify" in verified.stderr
    assert "status=VERIFIED_PUBLISHED_NOT_ADOPTED" in verified.stderr


def test_wrong_independent_execution_anchor_creates_no_publication_root(
    self_contained_completed_attempt,
):
    context = self_contained_completed_attempt
    bad_id = "publication-bad-anchor-0001"
    bad_root = (
        context["state_home"]
        / "kg-op/structural-hypothesis-reingestion/v1"
        / bad_id
    )
    args = _real_publish_args(
        context, publication=bad_root, publication_id=bad_id
    )
    index = args.index("--expected-execution-receipt-digest") + 1
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
    assert not bad_root.exists()


def test_captured_preflight_tamper_cannot_hide_behind_live_verifier(
    self_contained_completed_attempt, monkeypatch
):
    from performance import structural_hypothesis_reingestion_publisher as core

    context = self_contained_completed_attempt
    preflight_path = context["attempt"] / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["observed"]["memory_available_bytes"] += 1
    preflight_path.write_text(
        json.dumps(preflight, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    live_verifier_calls = []

    def always_live_valid(*args, **kwargs):
        live_verifier_calls.append((args, kwargs))
        return {"status": "VERIFIED_COMPLETED"}

    monkeypatch.setattr(core, "_verify_runtime_attempt", always_live_valid)
    with pytest.raises(
        core.ReingestionPublicationError,
        match="captured execution capsule failed strong verification",
    ):
        core.publish_single_task_reingestion(
            context["evidence"],
            context["attempt"],
            HYPOTHESIS_CONTRACT,
            EXECUTOR_CONTRACT,
            RUNTIME_CONTRACT,
            PUBLISHER_CONTRACT,
            BASE_MANIFEST,
            ASSET_ROOT,
            context["publication"],
            publication_id=context["publication_id"],
            expected_source_evidence_digest=(
                context["source_evidence_digest"]
            ),
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
        )
    assert live_verifier_calls == []
    assert not context["publication"].exists()


def test_semantically_equivalent_csv_raw_rewrite_creates_no_publication_root(
    self_contained_completed_attempt,
):
    context = self_contained_completed_attempt
    context["evidence"].write_text(
        context["evidence"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(RUNNER), *_real_publish_args(context)],
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
    assert "raw evidence CSV digest differs" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not context["publication"].exists()


@pytest.mark.parametrize(
    ("option", "target_kind"),
    [
        ("--base-evidence-csv", "base-evidence"),
        ("--hypothesis-contract", "hypothesis-contract"),
    ],
)
def test_dotdot_input_aliases_are_rejected_before_publication(
    self_contained_completed_attempt, option, target_kind
):
    context = self_contained_completed_attempt
    if target_kind == "base-evidence":
        target = context["evidence"]
    else:
        target = context["state_home"] / "hypothesis-contract.json"
        shutil.copyfile(HYPOTHESIS_CONTRACT, target)
        target.chmod(0o600)
    alias_component = context["state_home"] / f"{target_kind}-alias"
    alias_component.mkdir(mode=0o700)
    dotdot_path = alias_component / ".." / target.name
    assert ".." in dotdot_path.parts

    publication_id = f"publication-dotdot-{target_kind}"
    publication = (
        context["state_home"]
        / "kg-op/structural-hypothesis-reingestion/v1"
        / publication_id
    )
    args = _real_publish_args(
        context,
        publication=publication,
        publication_id=publication_id,
    )
    args[args.index(option) + 1] = str(dotdot_path)
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
        timeout=5,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "canonical absolute path" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not publication.exists()


@pytest.mark.parametrize(
    ("option", "fifo_name"),
    [
        ("--base-evidence-csv", "base-evidence.fifo"),
        ("--hypothesis-contract", "hypothesis-contract.fifo"),
    ],
)
def test_fifo_inputs_fail_fast_without_opening_or_publishing(
    self_contained_completed_attempt, option, fifo_name
):
    context = self_contained_completed_attempt
    publication_id = "publication-fifo-" + fifo_name.split(".")[0]
    publication = (
        context["state_home"]
        / "kg-op/structural-hypothesis-reingestion/v1"
        / publication_id
    )
    fifo = context["state_home"] / fifo_name
    os.mkfifo(fifo, mode=0o600)
    args = _real_publish_args(
        context,
        publication=publication,
        publication_id=publication_id,
    )
    args[args.index(option) + 1] = str(fifo)
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
        timeout=5,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "not a regular file" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not publication.exists()
