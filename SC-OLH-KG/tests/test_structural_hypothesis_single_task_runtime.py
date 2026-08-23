import builtins
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

import runners.run_structural_hypothesis_single_task as runner  # noqa: E402


RUNNER = ROOT / "runners/run_structural_hypothesis_single_task.py"
CORE = ROOT / "performance/structural_hypothesis_single_task_runtime.py"
HYPOTHESIS_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_loop_v1.json"
)
EXECUTOR_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_executor_v1.json"
)
MATERIALIZER_CONTRACT = (
    ROOT / "performance/manifests/structural_hypothesis_task_materializer_v1.json"
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


def _json_file(path, payload=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {}), encoding="utf-8")
    return path


def _pin_runtime_environment(monkeypatch):
    for key, value in runner.REQUIRED_EXECUTION_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _real_case(tmp_path, *, state_home=None):
    from performance.structural_hypothesis_loop import (
        run_structural_hypothesis_loop,
        verify_report_integrity,
    )
    from performance.structural_hypothesis_task_materializer import (
        materialize_task_bundle,
        verify_materialized_task_bundle,
    )

    hypothesis = _load_json(HYPOTHESIS_CONTRACT)
    executor = _load_json(EXECUTOR_CONTRACT)
    materializer = _load_json(MATERIALIZER_CONTRACT)
    scope = hypothesis["evidence_scope"]
    rows = []
    for profile in PROFILES:
        for domain in scope["domains"]:
            for seed in scope["seeds"]:
                rows.append({
                    "track": scope["track"],
                    "run_id": scope["run_id"],
                    "variant": scope["variant_template"].format(
                        profile=profile
                    ),
                    "method": profile,
                    "structural_prior_profile": profile,
                    "domain": domain,
                    "seed": str(seed),
                    "d": str(scope["d"]),
                    "N": str(scope["N"]),
                    "n0": str(scope["n0"]),
                    "source_calls": str(scope["source_calls"]),
                    "implementation": scope["implementation"],
                    "initial_design": scope["initial_design"],
                    "decision_backend": scope["decision_backend"],
                    "status": "ok",
                    "true_feasible": seed < 9,
                    "adaptive_loss": False,
                    "feasible_regret": "1.0",
                    **scope["fixed_row_values"],
                })
    report = run_structural_hypothesis_loop(rows, hypothesis).to_dict()
    assert verify_report_integrity(report)

    state_home = state_home or (tmp_path / "state")
    attempt = (
        state_home
        / "kg-op/structural-hypothesis-execution/v1/attempt-0001"
    )
    checkpoint_root = attempt / "checkpoints"
    bundle = materialize_task_bundle(
        report,
        hypothesis,
        executor,
        materializer,
        BASE_MANIFEST,
        ASSET_ROOT,
        checkpoint_root,
    )
    assert verify_materialized_task_bundle(
        bundle,
        report,
        hypothesis,
        executor,
        materializer,
        BASE_MANIFEST,
        ASSET_ROOT,
        checkpoint_root,
    )
    report_path = _json_file(tmp_path / "report.json", report)
    bundle_path = _json_file(tmp_path / "bundle.json", bundle)
    first = bundle["plan"]["tasks"][0]
    return {
        "state_home": state_home,
        "attempt": attempt,
        "report": report_path,
        "bundle": bundle_path,
        "task_id": first["task_id"],
        "bundle_digest": bundle["integrity"]["bundle_digest"],
        "plan_digest": bundle["plan"]["integrity"]["plan_digest"],
    }


@pytest.fixture
def prepared_core_attempt(tmp_path, monkeypatch):
    from performance import structural_hypothesis_single_task_runtime as core

    state_home = Path(tempfile.mkdtemp(
        prefix="kgop-single-task-core-test.", dir="/tmp"
    ))
    state_home.chmod(0o700)
    _pin_runtime_environment(monkeypatch)
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    case = _real_case(tmp_path, state_home=state_home)
    bundle = _load_json(case["bundle"])
    runtime_contract = _load_json(RUNTIME_CONTRACT)
    prepared = core.prepare_single_task_attempt(
        _load_json(case["report"]),
        bundle,
        _load_json(HYPOTHESIS_CONTRACT),
        _load_json(EXECUTOR_CONTRACT),
        _load_json(MATERIALIZER_CONTRACT),
        runtime_contract,
        BASE_MANIFEST,
        ASSET_ROOT,
        case["attempt"],
        task_id=case["task_id"],
        expected_bundle_digest=case["bundle_digest"],
        expected_plan_digest=case["plan_digest"],
        authorization_id="focused-fake-execution-0001",
    )
    context = {
        **case,
        "core": core,
        "bundle_object": bundle,
        "task": bundle["plan"]["tasks"][0],
        "runtime_contract": runtime_contract,
        "prepared": prepared,
    }
    try:
        yield context
    finally:
        shutil.rmtree(state_home, ignore_errors=True)


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


def _valid_fake_preflight(context):
    core = context["core"]
    task = context["task"]
    contract = context["runtime_contract"]
    binding = contract["runtime_binding"]
    callable_source = (
        ROOT / "performance/benchmark_lodo_meta_prior.py"
    ).resolve()
    body = {
        "schema_version": core.PREFLIGHT_SCHEMA_VERSION,
        "status": "PASSED_LOCAL_PREFLIGHT",
        "attempt_digest": context["prepared"]["attempt_digest"],
        "authorization_digest": context["prepared"][
            "authorization_digest"
        ],
        "task_id": task["task_id"],
        "task_digest": task["task_digest"],
        "requirements": contract["preflight"],
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
                "source_file": str(callable_source),
                "source_sha256": binding[
                    "executor_callable_source_sha256"
                ],
                "code_sha256": binding["executor_callable_code_sha256"],
                "firstlineno": binding[
                    "executor_callable_firstlineno"
                ],
            },
            "required_environment": dict(core._REQUIRED_ENVIRONMENT),
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
            "preflight_digest": core._digest(body),
        },
    }


def _install_fake_execution_boundary(monkeypatch, context, executor):
    core = context["core"]
    monkeypatch.setattr(core, "_load_real_executor", lambda _contract: executor)
    monkeypatch.setattr(
        core,
        "_run_preflight",
        lambda *args, **kwargs: _valid_fake_preflight(context),
    )


def _pin_good_resource_probes(monkeypatch, core):
    import threadpoolctl

    observed = {
        "pools": [{
            "user_api": "blas",
            "internal_api": "openblas",
            "prefix": "libopenblas",
            "version": "test-only",
            "threading_layer": "pthreads",
            "num_threads": 1,
        }],
    }
    monkeypatch.setattr(core.os, "sched_getaffinity", lambda _pid: set(range(12)))
    monkeypatch.setattr(
        core, "_memory_available_bytes", lambda: 12884901888
    )
    monkeypatch.setattr(
        core.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=2147483648, f_frsize=1),
    )
    monkeypatch.setattr(
        threadpoolctl, "threadpool_info", lambda: observed["pools"]
    )
    monkeypatch.setattr(core, "_fork_probe", lambda process_count: process_count)
    return observed


def _prepare_argv(tmp_path, *, attempt_root=None):
    report = _json_file(tmp_path / "report.json")
    bundle = _json_file(tmp_path / "bundle.json")
    hypothesis = _json_file(tmp_path / "hypothesis.json")
    executor = _json_file(tmp_path / "executor.json")
    materializer = _json_file(tmp_path / "materializer.json")
    runtime = _json_file(tmp_path / "runtime.json")
    base = _json_file(tmp_path / "base.json")
    assets = tmp_path / "assets"
    assets.mkdir()
    attempt = attempt_root or (tmp_path / "fresh-attempt")
    return [
        "prepare",
        "--report", str(report),
        "--bundle", str(bundle),
        "--hypothesis-contract", str(hypothesis),
        "--executor-contract", str(executor),
        "--materializer-contract", str(materializer),
        "--runtime-contract", str(runtime),
        "--base-manifest", str(base),
        "--asset-root", str(assets),
        "--attempt-root", str(attempt),
        "--task-id", "task:first-pending",
        "--expected-bundle-digest", "sha256:" + "1" * 64,
        "--expected-plan-digest", "sha256:" + "2" * 64,
        "--authorization-id", "local-consent-0001",
    ]


def test_runner_import_surface_is_stdlib_only_and_has_exact_actions(tmp_path):
    source = RUNNER.read_text(encoding="utf-8")
    assert "benchmark_lodo_meta_prior" not in source
    assert "benchmark_quality" not in source
    assert "import subprocess" not in source
    assert "import numpy" not in source
    assert "import scheduler" not in source.lower()
    assert "from scheduler" not in source.lower()

    choices = set(runner._parser()._subparsers._group_actions[0].choices)
    assert choices == {"prepare", "execute", "verify"}

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
        raise AssertionError("runner startup crossed the runtime boundary")
    return original_import(name, *args, **kwargs)

builtins.__import__ = bomb_import
from runners.run_structural_hypothesis_single_task import _parser
assert set(_parser()._subparsers._group_actions[0].choices) == {
    "prepare", "execute", "verify"
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


def test_environment_gate_is_exact_and_runs_before_core_import(
    tmp_path, monkeypatch, capsys
):
    argv = _prepare_argv(tmp_path)
    for key in runner.REQUIRED_EXECUTION_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)

    imported = False

    def forbidden_import():
        nonlocal imported
        imported = True
        raise AssertionError("core must not import before the environment gate")

    monkeypatch.setattr(runner, "_load_runtime_core", forbidden_import)
    assert runner.main(argv) == 2
    assert not imported
    stderr = capsys.readouterr().err
    assert "before Python startup" in stderr
    assert "OPENBLAS_NUM_THREADS=None" in stderr
    assert "Traceback" not in stderr

    _pin_runtime_environment(monkeypatch)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    assert runner.main(argv) == 2
    assert not imported
    assert "OPENBLAS_NUM_THREADS='2'" in capsys.readouterr().err


def test_prepare_wires_one_explicit_task_and_digest_set(
    tmp_path, monkeypatch, capsys
):
    argv = _prepare_argv(tmp_path)
    _pin_runtime_environment(monkeypatch)
    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "status": "PREPARED_NOT_EXECUTED",
            "integrity": {
                "attempt_digest": "sha256:" + "4" * 64,
                "authorization_digest": "sha256:" + "3" * 64,
            },
        }

    monkeypatch.setattr(
        runner,
        "_load_runtime_core",
        lambda: SimpleNamespace(prepare_single_task_attempt=prepare),
    )
    assert runner.main(argv) == 0
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 9
    assert kwargs == {
        "task_id": "task:first-pending",
        "expected_bundle_digest": "sha256:" + "1" * 64,
        "expected_plan_digest": "sha256:" + "2" * 64,
        "authorization_id": "local-consent-0001",
    }
    assert not Path(args[-1]).exists()
    captured = capsys.readouterr()
    assert captured.out == json.dumps({
        "attempt_root": str((tmp_path / "fresh-attempt").resolve()),
        "attempt_digest": "sha256:" + "4" * 64,
        "authorization_digest": "sha256:" + "3" * 64,
        "bundle_digest": "sha256:" + "1" * 64,
        "plan_digest": "sha256:" + "2" * 64,
        "status": "PREPARED_NOT_EXECUTED",
        "task_id": "task:first-pending",
    }, sort_keys=True, separators=(",", ":")) + "\n"
    stderr = captured.err
    assert "action=prepare" in stderr
    assert "status=PREPARED_NOT_EXECUTED" in stderr


@pytest.mark.parametrize(
    "omitted",
    [
        "--task-id",
        "--expected-bundle-digest",
        "--expected-plan-digest",
        "--authorization-id",
    ],
)
def test_prepare_requires_every_explicit_authorization_field(
    tmp_path, omitted
):
    argv = _prepare_argv(tmp_path)
    index = argv.index(omitted)
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as exc:
        runner._parser().parse_args(argv)
    assert exc.value.code == 2


def test_execute_is_explicit_real_only_and_forwards_exact_authorization(
    tmp_path, monkeypatch, capsys
):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    runtime = _json_file(tmp_path / "runtime.json")
    digest = "sha256:" + "4" * 64
    argv = [
        "execute",
        "--attempt-root", str(attempt),
        "--runtime-contract", str(runtime),
        "--expected-authorization-digest", digest,
        "--confirm-real-local-execution",
    ]
    _pin_runtime_environment(monkeypatch)
    calls = []

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "status": "EXECUTED_RECEIPT_WRITTEN",
            "authorization_digest": digest,
            "receipt_digest": "sha256:" + "5" * 64,
            "journal_head_digest": "sha256:" + "6" * 64,
            "attempt_digest": "sha256:" + "7" * 64,
        }

    monkeypatch.setattr(
        runner,
        "_load_runtime_core",
        lambda: SimpleNamespace(execute_single_task_attempt=execute),
    )
    assert runner.main(argv) == 0
    assert calls == [
        (
            (attempt, {}),
            {"expected_authorization_digest": digest},
        )
    ]
    captured = capsys.readouterr()
    assert captured.out == json.dumps({
        "attempt_digest": "sha256:" + "7" * 64,
        "authorization_digest": digest,
        "journal_head_digest": "sha256:" + "6" * 64,
        "receipt_digest": "sha256:" + "5" * 64,
    }, sort_keys=True, separators=(",", ":")) + "\n"
    assert "action=execute" in captured.err

    without_confirmation = argv[:-1]
    with pytest.raises(SystemExit) as exc:
        runner._parser().parse_args(without_confirmation)
    assert exc.value.code == 2
    choices = runner._parser()._subparsers._group_actions[0].choices["execute"]
    assert not any(action.dest in {"executor", "dry_run", "dry-run"}
                   for action in choices._actions)


def test_verify_needs_no_execution_environment_and_uses_no_executor(
    tmp_path, monkeypatch, capsys
):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    runtime = _json_file(tmp_path / "runtime.json")
    base = _json_file(tmp_path / "base.json")
    assets = tmp_path / "assets"
    assets.mkdir()
    digest = "sha256:" + "6" * 64
    for key in runner.REQUIRED_EXECUTION_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)
    calls = []

    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(
        runner,
        "_load_runtime_core",
        lambda: SimpleNamespace(verify_single_task_attempt=verify),
    )
    assert runner.main([
        "verify",
        "--attempt-root", str(attempt),
        "--runtime-contract", str(runtime),
        "--base-manifest", str(base),
        "--asset-root", str(assets),
        "--expected-authorization-digest", digest,
    ]) == 0
    assert calls == [
        (
            (attempt, {}, base, assets),
            {
                "expected_authorization_digest": digest,
                "expected_receipt_digest": None,
                "expected_journal_head_digest": None,
                "expected_attempt_digest": None,
            },
        )
    ]
    assert "status=VERIFIED" in capsys.readouterr().err


def test_attempt_root_is_absolute_fresh_and_never_a_symlink(
    tmp_path, monkeypatch, capsys
):
    _pin_runtime_environment(monkeypatch)
    existing = tmp_path / "existing"
    existing.mkdir()
    assert runner.main(_prepare_argv(tmp_path / "case1", attempt_root=existing)) == 2
    assert "fresh attempt is required" in capsys.readouterr().err

    relative = _prepare_argv(tmp_path / "case2")
    relative[relative.index("--attempt-root") + 1] = "relative-attempt"
    assert runner.main(relative) == 2
    assert "absolute path" in capsys.readouterr().err

    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    runtime = _json_file(tmp_path / "runtime.json")
    assert runner.main([
        "execute",
        "--attempt-root", str(alias),
        "--runtime-contract", str(runtime),
        "--expected-authorization-digest", "sha256:" + "7" * 64,
        "--confirm-real-local-execution",
    ]) == 2
    assert "non-symlink directory" in capsys.readouterr().err


def test_duplicate_prepare_json_is_rejected_without_traceback(
    tmp_path, monkeypatch, capsys
):
    argv = _prepare_argv(tmp_path)
    report = Path(argv[argv.index("--report") + 1])
    report.write_text('{"audit": {}, "audit": {}}\n', encoding="utf-8")
    _pin_runtime_environment(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_load_runtime_core",
        lambda: SimpleNamespace(
            prepare_single_task_attempt=lambda *args, **kwargs: None
        ),
    )
    assert runner.main(argv) == 2
    stderr = capsys.readouterr().err
    assert "duplicate key 'audit'" in stderr
    assert "Traceback" not in stderr


@pytest.mark.skipif(not CORE.is_file(), reason="runtime core is landing concurrently")
def test_real_prepare_and_authorized_verify_are_benchmark_free_no_clobber(
    tmp_path, request,
):
    # The app-level pytest temp root can be a drvfs mount whose reported mode
    # cannot satisfy the runtime's POSIX 0700 policy.  Runtime state itself is
    # exercised on the native Linux filesystem.
    state_home = Path(tempfile.mkdtemp(
        prefix="kgop-single-task-runtime-test.", dir="/tmp"
    ))
    state_home.chmod(0o700)
    request.addfinalizer(lambda: shutil.rmtree(state_home, ignore_errors=True))
    case = _real_case(tmp_path, state_home=state_home)
    prepare_args = [
        "prepare",
        "--report", str(case["report"]),
        "--bundle", str(case["bundle"]),
        "--hypothesis-contract", str(HYPOTHESIS_CONTRACT),
        "--executor-contract", str(EXECUTOR_CONTRACT),
        "--materializer-contract", str(MATERIALIZER_CONTRACT),
        "--runtime-contract", str(RUNTIME_CONTRACT),
        "--base-manifest", str(BASE_MANIFEST),
        "--asset-root", str(ASSET_ROOT),
        "--attempt-root", str(case["attempt"]),
        "--task-id", case["task_id"],
        "--expected-bundle-digest", case["bundle_digest"],
        "--expected-plan-digest", case["plan_digest"],
        "--authorization-id", "focused-local-consent-0001",
    ]
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
        raise AssertionError("non-execution command imported the benchmark")
    return original_import(name, *args, **kwargs)

builtins.__import__ = bomb_import
from runners.run_structural_hypothesis_single_task import main
raise SystemExit(main(sys.argv[1:]))
'''
    env = {
        **os.environ,
        **runner.REQUIRED_EXECUTION_ENVIRONMENT,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT),
        "XDG_STATE_HOME": str(case["state_home"]),
    }
    made = subprocess.run(
        [sys.executable, "-c", script, *prepare_args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert made.returncode == 0, made.stderr
    summary = json.loads(made.stdout)
    assert set(summary) == {
        "status",
        "attempt_root",
        "attempt_digest",
        "task_id",
        "authorization_digest",
        "plan_digest",
        "bundle_digest",
    }
    assert summary["status"] == "AUTHORIZED"
    assert summary["attempt_root"] == str(case["attempt"].resolve())
    assert summary["task_id"] == case["task_id"]
    assert summary["plan_digest"] == case["plan_digest"]
    assert summary["bundle_digest"] == case["bundle_digest"]
    assert summary["authorization_digest"].startswith("sha256:")
    assert summary["attempt_digest"].startswith("sha256:")
    assert "action=prepare" in made.stderr

    attempt = case["attempt"]
    assert oct(attempt.stat().st_mode & 0o777) == "0o700"
    assert {item.name for item in attempt.iterdir()} == {
        "attempt.json",
        "bundle.json",
        "authorization.json",
        "inputs",
        "journal",
        "checkpoints",
    }
    assert {item.name for item in (attempt / "inputs").iterdir()} == {
        "report.json",
        "hypothesis_contract.json",
        "executor_contract.json",
        "materializer_contract.json",
    }
    assert {item.name for item in (attempt / "journal").iterdir()} == {
        "0000_AUTHORIZED.json",
    }
    assert not (attempt / "raw_result.json").exists()
    assert not (attempt / "receipt.json").exists()
    assert all(
        oct(path.stat().st_mode & 0o777) == "0o600"
        for path in attempt.rglob("*.json")
    )
    task = _load_json(case["bundle"])["plan"]["tasks"][0]["run_one_task"]
    checkpoint_dir = Path(task["args"]["runtime_checkpoint_dir"])
    assert checkpoint_dir.is_dir()
    assert not any(checkpoint_dir.iterdir())

    before = {
        str(path.relative_to(attempt)): path.read_bytes()
        for path in attempt.rglob("*")
        if path.is_file()
    }
    repeated = subprocess.run(
        [sys.executable, "-c", script, *prepare_args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode == 2
    assert "fresh attempt is required" in repeated.stderr
    assert before == {
        str(path.relative_to(attempt)): path.read_bytes()
        for path in attempt.rglob("*")
        if path.is_file()
    }

    verified = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "verify",
            "--attempt-root", str(attempt),
            "--runtime-contract", str(RUNTIME_CONTRACT),
            "--base-manifest", str(BASE_MANIFEST),
            "--asset-root", str(ASSET_ROOT),
            "--expected-authorization-digest",
            summary["authorization_digest"],
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout == ""
    assert "action=verify" in verified.stderr


def test_fake_execute_publishes_full_chain_and_verifies_external_anchors(
    prepared_core_attempt, monkeypatch
):
    import numpy as np

    context = prepared_core_attempt
    core = context["core"]
    calls = []

    def fake_executor(task):
        calls.append(task)
        raw = _fake_native_result(task)
        raw["numpy_scalar_probe"] = np.int64(7)
        return raw

    _install_fake_execution_boundary(monkeypatch, context, fake_executor)
    receipt = core.execute_single_task_attempt(
        context["attempt"],
        context["runtime_contract"],
        expected_authorization_digest=context["prepared"][
            "authorization_digest"
        ],
    )
    assert len(calls) == 1
    assert calls[0] == context["task"]["run_one_task"]
    attempt = context["attempt"]
    assert {
        "preflight.json",
        "raw_result.json",
        "receipt.json",
    }.issubset({item.name for item in attempt.iterdir()})
    assert {item.name for item in (attempt / "journal").iterdir()} == {
        "0000_AUTHORIZED.json",
        "0001_RUNNING.json",
        "0002_COMPLETED.json",
    }
    raw = _load_json(attempt / "raw_result.json")
    assert raw["native_result"]["numpy_scalar_probe"] == 7
    assert type(raw["native_result"]["numpy_scalar_probe"]) is int
    saved_receipt = _load_json(attempt / "receipt.json")
    assert saved_receipt["status"] == "COMPLETED"
    assert saved_receipt["summary"] == {
        "authorized": 1,
        "succeeded": 1,
        "failed": 0,
    }
    verified = core.verify_single_task_attempt(
        attempt,
        context["runtime_contract"],
        BASE_MANIFEST,
        ASSET_ROOT,
        expected_authorization_digest=receipt["authorization_digest"],
        expected_receipt_digest=receipt["receipt_digest"],
        expected_journal_head_digest=receipt["journal_head_digest"],
        expected_attempt_digest=receipt["attempt_digest"],
    )
    assert verified == {
        "status": "VERIFIED_COMPLETED",
        "authorization_digest": receipt["authorization_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "journal_head_digest": receipt["journal_head_digest"],
        "attempt_digest": receipt["attempt_digest"],
    }


def test_fake_callback_secret_is_not_persisted_and_failure_verifies(
    prepared_core_attempt, monkeypatch
):
    context = prepared_core_attempt
    core = context["core"]
    secret = "SYNTHETIC-SECRET-MUST-NOT-PERSIST-9741"
    calls = []

    def crash(task):
        calls.append(task)
        raise RuntimeError(secret)

    _install_fake_execution_boundary(monkeypatch, context, crash)
    result = core.execute_single_task_attempt(
        context["attempt"],
        context["runtime_contract"],
        expected_authorization_digest=context["prepared"][
            "authorization_digest"
        ],
    )
    assert len(calls) == 1
    attempt = context["attempt"]
    assert not (attempt / "raw_result.json").exists()
    receipt = _load_json(attempt / "receipt.json")
    assert receipt["status"] == "COMPLETED_WITH_FAILURES"
    assert receipt["summary"] == {
        "authorized": 1,
        "succeeded": 0,
        "failed": 1,
    }
    assert receipt["results"][0]["error"] == {
        "code": "EXECUTOR_EXCEPTION",
        "type": "RuntimeError",
    }
    assert all(
        secret.encode("utf-8") not in path.read_bytes()
        for path in attempt.rglob("*")
        if path.is_file()
    )
    verified = core.verify_single_task_attempt(
        attempt,
        context["runtime_contract"],
        BASE_MANIFEST,
        ASSET_ROOT,
        expected_authorization_digest=result["authorization_digest"],
        expected_receipt_digest=result["receipt_digest"],
        expected_journal_head_digest=result["journal_head_digest"],
        expected_attempt_digest=result["attempt_digest"],
    )
    assert verified["status"] == "VERIFIED_COMPLETED"


def test_running_attempt_is_verifiable_incomplete_and_never_reentered(
    prepared_core_attempt, monkeypatch
):
    context = prepared_core_attempt
    core = context["core"]
    callback_calls = []
    loader_calls = []

    def fake_executor(task):
        callback_calls.append(task)
        return _fake_native_result(task)

    def fake_loader(_contract):
        loader_calls.append(True)
        return fake_executor

    monkeypatch.setattr(core, "_load_real_executor", fake_loader)
    monkeypatch.setattr(
        core,
        "_run_preflight",
        lambda *args, **kwargs: _valid_fake_preflight(context),
    )
    captured_execute = core._CAPTURED_EXECUTE_AUTHORIZED_PLAN
    monkeypatch.setattr(
        core,
        "_CAPTURED_EXECUTE_AUTHORIZED_PLAN",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic interruption after RUNNING")
        ),
    )
    with pytest.raises(RuntimeError, match="after RUNNING"):
        core.execute_single_task_attempt(
            context["attempt"],
            context["runtime_contract"],
            expected_authorization_digest=context["prepared"][
                "authorization_digest"
            ],
        )
    monkeypatch.setattr(
        core, "_CAPTURED_EXECUTE_AUTHORIZED_PLAN", captured_execute
    )
    assert callback_calls == []
    assert loader_calls == [True]
    attempt = context["attempt"]
    assert (attempt / "preflight.json").is_file()
    assert (attempt / "journal/0001_RUNNING.json").is_file()
    assert not (attempt / "receipt.json").exists()
    verified = core.verify_single_task_attempt(
        attempt,
        context["runtime_contract"],
        BASE_MANIFEST,
        ASSET_ROOT,
        expected_authorization_digest=context["prepared"][
            "authorization_digest"
        ],
    )
    assert verified["status"] == "VERIFIED_RUNNING_INCOMPLETE_NO_REENTRY"
    with pytest.raises(core.SingleTaskRuntimeValidationError):
        core.execute_single_task_attempt(
            attempt,
            context["runtime_contract"],
            expected_authorization_digest=context["prepared"][
                "authorization_digest"
            ],
        )
    assert callback_calls == []
    assert loader_calls == [True]


def test_preflight_only_crash_is_verifiable_but_requires_new_attempt(
    prepared_core_attempt, monkeypatch
):
    context = prepared_core_attempt
    core = context["core"]
    callback_calls = []
    loader_calls = []

    def fake_executor(task):
        callback_calls.append(task)
        return _fake_native_result(task)

    def fake_loader(_contract):
        loader_calls.append(True)
        return fake_executor

    monkeypatch.setattr(core, "_load_real_executor", fake_loader)
    monkeypatch.setattr(
        core,
        "_run_preflight",
        lambda *args, **kwargs: _valid_fake_preflight(context),
    )
    write_new_json = core._write_new_json

    def interrupt_running(path, payload):
        if path.name == "0001_RUNNING.json":
            raise RuntimeError("synthetic interruption before RUNNING")
        return write_new_json(path, payload)

    monkeypatch.setattr(core, "_write_new_json", interrupt_running)
    with pytest.raises(RuntimeError, match="before RUNNING"):
        core.execute_single_task_attempt(
            context["attempt"],
            context["runtime_contract"],
            expected_authorization_digest=context["prepared"][
                "authorization_digest"
            ],
        )
    monkeypatch.setattr(core, "_write_new_json", write_new_json)
    assert callback_calls == []
    assert loader_calls == [True]
    attempt = context["attempt"]
    assert (attempt / "preflight.json").is_file()
    assert not (attempt / "journal/0001_RUNNING.json").exists()
    verified = core.verify_single_task_attempt(
        attempt,
        context["runtime_contract"],
        BASE_MANIFEST,
        ASSET_ROOT,
        expected_authorization_digest=context["prepared"][
            "authorization_digest"
        ],
    )
    assert verified["status"] == "VERIFIED_PREFLIGHT_PASSED_NO_CALLBACK"
    with pytest.raises(core.SingleTaskRuntimeValidationError):
        core.execute_single_task_attempt(
            attempt,
            context["runtime_contract"],
            expected_authorization_digest=context["prepared"][
                "authorization_digest"
            ],
        )
    assert callback_calls == []
    assert loader_calls == [True]


def test_checkpoint_injection_refuses_before_executor_load_or_callback(
    prepared_core_attempt, monkeypatch
):
    context = prepared_core_attempt
    core = context["core"]
    loader_calls = []
    callback_calls = []

    def fake_executor(task):
        callback_calls.append(task)
        return _fake_native_result(task)

    def fake_loader(_contract):
        loader_calls.append(True)
        return fake_executor

    monkeypatch.setattr(core, "_load_real_executor", fake_loader)
    checkpoint = Path(
        context["task"]["run_one_task"]["args"]["runtime_checkpoint_dir"]
    ) / "checkpoint_latest.pkl"
    checkpoint.write_bytes(b"unbound-pickle-must-never-be-loaded")
    checkpoint.chmod(0o600)
    with pytest.raises(core.SingleTaskRuntimeValidationError):
        core.execute_single_task_attempt(
            context["attempt"],
            context["runtime_contract"],
            expected_authorization_digest=context["prepared"][
                "authorization_digest"
            ],
        )
    assert loader_calls == []
    assert callback_calls == []
    assert not (context["attempt"] / "preflight.json").exists()
    assert not (context["attempt"] / "journal/0001_RUNNING.json").exists()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("blas", "thread pool"),
        ("cpu", "CPU affinity"),
        ("memory", "MemAvailable"),
        ("disk", "less than 2 GiB"),
        ("fork", "fork probe failed"),
    ],
)
def test_resource_preflight_failures_are_before_running_and_callback(
    prepared_core_attempt, monkeypatch, failure, message
):
    context = prepared_core_attempt
    core = context["core"]
    callback_calls = []

    def fake_executor(task):
        callback_calls.append(task)
        return _fake_native_result(task)

    monkeypatch.setattr(
        core, "_load_real_executor", lambda _contract: fake_executor
    )
    observed = _pin_good_resource_probes(monkeypatch, core)
    if failure == "blas":
        observed["pools"][0]["num_threads"] = 2
    elif failure == "cpu":
        monkeypatch.setattr(
            core.os, "sched_getaffinity", lambda _pid: set(range(11))
        )
    elif failure == "memory":
        monkeypatch.setattr(
            core, "_memory_available_bytes", lambda: 12884901887
        )
    elif failure == "disk":
        monkeypatch.setattr(
            core.os,
            "statvfs",
            lambda _path: SimpleNamespace(
                f_bavail=2147483647, f_frsize=1
            ),
        )
    else:
        def fail_fork(_process_count):
            raise core.SingleTaskRuntimeValidationError(
                "12-process local fork probe failed"
            )

        monkeypatch.setattr(core, "_fork_probe", fail_fork)
    with pytest.raises(core.SingleTaskRuntimeValidationError, match=message):
        core.execute_single_task_attempt(
            context["attempt"],
            context["runtime_contract"],
            expected_authorization_digest=context["prepared"][
                "authorization_digest"
            ],
        )
    assert callback_calls == []
    assert not (context["attempt"] / "preflight.json").exists()
    assert not (context["attempt"] / "journal/0001_RUNNING.json").exists()


@pytest.mark.parametrize("mutation", ["child_symlink", "json_hardlink", "extra"])
def test_layout_alias_and_extra_artifacts_refuse_before_executor_load(
    prepared_core_attempt, monkeypatch, mutation
):
    context = prepared_core_attempt
    core = context["core"]
    load_calls = []

    def forbidden_loader(_contract):
        load_calls.append(True)
        raise AssertionError("layout rejection must precede executor load")

    monkeypatch.setattr(core, "_load_real_executor", forbidden_loader)
    attempt = context["attempt"]
    if mutation == "child_symlink":
        inputs = attempt / "inputs"
        saved = attempt.parent / "saved-inputs"
        inputs.rename(saved)
        inputs.symlink_to(saved, target_is_directory=True)
    elif mutation == "json_hardlink":
        os.link(
            attempt / "attempt.json",
            attempt.parent / "attempt-hardlink.json",
        )
    else:
        unexpected = attempt / "unexpected.json"
        unexpected.write_text("{}\n", encoding="utf-8")
        unexpected.chmod(0o600)
    with pytest.raises(core.SingleTaskRuntimeValidationError):
        core.execute_single_task_attempt(
            attempt,
            context["runtime_contract"],
            expected_authorization_digest=context["prepared"][
                "authorization_digest"
            ],
        )
    assert load_calls == []
    assert not (attempt / "preflight.json").exists()
    assert not (attempt / "journal/0001_RUNNING.json").exists()


def test_callable_label_spoof_is_rejected_before_running_and_callback(
    prepared_core_attempt, monkeypatch
):
    context = prepared_core_attempt
    core = context["core"]
    callback_calls = []

    def spoofed(task):
        callback_calls.append(task)
        return _fake_native_result(task)

    spoofed.__module__ = "performance.benchmark_lodo_meta_prior"
    spoofed.__name__ = "run_one"
    monkeypatch.setattr(core, "_load_real_executor", lambda _contract: spoofed)
    _pin_good_resource_probes(monkeypatch, core)
    with pytest.raises(
        core.SingleTaskRuntimeValidationError,
        match="callable source binding differs",
    ):
        core.execute_single_task_attempt(
            context["attempt"],
            context["runtime_contract"],
            expected_authorization_digest=context["prepared"][
                "authorization_digest"
            ],
        )
    assert callback_calls == []
    assert not (context["attempt"] / "preflight.json").exists()
    assert not (context["attempt"] / "journal/0001_RUNNING.json").exists()
