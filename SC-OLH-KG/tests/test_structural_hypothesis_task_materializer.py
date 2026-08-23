import builtins
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.structural_hypothesis_loop import (  # noqa: E402
    run_structural_hypothesis_loop,
    verify_report_integrity,
)
from performance.structural_hypothesis_task_materializer import (  # noqa: E402
    materialize_task_bundle,
    validate_materializer_contract,
    verify_materialized_task_bundle,
)


HYPOTHESIS_CONTRACT_PATH = (
    ROOT / "performance/manifests/structural_hypothesis_loop_v1.json"
)
EXECUTOR_CONTRACT_PATH = (
    ROOT / "performance/manifests/structural_hypothesis_executor_v1.json"
)
MATERIALIZER_CONTRACT_PATH = (
    ROOT
    / "performance/manifests/structural_hypothesis_task_materializer_v1.json"
)
BASE_MANIFEST_PATH = ROOT / "performance/manifests/v18b_exactkg_mcdiag.json"
ASSET_ROOT = (
    ROOT / "performance/task_inputs/structural_hypothesis_materializer_v1"
)
RUNNER = ROOT / "runners/run_structural_hypothesis_task_materializer.py"
PROFILES = (
    "none",
    "low_frequency_only",
    "orthogonality_only",
    "sparsity_only",
    "additivity_only",
    "full",
    "leave_out_low_frequency",
    "leave_out_orthogonality",
    "leave_out_sparsity",
    "leave_out_additivity",
)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _contracts():
    return (
        _json(HYPOTHESIS_CONTRACT_PATH),
        _json(EXECUTOR_CONTRACT_PATH),
        _json(MATERIALIZER_CONTRACT_PATH),
    )


def _rows(*, include_first_full=False):
    contract = _json(HYPOTHESIS_CONTRACT_PATH)
    scope = contract["evidence_scope"]
    rows = []
    for profile in PROFILES:
        if profile == "full":
            continue
        for domain in scope["domains"]:
            for seed in scope["seeds"]:
                rows.append(_row(scope, profile, domain, seed))
    if include_first_full:
        rows.append(_row(scope, "full", scope["domains"][0], 0))
    return rows


def _row(scope, profile, domain, seed):
    return {
        "track": scope["track"],
        "run_id": scope["run_id"],
        "variant": scope["variant_template"].format(profile=profile),
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
    }


def _report(*, include_first_full=False):
    report = run_structural_hypothesis_loop(
        _rows(include_first_full=include_first_full),
        _json(HYPOTHESIS_CONTRACT_PATH),
    ).to_dict()
    assert verify_report_integrity(report)
    return report


def _local_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "v18b_exactkg_mcdiag.json"
    assets = tmp_path / "assets"
    shutil.copy2(BASE_MANIFEST_PATH, base)
    shutil.copytree(ASSET_ROOT, assets)
    checkpoint = tmp_path / "not-created-checkpoints"
    return base, assets, checkpoint


def _materialize(tmp_path, report=None):
    hypothesis, executor, materializer = _contracts()
    base, assets, checkpoint = _local_inputs(tmp_path)
    source_report = report or _report()
    bundle = materialize_task_bundle(
        source_report,
        hypothesis,
        executor,
        materializer,
        base,
        assets,
        checkpoint,
    )
    return {
        "bundle": bundle,
        "report": source_report,
        "hypothesis": hypothesis,
        "executor": executor,
        "materializer": materializer,
        "base": base,
        "assets": assets,
        "checkpoint": checkpoint,
    }


def _verify(case, *, bundle=None, report=None, materializer=None):
    return verify_materialized_task_bundle(
        bundle or case["bundle"],
        report or case["report"],
        case["hypothesis"],
        case["executor"],
        materializer or case["materializer"],
        case["base"],
        case["assets"],
        case["checkpoint"],
    )


def _cli_common(case, report_path):
    return [
        "--report", str(report_path),
        "--hypothesis-contract", str(HYPOTHESIS_CONTRACT_PATH),
        "--executor-contract", str(EXECUTOR_CONTRACT_PATH),
        "--materializer-contract", str(MATERIALIZER_CONTRACT_PATH),
        "--base-manifest", str(case["base"]),
        "--asset-root", str(case["assets"]),
        "--checkpoint-root", str(case["checkpoint"]),
    ]


def test_materializes_30_complete_native_tasks_without_execution(tmp_path):
    case = _materialize(tmp_path)
    bundle = case["bundle"]

    assert bundle["status"] == "MATERIALIZED_NOT_AUTHORIZED"
    assert bundle["plan"]["status"] == "READY_FOR_AUTHORIZATION"
    assert _verify(case)
    assert not case["checkpoint"].exists()
    tasks = bundle["plan"]["tasks"]
    assert len(tasks) == 30
    assert [
        (task["cell"]["domain"], task["cell"]["seed"])
        for task in tasks
    ] == [
        (domain, seed)
        for domain in case["hypothesis"]["evidence_scope"]["domains"]
        for seed in range(10)
    ]
    for task in tasks:
        native = task["run_one_task"]
        assert set(native) == {"args", "heldout", "line", "seed"}
        assert native["heldout"] == task["cell"]["domain"]
        assert native["line"] == "lodo"
        assert native["seed"] == task["cell"]["seed"]
        args = native["args"]
        assert isinstance(args, dict) and len(args) > 300
        assert args["initial_design"] == "source_informed"
        assert len(args["initial_design_points"]) == 10
        assert all(len(point) == 50 for point in args["initial_design_points"])
        assert args["structural_prior_profile"] == "full"
        assert args["decision_backend"] == "risk_ts"
        assert args["offline_only"] is True


def test_isolated_materialization_never_imports_or_calls_run_one(tmp_path):
    case = _materialize(tmp_path / "fixture")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(case["report"]), encoding="utf-8")
    output = tmp_path / "isolated-bundle.json"
    script = r'''
import builtins
import sys

blocked = {
    "performance.benchmark_lodo_meta_prior",
    "performance.benchmark_quality",
}
assert blocked.isdisjoint(sys.modules)
original_import = builtins.__import__

def bomb_import(name, *args, **kwargs):
    if name in blocked:
        raise AssertionError("materialize-only crossed the run_one boundary")
    return original_import(name, *args, **kwargs)

builtins.__import__ = bomb_import
from runners.run_structural_hypothesis_task_materializer import main
rc = main(sys.argv[1:])
assert rc == 0
assert blocked.isdisjoint(sys.modules)
'''
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "materialize",
            *_cli_common(case, report_path),
            "--out", str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()


def test_bundle_task_tampering_and_source_report_mismatch_are_rejected(tmp_path):
    case = _materialize(tmp_path)
    tampered = copy.deepcopy(case["bundle"])
    tampered["plan"]["tasks"][0]["run_one_task"]["args"][
        "initial_design_points"
    ][0][0] += 1
    assert not _verify(case, bundle=tampered)

    different_valid_report = _report(include_first_full=True)
    assert len(different_valid_report["pending_evidence"]) == 29
    assert not _verify(case, report=different_valid_report)

    recursive_case = _materialize(
        tmp_path / "recursive", report=different_valid_report
    )
    assert recursive_case["bundle"]["task_count"] == 29
    assert _verify(recursive_case)
    assert [
        (task["cell"]["domain"], task["cell"]["seed"])
        for task in recursive_case["bundle"]["plan"]["tasks"]
    ] == [
        (cell["domain"], cell["seed"])
        for cell in different_valid_report["pending_evidence"]
    ]


def test_runtime_dependency_drift_cannot_redefine_complete_task(monkeypatch, tmp_path):
    import performance.run_lodo_manifest_shard as lodo_runner

    original = lodo_runner.apply_structural_prior_profile

    def drift(config, name):
        resolved = original(config, name)
        resolved["meta_spectral_coefficient_shrinkage"] = False
        return resolved

    monkeypatch.setattr(lodo_runner, "apply_structural_prior_profile", drift)
    hypothesis, executor, materializer = _contracts()
    base, assets, checkpoint = _local_inputs(tmp_path)
    with pytest.raises(ValueError, match="frozen domain template"):
        materialize_task_bundle(
            _report(),
            hypothesis,
            executor,
            materializer,
            base,
            assets,
            checkpoint,
        )


def test_public_runner_callable_rebinding_is_rejected_before_call(
    monkeypatch, tmp_path
):
    import performance.run_lodo_manifest_shard as lodo_runner

    original = lodo_runner.build_run_one_task
    observed = []

    def rebound(argv):
        observed.append("called")
        return original(argv)

    monkeypatch.setattr(lodo_runner, "build_run_one_task", rebound)
    hypothesis, executor, materializer = _contracts()
    base, assets, checkpoint = _local_inputs(tmp_path)
    with pytest.raises(ValueError, match="definition-time capture"):
        materialize_task_bundle(
            _report(),
            hypothesis,
            executor,
            materializer,
            base,
            assets,
            checkpoint,
        )
    assert observed == []


@pytest.mark.parametrize("changed_input", ["base", "design"])
def test_raw_inputs_are_rechecked_after_task_construction(
    monkeypatch, tmp_path, changed_input
):
    import performance.structural_hypothesis_task_materializer as module

    hypothesis, executor, materializer = _contracts()
    base, assets, checkpoint = _local_inputs(tmp_path)
    target = base
    if changed_input == "design":
        target = (
            assets
            / "FactorShockStatePolicyRZDT1/source_initial_designs.json"
        )
    original = module._validate_resolved_plan

    def validate_then_change_bytes(*args, **kwargs):
        original(*args, **kwargs)
        target.write_bytes(target.read_bytes() + b"\n")

    monkeypatch.setattr(
        module, "_validate_resolved_plan", validate_then_change_bytes
    )
    with pytest.raises(ValueError, match="post-materialization raw SHA-256"):
        materialize_task_bundle(
            _report(),
            hypothesis,
            executor,
            materializer,
            base,
            assets,
            checkpoint,
        )


def test_wrong_raw_sha_and_missing_or_extra_design_seed_are_rejected(tmp_path):
    hypothesis, executor, materializer = _contracts()
    source_report = _report()

    for mutation in ("raw-sha", "missing-seed", "extra-seed"):
        local = tmp_path / mutation
        base, assets, checkpoint = _local_inputs(local)
        design_path = (
            assets / "FactorShockStatePolicyRZDT1/source_initial_designs.json"
        )
        if mutation == "raw-sha":
            design_path.write_bytes(design_path.read_bytes() + b"\n")
        else:
            payload = _json(design_path)
            if mutation == "missing-seed":
                del payload["designs"]["9"]
            else:
                payload["designs"]["20"] = copy.deepcopy(
                    payload["designs"]["0"]
                )
            design_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            materialize_task_bundle(
                source_report,
                hypothesis,
                executor,
                materializer,
                base,
                assets,
                checkpoint,
            )


def test_companion_mismatch_base_manifest_change_and_asset_alias_are_rejected(
    tmp_path,
):
    hypothesis, executor, materializer = _contracts()
    source_report = _report()

    companion_case = tmp_path / "companion"
    base, assets, checkpoint = _local_inputs(companion_case)
    shutil.copy2(
        assets / "InventorySupplyChain/heldout_InventorySupplyChain.json",
        assets
        / "FactorShockStatePolicyRZDT1/heldout_FactorShockStatePolicyRZDT1.json",
    )
    with pytest.raises(ValueError):
        materialize_task_bundle(
            source_report, hypothesis, executor, materializer,
            base, assets, checkpoint,
        )

    manifest_case = tmp_path / "manifest"
    base, assets, checkpoint = _local_inputs(manifest_case)
    payload = _json(base)
    payload["config"]["N"] = 21
    base.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        materialize_task_bundle(
            source_report, hypothesis, executor, materializer,
            base, assets, checkpoint,
        )

    alias_case = tmp_path / "alias"
    base, assets, checkpoint = _local_inputs(alias_case)
    design = assets / "QueueResourceControl/source_initial_designs.json"
    design.unlink()
    design.symlink_to(
        ASSET_ROOT / "QueueResourceControl/source_initial_designs.json"
    )
    with pytest.raises(ValueError):
        materialize_task_bundle(
            source_report, hypothesis, executor, materializer,
            base, assets, checkpoint,
        )


def test_runner_hash_and_runner_relative_path_are_frozen(tmp_path):
    materializer = _json(MATERIALIZER_CONTRACT_PATH)
    changed_hash = copy.deepcopy(materializer)
    changed_hash["materializer_binding"]["runner_raw_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_materializer_contract(changed_hash)

    aliased_path = copy.deepcopy(materializer)
    aliased_path["materializer_binding"]["runner_relative_path"] = (
        "performance/../performance/run_lodo_manifest_shard.py"
    )
    with pytest.raises(ValueError):
        validate_materializer_contract(aliased_path)

    case = _materialize(tmp_path)
    assert _verify(case)


def test_cli_materialize_and_verify_with_atomic_nonreplacing_output(tmp_path):
    case = _materialize(tmp_path / "inputs")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(case["report"]), encoding="utf-8")
    output = tmp_path / "bundle.json"

    made = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "materialize",
            *_cli_common(case, report_path),
            "--out", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert made.returncode == 0, made.stderr
    assert made.stdout == ""
    assert "status=MATERIALIZED_NOT_AUTHORIZED" in made.stderr
    before = output.read_bytes()

    verified = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "verify",
            "--bundle", str(output),
            *_cli_common(case, report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout == ""
    assert "action=verify" in verified.stderr

    refused = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "materialize",
            *_cli_common(case, report_path),
            "--out", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode == 2
    assert "refusing to overwrite" in refused.stderr
    assert output.read_bytes() == before


def test_cli_rejects_duplicate_keys_relative_checkpoint_and_output_alias(
    tmp_path,
):
    case = _materialize(tmp_path / "inputs")
    duplicate_report = tmp_path / "duplicate-report.json"
    duplicate_report.write_text(
        '{"audit": {}, "audit": {}}\n', encoding="utf-8"
    )
    output = tmp_path / "output.json"
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "materialize",
            *_cli_common(case, duplicate_report),
            "--out", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "duplicate key 'audit'" in rejected.stderr
    assert "Traceback" not in rejected.stderr

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(case["report"]), encoding="utf-8")
    relative_args = _cli_common(case, report_path)
    checkpoint_index = relative_args.index("--checkpoint-root") + 1
    relative_args[checkpoint_index] = "relative-checkpoints"
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "materialize",
            *relative_args,
            "--out", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "absolute path" in rejected.stderr

    checkpoint_output = case["checkpoint"] / "result.json"
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "materialize",
            *_cli_common(case, report_path),
            "--out", str(checkpoint_output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "outside --checkpoint-root" in rejected.stderr
    assert not case["checkpoint"].exists()

    asset_output = case["assets"] / "bundle.json"
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "materialize",
            *_cli_common(case, report_path),
            "--out", str(asset_output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "descend from an input path" in rejected.stderr
    assert not asset_output.exists()

    alias = tmp_path / "alias.json"
    os.link(report_path, alias)
    before = report_path.read_bytes()
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "materialize",
            *_cli_common(case, report_path),
            "--out", str(alias),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "refusing to overwrite" in rejected.stderr
    assert report_path.read_bytes() == before


def test_runner_exposes_only_materialize_and_verify():
    import runners.run_structural_hypothesis_task_materializer as runner

    choices = set(runner._parser()._subparsers._group_actions[0].choices)
    assert choices == {"materialize", "verify"}
    source = RUNNER.read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "benchmark_lodo_meta_prior" not in source
    assert "import scheduler" not in source.lower()
    assert "from scheduler" not in source.lower()
