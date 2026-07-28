import importlib.util
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO / "scripts/submit_scolhkg_crossdim_backend_matrix_scheduler.py")
SPEC = importlib.util.spec_from_file_location(
    "crossdim_backend_submit", SCRIPT)
SUBMIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMIT)


def _args(tmp_path):
    return SimpleNamespace(
        deploy=tmp_path,
        run_id="crossdim_backend_test",
        archive_run_id="source_d50",
        design_run_id="design_d50_d1000",
        v64_run_id="v64_existing",
        heldouts=",".join(SUBMIT.DOMAINS),
        backends=",".join(SUBMIT.BACKENDS),
        seed_start=80,
        n_seeds=2,
        source_d=50,
        d=1000,
        N=13,
        n0=10,
        offline_source_calls=384,
        cpu=12,
        ram_mb=16384,
        gpu_cpu=12,
        gpu_ram_mb=24576,
        gpu_vram_mb=2048,
        terminal_profile="v7",
    )


def test_crossdim_backend_matrix_has_one_shared_design_and_no_ckpt_pull(
    tmp_path,
):
    specs = SUBMIT.build_specs(_args(tmp_path))
    assert len(specs) == 3 * 3 * 2
    assert len({spec["signature"] for spec in specs}) == len(specs)
    counts = {backend: 0 for backend in SUBMIT.BACKENDS}
    for spec in specs:
        backend = next(
            name for name in SUBMIT.BACKENDS
            if f"/{name}/" in spec["signature"]
        )
        counts[backend] += 1
        assert "checkpoints" in spec["stage_excludes"]
        assert "ckpt_dir" not in spec
        assert "ckpt_glob" not in spec
        assert "source_initial_designs.json" in spec["cmd"]
        assert "--d 1000" in spec["cmd"]
        assert "--n0 10" in spec["cmd"]
        assert "jtl311linux" not in spec["allowed_nodes"]
    assert counts == {
        "proposal_only": 6,
        "stacked_gp": 6,
        "saasbo": 6,
    }


def test_crossdim_backend_contracts_keep_backend_information_explicit(
    tmp_path,
):
    specs = SUBMIT.build_specs(_args(tmp_path))
    proposal = next(
        spec for spec in specs if "/proposal_only/" in spec["signature"])
    stacked = next(
        spec for spec in specs if "/stacked_gp/" in spec["signature"])
    saas = next(
        spec for spec in specs if "/saasbo/" in spec["signature"])
    assert "benchmark_frozen_proposal_only.py" in proposal["cmd"]
    assert "--offline-source-calls 384" in proposal["cmd"]
    assert proposal["cpu"] == 1
    assert proposal["vram"] == 0

    assert "--method stacked_transfer_gp_cbo" in stacked["cmd"]
    assert "--source-dimension-adapter ordered_dct_quadratic" in (
        stacked["cmd"])
    assert "--source-coordinate-max-frequency 8" in stacked["cmd"]
    assert stacked["allowed_nodes"] == list(SUBMIT.CPU_NODES)

    assert "--protocol shared_archive_n13" in saas["cmd"]
    assert "--method botorch_saasbo" in saas["cmd"]
    assert "--target-budget 13" in saas["cmd"]
    assert "SCOLHKG_TORCH_DETERMINISTIC=1" in saas["cmd"]
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in saas["cmd"]
    assert "--torch-deterministic" in saas["cmd"]
    assert saas["allowed_nodes"] == list(SUBMIT.GPU_NODES)
    assert saas["vram"] == 2048
    assert str(
        tmp_path / "SC-OLH-KG/performance/manifests/"
        "v18b_exactkg_mcdiag.json"
    ) in saas["cmd"]
    assert str(SUBMIT.REMOTE_ROOT / "SC-OLH-KG/performance/manifests") not in (
        saas["cmd"])
    assert saas["result_dir"].startswith(str(tmp_path / "SC-OLH-KG"))


def test_crossdim_v9_uses_identical_three_policy_verifier(tmp_path):
    args = _args(tmp_path)
    args.terminal_profile = "v9"
    specs = SUBMIT.build_specs(args)
    assert specs
    for spec in specs:
        assert (
            "--terminal-verification-candidate-budgets 80,128,128"
            in spec["cmd"]
        )
        assert (
            "--terminal-verification-shortlist-mode "
            "posterior_objective_challenger_then_safe"
            in spec["cmd"]
        )
        assert "--terminal-verification-shortlist-size 3" in spec["cmd"]
        assert (
            "--terminal-objective-challenger-"
            "max-violation-probability 0.5"
            in spec["cmd"]
        )
        assert "--terminal-verification-delta 0.05" in spec["cmd"]


def test_crossdim_recovery_skips_only_compact_success_results(tmp_path):
    args = _args(tmp_path)
    args.heldouts = "FactorShockStatePolicyRZDT1"
    args.backends = "saasbo"
    args.skip_existing_success = True
    complete = (
        tmp_path / "SC-OLH-KG" / "profiles" / args.run_id / "saasbo"
        / "FactorShockStatePolicyRZDT1" / "seed0080" / "result.json"
    )
    complete.parent.mkdir(parents=True)
    complete.write_text('{"status":"ok"}')
    failed = complete.parent.parent / "seed0081" / "result.json"
    failed.parent.mkdir(parents=True)
    failed.write_text('{"status":"failed"}')

    specs = SUBMIT.build_specs(args)

    assert len(specs) == 1
    assert specs[0]["signature"].endswith("seed0081")
