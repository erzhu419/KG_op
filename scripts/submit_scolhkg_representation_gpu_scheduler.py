#!/usr/bin/env python3
"""Submit SC-OLH-KG representation sweeps to GPU-capable nodes only."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_SCHEDULER = Path.home() / ".claude/skills/scheduler/scheduler.py"
DEFAULT_DEPLOY = Path("/home/erzhu419/mine_code/KG_op_scheduler_deploy")
DEFAULT_LOCAL_RESULTS = Path("/home/erzhu419/mine_code/KG_op/SC-OLH-KG/profiles")
PYTHON_CANDIDATES = (
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python",
    "/home/erzhu419/miniconda3/envs/csbapr/bin/python",
    "/home/erzhu419/.conda/envs/offline-sumo/bin/python",
    "/home/erzhu419/miniconda3/envs/offline-sumo/bin/python",
)


def parse_csv(text):
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def run_cmd(cmd, dry_run=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    if dry_run:
        return ""
    return subprocess.check_output([str(c) for c in cmd], text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", default=str(DEFAULT_SCHEDULER))
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--nodes",
        default="jtl110gpu,jtl110gpu2,jtl311linux,node007",
        help="GPU-capable nodes only. Passed as repeated --allowed-node.",
    )
    parser.add_argument(
        "--python",
        default="auto",
        help="Python path on target node, or auto to probe known GPU envs.",
    )
    parser.add_argument("--dims", default="1000,10000")
    parser.add_argument(
        "--encoder-kinds",
        default=(
            "synthetic,pca_manifold,kernel_manifold,graph_laplacian,"
            "ssl_masked,ssl_contrastive,ssl_next_risk,ssl_transformer,ssl_hybrid"
        ),
    )
    parser.add_argument("--calibration-modes", default="none,recommendation,certified")
    parser.add_argument("--N1000", type=int, default=40)
    parser.add_argument("--N10000", type=int, default=30)
    parser.add_argument("--seeds1000", type=int, default=10)
    parser.add_argument("--seeds10000", type=int, default=5)
    parser.add_argument("--n0", type=int, default=10)
    parser.add_argument("--K1", type=int, default=20)
    parser.add_argument("--K2", type=int, default=0)
    parser.add_argument("--posterior-pool-size", type=int, default=260)
    parser.add_argument("--posterior-keep", type=int, default=12)
    parser.add_argument("--state-candidate-count", type=int, default=24)
    parser.add_argument("--state-inverse-pool-size", type=int, default=600)
    parser.add_argument("--state-inverse-neighbors", type=int, default=2)
    parser.add_argument("--eval-pool-size", type=int, default=300)
    parser.add_argument("--raw-basis-dim", type=int, default=128)
    parser.add_argument("--encoder-latent-dim", type=int, default=8)
    parser.add_argument("--encoder-fit-pool-size", type=int, default=512)
    parser.add_argument("--exact-kg-jobs", type=int, default=1)
    parser.add_argument("--checkpoint-keep-last", type=int, default=2)
    parser.add_argument("--numeric-backend", default="torch_cuda")
    parser.add_argument("--numeric-backend-device", default="auto")
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--torch-min-rows", type=int, default=64)
    parser.add_argument("--vram", type=int, default=4096)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--ram-mb", type=int, default=16384)
    parser.add_argument("--local-result-dir", type=Path, default=DEFAULT_LOCAL_RESULTS)
    parser.add_argument("--priority", choices=("low", "normal", "high"), default="normal")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    cwd = args.deploy
    remote_out = cwd / "SC-OLH-KG" / "profiles" / run_id
    local_out = args.local_result_dir / run_id
    checkpoint_root = cwd / "SC-OLH-KG" / "checkpoints" / "repr_gpu" / run_id
    nodes = parse_csv(args.nodes)
    encoders = parse_csv(args.encoder_kinds)
    calibration_modes = parse_csv(args.calibration_modes)
    task_ids = []

    for dim_text in parse_csv(args.dims):
        dim = int(dim_text)
        N = args.N10000 if dim >= 10000 else args.N1000
        n_seeds = args.seeds10000 if dim >= 10000 else args.seeds1000
        for encoder in encoders:
            for calibration in calibration_modes:
                prefix = f"repr_gpu_{run_id}_d{dim}_{encoder}_{calibration}"
                if str(args.python).lower() == "auto":
                    candidates = " ".join(PYTHON_CANDIDATES)
                    python_setup = (
                        "PYTHON_BIN=${SCOLHKG_PYTHON:-}; "
                        "if [ -z \"$PYTHON_BIN\" ]; then "
                        f"for PY in {candidates}; do "
                        "if [ -x \"$PY\" ] && \"$PY\" -c "
                        "\"import numpy, scipy, torch; "
                        "assert torch.cuda.is_available(), 'cuda unavailable'\" "
                        ">/tmp/scolhkg_py_probe.log 2>&1; "
                        "then PYTHON_BIN=\"$PY\"; break; fi; "
                        "done; fi; "
                        "test -n \"$PYTHON_BIN\"; "
                        "export LD_LIBRARY_PATH=\"$(dirname \"$PYTHON_BIN\")/../lib:${LD_LIBRARY_PATH:-}\""
                    )
                else:
                    python_setup = (
                        f"PYTHON_BIN={args.python}; test -x \"$PYTHON_BIN\"; "
                        "export LD_LIBRARY_PATH=\"$(dirname \"$PYTHON_BIN\")/../lib:${LD_LIBRARY_PATH:-}\""
                    )
                preflight = (
                    "$PYTHON_BIN -c "
                    "\"import numpy, scipy, torch; "
                    "assert torch.cuda.is_available(), 'cuda unavailable'\""
                )
                command = (
                    f"cd SC-OLH-KG && $PYTHON_BIN -u performance/benchmark_encoder_suite.py "
                    f"--problem HighDimStatePolicyRZDT1 --d {dim} "
                    f"--N {N} --n0 {args.n0} --K1 {args.K1} --K2 {args.K2} "
                    f"--posterior_pool_size {args.posterior_pool_size} "
                    f"--posterior_keep {args.posterior_keep} "
                    f"--state_candidate_count {args.state_candidate_count} "
                    f"--state_inverse_pool_size {args.state_inverse_pool_size} "
                    f"--state_inverse_neighbors {args.state_inverse_neighbors} "
                    f"--eval_pool_size {args.eval_pool_size} "
                    f"--use_state_basis --state_basis_mode raw+manifold "
                    f"--raw_basis_dim {args.raw_basis_dim} "
                    f"--encoder_kinds {encoder} "
                    f"--calibration_modes {calibration} "
                    f"--encoder_latent_dim {args.encoder_latent_dim} "
                    f"--encoder_fit_pool_size {args.encoder_fit_pool_size} "
                    f"--numeric_backend {args.numeric_backend} "
                    f"--numeric_backend_device {args.numeric_backend_device} "
                    f"--torch_dtype {args.torch_dtype} "
                    f"--torch_min_rows {args.torch_min_rows} "
                    f"--sc_modes factor --modes '' --n_seeds {n_seeds} "
                    f"--exact_kg_jobs {args.exact_kg_jobs} "
                    f"--checkpoint_dir {checkpoint_root / f'd{dim}' / encoder / calibration} "
                    f"--checkpoint_resume --checkpoint_interval 1 "
                    f"--checkpoint_keep_last {args.checkpoint_keep_last} "
                    f"--out_dir {Path('profiles') / run_id / f'd{dim}' / encoder / calibration} "
                    f"--out_prefix {prefix}"
                )
                cmd = "; ".join([
                    "set -e",
                    "export LC_ALL=C LANG=C",
                    "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                    python_setup,
                    preflight,
                    command,
                    "echo DONE",
                ])
                submit = [
                    sys.executable, args.scheduler, "submit",
                    "--description", f"SC-OLH-KG repr GPU d={dim} {encoder} {calibration}",
                    "--cmd", cmd,
                    "--cwd", str(cwd),
                    "--signature", f"SC-OLH-KG/repr-gpu/{run_id}/d{dim}/{encoder}/{calibration}",
                    "--project", "SC-OLH-KG",
                    "--vram", str(args.vram),
                    "--cpu", str(args.cpu),
                    "--ram-mb", str(args.ram_mb),
                    "--priority", args.priority,
                    "--result-dir", str(remote_out / f"d{dim}" / encoder / calibration),
                    "--local-result-dir", str(local_out / f"d{dim}" / encoder / calibration),
                    "--stage-exclude", "SC-OLH-KG/profiles",
                    "--stage-exclude", "SC-OLH-KG/results",
                    "--stage-exclude", "SC-OLH-KG/checkpoints",
                    "--stage-exclude", "**/__pycache__",
                    "--stage-exclude", "*.pyc",
                    "--allow-no-ckpt",
                    "--allow-no-resume",
                    "--allow-duplicate",
                ]
                for node in nodes:
                    submit.extend(["--allowed-node", node])
                out = run_cmd(submit, dry_run=args.dry_run)
                if out:
                    print(out, end="")
                    parts = out.split()
                    if len(parts) > 1:
                        task_ids.append(parts[1])

    if args.dispatch:
        run_cmd([sys.executable, args.scheduler, "dispatch"], dry_run=args.dry_run)
    print({"run_id": run_id, "task_ids": task_ids, "result_dir": str(local_out)})


if __name__ == "__main__":
    main()
