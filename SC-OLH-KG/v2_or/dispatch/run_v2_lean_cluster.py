#!/usr/bin/env python3
"""Cluster Lean 4.31.0 proof checker for `SCOLHKG.V2ORProofInterface`.

Reuses the locked Mathlib v4.31.0 toolchain & unpacked `.olean` cache on the cluster
while writing build outputs & receipts strictly under `KG_op_v2_or_cluster/lean_attempts`
so historical `KG_op_mainline_lean_20261020` receipts remain 100% untouched.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time

IMPORT_RE = re.compile(r"^\s*(?:public\s+)?(?:meta\s+)?import\s+([\w.]+)", re.M)


def child_cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def import_order(target: str, roots: dict[str, Path], prefixes: set[str]) -> list[tuple[str, Path, Path]]:
    visited: set[str] = set()
    ordered: list[tuple[str, Path, Path]] = []

    def visit(module: str) -> None:
        prefix = module.split(".")[0]
        if prefix not in prefixes or module in visited:
            return
        visited.add(module)
        pkg = roots[prefix]
        rel = Path(*module.split(".")).with_suffix(".lean")
        for dep in IMPORT_RE.findall((pkg / rel).read_text(encoding="utf-8")):
            visit(dep)
        ordered.append((module, pkg, rel))

    visit(target)
    return ordered


def main() -> int:
    t0_wall, t0_cpu = time.monotonic(), time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-proof",
        type=Path,
        default=Path("/home/zhengliang01/scheduleurm_work/KG_op_mainline_lean_20261020/proof"),
    )
    parser.add_argument(
        "--v2-proof",
        type=Path,
        default=Path("/home/zhengliang01/scheduleurm_work/KG_op_v2_or_cluster/proof"),
    )
    parser.add_argument(
        "--toolchain",
        type=Path,
        default=Path("/home/zhengliang01/scheduleurm_work/KG_op_mainline_lean_20261020/toolchain"),
    )
    parser.add_argument(
        "--attempts-dir",
        type=Path,
        default=Path("/home/zhengliang01/scheduleurm_work/KG_op_v2_or_cluster/lean_attempts"),
    )
    parser.add_argument("--target", default="SCOLHKG.V2ORProofInterface")
    args = parser.parse_args()

    base_proof = args.base_proof.resolve()
    v2_proof = args.v2_proof.resolve()
    toolchain = args.toolchain.resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = args.attempts_dir.resolve() / f"{stamp}_{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=True)
    out_lib = v2_proof / ".lake/build/lib/lean"
    out_lib.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "attempt_directory": str(attempt),
        "target": args.target,
        "v2_proof": str(v2_proof),
        "base_proof": str(base_proof),
        "toolchain": str(toolchain),
        "steps": [],
        "status": "running",
    }

    def save() -> None:
        record["wall_seconds"] = time.monotonic() - t0_wall
        record["runner_cpu_seconds"] = time.process_time() - t0_cpu
        record["proof_check_cpu_seconds"] = sum(s["cpu_seconds"] for s in record["steps"])
        (attempt / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    manifest = json.loads((base_proof / "lake-manifest.json").read_text(encoding="utf-8"))
    packages = {
        p["name"]: base_proof / manifest["packagesDir"] / p["name"]
        for p in manifest["packages"]
    }
    lean_bin = toolchain / "bin" / "lean"
    lean_paths = [
        out_lib,
        base_proof / ".lake/build/lib/lean",
        *(p / ".lake/build/lib/lean" for p in packages.values()),
    ]
    src_paths = [v2_proof, base_proof, *packages.values()]

    env = os.environ.copy()
    env.update({
        "PATH": str(toolchain / "bin") + os.pathsep + env.get("PATH", ""),
        "LEAN_PATH": os.pathsep.join(map(str, lean_paths)),
        "LEAN_SRC_PATH": os.pathsep.join(map(str, src_paths)),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })

    def run_step(name: str, cmd: list[str], cwd: Path) -> str:
        log = attempt / f"{len(record['steps']):02d}_{name}.log"
        w0, c0 = time.monotonic(), child_cpu()
        with log.open("w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        text = log.read_text(encoding="utf-8", errors="replace")
        record["steps"].append({
            "name": name,
            "command": cmd,
            "returncode": rc,
            "wall_seconds": time.monotonic() - w0,
            "cpu_seconds": child_cpu() - c0,
            "output": text,
        })
        save()
        if rc != 0:
            raise RuntimeError(f"Step {name} failed (rc={rc}):\n{text}")
        return text

    try:
        avail = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {avail[0]})
        ver_txt = run_step("lean_version", [str(lean_bin), "--version"], v2_proof)
        record["lean_version"] = ver_txt.strip()

        # Resolve SCOLHKG modules preferring `v2_proof` when present, else `base_proof`
        modules_to_build = [
            "SCOLHKG.V2OR.DebiasedSubspace",
            "SCOLHKG.V2OR.TemperedHVD",
            "SCOLHKG.V2OR.VerificationKGAmortization",
            "SCOLHKG.V2OR.FunctionalTransferRisk",
            "SCOLHKG.V2OR.SequentialTheory",
            "SCOLHKG.V2ORProofInterface",
        ]

        for mod in modules_to_build:
            rel = Path(*mod.split(".")).with_suffix(".lean")
            olean = out_lib / rel.with_suffix(".olean")
            olean.parent.mkdir(parents=True, exist_ok=True)
            out_txt = run_step(mod, [str(lean_bin), "-j1", "-o", str(olean), str(v2_proof / rel)], v2_proof)
            if mod == args.target:
                record["interface_check_output"] = out_txt

        record["status"] = "passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
    save()
    print(json.dumps({
        "status": record["status"],
        "lean_version": record.get("lean_version"),
        "proof_check_cpu_seconds": record.get("proof_check_cpu_seconds"),
        "wall_seconds": record.get("wall_seconds"),
        "interface_check_output": record.get("interface_check_output", ""),
        "error": record.get("error"),
        "result_json": str(attempt / "result.json"),
    }, indent=2))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
