#!/usr/bin/env python3
"""Inventory, export, or cold-check only the locked current Lean closure.

The build command never downloads, retries, or reuses project oleans. Dependency
oleans are reused explicitly. Repository operators run builds on scheduleurm.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time


TARGET = "SCOLHKG.CurrentMainlineProofInterface"
TOOLCHAIN = "leanprover/lean4:v4.31.0"
MATHLIB_REVISION = "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f"
IMPORT = re.compile(r"^\s*import\s+([\w.]+)", re.M)
THEOREM = re.compile(r"^theorem\s+(\w+)", re.M)
AXIOMS = re.compile(r"^#print axioms\s+([\w.]+)", re.M)
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
CURRENT_COUNTS = {
    "SCOLHKG.Real.CurrentPropagation": 13,
    "SCOLHKG.Real.CurrentCovariance": 6,
    "SCOLHKG.Measure.CurrentTerminalGate": 13,
    "SCOLHKG.Real.CurrentRepresentationStability": 15,
    "SCOLHKG.Real.CurrentQueryStability": 11,
    "SCOLHKG.Real.CurrentSourceKernel": 21,
    "SCOLHKG.Real.CurrentAffineMismatch": 15,
}


def module_path(module):
    return Path(*module.split(".")).with_suffix(".lean")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(proof):
    """Read the small current source closure; do not execute Lean or Lake."""
    ordered, dependencies, declarations = [], {}, {}

    def visit(module):
        if module in dependencies:
            return
        text = (proof / module_path(module)).read_text()
        dependencies[module] = IMPORT.findall(text)
        for dependency in dependencies[module]:
            if dependency.startswith("SCOLHKG."):
                visit(dependency)
        ordered.append(module)
        if module in CURRENT_COUNTS:
            namespace = re.search(r"^namespace\s+([\w.]+)", text, re.M).group(1)
            declarations[module] = [namespace + "." + n for n in THEOREM.findall(text)]
            if len(declarations[module]) != CURRENT_COUNTS[module]:
                raise ValueError(f"current declaration count changed: {module}")

    visit(TARGET)
    manifest = json.loads((proof / "lake-manifest.json").read_text())
    packages = {p["name"]: p for p in manifest["packages"]}
    if (proof / "lean-toolchain").read_text().strip() != TOOLCHAIN:
        raise ValueError("lean-toolchain differs from the frozen toolchain")
    if packages["mathlib"]["rev"] != MATHLIB_REVISION:
        raise ValueError("Mathlib revision differs from the frozen lock")
    headline = AXIOMS.findall((proof / module_path(TARGET)).read_text())
    if len(ordered) != 16 or sum(map(len, declarations.values())) != 94 or len(headline) != 12:
        raise ValueError("the frozen current closure is not 16 modules / 94 theorems / 12 axiom checks")
    files = [module_path(m) for m in ordered]
    files += [Path(p) for p in ("lean-toolchain", "lake-manifest.json", "lakefile.lean")]
    hashes = {p.as_posix(): sha256(proof / p) for p in sorted(files)}
    tree_key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {
        "interface": TARGET, "lean_toolchain": TOOLCHAIN,
        "mathlib_revision": MATHLIB_REVISION, "project_modules": ordered,
        "module_imports": dependencies, "current_theorems": declarations,
        "current_theorem_count": 94, "headline_axiom_checks": headline,
        "external_import_roots": sorted({d for ds in dependencies.values() for d in ds
                                         if d.split(".")[0] not in {"SCOLHKG", "Lean", "Init", "Std", "Lake"}}),
        "source_sha256": hashes, "source_tree_key_sha256": tree_key,
        "dependency_locks": {n: p["rev"] for n, p in packages.items()},
    }


def export_sources(proof, destination, info):
    """Export just the interface closure, a current-only Lake target, and locks."""
    destination.mkdir(parents=True, exist_ok=False)
    for module in info["project_modules"]:
        relative = module_path(module)
        target = destination / "proof" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(proof / relative, target)
    for name in ("lean-toolchain", "lake-manifest.json"):
        shutil.copyfile(proof / name, destination / "proof" / name)
    # The historical root imports unrelated modules. Export its package with the
    # current interface as the sole default root; theorem sources stay byte exact.
    (destination / "proof/lakefile.lean").write_text(
        "import Lake\nopen Lake DSL\n\n"
        "package «scolhkg-proof» where\n  version := v!\"0.1.0\"\n\n"
        "require mathlib from git\n"
        '  "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0"\n\n'
        "@[default_target]\nlean_lib SCOLHKG where\n"
        "  roots := #[`SCOLHKG.CurrentMainlineProofInterface]\n")
    for name in ("lean_delivery.py", "LEAN.md"):
        shutil.copyfile(Path(__file__).with_name(name), destination / name)
    report = dict(info)
    report["exported_lakefile_sha256"] = sha256(destination / "proof/lakefile.lean")
    report["export_scope"] = "16 source modules; current-only default target; no compiled artifacts or dependencies"
    (destination / "lean_source_manifest.json").write_text(json.dumps(report, indent=2) + "\n")


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def cold_build(proof, output, toolchain):
    started_wall, started_cpu = time.monotonic(), time.process_time()
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "exit_code": None,
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "steps": [], "automatic_retries": 0, "scientific_calls": 0,
        "project_olean_reuse": False, "official_dependency_cache_reused": True,
        "all_mathlib_rebuilt_from_source": False,
        "previous_failures_or_fees_modified": False,
        "scheduler_and_transport_cpu_seconds": None,
        "billing_currency_cost": None,
    }
    env = os.environ.copy()
    env.update({n: "1" for n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})

    def save():
        record["wall_seconds"] = time.monotonic() - started_wall
        record["runner_cpu_seconds"] = time.process_time() - started_cpu
        record["child_cpu_seconds"] = sum(s["cpu_seconds"] for s in record["steps"])
        record["peak_child_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        record["proof_build_cpu_seconds"] = sum(s["cpu_seconds"] for s in record["steps"]
                                                if s["phase"] == "proof")
        (output / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")

    def run(name, command, cwd, phase="environment"):
        log = output / f"{len(record['steps']):02d}_{name}.log"
        wall, cpu = time.monotonic(), child_cpu()
        with log.open("w") as stream:
            try:
                code = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                      stdout=stream, stderr=subprocess.STDOUT).returncode
            except OSError as error:
                stream.write(str(error) + "\n")
                code = 127
        record["steps"].append({
            "name": name, "phase": phase, "command": [str(x) for x in command],
            "exit_code": code, "log": log.name, "cpu_seconds": child_cpu() - cpu,
            "wall_seconds": time.monotonic() - wall,
        })
        save()
        print(json.dumps({"step": name, "exit_code": code}), flush=True)
        if code:
            raise RuntimeError(f"{name} exited {code}; failure retained in {log.name}")
        return log.read_text()

    try:
        record.update(inventory(proof))
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {available[0]})
        record["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        if toolchain is None:
            prefix = run("lean_prefix", ["lean", "--print-prefix"], proof).strip()
            toolchain = Path(prefix)
        lean = toolchain / "bin/lean"
        version = run("lean_version", [lean, "--version"], proof).strip()
        record["lean_version"] = version
        if not re.search(r"version 4\.31\.0(?:,|\s|\))", version):
            raise ValueError("the executed Lean is not version 4.31.0")
        record["lake_version"] = run("lake_version", [toolchain / "bin/lake", "--version"], proof).strip()
        manifest = json.loads((proof / "lake-manifest.json").read_text())
        packages = {p["name"]: proof / manifest["packagesDir"] / p["name"]
                    for p in manifest["packages"]}
        dependency_identity = {}
        for name, source in packages.items():
            if not source.is_dir():
                raise FileNotFoundError(f"dependency source missing: {name}; prepare dependencies before this offline build")
            actual_revision = None
            if (source / ".git").exists():
                actual_revision = run(f"revision_{name}", ["git", "rev-parse", "HEAD"], source).strip()
                if actual_revision != record["dependency_locks"][name]:
                    raise ValueError(f"dependency checkout differs from lock: {name}")
            entrypoints = [p for p in (source / "lean-toolchain", source / "lake-manifest.json",
                                      source / "lakefile.lean", source / "lakefile.toml") if p.is_file()]
            dependency_identity[name] = {
                "locked_revision": record["dependency_locks"][name],
                "actual_git_revision": actual_revision,
                "git_revision_verified": actual_revision is not None,
                "entrypoint_sha256": {p.name: sha256(p) for p in entrypoints},
            }
        record["dependency_source_identity"] = dependency_identity
        record["all_dependency_git_revisions_verified"] = all(
            p["git_revision_verified"] for p in dependency_identity.values())
        export_sources(proof, output / "source_snapshot", record)
        current = output / "source_snapshot/proof"
        compiled = current / ".lake/build/lib/lean"
        compiled.mkdir(parents=True)
        env["LEAN_PATH"] = os.pathsep.join([str(compiled), *(
            str(p / ".lake/build/lib/lean") for p in packages.values())])
        env["LEAN_SRC_PATH"] = os.pathsep.join([str(current), *(str(p) for p in packages.values())])
        for module in record["project_modules"]:
            relative = module_path(module)
            target = compiled / relative.with_suffix(".olean")
            target.parent.mkdir(parents=True, exist_ok=True)
            run(module, [lean, "-j1", "-o", target, relative], current, "proof")
        declarations = [d for ds in record["current_theorems"].values() for d in ds]
        audit = current / "ReleaseAudit.lean"
        audit.write_text(f"import {TARGET}\n\n" + "\n".join(
            [f"#check {d}" for d in declarations] +
            [f"#print axioms {d}" for d in record["headline_axiom_checks"]]) + "\n")
        text = run("ReleaseAudit", [lean, "-j1", audit.name], current, "proof")
        reported = re.findall(r"'([^']+)' depends on axioms:\s*\[([^]]*)\]", text)
        axioms = {name: re.findall(r"[\w.]+", items) for name, items in reported}
        if set(axioms) != set(record["headline_axiom_checks"]):
            raise ValueError("the 12 headline axiom reports are incomplete")
        if any(set(items) != ALLOWED_AXIOMS for items in axioms.values()):
            raise ValueError("headline axiom dependencies differ from the frozen three")
        record["headline_axiom_dependencies"] = axioms
        record["all_94_current_declarations_checked"] = True
        record["all_16_project_modules_cold_compiled"] = True
        record["status"], record["exit_code"] = "passed", 0
    except Exception as error:
        record["status"], record["exit_code"], record["error"] = "failed", 1, str(error)
    record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(json.dumps({k: record.get(k) for k in (
        "status", "exit_code", "proof_build_cpu_seconds", "child_cpu_seconds", "error")}) |
        {"receipt": str(output / "receipt.json")}, flush=True)
    return record["exit_code"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "export", "build"))
    parser.add_argument("--proof", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="new destination; never reuse a failed attempt directory")
    parser.add_argument("--toolchain", type=Path, help="Lean toolchain prefix; default: lean --print-prefix")
    args = parser.parse_args(argv)
    proof = args.proof.resolve()
    if args.command == "inventory":
        print(json.dumps(inventory(proof), indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for export/build")
    output = args.output.resolve()
    if args.command == "export":
        export_sources(proof, output, inventory(proof))
        print(json.dumps({"source_snapshot": str(output), "compiled_artifacts_included": False}))
        return 0
    return cold_build(proof, output, args.toolchain.resolve() if args.toolchain else None)


if __name__ == "__main__":
    sys.exit(main())
