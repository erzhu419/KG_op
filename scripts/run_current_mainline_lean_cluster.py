#!/usr/bin/env python3
"""One CPU, one attempt: plan locked cache or check current proofs offline.

Run only as an existing scheduleurm CPU job. No Lake configuration, native C,
lock-file edits, scientific computation, or runner retries are performed.
Cache planning emits official URLs; downloading is a separate, recorded step.
Deploy changed Lean files with fresh receiver mtimes (--no-times --ignore-times
on those selected files), including edits made while an older build was queued.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time


IMPORT = re.compile(r"^\s*(?:public\s+)?(?:meta\s+)?import\s+([\w.]+)", re.M)


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def import_order(target, roots, prefixes):
    """Only the Cache/Batteries bootstrap or SCOLHKG import closure."""
    visited, ordered = set(), []

    def visit(module):
        prefix = module.split(".")[0]
        if prefix not in prefixes or module in visited:
            return
        visited.add(module)
        package = roots[prefix]
        relative = Path(*module.split(".")).with_suffix(".lean")
        for dependency in IMPORT.findall((package / relative).read_text()):
            visit(dependency)
        ordered.append((module, package, relative))

    visit(target)
    return ordered


def main():
    started_wall, started_cpu = time.monotonic(), time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--attempts-dir", type=Path)
    parser.add_argument("--phase", choices=("cache-plan", "cache-diagnose", "proof-check"),
                        default="cache-plan")
    args = parser.parse_args()
    proof, toolchain = args.proof.resolve(), args.toolchain.resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = (args.attempts_dir or proof.parent / "lean_attempts") / f"{stamp}_{os.getpid()}"
    attempt.mkdir(parents=True)
    record = {
        "attempt_directory": str(attempt), "proof": str(proof),
        "toolchain": str(toolchain), "steps": [], "status": "running",
        "scientific_computation": False, "runner_automatic_retries": 0,
        "phase": args.phase, "network_requests": False,
    }
    env = os.environ.copy()

    def save():
        record["wall_seconds"] = time.monotonic() - started_wall
        record["runner_cpu_seconds"] = time.process_time() - started_cpu
        record["child_cpu_seconds"] = sum(s["cpu_seconds"] for s in record["steps"])
        record["environment_cpu_seconds"] = sum(
            s["cpu_seconds"] for s in record["steps"] if s["phase"] == "environment")
        record["proof_check_cpu_seconds"] = sum(
            s["cpu_seconds"] for s in record["steps"] if s["phase"] == "proof_check")
        (attempt / "result.json").write_text(json.dumps(record, indent=2) + "\n")

    def run_step(name, command, cwd, phase):
        log = attempt / f"{len(record['steps']):02d}_{name}.log"
        wall, cpu = time.monotonic(), child_cpu()
        print(json.dumps({"step": name, "phase": phase, "status": "started"}), flush=True)
        with log.open("w") as stream:
            try:
                code = subprocess.run(command, cwd=cwd, env=env,
                                      stdout=stream, stderr=subprocess.STDOUT).returncode
            except OSError as error:
                stream.write(str(error) + "\n")
                code = 127
        record["steps"].append({
            "name": name, "phase": phase, "command": [str(x) for x in command],
            "cwd": str(cwd), "returncode": code, "log": str(log),
            "wall_seconds": time.monotonic() - wall,
            "cpu_seconds": child_cpu() - cpu,
        })
        save()
        print(json.dumps({"step": name, "returncode": code}), flush=True)
        if code:
            raise RuntimeError(f"{name} returned {code}; see {log}")

    try:
        # Affinity is inherited by Lean, curl and leantar, including their threads.
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {available[0]})
        record["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        manifest = json.loads((proof / "lake-manifest.json").read_text())
        packages = {p["name"]: proof / manifest["packagesDir"] / p["name"]
                    for p in manifest["packages"]}
        sources = [proof, *packages.values()]
        lean = toolchain / "bin" / "lean"
        required = [lean, toolchain / "bin" / "leantar",
                    *(p / "lake-manifest.json" for p in (proof, packages["mathlib"]))]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        for source in sources:
            if not source.is_dir():
                raise FileNotFoundError(source)
        env.update({
            "PATH": str(toolchain / "bin") + os.pathsep + env.get("PATH", ""),
            "LEAN_PATH": os.pathsep.join(str(p / ".lake/build/lib/lean") for p in sources),
            "LEAN_SRC_PATH": os.pathsep.join(map(str, sources)),
            "MATHLIB_CACHE_DIR": str(proof.parent / "cache"),
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        })
        record["source_roots"] = list(map(str, sources))
        run_step("lean_version", [str(lean), "--version"], proof, "environment")
        roots = {"Cache": packages["mathlib"], "Batteries": packages["batteries"],
                 "SCOLHKG": proof}

        def compile_modules(target, prefixes, phase):
            modules = import_order(target, roots, prefixes)
            record[f"{phase}_modules"] = [m for m, _, _ in modules]
            rebuilt = set()
            for module, source, relative in modules:
                output = source / ".lake/build/lib/lean" / relative.with_suffix(".olean")
                if phase == "environment" and output.is_file():
                    record.setdefault("reused_bootstrap_modules", []).append(module)
                    continue
                dependencies = IMPORT.findall((source / relative).read_text())
                project_dependencies = [d for d in dependencies if d.split(".")[0] in prefixes]
                # Fixed toolchain/package locks; changed sources are deployed
                # with fresh receiver mtimes, not their older local edit times.
                # Rebuild parents after any changed dependency, and always run
                # the interface so its #check / #print axioms output is fresh.
                if (phase == "proof_check" and module != target and output.is_file()
                        and output.stat().st_mtime_ns >= (source / relative).stat().st_mtime_ns
                        and not rebuilt.intersection(project_dependencies)
                        and all(output.stat().st_mtime_ns >= (
                            roots[d.split(".")[0]] / ".lake/build/lib/lean" /
                            Path(*d.split(".")).with_suffix(".olean")
                        ).stat().st_mtime_ns for d in project_dependencies)):
                    record.setdefault("reused_proof_modules", []).append(module)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                run_step(module, [str(lean), "-j1", "-o", str(output), str(relative)],
                         source, phase)
                rebuilt.add(module)

        compile_modules("Cache.Main", {"Cache", "Batteries"}, "environment")
        if args.phase in {"cache-plan", "cache-diagnose"}:
            project_modules = import_order("SCOLHKG.CurrentMainlineProofInterface", roots,
                                           {"SCOLHKG"})
            external_roots = sorted({
                dependency
                for _, source, relative in project_modules
                for dependency in IMPORT.findall((source / relative).read_text())
                if dependency.split(".")[0] not in {"SCOLHKG", "Lean", "Init", "Std", "Lake"}
            })
            record["external_roots"] = external_roots
            common_source = (
                "import Cache.Requests\n\n"
                "def main : IO Unit := Cache.IO.CacheM.run do\n"
                # The official CLI parser drops its first, command-name argument.
                f"  let roots ← Cache.IO.parseArgs {json.dumps(['get', *external_roots])}\n"
                "  let memo ← Cache.Hashing.getHashMemo roots\n"
                "  let selected ← memo.filterByRootModules roots.keys\n"
            )
        if args.phase == "cache-plan":
            config = proof.parent / "cache_download.cfg"
            plan_source = attempt / "CachePlan.lean"
            plan_source.write_text(
                common_source +
                "  let config ← Cache.Requests.mkGetConfigContent\n"
                "    Cache.Requests.MATHLIBREPO selected\n"
                "  IO.FS.createDirAll Cache.IO.CACHEDIR\n"
                f"  IO.FS.writeFile {json.dumps(str(config))} config\n"
                '  IO.println s!"cache plan ready: {selected.size} modules"\n'
            )
            record["cache_download_config"] = str(config)
            run_step("mathlib_cache_plan", [str(lean), "-j1", "--run", str(plan_source)],
                     proof, "environment")
        elif args.phase == "cache-diagnose":
            diagnostic = attempt / "CacheDiagnose.lean"
            diagnostic.write_text(
                common_source +
                "  let field := `Mathlib.Algebra.BigOperators.Field\n"
                "  let some key := selected[field]?\n"
                '    | throw (IO.userError "Field absent from selected cache map")\n'
                "  let single : Cache.IO.ModuleHashMap := (∅ : Cache.IO.ModuleHashMap).insert field key\n"
                "  let config ← Cache.IO.mkLeanTarConfig single\n"
                "  let cached := Cache.IO.CACHEDIR / key.asLTar\n"
                "  let cachedPresent ← cached.pathExists\n"
                "  let context ← read\n"
                "  let source ← Cache.IO.getSrcDir context.srcSearchPath field\n"
                '  let olean := source / ".lake/build/lib/lean/Mathlib/Algebra/BigOperators/Field.olean"\n'
                "  let oleanExists ← olean.pathExists\n"
                "  let needsDecompression ← Cache.IO.needsDecompression field key\n"
                "  IO.println (Lean.Json.compress (Lean.Json.mkObj [\n"
                '    ("selected_modules", Lean.toJson selected.size),\n'
                '    ("field_cache_file", Lean.toJson cached.toString),\n'
                '    ("cache_file_exists", Lean.toJson cachedPresent),\n'
                '    ("olean", Lean.toJson olean.toString),\n'
                '    ("olean_exists", Lean.toJson oleanExists),\n'
                '    ("needs_decompression", Lean.toJson needsDecompression),\n'
                '    ("leantar_config", Lean.Json.arr config)]))\n'
            )
            run_step("mathlib_cache_diagnose", [str(lean), "-j1", "--run", str(diagnostic)],
                     proof, "environment")
            diagnosis = json.loads(Path(record["steps"][-1]["log"]).read_text().splitlines()[-1])
            record["field_cache_diagnosis"] = diagnosis
            print(json.dumps({"field_cache_diagnosis": diagnosis}), flush=True)
            archive_output = attempt / "field_archive"
            archive_output.mkdir()
            run_step("field_archive_contents", [str(toolchain / "bin/leantar"),
                     "--jobs", "1", "-v", "-x", "-C", str(archive_output),
                     diagnosis["field_cache_file"]], proof, "environment")
        else:
            run_step("mathlib_cache_unpack", [str(lean), "-j1", "--run",
                     str(packages["mathlib"] / "Cache/Main.lean"), "unpack"],
                     proof, "environment")
            compile_modules("SCOLHKG.CurrentMainlineProofInterface", {"SCOLHKG"}, "proof_check")
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
    save()
    last = record["steps"][-1] if record["steps"] else None
    tail = ""
    if last:
        with Path(last["log"]).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 2500))
            tail = stream.read().decode(errors="replace")
    print(json.dumps({
        "status": record["status"], "result_json": str(attempt / "result.json"),
        "phase": args.phase, "cache_download_config": record.get("cache_download_config"),
        "completed_steps": len(record["steps"]), "last_step": last,
        "environment_cpu_seconds": record["environment_cpu_seconds"],
        "proof_check_cpu_seconds": record["proof_check_cpu_seconds"],
        "runner_cpu_seconds": record["runner_cpu_seconds"],
        "wall_seconds": record["wall_seconds"], "error": record.get("error"),
        "last_log_tail": tail,
    }))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
