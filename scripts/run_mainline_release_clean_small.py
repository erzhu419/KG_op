"""Operator-only, once-through clean-environment acceptance of the release.

Run this through the project scheduler. It exports the preselected paid inputs,
builds a candidate archive, installs the pinned public dependencies into a fresh
venv and runs that archive's real small case once. Historical studies are read
only. The public run_small entrypoint itself requires no scheduler or SSH.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tarfile
import time


RELEASE = Path("SC-OLH-KG/reproducibility/mainline_release")
BASE_COMMIT = "d3fac7622462d50c2a048833b91b33bdcccf2dab"


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def execute(root, work, source_commit, wheelhouse=None):
    root, work = Path(root).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    started, wall = time.process_time(), time.perf_counter()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    release = root / RELEASE
    receipt_dir = release / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    public_receipt = receipt_dir / "small_run.json"
    operator_receipt = receipt_dir / f"small_run_operator_{work.name}.json"
    receipt = dict(schema="mainline-clean-small-operator-v1", status="failed",
        exit_code=1, kind="release_acceptance_only", source_commit=source_commit,
        automatic_retries=0, scientific_cohort_overwritten=False,
        started_utc=datetime.now(timezone.utc).isoformat(), phases=[],
        operator_work_directory=str(work), new_source_calls=0, new_source_fits=0,
        original_scientific_budgets_reset=False, phase="preflight")
    small_output = work / "small_attempt"
    small_exit = None
    environment = dict(os.environ)
    for name in list(environment):
        if name in ("PYTHONPATH", "PYTHONHOME") or name.startswith("PIP_"):
            environment.pop(name)
    environment.update(PYTHONNOUSERSITE="1", PIP_CONFIG_FILE=os.devnull,
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def phase(name, command, *, cwd=root, allowed=(0,)):
        receipt["phase"] = name
        row = dict(name=name, status="failed", command=list(map(str, command)),
                   log_file=f"{name}.log", exit_code=None)
        receipt["phases"].append(row)
        cpu, clock = time.process_time(), time.perf_counter()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        print(json.dumps(dict(phase=name, status="started")), flush=True)
        try:
            with (work / row["log_file"]).open("w") as log:
                result = subprocess.run(list(map(str, command)), cwd=cwd, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
            row["exit_code"] = result.returncode
            if allowed is not None and result.returncode not in allowed:
                raise RuntimeError(f"{name} exited {result.returncode}; see {row['log_file']}")
            row["status"] = ("complete" if result.returncode == 0 else
                "candidate_pending_small_acceptance" if name == "build_candidate" else "failed")
            return result.returncode
        except BaseException as error:
            row["error"] = repr(error)
            raise
        finally:
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            row.update(operator_cpu_seconds=time.process_time()-cpu,
                child_cpu_seconds=(after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime),
                wall_seconds=time.perf_counter()-clock,
                cumulative_child_peak_rss_bytes=int(after.ru_maxrss) * (1 if sys.platform == "darwin" else 1024))
            save(work / "operator_receipt.json", receipt)

    try:
        if public_receipt.exists() or operator_receipt.exists():
            raise FileExistsError("Existing acceptance receipts are retained; this operator never overwrites an attempt")
        case_directory = release / "case"
        if case_directory.exists():
            receipt["phase"] = "reuse_completed_case"
            cpu, clock = time.process_time(), time.perf_counter()
            preparation = json.loads((case_directory / "preparation_receipt.json").read_text())
            case = json.loads((case_directory / "case.json").read_text())
            expected = dict(selected_domain="InventorySupplyChain", geometry_task_seed=2026120001,
                selected_stream="observation_1", reused_initial_rows=10, reused_shared_neighbors=64,
                source_commit=source_commit, status="complete")
            if any(preparation.get(key) != value for key, value in expected.items()):
                raise ValueError("Existing case preparation is incomplete or differs from the preselected release case")
            if (case["domain"], case["geometry_task_seed"], case["stream"], case["source_commit"]) != (
                    "InventorySupplyChain", 2026120001, "observation_1", source_commit):
                raise ValueError("Existing portable case differs from its successful preparation receipt")
            for field in ("initial_rows_file", "shared_neighbors_file", "resolved_task_file"):
                if not (case_directory / case[field]).is_file():
                    raise FileNotFoundError(f"The completed case is missing {case[field]}")
            receipt["phases"].append(dict(name="reuse_completed_case", status="complete", exit_code=0,
                source_preparation_receipt=str(case_directory / "preparation_receipt.json"),
                source_preparation_unchanged=True, new_source_calls=0, new_target_calls=0,
                operator_cpu_seconds=time.process_time()-cpu, child_cpu_seconds=0.,
                wall_seconds=time.perf_counter()-clock))
            save(work / "operator_receipt.json", receipt)
        else:
            phase("prepare_case", [sys.executable, root / "scripts/prepare_mainline_submission_small_case.py",
                "--project-root", root / "SC-OLH-KG", "--output", case_directory,
                "--source-commit", source_commit, "--asset-prefix", "../assets"])
        archive = work / "candidate.tar.gz"
        build_exit = phase("build_candidate", [sys.executable, release / "package.py", "build",
            "--root", root, "--output", archive], allowed=(0, 2))
        receipt["phase"] = "extract_candidate"
        cpu, clock = time.process_time(), time.perf_counter()
        clean = work / "clean"
        clean.mkdir()
        extraction = dict(name="extract_candidate", status="failed", exit_code=1)
        receipt["phases"].append(extraction)
        try:
            with tarfile.open(archive, "r:gz") as source:
                manifest = json.load(source.extractfile("release_manifest.json"))
                missing = set(manifest["missing"])
                expected_missing = {str(RELEASE / "receipts/small_run.json")}
                if build_exit == 2 and missing != expected_missing:
                    raise RuntimeError(f"Candidate lacks more than the pending small receipt: {sorted(missing)}")
                if build_exit == 0 and missing:
                    raise RuntimeError("Candidate build reported success with missing files")
                for name in ("saved_evidence.json", "lean_cold.json"):
                    if not manifest["acceptance_receipts"][name]["accepted"]:
                        raise RuntimeError(f"The preceding release gate is unfinished: {name}")
                # This is the cooperating operator's just-built archive.
                source.extractall(clean)
            extraction.update(status="complete", exit_code=0)
        except BaseException as error:
            extraction["error"] = repr(error)
            raise
        finally:
            extraction.update(operator_cpu_seconds=time.process_time()-cpu,
                child_cpu_seconds=0., wall_seconds=time.perf_counter()-clock)
            save(work / "operator_receipt.json", receipt)
        receipt["candidate_archive"] = dict(path=str(archive), build_exit_code=build_exit,
            missing=sorted(missing), delivery_gate_passed=False,
            meaning="Candidate for G3 only; final archive must include its acceptance receipt")
        save(work / "operator_receipt.json", receipt)

        venv = work / "venv"
        phase("create_venv", [sys.executable, "-m", "venv", venv], cwd=clean)
        configuration = (venv / "pyvenv.cfg").read_text()
        if "include-system-site-packages = false" not in configuration.lower():
            raise RuntimeError("Clean acceptance requires a venv without system site packages")
        python = venv / "bin/python"
        environment["VIRTUAL_ENV"] = str(venv)
        requirements = clean / RELEASE / "requirements.txt"
        install = [python, "-I", "-m", "pip", "install", "--disable-pip-version-check",
            "--no-input", "--retries", "0", "--no-cache-dir", "--only-binary=:all:"]
        if wheelhouse is None:
            install += ["--index-url", "https://pypi.org/simple"]
            receipt["dependency_source"] = "https://pypi.org/simple"
        else:
            wheelhouse = Path(wheelhouse).resolve()
            install += ["--no-index", "--find-links", wheelhouse]
            receipt["dependency_source"] = dict(wheelhouse=str(wheelhouse), network_used=False)
        install += ["--requirement", requirements]
        phase("install_dependencies", install, cwd=clean)
        probe_path = work / "environment.json"
        probe = "\n".join([
            "import importlib.metadata as m, json, pathlib, platform, site, sys",
            "import numpy, scipy, requests",
            "requirements = pathlib.Path(sys.argv[1]).read_text().splitlines()",
            "expected = dict(line.strip().split('==') for line in requirements if line.strip() and not line.startswith('#'))",
            "actual = {name:m.version(name) for name in expected}",
            "assert actual == expected, (actual, expected)",
            "assert sys.prefix != sys.base_prefix and site.ENABLE_USER_SITE is False",
            "assert platform.python_version() == sys.argv[3], (platform.python_version(), sys.argv[3])",
            "value = dict(status='passed',python=platform.python_version(),prefix=sys.prefix,base_prefix=sys.base_prefix,user_site_enabled=site.ENABLE_USER_SITE,isolated=bool(sys.flags.isolated),versions=actual,sys_path=sys.path)",
            "pathlib.Path(sys.argv[2]).write_text(json.dumps(value,indent=2)+'\\n')",
        ])
        phase("check_environment", [python, "-I", "-c", probe, requirements, probe_path,
            manifest["runtime"]["python"]], cwd=clean)
        receipt["clean_environment"] = json.loads(probe_path.read_text())
        runner = clean / RELEASE / "run_small.py"
        case = clean / RELEASE / "case/case.json"
        small_exit = phase("real_small_case", [python, "-I", runner,
            "--case", case, "--output", small_output], cwd=clean, allowed=None)
        if small_exit != 0:
            raise RuntimeError(f"Real small-case acceptance exited {small_exit}; the attempt is retained without retry")
        result = json.loads((small_output / "receipt.json").read_text())
        if result["status"] != "complete" or result["new_target_search_calls"] != 3:
            raise RuntimeError("The real small case did not complete its frozen three search calls")
        if result["automatic_retries"] != 0 or result["new_verification_calls"] > 352:
            raise RuntimeError("The real small case changed retry or terminal budget rules")
        receipt.update(status="complete", exit_code=0, phase="complete",
            small_run_receipt=str(small_output / "receipt.json"),
            preparation_receipt=str(case_directory / "preparation_receipt.json"),
            new_target_search_calls=result["new_target_search_calls"],
            new_verification_calls=result["new_verification_calls"],
            new_physical_target_calls=result["new_physical_target_calls"],
            reused_initial_calls=result["reused_initial_calls"],
            terminal_maximum_calls=352, public_entrypoint_requires_scheduler=False,
            candidate_delivery_gate_passed=False)
        return receipt
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        if (small_output / "receipt.json").is_file():
            actual = json.loads((small_output / "receipt.json").read_text())
            for key in ("new_target_search_calls", "new_verification_calls", "new_physical_target_calls",
                        "reused_initial_calls", "logical_total_calls"):
                receipt[key] = actual[key]
            actual.update(exit_code=small_exit if small_exit is not None else 1,
                clean_environment=receipt.get("clean_environment"),
                operator_receipt=operator_receipt.name, candidate_delivery_gate_passed=False)
            if receipt["status"] != "complete":
                actual.update(run_small_status=actual["status"], status="failed", exit_code=1,
                              operator_error=receipt.get("error"))
        else:
            actual = dict(status="failed", exit_code=1, kind="release_acceptance_only",
                phase=receipt["phase"], error=receipt.get("error"), automatic_retries=0,
                new_target_search_calls=0, new_verification_calls=0,
                operator_receipt=operator_receipt.name)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt.update(operator_cpu_seconds=time.process_time()-started,
            child_cpu_seconds=children.ru_utime+children.ru_stime-child_start.ru_utime-child_start.ru_stime,
            wall_seconds=time.perf_counter()-wall,
            operator_peak_rss_bytes=int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
            child_peak_rss_bytes=int(children.ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
            finished_utc=datetime.now(timezone.utc).isoformat())
        save(work / "operator_receipt.json", receipt)
        if not operator_receipt.exists():
            save(operator_receipt, receipt)
        if not public_receipt.exists():
            save(public_receipt, actual)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="KG_op repository root, containing SC-OLH-KG and scripts")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="New operator directory outside scientific experiment outputs")
    parser.add_argument("--source-commit", default=BASE_COMMIT)
    parser.add_argument("--wheelhouse", type=Path,
                        help="Optional existing directory of the pinned official wheels; disables network installation")
    args = parser.parse_args(argv)

    def terminate(signum, _frame):
        raise KeyboardInterrupt(f"External termination signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    result = execute(args.root, args.work_dir, args.source_commit, args.wheelhouse)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
