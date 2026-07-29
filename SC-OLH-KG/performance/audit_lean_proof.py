#!/usr/bin/env python3
"""Build and hash the Lean proof artifact used by the paper release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom)\b")


def lean_sources(proof_root):
    proof_root = Path(proof_root).resolve()
    paths = sorted((proof_root / "SCOLHKG").rglob("*.lean"))
    entrypoint = proof_root / "SCOLHKG.lean"
    if entrypoint.exists():
        paths.append(entrypoint)
    return sorted(set(paths))


def source_tree_sha256(proof_root):
    proof_root = Path(proof_root).resolve()
    digest = hashlib.sha256()
    for path in lean_sources(proof_root):
        relative = path.relative_to(proof_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def forbidden_declarations(proof_root):
    proof_root = Path(proof_root).resolve()
    findings = []
    for path in lean_sources(proof_root):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if FORBIDDEN.search(line):
                findings.append({
                    "path": path.relative_to(proof_root).as_posix(),
                    "line": line_number,
                    "text": line.strip(),
                })
    return findings


def build_receipt(proof_root, *, run_build=True):
    proof_root = Path(proof_root).resolve()
    sources = lean_sources(proof_root)
    if not sources:
        raise ValueError("Lean proof tree contains no sources")
    findings = forbidden_declarations(proof_root)
    started = time.time()
    build = {
        "command": ["lake", "build"],
        "executed": bool(run_build),
        "returncode": None,
        "wall_time_sec": 0.0,
    }
    if run_build:
        completed = subprocess.run(
            ["lake", "build"],
            cwd=proof_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        build.update({
            "returncode": int(completed.returncode),
            "wall_time_sec": float(time.time() - started),
            "output_tail": completed.stdout.splitlines()[-40:],
        })
    passed = not findings and (
        not run_build or build["returncode"] == 0
    )
    return {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "proof_root": str(proof_root),
        "lean_source_count": len(sources),
        "lean_source_tree_sha256": source_tree_sha256(proof_root),
        "forbidden_tokens": ["sorry", "admit", "axiom"],
        "forbidden_declaration_count": len(findings),
        "forbidden_declarations": findings,
        "build": build,
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    receipt = build_receipt(args.proof_root, run_build=True)
    _atomic_json(args.out, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "lean_source_count": receipt["lean_source_count"],
        "lean_source_tree_sha256": receipt["lean_source_tree_sha256"],
        "out": args.out,
    }, indent=2))
    if receipt["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
