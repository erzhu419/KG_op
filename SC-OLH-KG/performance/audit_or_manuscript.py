#!/usr/bin/env python3
"""Compile and audit the frozen Operations Research manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


CONTRACT_ID = "or_manuscript_compilation_receipt_v1"
FORBIDDEN_LOG_PATTERNS = (
    "LaTeX Warning",
    "Package natbib Warning",
    "Overfull",
    "Underfull",
    "undefined references",
    "undefined citations",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _abstract_word_count(source: str) -> int:
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        source,
        flags=re.DOTALL,
    )
    if match is not None:
        abstract = match.group(1)
    else:
        command = re.search(r"\\ABSTRACT\s*\{", source)
        if command is None:
            raise ValueError("abstract environment or OPRE ABSTRACT command not found")
        start = command.end()
        depth = 1
        cursor = start
        while cursor < len(source) and depth:
            if source[cursor] == "{" and source[cursor - 1] != "\\":
                depth += 1
            elif source[cursor] == "}" and source[cursor - 1] != "\\":
                depth -= 1
            cursor += 1
        if depth:
            raise ValueError("unterminated OPRE ABSTRACT command")
        abstract = source[start:cursor - 1]
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", abstract)
    text = re.sub(r"[{}~$^_\\]", " ", text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def _reference_start_page(aux_text: str) -> int:
    match = re.search(
        r"\\newlabel\{page:references\}\{\{[^}]*\}\{([0-9]+)\}",
        aux_text,
    )
    if match is None:
        raise ValueError("reference-start page label not found")
    return int(match.group(1))


def _source_files(manuscript_dir: Path) -> list[Path]:
    rows = [
        manuscript_dir / "main.tex",
        manuscript_dir / "supplement.tex",
        manuscript_dir / "references.bib",
        manuscript_dir / "main.bbl",
        manuscript_dir / "informs4.cls",
        manuscript_dir / "informs2014.bst",
        manuscript_dir / "informs_Logo.pdf",
        manuscript_dir / "figures" / "figure1_profile_space.pdf",
        manuscript_dir / "figures" / "figure2_atlas_coverage.pdf",
    ]
    rows.extend(sorted((manuscript_dir / "sections").glob("*.tex")))
    rows.extend(sorted((manuscript_dir / "tables").glob("*.tex")))
    return rows


def build_receipt(
    *,
    manuscript_dir: Path,
    artifact_manifest_path: Path,
    compile_manuscript: bool = True,
) -> dict:
    manuscript_dir = Path(manuscript_dir).resolve()
    main_tex = manuscript_dir / "main.tex"
    pdf = manuscript_dir / "main.pdf"
    log = manuscript_dir / "main.log"
    aux = manuscript_dir / "main.aux"
    supplement_tex = manuscript_dir / "supplement.tex"
    supplement_pdf = manuscript_dir / "supplement.pdf"
    supplement_log = manuscript_dir / "supplement.log"
    failures: list[str] = []

    main_source = (
        main_tex.read_text(encoding="utf-8") if main_tex.is_file() else ""
    )
    supplement_source = (
        supplement_tex.read_text(encoding="utf-8")
        if supplement_tex.is_file()
        else ""
    )
    opre_class = (
        r"\documentclass[opre,dblanonrev]{informs4}" in main_source
    )
    opre_bibliography = (
        r"\bibliographystyle{informs2014}" in main_source
    )
    opre_spacing = r"\OneAndAHalfSpacedXII" in main_source
    ec_numbering = r"\ECSwitch" in supplement_source
    width_validator_removed = all(
        "eqndefns-left" not in source
        for source in (main_source, supplement_source)
    )
    format_contract = {
        "opre_double_anonymous_class": opre_class,
        "opre_bibliography_style": opre_bibliography,
        "opre_default_review_spacing": opre_spacing,
        "ec_supplement_numbering": ec_numbering,
        "width_validator_removed_from_final_sources": width_validator_removed,
    }
    failed_format = [
        name for name, passed in format_contract.items() if not passed
    ]
    if failed_format:
        failures.append(f"OPRE format contract failed: {failed_format}")

    compile_result = None
    supplement_compile_result = None
    if compile_manuscript:
        compile_result = subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "main.tex",
            ],
            cwd=manuscript_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        if compile_result.returncode != 0:
            failures.append(
                f"latexmk returned {compile_result.returncode}"
            )
        supplement_compile_result = subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "supplement.tex",
            ],
            cwd=manuscript_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        if supplement_compile_result.returncode != 0:
            failures.append(
                "supplement latexmk returned "
                f"{supplement_compile_result.returncode}"
            )

    required = [
        main_tex,
        pdf,
        log,
        aux,
        supplement_tex,
        supplement_pdf,
        supplement_log,
        Path(artifact_manifest_path),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        failures.append(f"missing manuscript artifacts: {missing}")

    abstract_words = None
    reference_start_page = None
    body_pages = None
    log_hits: list[str] = []
    if main_tex.is_file():
        try:
            abstract_words = _abstract_word_count(main_source)
            if abstract_words > 200:
                failures.append(
                    f"abstract has {abstract_words} words, limit is 200"
                )
        except ValueError as error:
            failures.append(str(error))
    if aux.is_file():
        try:
            reference_start_page = _reference_start_page(
                aux.read_text(encoding="utf-8", errors="replace")
            )
            body_pages = reference_start_page - 1
            if body_pages > 30:
                failures.append(
                    f"body has {body_pages} pages before references, limit is 30"
                )
        except ValueError as error:
            failures.append(str(error))
    for label, log_path in (("main", log), ("supplement", supplement_log)):
        if not log_path.is_file():
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        hits = [
            pattern for pattern in FORBIDDEN_LOG_PATTERNS
            if pattern in log_text
        ]
        log_hits.extend(f"{label}:{pattern}" for pattern in hits)
    if log_hits:
        failures.append(f"forbidden LaTeX log diagnostics: {log_hits}")

    source_files = _source_files(manuscript_dir)
    missing_sources = [str(path) for path in source_files if not path.is_file()]
    if missing_sources:
        failures.append(f"missing manuscript source files: {missing_sources}")

    artifact_manifest = None
    if Path(artifact_manifest_path).is_file():
        artifact_manifest = json.loads(
            Path(artifact_manifest_path).read_text(encoding="utf-8")
        )
        if (
            artifact_manifest.get("status") != "complete"
            or artifact_manifest.get("contracts", {}).get(
                "reads_compact_audited_artifacts_only"
            ) is not True
        ):
            failures.append("manuscript artifact manifest is not complete")

    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "compile": {
            "executed": bool(compile_manuscript),
            "returncode": (
                None if compile_result is None else compile_result.returncode
            ),
            "supplement_returncode": (
                None
                if supplement_compile_result is None
                else supplement_compile_result.returncode
            ),
        },
        "journal_format_checks": {
            "abstract_word_count": abstract_words,
            "abstract_word_limit": 200,
            "reference_start_page": reference_start_page,
            "body_pages_excluding_references": body_pages,
            "body_page_limit": 30,
            "forbidden_log_diagnostics": log_hits,
            **format_contract,
        },
        "pdf": {
            "path": str(pdf),
            "sha256": _sha256(pdf) if pdf.is_file() else None,
            "size_bytes": pdf.stat().st_size if pdf.is_file() else None,
        },
        "supplement": {
            "path": str(supplement_pdf),
            "sha256": (
                _sha256(supplement_pdf)
                if supplement_pdf.is_file()
                else None
            ),
            "size_bytes": (
                supplement_pdf.stat().st_size
                if supplement_pdf.is_file()
                else None
            ),
        },
        "source_files": [
            {
                "path": str(path.relative_to(manuscript_dir)),
                "sha256": _sha256(path),
            }
            for path in source_files
            if path.is_file()
        ],
        "artifact_manifest": {
            "path": str(Path(artifact_manifest_path).resolve()),
            "sha256": (
                _sha256(Path(artifact_manifest_path))
                if Path(artifact_manifest_path).is_file()
                else None
            ),
            "contract_id": (
                artifact_manifest.get("contract_id")
                if artifact_manifest is not None
                else None
            ),
        },
        "claims": {
            "evidence_and_theory_frozen_before_drafting": True,
            "source_search_verification_costs_separated": True,
            "hvd_headline_claim_prohibited": True,
            "unconditional_high_dimensional_claim_prohibited": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript-dir", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    receipt = build_receipt(
        manuscript_dir=Path(args.manuscript_dir),
        artifact_manifest_path=Path(args.artifact_manifest),
        compile_manuscript=not args.no_compile,
    )
    _atomic_json(Path(args.out), receipt)
    print(json.dumps({
        "status": receipt["status"],
        "failure_count": len(receipt["failures"]),
        "abstract_word_count": receipt["journal_format_checks"][
            "abstract_word_count"
        ],
        "body_pages_excluding_references": receipt[
            "journal_format_checks"
        ]["body_pages_excluding_references"],
        "out": args.out,
    }, indent=2))
    raise SystemExit(0 if receipt["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
