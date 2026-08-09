import json

from performance.audit_or_manuscript import (
    _abstract_word_count,
    _reference_start_page,
    build_receipt,
)


def test_abstract_word_count_ignores_latex_commands():
    source = r"""
    \begin{abstract}
    A small \textbf{structural} proposal solves a held-out task.
    \end{abstract}
    """
    assert _abstract_word_count(source) == 8


def test_reference_start_page_reads_aux_label():
    aux = r"\newlabel{page:references}{{}{26}{}{section*.8}{}}"
    assert _reference_start_page(aux) == 26


def test_receipt_passes_for_clean_precompiled_fixture(tmp_path):
    manuscript = tmp_path / "manuscript"
    sections = manuscript / "sections"
    tables = manuscript / "tables"
    sections.mkdir(parents=True)
    tables.mkdir()
    (manuscript / "main.tex").write_text(
        "\\begin{abstract}A short abstract.\\end{abstract}",
        encoding="utf-8",
    )
    (manuscript / "supplement.tex").write_text(
        "Supplement fixture.",
        encoding="utf-8",
    )
    (manuscript / "references.bib").write_text("", encoding="utf-8")
    (sections / "01.tex").write_text("text", encoding="utf-8")
    (tables / "table.tex").write_text("table", encoding="utf-8")
    (manuscript / "main.pdf").write_bytes(b"%PDF-1.4\nfixture")
    (manuscript / "main.log").write_text("clean log", encoding="utf-8")
    (manuscript / "supplement.pdf").write_bytes(b"%PDF-1.4\nfixture")
    (manuscript / "supplement.log").write_text(
        "clean supplement log",
        encoding="utf-8",
    )
    (manuscript / "main.aux").write_text(
        r"\newlabel{page:references}{{}{12}{}{section*.1}{}}",
        encoding="utf-8",
    )
    artifact_manifest = manuscript / "artifact_manifest.json"
    artifact_manifest.write_text(json.dumps({
        "status": "complete",
        "contract_id": "artifact",
        "contracts": {"reads_compact_audited_artifacts_only": True},
    }), encoding="utf-8")

    receipt = build_receipt(
        manuscript_dir=manuscript,
        artifact_manifest_path=artifact_manifest,
        compile_manuscript=False,
    )
    assert receipt["status"] == "pass"
    assert receipt["journal_format_checks"][
        "body_pages_excluding_references"
    ] == 11
    assert receipt["supplement"]["sha256"] is not None
