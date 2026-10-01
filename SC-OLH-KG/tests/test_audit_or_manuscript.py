import json

import pytest

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


def test_abstract_word_count_supports_opre_command_and_nested_braces():
    source = r"""
    \ABSTRACT{A small \textbf{structural design} solves a held-out task.}
    """
    assert _abstract_word_count(source) == 8


def test_reference_start_page_reads_aux_label():
    aux = r"\newlabel{page:references}{{}{26}{}{section*.8}{}}"
    assert _reference_start_page(aux) == 26


@pytest.fixture
def precompiled_manuscript(tmp_path, monkeypatch):
    manuscript = tmp_path / "manuscript"
    sections = manuscript / "sections"
    tables = manuscript / "tables"
    figures = manuscript / "figures"
    sections.mkdir(parents=True)
    tables.mkdir()
    figures.mkdir()
    (manuscript / "main.tex").write_text(
        "\\documentclass[opre,dblanonrev]{informs4}\n"
        "\\OneAndAHalfSpacedXI\n"
        "\\SUBJECTCLASS{Simulation: initial designs.}\n"
        "\\AREAOFREVIEW{Simulation}\n"
        "\\begin{abstract}A short abstract.\\end{abstract}\n"
        "\\bibliographystyle{informs2014}\n",
        encoding="utf-8",
    )
    (manuscript / "supplement.tex").write_text(
        "\\OneAndAHalfSpacedXI\n\\ECSwitch\nSupplement fixture.",
        encoding="utf-8",
    )
    (manuscript / "references.bib").write_text("", encoding="utf-8")
    (manuscript / "main.bbl").write_text("", encoding="utf-8")
    (manuscript / "informs4.cls").write_text("class", encoding="utf-8")
    (manuscript / "informs2014.bst").write_text("style", encoding="utf-8")
    (manuscript / "informs_Logo.pdf").write_bytes(b"logo")
    (figures / "figure1_profile_space.pdf").write_bytes(b"figure one")
    (figures / "figure2_atlas_coverage.pdf").write_bytes(b"figure two")
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
        r"\newlabel{page:references}{{}{12}{}{section*.1}{}}" + "\n"
        + r"\newlabel{page:tables}{{}{14}{}{section*.2}{}}",
        encoding="utf-8",
    )
    artifact_manifest = manuscript / "artifact_manifest.json"
    artifact_manifest.write_text(json.dumps({
        "status": "complete",
        "contract_id": "artifact",
        "contracts": {"reads_compact_audited_artifacts_only": True},
    }), encoding="utf-8")
    pages = {"main.pdf": 17, "supplement.pdf": 8}
    monkeypatch.setattr(
        "performance.audit_or_manuscript._pdf_page_count",
        lambda path: pages[path.name],
    )
    return manuscript, artifact_manifest, pages


def test_receipt_counts_tables_after_references(precompiled_manuscript):
    manuscript, artifact_manifest, _ = precompiled_manuscript

    receipt = build_receipt(
        manuscript_dir=manuscript,
        artifact_manifest_path=artifact_manifest,
        compile_manuscript=False,
    )
    assert receipt["status"] == "pass"
    assert receipt["journal_format_checks"][
        "body_pages_excluding_references"
    ] == 15
    assert receipt["journal_format_checks"]["table_pages_after_references"] == 4
    assert receipt["journal_format_checks"]["supplement_page_limit"] == 15
    assert receipt["supplement"]["sha256"] is not None


def test_postreference_tables_can_exceed_article_page_limit(precompiled_manuscript):
    manuscript, artifact_manifest, pages = precompiled_manuscript
    (manuscript / "main.aux").write_text(
        r"\newlabel{page:references}{{}{26}{}{section*.1}{}}" + "\n"
        + r"\newlabel{page:tables}{{}{29}{}{section*.2}{}}",
        encoding="utf-8",
    )
    pages["main.pdf"] = 34
    receipt = build_receipt(
        manuscript_dir=manuscript, artifact_manifest_path=artifact_manifest,
        compile_manuscript=False,
    )
    assert receipt["status"] == "fail"
    assert receipt["journal_format_checks"]["body_pages_excluding_references"] == 31
    assert any("31 non-reference pages" in message for message in receipt["failures"])


def test_electronic_companion_cannot_exceed_article(precompiled_manuscript):
    manuscript, artifact_manifest, pages = precompiled_manuscript
    pages["supplement.pdf"] = 16
    receipt = build_receipt(
        manuscript_dir=manuscript, artifact_manifest_path=artifact_manifest,
        compile_manuscript=False,
    )
    assert receipt["status"] == "fail"
    assert any("supplement has 16 pages" in message for message in receipt["failures"])


@pytest.mark.parametrize("filename", ["main.tex", "supplement.tex"])
def test_twelve_point_command_does_not_match_eleven_point_requirement(
    precompiled_manuscript, filename,
):
    manuscript, artifact_manifest, _ = precompiled_manuscript
    path = manuscript / filename
    path.write_text(path.read_text().replace("OneAndAHalfSpacedXI", "OneAndAHalfSpacedXII"))
    receipt = build_receipt(
        manuscript_dir=manuscript, artifact_manifest_path=artifact_manifest,
        compile_manuscript=False,
    )
    assert receipt["status"] == "fail"
    assert receipt["journal_format_checks"]["opre_11pt_one_and_a_half_spacing"] is False


@pytest.mark.parametrize("command, field", [
    (r"\SUBJECTCLASS{Simulation: initial designs.}", "subject_classifications_present"),
    (r"\AREAOFREVIEW{Simulation}", "area_of_review_present"),
])
def test_required_journal_metadata(precompiled_manuscript, command, field):
    manuscript, artifact_manifest, _ = precompiled_manuscript
    path = manuscript / "main.tex"
    path.write_text(path.read_text().replace(command, ""))
    receipt = build_receipt(
        manuscript_dir=manuscript, artifact_manifest_path=artifact_manifest,
        compile_manuscript=False,
    )
    assert receipt["status"] == "fail"
    assert receipt["journal_format_checks"][field] is False


def test_abstract_word_limit_is_enforced(precompiled_manuscript):
    manuscript, artifact_manifest, _ = precompiled_manuscript
    path = manuscript / "main.tex"
    path.write_text(path.read_text().replace("A short abstract.", " ".join(["word"] * 201)))
    receipt = build_receipt(
        manuscript_dir=manuscript, artifact_manifest_path=artifact_manifest,
        compile_manuscript=False,
    )
    assert receipt["status"] == "fail"
    assert "abstract has 201 words, limit is 200" in receipt["failures"]
