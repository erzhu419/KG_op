from pathlib import Path

from performance.audit_lean_proof import (
    forbidden_declarations,
    source_tree_sha256,
)


def test_lean_source_audit_is_deterministic_and_rejects_placeholders(tmp_path):
    root = Path(tmp_path)
    package = root / "SCOLHKG"
    package.mkdir()
    source = package / "Proof.lean"
    source.write_text("theorem complete : True := by trivial\n")
    first = source_tree_sha256(root)
    assert first == source_tree_sha256(root)
    assert forbidden_declarations(root) == []

    source.write_text("axiom unfinished : True\n")
    findings = forbidden_declarations(root)
    assert len(findings) == 1
    assert findings[0]["line"] == 1
    assert source_tree_sha256(root) != first
