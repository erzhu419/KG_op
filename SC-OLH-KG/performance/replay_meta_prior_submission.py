"""Artifact-level replay and submission consistency check.

Default mode does not run simulators.  It regenerates manuscript tables from
the saved development artifact and runs the claim audit.  ``--compile`` also
builds the independent meta-prior main and supplement with latexmk.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.audit_meta_prior_claims import run as audit_claims
from performance.render_meta_prior_tables import render
from performance.render_meta_prior_baseline import render as render_baseline
from performance.summarize_meta_prior_uncertainty import run as summarize_uncertainty


def run(compile_pdf=False):
    manifest = json.loads((ROOT / "performance/meta_prior_submission_manifest.json").read_text())
    for path in manifest["source_of_truth"].values():
        if path.endswith("source_structure_confirmation_v2.json") and not (ROOT / path).exists():
            continue
        if not (ROOT / path).exists():
            raise FileNotFoundError(ROOT / path)
    tables = render(output_confirmation="manuscript/meta_prior_confirmation_table.tex")
    baseline = render_baseline()
    uncertainty = summarize_uncertainty()
    audit = audit_claims()
    if not audit["passed"]:
        raise RuntimeError(json.dumps(audit, indent=2))
    if compile_pdf:
        manuscript_dir = ROOT / "manuscript"
        for entrypoint in ("meta_prior_main.tex", "meta_prior_supplement.tex"):
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", entrypoint],
                           cwd=manuscript_dir, check=True)
    result = {"passed": True, "tables": tables, "baseline": baseline,
              "uncertainty": uncertainty, "claim_audit": audit,
              "original_confirmation_block_read": False,
              "compile_requested": bool(compile_pdf)}
    (ROOT / "paper_artifacts/meta_prior_submission_replay.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.compile), indent=2))
