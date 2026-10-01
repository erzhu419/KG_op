#!/usr/bin/env python3
"""Build the compact anonymous OR attachment from local completed evidence.

Per-cell exports preserve analysis inputs, not byte-identical raw results.
The original frozen receipts remain provenance for the original files.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
REGISTRY = PROJECT / "paper_artifacts/or_review/final_evidence_registry_v1.json"
DEFAULT_OUTPUT = PROJECT / "reproducibility/dist/or_submission_revision_controls_20261002.zip"

# These are the actual high-volume fields in the registered result schemas.
# All other scalar values, nested contracts, verification counts and IDs survive.
OMITTED_FIELDS = {
    "point", "coefficient_point", "x", "x_normalized", "x_recommended",
    "initial_points", "search_records", "history", "runtime_fingerprint",
    "execution_provenance", "frozen_design_path", "traceback",
}
ENTRYPOINTS = (
    "run_profile_stress_matrix.py", "run_functional_profile_scbo_matrix.py",
    "run_external_energy_v2_matrix.py", "run_external_energy_v3_matrix.py",
    "run_external_energy_functional_scbo_matrix.py",
    "run_external_energy_temporal_audit_matrix.py",
    "analyze_profile_stress_suite.py", "analyze_functional_profile_scbo.py",
    "analyze_external_energy_v2.py", "analyze_external_energy_v3.py",
    "analyze_external_energy_temporal_audits.py", "analyze_native_transfer_matrix.py",
    "analyze_binomial_verifier_power.py", "analyze_or_review_v2_diagnostics.py",
    "analyze_profile_all_in_frontier.py", "benchmark_transfer_fairness.py",
    "materialize_transfer_archives.py", "prepare_opsd_energy_data.py",
    "compact_or_review_analysis.py", "build_or_review_evidence_registry.py",
    "audit_frozen_evidence.py", "audit_or_manuscript.py",
    "render_or_review_final_artifacts.py", "export_submission_revision_bundle.py",
    "submission_revision_attribution.py", "submission_revision_energy_v5.py",
    "render_submission_revision_20260930.py", "render_hvd_diagnostic.py",
    "benchmark_energy_archive_reuse.py",
    "compare_source_first_remainders.py", "reaudit_profile_decision_chain.py",
    "diagnose_first_center_margin_transfer.py",
    "replay_public_battery_target_search.py", "replay_public_battery_simple_controls.py",
    "run_public_battery_source_archive.py",
)


def portable_text(value):
    """Remove observed workstation prefixes while retaining repository paths."""
    value = value.replace(str(REPOSITORY) + "/", "")
    # JSON metadata contains both local and cluster checkout locations.
    value = re.sub(r"/home/[A-Za-z0-9_.-]+/(?:[^/\s\"']+/)*KG_op[^/\s\"']*/", "", value)
    value = re.sub(r"/home/[A-Za-z0-9_.-]+", "/home/operator", value)
    value = re.sub(r"(?:/mnt/[a-z]/Users/|[A-Z]:\\+Users\\+)[^/\\\s\"']+", "<user-home>", value)
    return value


def compact_cell(value):
    if isinstance(value, dict):
        return {key: compact_cell(item) for key, item in value.items()
                if key not in OMITTED_FIELDS}
    if isinstance(value, list):
        return [compact_cell(item) for item in value]
    if isinstance(value, str):
        return portable_text(value)
    return value


def encoded_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_cells(archive, results_root, registry):
    matrices = []
    for matrix in registry["matrices"]:
        paths = sorted(results_root.glob(matrix["relative_glob"]))
        expected = int(matrix["expected_cell_count"])
        if len(paths) != expected:
            raise ValueError(f"{matrix['name']}: {len(paths)} cells; expected {expected}")
        member = f"SC-OLH-KG/reproducibility/cells/{matrix['name']}.jsonl"
        output_bytes = 0
        failures = 0
        with archive.open(member, "w") as handle:
            for path in paths:
                original = json.loads(path.read_text(encoding="utf-8"))
                failures += original.get("status") == "error"
                data = encoded_json({
                    "relative_result_path": path.relative_to(results_root).as_posix(),
                    "cell": compact_cell(original),
                })
                handle.write(data)
                output_bytes += len(data)
        matrices.append({
            "name": matrix["name"], "cell_count": len(paths),
            "algorithmic_failure_count": failures,
            "path": member, "uncompressed_bytes": output_bytes,
        })
    return matrices


def materialize_cells(cells_dir, results_root):
    """Expand projected analysis inputs into a new directory, never raw results."""
    if results_root.exists() and any(results_root.iterdir()):
        raise ValueError("choose a new empty results directory for projected cells")
    count = 0
    for path in sorted(cells_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                target = results_root / row["relative_result_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(encoded_json(row["cell"]))
                count += 1
    if not count:
        raise ValueError(f"no compact cells found in {cells_dir}")
    return count


def source_closure(entrypoints):
    """Include repository imports of the actual runners and analysis entrypoints."""
    pending = list(entrypoints)
    included = set()
    while pending:
        path = pending.pop()
        if path in included or not path.is_file():
            continue
        included.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parents[node.level - 1]
                    parts = node.module.split(".") if node.module else []
                    module_path = base.joinpath(*parts)
                    pending.append(module_path.with_suffix(".py"))
                    pending.append(module_path / "__init__.py")
                    pending.extend(module_path / (alias.name + ".py") for alias in node.names)
                elif node.module:
                    modules.append(node.module)
                    modules.extend(node.module + "." + alias.name for alias in node.names)
        for module in modules:
            parts = module.split(".")
            for base in (PROJECT, REPOSITORY):
                pending.append(base.joinpath(*parts).with_suffix(".py"))
                for length in range(1, len(parts) + 1):
                    pending.append(base.joinpath(*parts[:length], "__init__.py"))
    return included


def selected_files():
    entrypoints = [PROJECT / "performance" / name for name in ENTRYPOINTS]
    entrypoints.extend((PROJECT / "performance").glob("submission_revision*.py"))
    selected = source_closure(entrypoints)
    # Third-party backend implementations are imported dynamically.
    selected.update((PROJECT / "baselines").glob("*.py"))
    selected.update((PROJECT / "reproducibility").glob("requirements*.txt"))
    selected.update({PROJECT / "README.md", PROJECT / "reproducibility/README.md",
                     PROJECT / "reproducibility/current_protocol_errata_20260907.md"})
    manifests = PROJECT / "performance/manifests"
    for pattern in ("or_review*.json", "profile_atlas_v2*.json", "profile_stress_v2*.json",
                    "external_energy_region_holdout_v2*.json", "submission_revision*.json", "energy_archive_reuse*.json",
                    "source_first_remainders*.json", "public_battery*.json"):
        selected.update(manifests.glob(pattern))
    selected.add(manifests / "v18b_exactkg_mcdiag.json")
    selected.update((PROJECT / "paper_artifacts/or_review").glob("*.json"))
    selected.update((PROJECT / "paper_artifacts/submission_revision_20260907").glob("*.json"))
    current = PROJECT / "paper_artifacts/submission_revision_20260930"
    selected.update(current.glob("*.json"))
    selected.update((current / "energy_archives").glob("*.json"))
    selected.update((current / "energy_designs").glob("*.json"))
    corrected = PROJECT / "paper_artifacts/submission_revision_energy_v5_20260930"
    selected.update(corrected.glob("*.json"))
    selected.update((corrected / "energy_archives").glob("*.json"))
    selected.update((corrected / "energy_designs").glob("*.json"))
    reuse = PROJECT / "paper_artifacts/energy_archive_reuse_v2_20260930"
    selected.update(reuse.glob("*.json"))
    selected.update((reuse / "archives").glob("*.json"))
    for directory in ("decision_chain_reaudit_20261002", "first_center_margin_diagnostic_20261002",
                      "source_first_remainders_v1_20261002", "public_battery_target_search_replay_v1_20261002",
                      "public_battery_simple_controls_v1_20261002", "public_battery_source_archive_population_v1_20261002",
                      "public_battery_source_archive_assessment_v1_20261002",
                      "public_battery_frozen_design_coverage_v1_20261002"):
        for name in ("plan.json", "designs.json", "source_designs.json", "population_support.json",
                     "source_designs_snapshot.json", "rows.json", "new_rows.json", "summary.json"):
            selected.add(PROJECT / "paper_artifacts" / directory / name)
    population = PROJECT / "paper_artifacts/public_battery_announced_population_screen_v1_20261001"
    for name in ("summary.json", "population_reliability.json", "artifact_accounting.json", "execution_interruptions.json"):
        selected.add(population / name)
    selected.add(PROJECT / "paper_artifacts/hvd_diagnostic_summary.csv")
    selected.add(PROJECT / "tests/test_energy_archive_reuse.py")
    selected.add(PROJECT / "tests/test_external_energy_v3.py")
    selected.add(PROJECT / "docs/submission_review_20260907/kg_standardized_generic_diagnostic.json")
    manuscript = PROJECT / "manuscript"
    for name in ("main.tex", "supplement.tex", "main.pdf", "supplement.pdf", "main.bbl", "supplement.bbl",
                 "README.md", "review_artifact_manifest.json",
                 "references.bib", "informs4.cls", "informs2014.bst", "informs_Logo.pdf"):
        selected.add(manuscript / name)
    selected.update((manuscript / "sections").glob("*.tex"))
    # Include exactly the figure/table dependencies of current manuscript sources.
    for path in list(selected):
        if path.suffix != ".tex":
            continue
        source = path.read_text(encoding="utf-8")
        for command, name in re.findall(r"\\(input|includegraphics|begin\{overpic\})(?:\[[^\]]*\])?\{([^}]+)\}", source):
            candidate = manuscript / name
            if command != "input" and not candidate.is_file():
                candidate = manuscript / "figures" / name
            if not candidate.suffix:
                candidate = candidate.with_suffix(".tex" if command == "input" else ".pdf")
            if not candidate.is_file():
                raise FileNotFoundError(f"manuscript dependency missing: {candidate}")
            selected.add(candidate)
    proof = REPOSITORY / "proof"
    selected.update((proof / "SCOLHKG").rglob("*.lean"))
    for name in ("SCOLHKG.lean", "lean-toolchain", "lakefile.lean", "lake-manifest.json",
                 "final_or_theory.md", "code_map.md"):
        selected.add(proof / name)
    selected.add(PROJECT / "data/external/opsd_time_series_extended_v2.npz")
    selected.add(PROJECT / "data/external/opsd_time_series_extended_v2.npz.manifest.json")
    return sorted(path for path in selected if path.is_file())


BUNDLE_README = """# Anonymous Operations Research supplementary code and data

Start with `SC-OLH-KG/reproducibility/README.md`. The article and electronic
companion are `SC-OLH-KG/manuscript/main.pdf` and `supplement.pdf`.

`reproducibility/cells/*.jsonl` contains one projected record per original
registered experiment cell (17,360 records, including the recorded failed
functional-control run). Each line stores `relative_result_path` and `cell`.
Projection retains task/seed/arm/configuration identifiers, every scalar
outcome, oracle objective, cost, nested deployment truth, verifier counts and
small profile identifiers. It omits policy vectors, coefficient vectors,
search histories and workstation runtime metadata. The added comparison's
per-task inputs and summaries are under `paper_artifacts/submission_revision_20260907`.
The September 30 revision is under `paper_artifacts/submission_revision_20260930`:
4,160 profile records include attribution, 240 prospective latent tasks crossed
with six arms, and library-size diagnostics. The historical 720 Energy V4
records are preserved, but their simulator reused hourly power capacity across
dispatch substeps. Current corrected evidence is the separate 720 Energy V5
records in `paper_artifacts/submission_revision_energy_v5_20260930`, with common
50% initial inventory and shared hourly power. Its five source archives and
18 designs are included. `paper_artifacts/energy_archive_reuse_v2_20260930`
contains 5,400 records for the retrospective same-market monthly reuse study,
its 18 source archives, and its exact analysis inputs. The interrupted reuse V1
is excluded from current physical evidence.
The frozen local protocol is `performance/manifests/submission_revision_20260930.json`.

October 2 adds the 1,440-decision reconstruction, 240-task first-center margin
diagnostic, and fixed-first remainder controls (250 new runs, 230 reused ordered
designs). Their compact outcomes, frozen plan, protocol and replay code are
included. Public-battery development controls include compact per-run outcomes,
source accounting and census summaries. Large source/target population journals
stay on the server; battery window-level replay requires those journals and
is not supported by the compact attachment alone. The 2025 year remains sealed.

The source package contains the actual benchmark/analysis dependencies,
registered protocols and erratum, current compact evidence, manuscript sources
and PDFs, compact OPSD data and manifest, and Lean sources/configuration.
Full raw results, checkpoints, model weights, raw OPSD CSV, development notes,
private scheduler scripts and author-specific submission metadata are excluded.

The compact cells support fresh paired and cost analysis. They cannot reproduce
raw-result byte receipts or rerun a policy-level temporal audit; frozen original
receipts refer to the original files. Statistical temporal block summaries are
retained per cell. The current statistical commands and pairing units are in
the reproduction guide. Re-executing simulators or fitting third-party methods
requires the documented dependencies and compute time.

`paper_artifacts/hvd_diagnostic_summary.csv` contains the six aggregate inputs
needed by `performance/render_hvd_diagnostic.py`; historical seed-level HVD
records are outside this attachment. The added rank/design-stability proposition
has an analytic proof; the included Lean interface covers the original results.
"""


def export_bundle(results_root, output, previous_bundle=None):
    required = [
        PROJECT / "manuscript/main.pdf", PROJECT / "manuscript/supplement.pdf",
        PROJECT / "paper_artifacts/submission_revision_20260907/rows.json",
        PROJECT / "paper_artifacts/submission_revision_20260907/analysis.json",
        PROJECT / "performance/manifests/submission_revision_controls_20260907.json",
        PROJECT / "paper_artifacts/submission_revision_20260907/energy_rows.json",
        PROJECT / "paper_artifacts/submission_revision_20260907/energy_analysis.json",
        PROJECT / "performance/manifests/submission_revision_energy_20260907.json",
        PROJECT / "docs/submission_review_20260907/kg_standardized_generic_diagnostic.json",
        PROJECT / "paper_artifacts/decision_chain_reaudit_20261002/rows.json",
        PROJECT / "paper_artifacts/first_center_margin_diagnostic_20261002/rows.json",
        PROJECT / "paper_artifacts/source_first_remainders_v1_20261002/plan.json",
        PROJECT / "paper_artifacts/source_first_remainders_v1_20261002/new_rows.json",
        PROJECT / "paper_artifacts/source_first_remainders_v1_20261002/summary.json",
        PROJECT / "paper_artifacts/public_battery_simple_controls_v1_20261002/summary.json",
    ]
    for path in required:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"complete revision artifact required: {path.relative_to(REPOSITORY)}")
    current = PROJECT / "paper_artifacts/submission_revision_20260930"
    for stem, expected, directory in (("profile", 4160, current), ("energy", 720, PROJECT / "paper_artifacts/submission_revision_energy_v5_20260930")):
        rows = json.loads((directory / (stem + "_rows.json")).read_text())["rows"]
        analysis = json.loads((directory / (stem + "_analysis.json")).read_text())
        if len(rows) != expected or analysis["status"] != "complete":
            raise ValueError(f"current {stem} evidence is incomplete")
    reuse = PROJECT / "paper_artifacts/energy_archive_reuse_v2_20260930"
    if len(json.loads((reuse / "rows.json").read_text())["rows"]) != 5400 or json.loads((reuse / "analysis.json").read_text())["status"] != "complete":
        raise ValueError("current monthly archive reuse evidence is incomplete")
    revision = PROJECT / "paper_artifacts/submission_revision_20260907"
    for stem, expected in (("", 1920), ("energy_", 360)):
        rows = json.loads((revision / (stem + "rows.json")).read_text())["rows"]
        analysis = json.loads((revision / (stem + "analysis.json")).read_text())
        if len(rows) != expected or analysis.get("status") != "complete":
            raise ValueError(f"{stem or 'synthetic'} revision is not complete: expected {expected} rows")
    registry = json.loads(REGISTRY.read_text())
    remainder = json.loads((PROJECT / "paper_artifacts/source_first_remainders_v1_20261002/summary.json").read_text())
    if remainder["status"] != "complete" or remainder["new_arm_task_runs"] != 250 or remainder["reused_arm_task_runs"] != 230:
        raise ValueError("fixed-first remainder controls are incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("README.md", BUNDLE_README)
        files = []
        for path in selected_files():
            name = path.relative_to(REPOSITORY).as_posix()
            content = path.read_bytes()
            if path.suffix in {".py", ".json", ".md", ".tex", ".bbl", ".txt", ".lean"}:
                content = portable_text(content.decode("utf-8")).encode("utf-8")
            archive.writestr(name, content)
            files.append(name)
        if previous_bundle is None:
            matrices = write_cells(archive, results_root, registry)
        else:
            # Unchanged projected cells already exist; avoid reading full raw histories.
            with zipfile.ZipFile(previous_bundle) as previous:
                old_inventory = json.loads(previous.read("bundle_inventory.json"))
                matrices = old_inventory["matrices"]
                if sum(m["cell_count"] for m in matrices) != 17360:
                    raise ValueError("previous attachment must contain the 17360 historical cells")
                for matrix in matrices:
                    archive.writestr(matrix["path"], previous.read(matrix["path"]))
        inventory = {
            "format": "or_submission_revision_compact_bundle_20261002",
            "current_revision": {"profile_rows": 4160, "energy_rows": 720,
                                 "energy_version": "V5 shared hourly power", "monthly_archive_reuse_rows": 5400,
                                 "prospective_latent_tasks": 240,
                                 "hvd_table_inputs": "six aggregate rows",
                                 "remainder_diagnostic": {"tasks": 240, "new_runs": 250, "reused_runs": 230},
                                 "reconstructed_original_decisions": 1440,
                                 "first_center_margin_tasks": 240,
                                 "battery_population_journals_included": False},
            "original_registered_cell_count": sum(row["cell_count"] for row in matrices),
            "algorithmic_failure_count": sum(row["algorithmic_failure_count"] for row in matrices),
            "projection_omitted_fields": sorted(OMITTED_FIELDS),
            "original_receipts_apply_to": "original full result files; not projected cells",
            "files": files, "matrices": matrices,
        }
        archive.writestr("bundle_inventory.json", encoded_json(inventory))
    temporary.replace(output)
    return {"output": str(output), "bytes": output.stat().st_size,
            "files": len(files), "original_cells": inventory["original_registered_cell_count"],
            "algorithmic_failures": inventory["algorithmic_failure_count"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-root", type=Path, default=PROJECT / "results")
    parser.add_argument("--reuse-cells-from", type=Path,
                        help="Copy unchanged compact historical cells from a previous attachment.")
    parser.add_argument("--materialize-cells", type=Path,
                        help="Expand compact JSONL records into --results-root (must be empty).")
    args = parser.parse_args()
    if args.materialize_cells:
        result = {"materialized_projected_cells": materialize_cells(
            args.materialize_cells, args.results_root)}
    else:
        result = export_bundle(args.results_root, args.out, args.reuse_cells_from)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
