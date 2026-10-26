#!/usr/bin/env python3
"""Assemble or verify the fixed current-mainline review attachment.

Only source inspection, selected file reads and archive compression occur here.
No scientific modules are imported, models restored, or computations scheduled.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

RELEASE = Path('SC-OLH-KG/reproducibility/mainline_release')
BASE_COMMIT = 'd3fac7622462d50c2a048833b91b33bdcccf2dab'
FAMILIES = ('InventorySupplyChain', 'QueueResourceControl', 'FactorShockStatePolicyRZDT1')
GUIDES = ('README.md', 'DATA_AVAILABILITY.md', 'AUTHOR_DISCLOSURE.md',
          'THIRD_PARTY.md', 'LICENSE', 'DATA_LICENSE.md', 'LEAN.md', 'COSTS.md')
TOOLS = ('package.py', 'statistics.py', 'export_cells.py', 'run_small.py',
         'lean_delivery.py', 'verify_statistics.py', 'frozen_behavior_parity.py')


def read(path):
    return json.loads(path.read_text())


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def digest(source):
    value = hashlib.sha256()
    if isinstance(source, bytes):
        value.update(source)
    else:
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                value.update(block)
    return value.hexdigest()


def python_closure(root):
    """Include production imports reachable from the public small-case runner.

    Conditional imports stay present even when this case does not execute them.
    Pickled native/source classes use the same representation/production modules.
    External packages are installed from requirements, rather than redistributed.
    """
    project, visited = root / 'SC-OLH-KG', set()
    pending = [root / RELEASE / 'run_small.py']
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module] + [node.module + '.' + item.name for item in node.names]
            for name in names:
                stem = project.joinpath(*name.split('.'))
                for dependency in (stem.with_suffix('.py'), stem / '__init__.py'):
                    if dependency.is_file():
                        pending.append(dependency)
    return sorted(visited)


def git(root, *args):
    try:
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def assemble(root, publication_commit=None):
    files, missing, transformations = {}, [], {}

    def add(relative, target=None):
        path = root / relative
        if path.is_file():
            files[str(target or relative)] = path
        else:
            missing.append(str(relative))

    for path in python_closure(root):
        add(path.relative_to(root))
    for name in (*GUIDES, *TOOLS, 'acceptance_registration.json', 'requirements.txt'):
        add(RELEASE / name)
    for name in GUIDES:
        add(RELEASE / name, Path(name))
    add(Path('SC-OLH-KG/reproducibility/requirements-mainline-cluster.txt'))
    for name in ('cells.json.gz', 'statistics.json.gz'):
        add(RELEASE / 'data' / name)
    add(RELEASE / 'data/cost_metadata.json')
    for name in ('case.json', 'initial_rows.json', 'neighbors.json', 'resolved_task.json'):
        add(RELEASE / 'case' / name)
    add(RELEASE / 'case/reference/initialization.json')
    add(RELEASE / 'case/reference/search/paid_search_raw.json')
    add(RELEASE / 'case/reference/input.json')
    add(RELEASE / 'case/reference/verification.json')
    receipts = sorted(path for path in (root / RELEASE / 'receipts').glob('*')
                      if path.suffix in ('.json', '.jsonl'))
    accepted = {}
    for name in ('saved_evidence.json', 'lean_cold.json', 'small_run.json'):
        path = root / RELEASE / 'receipts' / name
        if not path.is_file():
            missing.append(str(RELEASE / 'receipts' / name))
            accepted[name] = dict(status='missing', accepted=False)
        else:
            receipt = read(path)
            status = receipt.get('status')
            accepted[name] = dict(status=status, exit_code=receipt.get('exit_code'),
                accepted=status in ('complete', 'passed') and receipt.get('exit_code', 0) == 0)
    parity = root / RELEASE / 'receipts/frozen_behavior_parity.json'
    if parity.is_file():
        receipt = read(parity)
        accepted[parity.name] = dict(status=receipt.get('status'), exit_code=receipt.get('exit_code'),
            accepted=receipt.get('status') == 'passed' and receipt.get('exit_code') == 0)
    for path in receipts:
        add(path.relative_to(root))

    # Map assets into one portable directory without making a second local copy.
    assets = []
    source = read(root / 'SC-OLH-KG/performance/manifests/mainline_source_alignment_targets_v1.json')
    for family in FAMILIES:
        for kind in ('real', 'permuted'):
            base = Path('paper_artifacts/mainline_legacy_backend_bridge_v1/source_models') / family
            paths = ({'source_models_file': str(base / 'source_models.pkl'),
                      'native_source_prior_file': str(base / 'native_source_prior.pkl')}
                     if kind == 'real' else source['source_permuted_assets_by_family'][family])
            for role, relative in paths.items():
                destination = RELEASE / 'assets'
                if kind == 'permuted':
                    destination /= 'permuted'
                destination = destination / family / Path(relative).name
                add(Path('SC-OLH-KG') / relative, destination)
                assets.append(dict(family=family, source_relationship=kind, role=role,
                                   file=str(destination), scientific_origin=relative))
        for name in ('archive.json', 'design.json'):
            add(Path('SC-OLH-KG/paper_artifacts/legacy_joint_bridge_v1/inputs') / family / name,
                RELEASE / 'assets' / family / name)

    # Only selected scientific fields leave the operational manifests. Runtime
    # scheduler accounts and unrelated historical result directories stay behind.
    selected = ('name', 'registered_utc', 'search_arms', 'terminal_arms', 'old_protocol',
                'terminal_protocols', 'geometry_seed', 'tasks', 'frozen', 'randomness',
                'primary_candidate', 'primary_reference', 'primary_terminal', 'objective',
                'scope', 'geometry_description', 'statistical_unit', 'analysis', 'streams',
                'budget', 'selected_N', 'source_permuted_assets_by_family')
    scientific = {}
    for name in ('mainline_precision_confirmation_v1', 'mainline_source_alignment_targets_v1'):
        relative = Path(f'SC-OLH-KG/performance/manifests/{name}.json')
        add(relative)
        value = read(root / relative)
        for stream in value['streams']:
            add(Path('SC-OLH-KG') / stream['manifest'])
        scientific[name] = {key: value[key] for key in selected if key in value}
    source_control = read(root / 'SC-OLH-KG/performance/manifests/mainline_source_alignment_control_v1.json')
    for family in source_control['families']:
        add(Path('SC-OLH-KG') / family['source_search_manifest'])
    files[str(RELEASE / 'scientific_configuration.json')] = encoded(scientific)
    case_name = str(RELEASE / 'case/case.json')
    if case_name in files:
        transformations[case_name] = dict(original_sha256=digest(files[case_name]),
            change='Only the two source-asset paths are made relative to the portable assets directory.')
        case = read(files[case_name])
        for field in ('source_models_file', 'native_source_prior_file'):
            case[field] = f'../assets/{case["domain"]}/{Path(case[field]).name}'
        files[case_name] = encoded(case)

    # Reuse the same source inventory as the separate current-only Lean export.
    specification = importlib.util.spec_from_file_location('release_lean', root / RELEASE / 'lean_delivery.py')
    lean = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(lean)
    info = lean.inventory(root / 'proof')
    for relative in info['source_sha256']:
        add(Path('proof') / relative)
    files['proof/lakefile.lean'] = (
        'import Lake\nopen Lake DSL\n\npackage «scolhkg-proof» where\n'
        '  version := v!"0.1.0"\n\nrequire mathlib from git\n'
        '  "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0"\n\n'
        '@[default_target]\nlean_lib SCOLHKG where\n'
        '  roots := #[`SCOLHKG.CurrentMainlineProofInterface]\n').encode()
    files[str(RELEASE / 'lean_source_manifest.json')] = encoded(info)

    # Redact workstation identities in the review copies, not historical files.
    # This applies only to release metadata; scientific values and code are exact.
    for name, source_file in list(files.items()):
        if not (name.endswith(('.json', '.jsonl')) and name.startswith('SC-OLH-KG/')):
            continue
        payload = source_file if isinstance(source_file, bytes) else source_file.read_bytes()
        text = payload.decode()
        reviewed = re.sub(r'/home/[^/\s"\\]+', '{HOME}', text)
        if reviewed != text:
            transformations[name] = dict(original_sha256=digest(source_file),
                change='Home-directory user identifiers redacted in metadata copy only.')
            files[name] = reviewed.encode()

    entries = {name: dict(bytes=len(value) if isinstance(value, bytes) else value.stat().st_size,
                          sha256=digest(value)) for name, value in sorted(files.items())}
    worktree_status = git(root, 'status', '--porcelain')
    manifest = dict(format='current-mainline-review-bundle-v1',
        status='incomplete' if missing else 'assembled', missing=missing,
        reviewed_base_commit=BASE_COMMIT, release_worktree_commit=git(root, 'rev-parse', 'HEAD'),
        publication_commit=publication_commit,
        publication_commit_source='operator_supplied' if publication_commit else None,
        runtime=dict(python='3.10.14', requirements=str(RELEASE / 'requirements.txt')),
        release_worktree_has_uncommitted_changes=None if worktree_status is None else bool(worktree_status),
        actual_historical_scientific_source_commit=None,
        scientific_commit_status='Not recoverable from the retained run receipts; base commit identifies the reviewed delivery, not a rerun.',
        source_assets=assets, files=entries, current_lean_modules=16, current_lean_theorems=94,
        metadata_transformations=transformations,
        acceptance_receipts=accepted,
        delivery_ready=not missing and all(row['accepted'] for row in accepted.values()),
        proof_lakefile_adjustment='Default root restricted to CurrentMainlineProofInterface; theorem sources and locks unchanged.',
        receipt_scope='Included receipts are evidence of their recorded attempts; archive assembly does not certify scientific or author approvals.',
        licensing=dict(code='MIT', generated_data='CC-BY-4.0'),
        historical_failures_and_fees='Retained in compact cell/account records; no reserve reset.',
        archive_assembly_operation=dict(scientific_calls=0, model_restorations=0,
            includes_prior_acceptance_execution_receipts=True))
    return files, manifest


def verify(archive):
    """Check bundled payload digests without extraction or scientific execution."""
    with tarfile.open(archive, 'r:gz') as stream:
        manifest = json.load(stream.extractfile('release_manifest.json'))
        for name, entry in manifest['files'].items():
            value = hashlib.sha256()
            source = stream.extractfile(name)
            count = 0
            for block in iter(lambda: source.read(1024 * 1024), b''):
                value.update(block)
                count += len(block)
            if value.hexdigest() != entry['sha256'] or count != entry['bytes']:
                raise ValueError(f'Bundle payload differs from its manifest: {name}')
    return dict(status='verified' if not manifest['missing'] else 'verified_incomplete',
                files=len(manifest['files']), missing=manifest['missing'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory', 'build', 'verify'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output', type=Path, help='New tar.gz file for build; existing file for verify')
    parser.add_argument('--release-commit',
                        help='Operator-supplied publication commit; not the historical scientific execution commit')
    args = parser.parse_args()
    if args.command == 'verify':
        print(json.dumps(verify(args.output), indent=2))
        return
    files, manifest = assemble(args.root.resolve(), publication_commit=args.release_commit)
    if args.command == 'inventory':
        print(json.dumps(manifest, indent=2))
        return
    if args.output is None:
        parser.error('build requires --output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as destination, tarfile.open(fileobj=destination, mode='w:gz') as archive:
        for name, value in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(value) if isinstance(value, bytes) else value.stat().st_size
            with io.BytesIO(value) if isinstance(value, bytes) else value.open('rb') as source:
                archive.addfile(entry, source)
        payload = encoded(manifest)
        entry = tarfile.TarInfo('release_manifest.json')
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    print(json.dumps(dict(status=manifest['status'], files=len(files),
                          missing=manifest['missing'], archive=str(args.output)), indent=2))
    if manifest['missing']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
