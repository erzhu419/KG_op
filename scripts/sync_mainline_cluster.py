#!/usr/bin/env python3
"""Stage current mainline code and the listed frozen inputs, then upload once."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT.parent / 'KG_op_mainline_cluster'
REMOTE_ROOT = '/home/zhengliang01/scheduleurm_work/KG_op_mainline_cluster'
TRANSPORT = ROOT.parent / 'scheduleurm/skill/transport_exec.py'
INPUTS = ROOT / 'SC-OLH-KG/reproducibility/mainline-cluster-inputs.json'


def files_to_stage(inputs, extra_manifests):
    files = set(inputs['files'])
    for directory in inputs['python_directories']:
        for path in (ROOT / directory).rglob('*.py'):
            if '__pycache__' not in path.parts:
                files.add(str(path.relative_to(ROOT)))
    files.add(str(INPUTS.relative_to(ROOT)))
    files.add('scripts/sync_mainline_cluster.py')
    for path in extra_manifests:
        files.add(str(path.resolve().relative_to(ROOT)))
    for relative in files:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', default=[],
                        help='Additional registered manifest to upload unchanged')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    files = files_to_stage(json.loads(INPUTS.read_text()), args.manifest)
    size = sum((ROOT / path).stat().st_size for path in files)
    print(json.dumps({'files': len(files), 'bytes': size, 'local_stage': str(STAGE),
                      'remote_root': REMOTE_ROOT, 'download_bytes': 0}), flush=True)
    if args.dry_run:
        return
    STAGE.mkdir(parents=True, exist_ok=True)
    listing = '\n'.join(files) + '\n'
    subprocess.run(['rsync', '-a', '--files-from=-', f'{ROOT}/', f'{STAGE}/'],
                   input=listing, text=True, check=True)
    transport = [sys.executable, str(TRANSPORT)]
    ssh = ['ssh', '-o', 'ConnectTimeout=30', '-o', 'BatchMode=yes', '-J', 'jtl110gpu']
    remote = 'zhengliang01@202.197.46.16'
    subprocess.run(transport + ['--'] + ssh[:1] + ['-n'] + ssh[1:] + [remote,
        shlex.join(['mkdir', '-p', f'{REMOTE_ROOT}/SC-OLH-KG'])], check=True)
    subprocess.run(transport + ['--bulk', '--', 'rsync', '-a', '--files-from=-',
        '-e', shlex.join(ssh), f'{STAGE}/', f'{remote}:{REMOTE_ROOT}/'],
        input=listing, text=True, check=True)
    print('Mainline upload complete; remote results were not downloaded.', flush=True)


if __name__ == '__main__':
    main()
