"""Meter the single identity regression fixture and saved-field supplement."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


OUTPUT = Path('paper_artifacts/mainline_observed_terminal_attribution_v1')


def cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def run(label, command):
    before, wall = cpu(), time.monotonic()
    result = subprocess.run(command, check=False)
    record = dict(label=label, command=command, returncode=result.returncode,
                  status='complete' if result.returncode == 0 else 'failed',
                  cpu_seconds=cpu()-before, wall_seconds=time.monotonic()-wall)
    with (OUTPUT/'cluster_control'/f'{label}.json').open('x') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    print(json.dumps(record), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


run('identity_pytest', [sys.executable, '-m', 'pytest', '-p', 'no:capture', '-q',
    'tests/test_mainline_observed_shortlist_attribution.py::test_incumbent_overwritten_contract_retains_challenger_identity_from_role'])
run('identity_read', [sys.executable,
    str(OUTPUT/'cluster_control/stage74_challenger_identity_supplement.py'),
    str(OUTPUT), str(OUTPUT/'metadata_identity_supplement.json')])
