"""Run the 18 new substitutes, then admit only the frozen execution metadata."""
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


ROOT = Path.cwd()
OUT = ROOT / 'paper_artifacts/mainline_terminal_recording_seed_replication_v1'
MANIFEST = ROOT / 'performance/manifests/mainline_terminal_recording_seed_replication_v1.json'
CHECKS = OUT / 'cluster_control/validation/attempt1/checks.json'
TESTS = ['tests/test_mainline_terminal_recording_cluster_control.py',
         'tests/test_mainline_terminal_recording_replication.py',
         'tests/test_mainline_terminal_recording_audit.py',
         'tests/test_mainline_terminal_recording_replication_summary.py']


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def main():
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    assert not CHECKS.exists(), 'Preserve the existing validation attempt'
    command = [sys.executable, '-m', 'pytest', '-p', 'no:capture', '-q', *TESTS]
    cpu, wall = child_cpu(), time.monotonic()
    result = subprocess.run(command, cwd=ROOT)
    report = dict(status='complete' if result.returncode == 0 else 'failed',
                  command=command, expected_instances=18, returncode=result.returncode,
                  cpu_seconds=child_cpu() - cpu, wall_seconds=time.monotonic() - wall,
                  checked_utc=datetime.now(timezone.utc).isoformat())
    with CHECKS.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    start = time.process_time()
    outer = json.loads(MANIFEST.read_text())
    paths = [MANIFEST, *(ROOT / stream['manifest'] for stream in outer['streams'])]
    for path in paths:
        plan = json.loads(path.read_text())
        plan['execution_preparation'].update(status='validated', execution_started=False,
            validation_file=str(CHECKS.relative_to(ROOT)), remaining=[],
            controller='../scripts/run_mainline_terminal_recording_cluster.py',
            entrypoint='performance.run_mainline_terminal_recording_replication')
        if path != MANIFEST:
            plan['registration_account'] = (
                'paper_artifacts/mainline_terminal_recording_preparation_v1/scheduler_execution.json')
        path.write_text(json.dumps(plan, indent=2) + '\n')
    admission = dict(status='validated', execution_started=False,
        manifests=[str(path.relative_to(ROOT)) for path in paths],
        cpu_seconds=time.process_time() - start, new_scientific_calls=0,
        new_fits=0, new_posterior_restorations=0, new_truth_calls=0)
    with (CHECKS.parent / 'admission.json').open('x') as output:
        json.dump(admission, output, indent=2)
        output.write('\n')
    print(json.dumps(admission), flush=True)


if __name__ == '__main__':
    main()
