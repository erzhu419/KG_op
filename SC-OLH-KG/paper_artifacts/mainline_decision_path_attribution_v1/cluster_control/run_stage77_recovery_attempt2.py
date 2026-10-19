"""Check the JSON boundary repair and retain failed attribution files before rereading."""
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


ROOT = Path.cwd()
OUT = ROOT / 'paper_artifacts/mainline_decision_path_attribution_v1'
CHECKS = OUT / 'cluster_control/validation/attempt2/checks.json'
TEST = ('tests/test_mainline_decision_path_attribution.py::'
        'test_reader_keeps_all_registered_controls_and_uses_corrected_recording')


def child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def main():
    CHECKS.parent.mkdir(parents=True, exist_ok=True)
    assert not CHECKS.exists(), 'Retain the existing correction check'
    command = [sys.executable, '-m', 'pytest', '-p', 'no:capture', '-q', TEST]
    cpu, wall = child_cpu(), time.monotonic()
    checked = subprocess.run(command, cwd=ROOT)
    report = dict(status='complete' if checked.returncode == 0 else 'failed',
        command=command, expected_instances=1, returncode=checked.returncode,
        cpu_seconds=child_cpu()-cpu, wall_seconds=time.monotonic()-wall,
        checked_utc=datetime.now(timezone.utc).isoformat(), new_scientific_calls=0,
        new_truth_calls=0, new_fits=0, new_posterior_restorations=0)
    with CHECKS.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report), flush=True)
    if checked.returncode:
        raise SystemExit(checked.returncode)
    cpu = time.process_time()
    plan = json.loads((ROOT/'performance/manifests/mainline_decision_path_attribution_v1.json').read_text())
    cases = [OUT/'cases'/f"seed_{task['seed']}"/stream['name']/'result.json'
             for task in plan['tasks'] for stream in plan['streams']]
    shards = list((OUT/'dispatch').glob('shard_*/result.json'))
    assert len(cases) == 48 and len(shards) == 4
    for path in cases:
        value = json.loads(path.read_text())
        assert value['status'] == 'failed'
        assert 'Object of type PosixPath is not JSON serializable' in value['error']
    for path in shards:
        value = json.loads(path.read_text())
        assert value['outcomes'] and all(row['status'] == 'failed' for row in value['outcomes'])
    moves = [(path, path.with_name('failed_result_attempt1.json')) for path in [*cases, *shards]]
    assert not any(destination.exists() for _, destination in moves)
    for source, destination in moves:
        source.rename(destination)
    archive = dict(status='complete', case_files=len(cases), shard_files=len(shards),
        files=[str(destination.relative_to(ROOT)) for _, destination in moves],
        cpu_seconds=time.process_time()-cpu, original_attempt_ledgers_retained=True,
        new_scientific_calls=0, new_truth_calls=0, new_fits=0, new_posterior_restorations=0)
    (CHECKS.parent/'archive.json').write_text(json.dumps(archive, indent=2)+'\n')
    report['archive_cpu_seconds'] = archive['cpu_seconds']
    report['archive_file'] = str((CHECKS.parent/'archive.json').relative_to(ROOT))
    CHECKS.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(archive), flush=True)


if __name__ == '__main__':
    main()
