"""Correct the saved safe-atlas audit; retain all original experiments and accounts."""
import json
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.run_mainline_safe_unsampled_audit import ZERO
from performance.mainline_terminal_recording_audit import audit_stream, compact_audit
from performance.summarize_mainline_terminal_recording_replication import summarize


MANIFEST = ROOT/'performance/manifests/mainline_terminal_recording_seed_replication_v1.json'
TEST = ('tests/test_mainline_terminal_recording_audit.py::'
        'test_initial_safe_pool_uses_registered_atlas_without_relaxing_observed_pools')


def main():
    assert socket.gethostname().split('.')[0] in {f'node{i:03d}' for i in range(1, 7)}
    plan = read(MANIFEST)
    output = ROOT/plan['output_dir']
    assert plan['status'] == 'complete'
    checks = output/'cluster_control/validation/audit_revision_1/checks.json'
    assert not checks.exists(), 'Keep the existing correction attempt and inspect before retry'
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    before_cpu, started = usage.ru_utime+usage.ru_stime, time.monotonic()
    result = subprocess.run([sys.executable, '-m', 'pytest', '-p', 'no:capture', '-q', TEST], cwd=ROOT)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    check = dict(status='complete' if result.returncode == 0 else 'failed', returncode=result.returncode,
        cpu_seconds=usage.ru_utime+usage.ru_stime-before_cpu, wall_seconds=time.monotonic()-started,
        test=TEST, **ZERO)
    save(checks, check)
    if result.returncode:
        raise SystemExit(result.returncode)
    saved_summary = read(output/'summary.json')
    original = read(output/'terminal_recording/summary.json')
    save(output/'terminal_recording/summary_before_revision_1.json', original)
    save(output/'terminal_recording/completion_before_revision_1.json', read(output/'completion.json'))
    corrected = []
    for stream in plan['streams']:
        child = read(ROOT/stream['manifest'])
        folder = output/'terminal_recording'/stream['name']/'revision_1'
        with attempt(folder, 'correct-saved-safe-atlas-audit') as cost:
            cost.update(ZERO)
            detail = audit_stream(ROOT, child)
            compact = compact_audit(detail)
            save(folder/'details.json', detail)
            save(folder/'summary.json', compact)
        corrected.append(dict(stream=stream['name'], **compact,
            detailed_file=str((folder/'details.json').relative_to(ROOT))))
    save(output/'terminal_recording/summary.json', dict(streams=corrected,
        correction='Safe fallback pool follows its saved initial-atlas or observed-history universe.',
        previous_summary='terminal_recording/summary_before_revision_1.json'))
    refreshed = summarize(plan)
    # Only diagnostic validity and newly paid audit CPU may change. These are
    # the actual saved scientific metrics, not a recomputed counterfactual.
    for key in ('geometry_averaged', 'streams', 'outcomes', 'failures', 'false_certificates', 'mechanism'):
        assert saved_summary[key] == refreshed[key], 'Scientific result changed during read-only correction: '+key
    record = dict(status='complete', original_issue_pairs=[s['coverage']['issue_pairs'] for s in original['streams']],
        corrected_coverage=[s['coverage'] for s in corrected], scientific_results_unchanged=True,
        checks_file=str(checks.relative_to(ROOT)), **ZERO)
    save(output/'terminal_recording/revision_1.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
