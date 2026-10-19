"""Scheduler-only decomposition of sealed source-neighbor mean differences."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
import traceback

from performance.analyze_mainline_three_stream_attribution import ZERO, read, save

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'performance/manifests/mainline_source_neighbor_propagation_v1.json'


def main():
    start, wall = time.process_time(), time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=DEFAULT)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    node = socket.gethostname().split('.')[0]
    if node not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Use scheduler on node001-node006 for saved scientific inputs')
    if (args.start, args.end, args.workers) != (0, 1, 1):
        raise ValueError('This small saved-pair analysis is one CPU item')
    plan = read(ROOT / args.manifest)
    output = ROOT / plan['output_dir']
    if (output / 'completion.json').is_file() and read(output / 'completion.json')['status'] == 'complete':
        print(json.dumps(dict(status='reused')), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    status, error, coverage = 'failed', None, None
    print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=0, total=1, unit='saved_source_pairs')), flush=True)
    try:
        from performance.mainline_source_neighbor_propagation import analyze_propagation, compact_propagation
        raw = analyze_propagation(ROOT, plan)
        if len(raw['cases']) != plan['expected']['source_groups']:
            raise ValueError('Decomposition omitted a registered source group')
        if sum(len(case['pairs']) for case in raw['cases']) != plan['expected']['fixed_pairs']:
            raise ValueError('Decomposition changed the fixed pair population')
        save(output / 'analysis.json', raw)
        summary = compact_propagation(raw)
        save(output / 'summary.json', summary)
        status, coverage = raw['status'], summary['coverage']
    except Exception:
        error = traceback.format_exc()
    account = dict(status=status, error=error, node=node, utc=datetime.now(timezone.utc).isoformat(),
        cpu_seconds=time.process_time()-start, wall_seconds=time.monotonic()-wall, **ZERO)
    with (output / 'attempts.jsonl').open('a') as handle:
        handle.write(json.dumps(account) + '\n')
    attempts = [json.loads(line) for line in (output / 'attempts.jsonl').read_text().splitlines()]
    validation = [dict(file=str(p.relative_to(ROOT)), **read(p)) for p in (output / 'validation').glob('*.json')]
    result = dict(status=status, coverage=coverage, attempts=attempts,
        failed_attempts=[a for a in attempts if a['status'] != 'complete'],
        analysis_cpu_seconds=sum(a['cpu_seconds'] for a in attempts),
        validation=validation, validation_cpu_seconds=sum(v.get('cpu_seconds') or 0 for v in validation),
        registration_cpu_seconds=read(output / 'registration.json')['cpu_seconds'],
        historical_accounts=plan['historical_accounts'], all_control_evidence=plan['all_control_evidence'],
        new_work=dict(ZERO), accounting_scope='New saved-difference analysis and fixtures only; all earlier costs remain linked.')
    result['total_new_recorded_cpu_seconds'] = sum(result[k] for k in (
        'analysis_cpu_seconds', 'validation_cpu_seconds', 'registration_cpu_seconds'))
    save(output / 'completion.json', result)
    if status == 'complete':
        plan.update(status='complete', completed_utc=account['utc'])
        save(ROOT / args.manifest, plan)
    print(json.dumps(dict(status=status, error=error, coverage=coverage,
                         total_new_recorded_cpu_seconds=result['total_new_recorded_cpu_seconds'])), flush=True)
    if status != 'complete':
        raise SystemExit(1)
    print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=1, total=1, unit='saved_source_pairs')), flush=True)


if __name__ == '__main__':
    main()
