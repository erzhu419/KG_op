"""Two scheduler stages: seal public neighbors, then score saved source forecasts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
import traceback

from performance.analyze_mainline_three_stream_attribution import ZERO, read, save

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'performance/manifests/mainline_source_neighbor_coverage_v1.json'


def execute(plan, phase):
    output = ROOT / plan['output_dir']
    if phase == 'geometry':
        from performance.mainline_source_neighbor_geometry import build_fixed_pairs
        fixed = build_fixed_pairs(ROOT, plan)
        if len(fixed['groups']) != plan['expected']['source_groups']:
            raise ValueError('Public pairing omitted a registered source group')
        save(ROOT / plan['inputs']['fixed_pairs'], fixed)
        compact = {k: v for k, v in fixed.items() if k != 'groups'}
        compact['groups'] = [dict({k: v for k, v in group.items() if k not in ('pairs', 'nodes')},
                                 pair_count=len(group['pairs'])) for group in fixed['groups']]
        save(output / 'geometry_summary.json', compact)
        return dict(status='complete', source_groups=len(fixed['groups']),
                    fixed_pairs=sum(len(group['pairs']) for group in fixed['groups']))
    if read(output / 'geometry_completion.json')['status'] != 'complete':
        raise ValueError('All public pairs must be sealed before source scoring')
    from performance.mainline_source_neighbor_coverage import (
        analyze_source_neighbors, compact_source_neighbors,
    )
    raw = analyze_source_neighbors(ROOT, plan)
    compact = compact_source_neighbors(raw)
    save(output / 'analysis.json', raw)
    save(output / 'summary.json', compact)
    return dict(status=raw['status'], coverage=compact['coverage'])


def main():
    start, wall = time.process_time(), time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=DEFAULT)
    parser.add_argument('--phase', choices=('geometry', 'score'), required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    node = socket.gethostname().split('.')[0]
    if node not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Saved scientific inputs run through scheduler on node001-node006')
    if (args.start, args.end, args.workers) != (0, 1, 1):
        raise ValueError('Each small source stage is one sequential CPU item')
    plan = read(ROOT / args.manifest)
    output = ROOT / plan['output_dir']
    completion_file = output / (args.phase + '_completion.json')
    if completion_file.is_file() and read(completion_file)['status'] == 'complete':
        print(json.dumps(dict(phase=args.phase, status='reused')), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    result = dict(status='failed')
    error = None
    print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=0, total=1, unit=args.phase)), flush=True)
    try:
        result = execute(plan, args.phase)
    except Exception:
        error = traceback.format_exc()
    account = dict(phase=args.phase, **result, error=error, node=node,
        utc=datetime.now(timezone.utc).isoformat(), cpu_seconds=time.process_time()-start,
        wall_seconds=time.monotonic()-wall, **ZERO)
    with (output / 'attempts.jsonl').open('a') as handle:
        handle.write(json.dumps(account) + '\n')
    save(completion_file, account)
    if args.phase == 'score':
        attempts = [json.loads(line) for line in (output / 'attempts.jsonl').read_text().splitlines()]
        validation = [dict(file=str(p.relative_to(ROOT)), **read(p))
                      for p in (output / 'validation').glob('*.json')]
        registration = read(output / 'registration.json')
        completion = dict(status=result['status'], coverage=result.get('coverage'),
            attempts=attempts, failed_attempts=[a for a in attempts if a['status'] != 'complete'],
            analysis_cpu_seconds=sum(a['cpu_seconds'] for a in attempts),
            registration_cpu_seconds=registration['cpu_seconds'], validation=validation,
            validation_cpu_seconds=sum(v.get('cpu_seconds') or 0 for v in validation),
            historical_accounts=plan['historical_accounts'], all_control_evidence=plan['all_control_evidence'],
            new_work=dict(ZERO), accounting_scope='New measured source analysis and fixtures only; retain historical accounts.')
        completion['total_new_recorded_cpu_seconds'] = sum(completion[k] for k in (
            'analysis_cpu_seconds', 'registration_cpu_seconds', 'validation_cpu_seconds'))
        save(output / 'completion.json', completion)
        if result['status'] == 'complete':
            plan.update(status='complete', completed_utc=account['utc'])
            save(ROOT / args.manifest, plan)
    print(json.dumps(account), flush=True)
    if result['status'] != 'complete':
        raise SystemExit(1)
    print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(current=1, total=1, unit=args.phase)), flush=True)


if __name__ == '__main__':
    main()
