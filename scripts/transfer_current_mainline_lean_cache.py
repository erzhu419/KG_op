#!/usr/bin/env python3
"""Stream the official cluster-generated cache plan through a TLS-valid client.

Package I/O only: no local Lean execution or local cache payload directory.
Use after the scheduler cache-plan phase; then run its proof-check phase.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import datetime
import io
import json
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import tarfile
import threading
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = '/home/zhengliang01/scheduleurm_work/KG_op_mainline_lean_20261020'
TRANSPORT = ROOT.parent / 'scheduleurm/skill/transport_exec.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    urls = list(dict.fromkeys(line[6:].strip() for line in args.config.read_text().splitlines()
                             if line.startswith('url = ')))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    record_path = args.config.parent / f'cache_transfer_{stamp}.json'
    log_path = record_path.with_suffix('.log')
    record = {'config': str(args.config), 'files': len(urls), 'status': 'running',
              'download_bytes': 0, 'streamed_files': 0, 'automatic_retries': 0,
              'local_lean_executed': False, 'payloads_saved_locally': False,
              'server_cache': REMOTE_ROOT + '/cache', 'log': str(log_path)}
    started, cpu = time.monotonic(), time.process_time()
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    context = threading.local()

    def fetch(url):
        if not hasattr(context, 'session'):
            context.session = requests.Session()
        response = context.session.get(url, timeout=(20, 60),
                                       verify='/etc/ssl/certs/ca-certificates.crt')
        response.raise_for_status()
        return url.rsplit('/', 1)[1], response.content

    command = [sys.executable, str(TRANSPORT), '--bulk', '--', 'ssh', '-T',
               '-o', 'ConnectTimeout=30', '-o', 'BatchMode=yes', '-J', 'jtl110gpu',
               'zhengliang01@202.197.46.16',
               shlex.join(['tar', '-xf', '-', '-C', REMOTE_ROOT + '/cache'])]
    process = None
    try:
        with log_path.open('w') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                       stdout=log, stderr=subprocess.STDOUT)
            # Four outstanding payloads bound memory while SSH applies backpressure.
            with tarfile.open(fileobj=process.stdin, mode='w|') as archive:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    remaining = iter(urls)
                    pending = {pool.submit(fetch, u) for u in [next(remaining, None)
                               for _ in range(4)] if u is not None}
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            name, payload = future.result()
                            entry = tarfile.TarInfo(name)
                            entry.size = len(payload)
                            archive.addfile(entry, io.BytesIO(payload))
                            record['download_bytes'] += len(payload)
                            record['streamed_files'] += 1
                            if record['streamed_files'] % 100 == 0:
                                print(json.dumps({k: record[k] for k in
                                      ('files', 'streamed_files', 'download_bytes')}), flush=True)
                            url = next(remaining, None)
                            if url is not None:
                                pending.add(pool.submit(fetch, url))
            process.stdin.close()
            record['receiver_returncode'] = process.wait()
            if process.returncode:
                raise RuntimeError(f'cache transfer receiver returned {process.returncode}')
        record['status'] = 'passed'
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = str(error)
        if process is not None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            record['receiver_returncode'] = process.wait()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update(wall_seconds=time.monotonic() - started,
                  client_cpu_seconds=time.process_time() - cpu,
                  client_children_cpu_seconds=usage.ru_utime + usage.ru_stime
                  - children.ru_utime - children.ru_stime)
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'receipt': str(record_path), **record}), flush=True)
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
