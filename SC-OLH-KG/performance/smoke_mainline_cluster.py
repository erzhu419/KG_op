"""Zero-simulation environment check for the frozen mainline's cluster deployment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def probe(item):
    started = time.thread_time()
    for module in ('performance.run_mainline_two_comparison_confirmation',
                   'performance.mainline_cluster_dispatch'):
        importlib.import_module(module)
    import numpy
    import scipy
    inventory = json.loads((ROOT / 'reproducibility/mainline-cluster-inputs.json').read_text())
    paths = [ROOT.parent / path for path in inventory['files'] if path.endswith('.pkl')]
    assert len(paths) == 6
    loaded = []
    for path in paths:
        with path.open('rb') as stream:
            value = pickle.load(stream)
        loaded.append(dict(path=str(path.relative_to(ROOT.parent)),
                           type=type(value).__name__, bytes=path.stat().st_size))
        del value
    return dict(status='passed', item=item, node=socket.gethostname().split('.')[0],
        python=sys.version, executable=sys.executable, numpy=numpy.__version__,
        scipy=scipy.__version__, source_pickles=loaded,
        thread_environment={name: os.environ.get(name) for name in
                            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        new_simulation_calls=0, fits=0, posterior_restorations=0,
        worker_thread_cpu_seconds=time.thread_time() - started,
        finished_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    if args.probe_only:
        print(json.dumps(probe(-1)))
        return
    assert args.out_dir is not None and 0 <= args.start < args.end <= 6
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Each item is an independent environment/readability check, never a search job.
    with ThreadPoolExecutor(max_workers=min(args.workers, args.end - args.start)) as pool:
        for result in pool.map(probe, range(args.start, args.end)):
            (args.out_dir / f"item_{result['item']}.json").write_text(json.dumps(result, indent=2) + '\n')
    print(f'DONE zero-simulation smoke [{args.start},{args.end})', flush=True)


if __name__ == '__main__':
    main()
