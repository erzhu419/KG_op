"""Read paid fixed355 statistics on the original F16/risk224 nodes.

This is a diagnostic reader, not an integration worker. Its CPU is support
cost and needs an available, authorized support budget before execution.
"""
import time

started = time.process_time()

import gzip
import json
import signal
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def stop(*args):
    raise TimeoutError('Cached diagonal reader reached its requested3CPU timer')


def main():
    result = dict(complete=False, new_integrations=0, stored_mix_recomputed=False)
    signal.signal(signal.SIGPROF, stop)
    signal.setitimer(signal.ITIMER_PROF, 3.0)
    try:
        source = 'performance/manifests/historical_positive_risk_uppervariance_axis_cached_pair_references_screen_v2.json'
        spec = read(source)
        bank = read(spec['bank_plan'])
        base = bank['output']
        # Origins are the first field. Skip the trailing local-moment arrays.
        marker = b',"actions":'
        prefix = b''
        with gzip.open(base + '/context.json.gz', 'rb') as stream:
            while marker not in prefix:
                chunk = stream.read(131072)
                if not chunk:
                    raise ValueError('Origin-prefix terminator absent')
                prefix += chunk
        origin_bytes = prefix.split(marker, 1)[0]
        assert origin_bytes.startswith(b'{"origins":')
        origins = json.loads(origin_bytes + b'}')['origins']
        receipt = read('paper_artifacts/historical_positive_risk_uppervariance_axis_prior_f16_source_v1/paid_prefix_controls.json')
        prior_weights = receipt['prior_weights']
        assert sorted(origins) == ['4', '5']
        shard = next(i for i, policies in enumerate(spec['policy_shards']) if 63 in policies)
        diagonal = 0.0
        rows = []
        for origin, data in origins.items():
            o = int(origin)
            assert len(data['objective']['raw']) == 16
            task = read(base + f'/tasks/{o}_{shard}.json')
            assert task['complete']
            fixed = {row['record']: row['fixed']['raw'] for row in task['rows'] if 'fixed' in row}
            f_mean = sum(row[o][355][0] for row in data['objective']['raw']) / 16
            risk_sum = sum(row['outer_weight'] * fixed[row['record']] for row in data['risk'])
            contribution = prior_weights[o] * f_mean * risk_sum
            diagonal += contribution
            rows.append(dict(origin=o, objective_count=16, risk_count=len(data['risk']),
                             objective_mean=f_mean, risk_weighted_sum=risk_sum,
                             risk_weight_sum=sum(row['outer_weight'] for row in data['risk']),
                             contribution=contribution))
        # This accepted point was already settled from the same frozen bank.
        # It is deliberately reused rather than evaluated again by this reader.
        mixed = 0.7452258287528934
        baseline = receipt['debited_prior_point']
        result.update(complete=True, source_bound_plan=source, source_bank_plan=spec['bank_plan'],
                      fixed_action=355, stored_mix=mixed, origin_diagonal=diagonal,
                      mixture_minus_diagonal=mixed - diagonal,
                      diagonal_minus_prior=diagonal - baseline,
                      stored_mix_minus_prior=mixed - baseline, origin_rows=rows,
                      prior_origin_mass_not_in_nodes=1 - prior_weights[4] - prior_weights[5],
                      weights_renormalized=False, density_ratio_added=False,
                      scope='Same-node fixed-action diagonal only. The difference includes cross-origin mixing and omitted-utility effects;it is not an absolute error bound or a bank replacement.')
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        result['reader_process_cpu_seconds'] = time.process_time() - started
        print(json.dumps(result))


if __name__ == '__main__':
    main()
