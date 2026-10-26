"""Compare an already recomputed release table with saved pre-release summaries.

This module does not aggregate cells or estimate effects. Frozen summaries are
validation references only, never inputs to ``statistics.recompute``.
"""
from __future__ import annotations

import argparse
from math import isclose
from pathlib import Path
import json
import time

from .statistics import METRICS, read_json, write_json

REFERENCES = dict(
    stage85='paper_artifacts/mainline_precision_confirmation_v1/headline.json',
    stage93='paper_artifacts/mainline_source_alignment_control_v1/cluster_control/stage93_results_compact_20261019.json',
    stage103='paper_artifacts/mainline_frozen_backend_comparison_v1/headline.json')
ALIASES = dict(real_AFTER_minus_POINT='AFTER_minus_POINT', real_FULL_minus_AFTER='FULL_minus_AFTER',
    AFTER_minus_POINT_reference='AFTER_minus_POINT')


def _normalized(value):
    if 'defined' in value:
        return dict(available=value['defined'], mean=value['mean'], conditional_mean=value['available_mean'])
    if 'conditional_available' in value:
        known = value['conditional_available']
        return dict(registered=value['registered_geometries'], available=value['available_geometries'],
            mean=value['all_registered_mean'], conditional_mean=known['mean'],
            interval=known['descriptive_95pct_t_interval'])
    return dict(registered=value['registered'], available=value['n'], mean=value['all_registered_mean'],
        conditional_mean=value['conditional_mean'], interval=value['conditional_ci95'])


def _equal(left, right):
    if left is None or right is None:
        return left is right
    if isinstance(left, dict):
        return set(left) == set(right) and all(_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return isclose(left, right, rel_tol=0., abs_tol=1e-12)


def verify(result, references):
    current = {cohort['name']: cohort for cohort in result['cohorts']}
    comparisons, checks = [], []

    def check(current_stat, saved_stat, label):
        actual, expected = _normalized(current_stat), _normalized(saved_stat)
        if not all(key in actual and _equal(actual[key], value) for key, value in expected.items()):
            raise ValueError(f'Frozen table parity failed: {label}: {actual} != {expected}')
        checks.append(label)

    def compare_unit(cohort, unit_name, reference_unit, origin, *, backend=False):
        unit = next(row for row in current[cohort]['units'] if row['name'] == unit_name)
        controls = {row['arm']: row for row in unit['controls']}
        effects = {row['name']: row for row in unit['comparisons']}
        for control in reference_unit.get('controls', []):
            for metric in control['metrics']:
                check(controls[control['arm']]['metrics'][metric], control['metrics'][metric],
                    f"{origin}/{cohort}/{unit_name}/{control['arm']}/{metric}")
        for saved in reference_unit['comparisons']:
            name = saved.get('name')
            if name is None:
                name = ('AFTER_minus_POINT' if saved['left'] == 'legacy_role_after_init_point_only'
                    else 'FULL_minus_AFTER')
            name = ALIASES.get(name, name)
            effect = effects[name]
            for metric in saved['delta']:
                check(effect['delta'][metric], saved['delta'][metric], f'{origin}/{cohort}/{unit_name}/{name}/{metric}')
            if unit_name == 'independent':
                comparisons.append(dict(reference=origin, cohort=cohort, comparison=name,
                    endpoints={metric: _normalized(effect['delta'][metric]) for metric in ('C', 'G')}))
            if backend:
                for stream in saved['per_stream']:
                    stream_effect = next(row for group in current[cohort]['units'] if group['name'] == stream['name']
                        for row in group['comparisons'] if row['name'] == name)
                    for metric in METRICS:
                        check(stream_effect['delta'][metric], stream['delta'][metric],
                            f"{origin}/{cohort}/{stream['name']}/{name}/{metric}")

    for cohort in ('stage85', 'stage93'):
        saved = references[cohort]
        if current[cohort]['registered_cells'] != saved['registered_terminal_cells']:
            raise ValueError('Frozen registered cell denominator changed')
        compare_unit(cohort, 'independent', saved['independent'], cohort)
        for unit in saved.get('per_stream', saved.get('streams', [])):
            compare_unit(cohort, unit['name'], unit, cohort)
    for cohort in references['stage103']['cohorts']:
        if current[cohort['name']]['terminal_counts'] != cohort['sealed_terminal_counts']:
            raise ValueError('Frozen terminal complete/failure states changed')
        compare_unit(cohort['name'], 'independent', cohort, 'stage103', backend=True)
    return dict(status='passed', scalar_table_checks=len(checks), references=REFERENCES,
        absolute_tolerance=1e-12, comparisons=comparisons,
        note='Only already computed summaries are compared. No cell aggregation, model restoration, simulation or inferential rerun.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--statistics', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    started, wall = time.process_time(), time.monotonic()
    attempt = dict(action='compare_saved_release_statistics', status='failed')
    try:
        report = verify(read_json(args.statistics), {name: read_json(root / path) for name, path in REFERENCES.items()})
        write_json(args.output, report)
        attempt['status'] = 'passed'
    except Exception as error:
        attempt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        attempt.update(cpu_seconds=time.process_time() - started, wall_seconds=time.monotonic() - wall)
        ledger = Path(args.output).parent / 'verification_attempts.jsonl'
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open('a') as stream:
            stream.write(json.dumps(attempt) + '\n')
    print(dict(status=report['status'], scalar_table_checks=report['scalar_table_checks']))


if __name__ == '__main__':
    main()
