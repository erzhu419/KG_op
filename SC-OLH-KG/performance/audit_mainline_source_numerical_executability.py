"""Read registered source diagnostics and retained failure logs, without models."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import socket
import time


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'performance/manifests/mainline_source_numerical_executability_audit_v1.json'
GUARD = 'Batched raw moment error exceeds frozen tolerance'
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
ZERO = dict(new_source_calls=0, new_target_initial_calls=0, new_target_search_calls=0,
            new_verification_calls=0, new_truth_function_calls=0, new_candidate_scores=0,
            new_source_fits=0, source_model_restorations=0, engine_restorations=0,
            posterior_restorations=0, raw_replays=0, new_rng_draws=0, new_integrations=0)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class SavedRecords:
    """One small, explicit JSON/log inventory; no model or raw-data readers."""
    def __init__(self, root):
        self.root, self.reads, self.issues = Path(root), [], []

    def read(self, name, kind='json'):
        path = self.root / name
        suffix = dict(json='.json', jsonl='.jsonl', log='.log')[kind]
        if path.suffix != suffix:
            raise ValueError(f'Expected saved {kind}, not {path}')
        self.reads.append(str(name))
        try:
            text = path.read_text()
            if kind == 'log':
                return text
            if kind == 'jsonl':
                return [json.loads(line) for line in text.splitlines() if line.strip()]
            return json.loads(text)
        except (OSError, ValueError) as error:
            self.issues.append(dict(file=str(name), error=str(error)))
            return None


def unwrapped(value):
    return value.get('result', value) if isinstance(value, dict) else None


def parse_log(text):
    """Extract only printed quantities; stack context is distinct from inference."""
    text = text or ''
    if '_decision_backend_terminal_recommendation' in text or '_terminal_bayes_risk_components' in text:
        phase = 'terminal_recommendation'
    elif '_initial_conditioning' in text:
        phase = 'initial_conditioning'
    elif 'after_init_point_only_moment_state' in text and 'model.update_raw' in text:
        phase = 'post_initial_own_update'
    else:
        phase = 'unknown'
    ratios = [float(value) for value in re.findall(re.escape(GUARD) + r':\s*(' + NUMBER + ')', text)]
    unique_ratios = list(dict.fromkeys(ratios))
    recorded = dict(reported_error_ratio=unique_ratios[-1] if unique_ratios else None,
                    head=None, identity_or_component_range=None, internal_raw_position=None,
                    integration_absolute_error=None, minimum_normalizer=None,
                    integration_status=None, integrand_evaluations=None)
    # The integration-failed exception prints these values; ordinary Python
    # traceback source lines containing placeholders do not count as values.
    for key, printed in [('integration_absolute_error', 'error'),
                         ('minimum_normalizer', 'minimum_normalizer'),
                         ('integration_status', 'status'),
                         ('integrand_evaluations', 'integrand_evaluations')]:
        matches = re.findall(r'(?<![\w])' + printed + r'\s*=\s*(' + NUMBER + ')', text)
        if matches:
            recorded[key] = float(matches[-1])
    return dict(traceback_phase=phase, recorded=recorded,
                unknown_fields=[key for key, value in recorded.items() if value is None],
                guard_message_occurrences=len(ratios), distinct_reported_ratios=unique_ratios,
                code_inference=(dict(integral_finite=True, component_normalizers_positive=True,
                    integrator_success=True, basis='Frozen vectorized_joint_function_risk.py:58-74 control flow; not recorded status')
                    if unique_ratios else None),
                scope='Repeated exception text is not an attempt count. No missing local quantities reconstructed.')


def source_rows(plan, reader, seal):
    models = {(family['family'], model['kind']): model
              for family in (seal or {}).get('families', []) for model in family['models']}
    rows = []
    for family in plan['source_diagnostics']:
        for association in ('real', 'permuted'):
            for kind in ('structural', 'native'):
                name = family[association][kind]
                diagnostic = reader.read(name)
                model = models.get((family['family'], kind), {}) if association == 'permuted' else {}
                receipt_name = model.get('receipt_file')
                receipt = reader.read(receipt_name) if receipt_name else None
                row = dict(family=family['family'], association=association, kind=kind,
                           file=name, read_status='complete' if isinstance(diagnostic, dict) else 'unavailable',
                           receipt_file=receipt_name, saved_receipt=receipt)
                if isinstance(diagnostic, dict):
                    if kind == 'structural':
                        keys = ('source_task_names', 'source_sites', 'reused_source_calls',
                                'new_source_simulator_calls', 'source_identity_heads',
                                'source_identities', 'source_identity_weights',
                                'source_quadrature_nodes', 'precision_certified', 'cpu_seconds')
                    else:
                        keys = ('status', 'component_stage', 'source_domains', 'n_records', 'training',
                                'precision_certified', 'cpu_seconds')
                    row['saved_diagnostics'] = {key: diagnostic.get(key) for key in keys}
                    row['missing_top_level_fields'] = [key for key in keys if key not in diagnostic]
                    row['saved_optimizer_diagnostics'] = optimizer_records(diagnostic)
                else:
                    row['saved_diagnostics'], row['missing_top_level_fields'] = None, None
                    row['saved_optimizer_diagnostics'] = None
                    if diagnostic is not None:
                        reader.issues.append(dict(file=name, error='Source diagnostic is not a JSON object'))
                row['importance_sampling_scope'] = ('eight_identity_f_g_heads' if kind == 'structural'
                                                   else 'not_applicable_to_native_schema')
                rows.append(row)
    return rows


def optimizer_records(value, path=''):
    """Retain explicitly saved optimizer fields, never infer from status='fit'."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f'{path}.{key}' if path else key
            if key in ('optimizer_converged', 'optimizer_message', 'optimizer_iterations'):
                result[name] = item
            elif isinstance(item, (dict, list)):
                result.update(optimizer_records(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(optimizer_records(item, f'{path}[{index}]'))
    return result


def source_summary(rows):
    result = []
    for row in rows:
        saved = row['saved_diagnostics'] or {}
        summary = {key: row[key] for key in ('family', 'association', 'kind', 'read_status')}
        summary.update(precision_certified=saved.get('precision_certified'),
                       saved_optimizer_diagnostics=row['saved_optimizer_diagnostics'] or None,
                       missing_top_level_fields=row['missing_top_level_fields'])
        if row['kind'] == 'structural':
            heads = saved.get('source_identity_heads')
            ess = [head.get('weight_ess') for head in heads] if heads is not None else None
            flags = [head.get('low_importance_ess') for head in heads] if heads is not None else None
            identities = saved.get('source_identities')
            summary.update(nodes=saved.get('source_quadrature_nodes'),
                identities=len(identities) if identities is not None else None,
                heads=len(heads) if heads is not None else None,
                low_ess_heads=sum(flags) if flags is not None and all(value is not None for value in flags) else None,
                minimum_ess=min(ess) if ess and all(value is not None for value in ess) else None)
        else:
            summary.update(status=saved.get('status'), n_records=saved.get('n_records'),
                           importance_sampling_scope=row['importance_sampling_scope'])
        result.append(summary)
    return result


def failure_summary(rows, registered_count):
    ratios = [row['recorded']['reported_error_ratio'] for row in rows
              if row['recorded']['reported_error_ratio'] is not None]
    counts = [row['attempt_count'] for row in rows]
    cpus = [row['recorded_attempt_cpu_seconds'] for row in rows]
    queries = [row['returned_query_count'] for row in rows]
    by_arm_phase = Counter((row['arm'], row['traceback_phase']) for row in rows)
    complete_coverage = len(rows) == registered_count
    return dict(arm_phase_counts=[dict(arm=arm, phase=phase, count=count)
                                 for (arm, phase), count in sorted(by_arm_phase.items())],
        registered_failure_rows=registered_count, retained_failure_rows=len(rows),
        recorded_error_ratio_count=len(ratios),
        minimum_recorded_error_ratio=min(ratios) if ratios else None,
        maximum_recorded_error_ratio=max(ratios) if ratios else None,
        recorded_attempt_count=sum(counts) if complete_coverage and all(value is not None for value in counts) else None,
        attempt_count_distribution=dict(Counter(str(value) for value in counts)),
        recorded_attempt_cpu_seconds=sum(cpus) if complete_coverage and all(value is not None for value in cpus) else None,
        known_attempt_cpu_seconds=sum(value for value in cpus if value is not None),
        rows_with_unknown_attempt_cpu=sum(value is None for value in cpus),
        returned_query_count=sum(queries) if complete_coverage and all(value is not None for value in queries) else None,
        accounting='Retained failed-search attempt CPU and returned queries are already included in Stage93. '
                   'Do not add index copies, source diagnostic CPU, fit receipts or inclusive wrappers to these totals.')


def seal_status(plan, seal, reader):
    """Project execution status only, never open terminal/truth/score files."""
    arms = plan['retained_terminal_arms']
    if not isinstance(seal, dict):
        return dict(status=None, arms=[dict(arm=arm, status_counts=None) for arm in arms]), {}
    cases = seal.get('cases', [])
    expected = plan['coverage']['registered_terminal_cells']
    if seal.get('status') != 'sealed' or len(cases) != expected:
        reader.issues.append(dict(file='execution_status.json', error='Registered terminal seal is incomplete'))
    index = {(row['stream'], row['seed'], row['arm']): row for row in cases}
    if len(index) != len(cases) or set(row['arm'] for row in cases) != set(arms):
        reader.issues.append(dict(file='execution_status.json', error='Terminal keys or arm set differ from registration'))
    return dict(status=seal.get('status'), registered_terminal_cells=expected,
                saved_terminal_cells=len(cases),
                status_counts=dict(Counter(row['old']['status'] for row in cases)),
                arms=[dict(arm=arm, status_counts=dict(Counter(
                    row['old']['status'] for row in cases if row['arm'] == arm))) for arm in arms]), index


def failure_rows(plan, reader, saved_index, execution_index, cases):
    rows = []
    output = Path(plan['source_output_dir'])
    shards = (execution_index or {}).get('online_shards', [])
    registered = (execution_index or {}).get('registered_failure_labels', [])
    index_rows = (saved_index or {}).get('rows', [])
    labels = [row['label'] for row in index_rows]
    if (len(labels) != plan['coverage']['retained_failures'] or len(set(labels)) != len(labels)
            or set(labels) != set(registered)):
        reader.issues.append(dict(file=plan['failure_index_file'], error='Failure index does not match registered failures'))
    for saved in index_rows:
        label = saved['label']
        match = re.fullmatch(r'(stream\d+)_r(\d+)_search_(\d+)_(.+)', label)
        if not match:
            reader.issues.append(dict(file=plan['failure_index_file'], error=f'Unrecognized saved label: {label}'))
            stream, seed, arm = None, None, saved.get('arm')
        else:
            stream, _, seed, arm = match.groups()
            seed = int(seed)
        cell = cases.get((stream, seed, arm))
        ledger_name = str(Path(cell['source_case']) / 'attempts.jsonl') if cell else None
        ledger = reader.read(ledger_name, 'jsonl') if ledger_name else None
        if ledger_name is None:
            reader.issues.append(dict(file=label, error='No matching sealed source case for attempt ledger'))
        log_names = [output / 'cluster/online' / f"shard_{shard['start']}_{shard['end']}" /
                     'execution_logs' / f'{label}.log' for shard in shards]
        found = [str(name) for name in log_names if (reader.root / name).is_file()]
        log_name = found[0] if len(found) == 1 else None
        text = reader.read(log_name, 'log') if log_name else None
        if log_name is None:
            reader.issues.append(dict(file=label, error=f'Expected one saved execution log; found {len(found)}',
                                      searched_paths=[str(name) for name in log_names]))
        parsed = parse_log(text)
        if ledger is not None and len(ledger) != saved.get('attempt_count'):
            reader.issues.append(dict(file=ledger_name, error='Attempt count differs from saved audit'))
        cpus = [entry.get('cpu_seconds') for entry in ledger] if ledger is not None else None
        rows.append(dict(label=label, arm=arm, seed=seed, stream=stream,
            domain=cell.get('domain') if cell else None,
            retained_scientific_status=cell['old']['status'] if cell else None,
            original_failure_index=saved, attempts_file=ledger_name, saved_attempts=ledger,
            attempt_count=len(ledger) if ledger is not None else None,
            returned_query_count=(sum(entry['new_target_search_calls'] for entry in ledger)
                                  if ledger is not None and all('new_target_search_calls' in entry for entry in ledger) else None),
            recorded_attempt_cpu_seconds=(sum(cpus) if cpus is not None and all(value is not None for value in cpus) else None),
            execution_log_file=log_name, log_read_status='complete' if text is not None else 'unavailable',
            log_excerpt='\n'.join(text.splitlines()[-65:]) if text is not None else None,
            **parsed))
    return rows


def audit(plan, *, root=ROOT, checks_file=None):
    root, output = Path(root), Path(root) / plan['output_dir']
    complete = output / 'completion.json'
    if complete.exists():
        return json.loads(complete.read_text())
    if (output / 'attempts.jsonl').exists():
        raise RuntimeError('Existing unfinished audit attempt retained; inspect it before any resubmission')
    if plan.get('saved_record_audit_authorized') is not True:
        raise ValueError('Saved-record audit execution must be authorized separately from registration')
    if not checks_file or json.loads((root / checks_file).read_text()).get('status') != 'passed':
        raise ValueError('Targeted parser substitutes must pass before reading real records')
    output.mkdir(parents=True, exist_ok=True)
    started, wall = time.process_time(), time.monotonic()
    attempt = dict(stage='audit-source-numerical-executability', started_utc=now(), status='failed', **ZERO)
    reader = SavedRecords(root)
    try:
        # Source diagnostics are read first. No scientific-score files enter this reader.
        source_seal = reader.read(plan['source_asset_seal_file'])
        sources = source_rows(plan, reader, source_seal)
        source_result = dict(rows=sources, registered_files=12,
            completed_files=sum(row['read_status'] == 'complete' for row in sources),
            retained_warning=plan['source_warning'], source_asset_seal_file=plan['source_asset_seal_file'],
            missing_fields_policy='Missing fields remain null; native importance-sampling fields are not applicable. No precision or optimizer status inferred from fit completion.')
        save(output / 'source_diagnostics.json', source_result)
        source_output = Path(plan['source_output_dir'])
        seal = reader.read(str(source_output / 'execution_status.json'))
        retained, cases = seal_status(plan, seal, reader)
        execution_index = reader.read(plan['failure_execution_index_file'])
        saved_index = unwrapped(reader.read(plan['failure_index_file']))
        failures = failure_rows(plan, reader, saved_index, execution_index, cases)
        failure_result = dict(rows=failures, registered_failures=plan['coverage']['retained_failures'],
            retained_failure_rows=len(failures), completed_logs=sum(row['log_read_status'] == 'complete' for row in failures),
            phase_counts=dict(Counter(row['traceback_phase'] for row in failures)),
            arm_counts=dict(Counter(row['arm'] for row in failures)), retained_terminal_status=retained,
            accounting='Attempt-ledger fees are already included in Stage93; index copies and wrapper CPU must not be added again. Missing fees remain null.')
        save(output / 'failure_locations_and_costs.json', failure_result)
        historical = [dict(file=name, saved_account=reader.read(name)) for name in plan['historical_accounts']]
        save(output / 'accounting.json', dict(retained_historical_accounts=historical,
            new_work=ZERO, note='History is retained, not summed. This audit attempt and outer wrapper are nested CPU views.'))
        missing = sorted({field for row in failures for field in row['unknown_fields']})
        status = 'complete_with_missing_records' if reader.issues else 'complete'
        diagnosis = 'cause_unidentified' if missing or reader.issues else 'recorded_quantities_require_review'
        conclusion = dict(status=status, diagnosis=diagnosis,
            source_diagnostic_files=dict(registered=12, read=source_result['completed_files']),
            failure_logs=dict(registered=plan['coverage']['retained_failures'], read=failure_result['completed_logs']),
            source_diagnostic_summary=source_summary(sources),
            failure_summary=failure_summary(failures, plan['coverage']['retained_failures']),
            retained_terminal_status=retained, phase_counts=failure_result['phase_counts'],
            missing_failure_quantities=missing, input_issues=reader.issues,
            scientific_coverage=plan['coverage'], new_work=ZERO,
            statements=[
                'The raw-moment guard is a shared integration-error estimate divided by the smallest component normalizer, not a certified posterior or decision error.',
                'Source importance ESS and target moment-integration failures describe different numerical layers.',
                ('Saved logs lack local quantities needed to identify a scale, tail, cancellation or source-weight cause; none were reconstructed.'
                 if missing else 'All requested local quantities were recorded; they require scientific review rather than automatic numerical certification.'),
                'Audit completion does not certify precision or authorize a correction, a failed-case replay or target validation.'],
            files_read=reader.reads, historical_accounts=plan['historical_accounts'])
        save(output / 'conclusion.json', conclusion)
        attempt['status'] = 'completed'
    except BaseException as error:
        attempt['error'] = repr(error)
        raise
    finally:
        attempt.update(cpu_seconds=time.process_time()-started, wall_seconds=time.monotonic()-wall, finished_utc=now())
        with (output / 'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(attempt, allow_nan=False) + '\n')
    completion = dict(status=status, diagnosis=conclusion['diagnosis'], completed_utc=now(),
        source_diagnostic_files=conclusion['source_diagnostic_files'], failure_logs=conclusion['failure_logs'],
        input_issues=reader.issues, retained_terminal_status=retained,
        phase_counts=conclusion['phase_counts'], missing_failure_quantities=missing,
        checks_file=str(checks_file), attempt=attempt, new_work=ZERO,
        output_files=['source_diagnostics.json', 'failure_locations_and_costs.json', 'conclusion.json', 'accounting.json'],
        accounting='Audit attempt CPU is included in scheduler wrapper CPU; preserved scientific fees are not new audit fees.')
    save(complete, completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--checks', required=True)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] not in {f'node{i:03d}' for i in range(1, 7)}:
        raise RuntimeError('Run the saved-record audit through scheduler on node001-node006')
    result = audit(json.loads((ROOT / args.manifest).read_text()), checks_file=args.checks)
    print(json.dumps({key: result[key] for key in ('status', 'diagnosis', 'source_diagnostic_files', 'failure_logs')}))


if __name__ == '__main__':
    main()
