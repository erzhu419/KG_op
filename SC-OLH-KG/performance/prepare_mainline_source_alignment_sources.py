"""Describe six source-only fits, or explicitly execute one future registered fit.

``prepare`` only writes metadata. ``fit`` is a separate scheduler action; it never
runs a target experiment. Original sources/models remain read-only, and a failed
or interrupted attempt cannot be retried automatically by calling this entrypoint.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from baselines.transfer_archive import FrozenTransferArchive
from performance.benchmark_lodo_meta_prior import build_scalarized_problem
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.mainline_legacy_source_models import DEFAULTS, fit_source_models
from performance.mainline_role_source_holdout import fit_native
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save, save_pickle
from performance import mainline_source_alignment_archive as archive_ops


MANIFEST = 'performance/manifests/mainline_source_alignment_control_v1.json'
KINDS = ('structural', 'native')
CONTROL_ARMS = ('source_permuted_role_after_init_point_only', 'source_permuted_role_point_only')
ZERO = dict(new_source_calls=0, new_target_initial_calls=0, new_target_search_calls=0,
    new_verification_calls=0, new_truth_function_calls=0, new_candidate_scores=0,
    source_model_restorations=0, engine_restorations=0, posterior_restorations=0,
    new_source_fits=0, current_bundle_fits=0, native_prior_fits=0,
    source_permutations_generated=0, simulation_rng_draws=0)


def describe_jobs(plan):
    """Pure metadata: no archive/model reads, random draws, fits or providers."""
    families = plan['families']
    if len(families) != 3 or len({row['domain'] for row in families}) != 3:
        raise ValueError('The registration requires three distinct held-out families')
    originals = {row['source_archive_file'] for row in families}
    originals.update(value for row in families for value in row['frozen_assets'].values())
    jobs = []
    for family in families:
        domain = family['domain']
        directory = Path(family['planned_permuted_model_directory'])
        assets = family['source_permuted_assets']
        expected = dict(source_models_file=str(directory / 'source_models.pkl'),
                        native_source_prior_file=str(directory / 'native_source_prior.pkl'))
        if (directory != Path(plan['output_dir']) / 'source_models' / domain
                or assets != expected or any(path in originals for path in assets.values())):
            raise ValueError('Both permuted models must use their registered separate family directory')
        if family['task']['sigma'] != .04 or family['anchor']['initial_design_seed'] != 80:
            raise ValueError('Retain the registered sigma .04 and native source recipe seed80')
        for kind in KINDS:
            artifact = assets['source_models_file' if kind == 'structural' else 'native_source_prior_file']
            jobs.append(dict(job_id=f'{domain}/{kind}', family=domain, kind=kind,
                source_archive_file=family['source_archive_file'],
                permutation_file=str(directory / 'permutation.json'),
                permutation_seed=family['source_permutation_seed'], nominal_sigma=family['task']['sigma'],
                output_file=artifact, directory=str(directory),
                receipt_directory=str(directory / f'fit_{kind}'),
                diagnostics_file=str(directory / f'{kind}_source_diagnostics.json'),
                dependencies=[] if kind == 'structural' else [f'{domain}/structural'],
                consumers=list(CONTROL_ARMS), new_source_simulations=0,
                recipe=dict(structural_settings=deepcopy(DEFAULTS), identities=8, quadrature_nodes=1024,
                    native_heldout_family=domain, native_seed=80, native_N=13,
                    native_decomposition='latent_shared'),
                source_provider=dict(dimension='archive task X width', L=100, sigma=.04,
                    alpha=.05, weights=[.5, .5], outcome_access='forbidden by no_outcomes')))
    return jobs


def _preparation_path(plan, root):
    return Path(root) / plan['output_dir'] / 'source_preparation' / 'preparation.json'


def prepare(plan, *, root=ROOT):
    """Write the six-job description; this does not materialize even a permutation."""
    path = _preparation_path(plan, root)
    jobs = describe_jobs(plan)
    if path.exists():
        saved = read(path)
        if saved['jobs'] != jobs:
            raise ValueError('The saved source preparation differs from the registered jobs')
        return saved
    with attempt(path.parent, 'describe-source-alignment-fits') as cost:
        cost.update(ZERO)
        result = dict(status='prepared_not_started', jobs=jobs, fit_pipeline_count=len(jobs),
            source_fitting_started=False, permutations_materialized=False,
            scientific_execution_started=False, target_execution_allowed=False,
            entrypoint='performance.prepare_mainline_source_alignment_sources --action fit',
            execution='Future explicit scheduler jobs only; node001-node006, vram0, BLAS threads1.',
            accounting='Preparation metadata is separate from source fitting and historical source simulation.')
    result['cost'] = deepcopy(cost)
    save(path, result)
    return result


def _registered_jobs(plan, root):
    path = _preparation_path(plan, root)
    if not path.is_file():
        raise ValueError('Prepare the registered six-job metadata before an explicit source fit')
    saved = read(path)
    if saved['jobs'] != describe_jobs(plan):
        raise ValueError('Frozen source job metadata changed after preparation')
    return saved['jobs']


def _permuted_archive(job, root, cost):
    original = FrozenTransferArchive.load(Path(root) / job['source_archive_file'])
    archive_ops.validate_source_archive(original, nominal_sigma=job['nominal_sigma'])
    if job['family'] in [task.name for task in original.tasks]:
        raise ValueError('The target held-out family must be absent from its source archive')
    path = Path(root) / job['permutation_file']
    if path.exists():
        record = read(path)
        permutation = record['permutation']
        expected = archive_ops.permutation_record(original, permutation,
            job['permutation_seed'], nominal_sigma=job['nominal_sigma'])
        if record != expected:
            raise ValueError('Reuse exactly the saved permutation and original registered source metadata')
    else:
        permutation = archive_ops.draw_source_permutation(job['permutation_seed'])
        cost['source_permutations_generated'] = 1
        record = archive_ops.permutation_record(original, permutation,
            job['permutation_seed'], nominal_sigma=job['nominal_sigma'])
        save(path, record)
    cost.update(source_archive_calls_reused=original.simulator_calls,
        source_sites_reused=sum(len(task.X) for task in original.tasks),
        permutation_file=job['permutation_file'],
        source_reuse_scope='Logical rows consumed by this fit only; not new or globally unique source calls')
    return archive_ops.permute_response_blocks(original, permutation, nominal_sigma=job['nominal_sigma'])


def _source_providers(archive, job):
    spec = job['source_provider']
    return {task.name: no_outcomes(build_scalarized_problem(task.name, task.X.shape[1],
        spec['L'], spec['sigma'], spec['alpha'], spec['weights'])) for task in archive.tasks}


def fit_job(plan, family, kind, *, root=ROOT):
    """Future explicit fit action; reuse completion, never retry a failed prefix."""
    root = Path(root)
    jobs = _registered_jobs(plan, root)
    job = next((row for row in jobs if row['family'] == family and row['kind'] == kind), None)
    if job is None:
        raise ValueError('Select one registered family and structural/native fit')
    directory = root / job['receipt_directory']
    receipt, started = directory / 'result.json', directory / 'started.json'
    if receipt.exists():
        saved = read(receipt)
        if saved['status'] == 'complete' and (root / job['output_file']).is_file():
            return saved
        raise RuntimeError('Retain the previous source-fit failure or incomplete artifact; no automatic retry')
    if started.exists():
        raise RuntimeError('An interrupted or running source fit already exists; do not retry it automatically')
    for other in jobs:
        other_receipt = root / other['receipt_directory'] / 'result.json'
        if other_receipt.exists() and read(other_receipt)['status'] != 'complete':
            raise RuntimeError('A registered source fit failed; retain it and stop additional source work')
    for dependency in job['dependencies']:
        parent = next(row for row in jobs if row['job_id'] == dependency)
        parent_receipt = root / parent['receipt_directory'] / 'result.json'
        if (not parent_receipt.exists() or read(parent_receipt)['status'] != 'complete'
                or not (root / parent['output_file']).is_file()):
            raise ValueError('The family structural fit must finish before its native fit')
    if (root / job['output_file']).exists():
        raise RuntimeError('A model artifact without a completed receipt is retained, never overwritten')
    cost = {}
    try:
        with attempt(directory, 'fit-source-alignment-' + kind, family) as cost:
            cost.update(ZERO, job_id=job['job_id'], output_file=job['output_file'])
            save(started, dict(job=job, status='started', automatic_retry=False))
            archive = _permuted_archive(job, root, cost)
            if kind == 'structural':
                providers = _source_providers(archive, job)
                cost['new_source_fits'] = 1
                cost['current_bundle_fits'] = 1
                fitted = fit_source_models(archive, providers,
                                           settings=deepcopy(job['recipe']['structural_settings']))
                diagnostics = fitted['diagnostics']
                if len(fitted['identities']) != 8 or diagnostics['source_quadrature_nodes'] != 1024:
                    raise ValueError('The permuted structural fit changed the eight-identity/1024-node recipe')
            else:
                cost['new_source_fits'] = 1
                cost['native_prior_fits'] = 1
                fitted = fit_native(archive, heldout_family=family)
                if fitted.n_records != sum(len(task.X) for task in archive.tasks):
                    raise ValueError('Native source fitting must consume all supplied permuted source rows')
                diagnostics = fitted.diagnostics()
            save(root / job['diagnostics_file'], diagnostics)
            save_pickle(root / job['output_file'], fitted)
    except BaseException as error:
        failure = dict(status='failed', job=job, error=repr(error), cost=deepcopy(cost),
            automatic_retry=False, target_execution_allowed=False)
        if hasattr(error, 'diagnostics'):
            failure['fit_diagnostics'] = error.diagnostics
        save(receipt, failure)
        raise
    result = dict(status='complete', job=job, cost=deepcopy(cost),
        output_file=job['output_file'], diagnostics_file=job['diagnostics_file'],
        source_only=True, target_execution_allowed=False)
    save(receipt, result)
    return result


def collect(plan, *, root=ROOT):
    """Summarize saved source receipts and fees; never admit or run target science."""
    root = Path(root)
    jobs = _registered_jobs(plan, root)
    outcomes, attempts, unknown = [], [], []
    for job in jobs:
        directory = root / job['receipt_directory']
        path = directory / 'result.json'
        if path.exists():
            row = read(path)
            if row['status'] == 'complete' and not (root / job['output_file']).is_file():
                row = dict(row, status='missing_completed_artifact')
        elif (directory / 'started.json').exists():
            row = dict(status='running_or_interrupted', job=job, cost=None)
            unknown.append(dict(job_id=job['job_id'], reason='Attempt has no closed receipt; CPU and fit count unknown'))
        else:
            row = dict(status='not_started', job=job)
        outcomes.append(row)
        ledger = directory / 'attempts.jsonl'
        if ledger.exists():
            attempts.extend(dict(json.loads(line), ledger=str(ledger.relative_to(root)))
                for line in ledger.read_text().splitlines() if line.strip())
    preparation_ledger = _preparation_path(plan, root).parent / 'attempts.jsonl'
    if preparation_ledger.exists():
        attempts.extend(dict(json.loads(line), ledger=str(preparation_ledger.relative_to(root)))
            for line in preparation_ledger.read_text().splitlines() if line.strip())
    fields = (*ZERO, 'cpu_seconds')
    known = {key: sum(row.get(key, 0) for row in attempts) for key in fields}
    completed = sum(row['status'] == 'complete' for row in outcomes)
    failed = [row for row in outcomes if row['status'] in ('failed', 'missing_completed_artifact')]
    result = dict(status='complete' if completed == 6 else 'failed' if failed else 'pending',
        completed_fit_pipelines=completed, registered_fit_pipelines=6, jobs=outcomes,
        failures=failed, attempts=attempts, unknown_attempt_costs=unknown,
        known_recorded=known, source_fit_count=known['new_source_fits'] if not unknown else None,
        cpu_seconds=known['cpu_seconds'] if not unknown else None,
        source_models_ready=completed == 6, target_execution_allowed=False,
        historical_source_accounts=plan.get('historical_accounts'),
        original_source_reference_manifest=plan.get('reference_manifest'),
        accounting='Each physical attempt ledger read once. Reused raw source rows are provenance, not new simulator '
            'calls. Receipt costs duplicate ledgers and are not added. Historical fees and future target expenses remain separate.')
    save(root / plan['output_dir'] / 'source_preparation' / 'status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--action', choices=('prepare', 'fit', 'collect'), required=True)
    parser.add_argument('--family')
    parser.add_argument('--kind', choices=KINDS)
    args = parser.parse_args()
    root = Path(args.root)
    plan = read(root / args.manifest)
    if args.action == 'prepare':
        result = prepare(plan, root=root)
    elif args.action == 'fit':
        if args.family is None or args.kind is None:
            parser.error('The explicit fit action requires --family and --kind')
        result = fit_job(plan, args.family, args.kind, root=root)
    else:
        result = collect(plan, root=root)
    print(json.dumps(dict(status=result['status'], target_execution_allowed=False)))


if __name__ == '__main__':
    main()
