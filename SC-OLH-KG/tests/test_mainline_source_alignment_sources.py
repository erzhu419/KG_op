"""Metadata and synthetic fit substitutes; no real archive, model fit or RNG."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baselines.transfer_archive import FrozenTransferArchive, TransferTaskArchive
from performance import prepare_mainline_source_alignment_sources as source


def forbidden(*args, **kwargs):
    raise AssertionError('Preparation or substitute reached real source work')


def plan_fixture():
    families = []
    for index, family in enumerate(('FamilyA', 'FamilyB', 'FamilyC'), 1):
        directory = f'alignment/source_models/{family}'
        families.append(dict(domain=family, source_archive_file=f'original/{family}/archive.json',
            task=dict(sigma=.04), anchor=dict(initial_design_seed=80), source_permutation_seed=100 + index,
            planned_permuted_model_directory=directory,
            frozen_assets=dict(source_models_file=f'original/{family}/source_models.pkl',
                native_source_prior_file=f'original/{family}/native_source_prior.pkl'),
            source_permuted_assets=dict(source_models_file=f'{directory}/source_models.pkl',
                native_source_prior_file=f'{directory}/native_source_prior.pkl')))
    return dict(output_dir='alignment', families=families, reference_manifest='historical/confirmation.json',
                historical_accounts=['historical/source_costs.json'])


def synthetic_archive():
    integer = np.repeat(np.arange(64)[:, None], 50, axis=1)
    tasks = []
    for family in range(2):
        blocks = tuple(np.array([[i + family + j / 10, family - i + j / 5] for j in range(3)]) for i in range(64))
        values = np.asarray(blocks)
        means = values.mean(axis=1)
        variance = np.maximum(values.var(axis=1, ddof=1), 1e-12)
        sigma = np.sqrt(np.maximum(((values[:, :, 1] - means[:, 1, None]) ** 2).sum(axis=1) / 4 + 2 * .04 ** 2 / 4, 1e-12))
        tasks.append(TransferTaskArchive(name=f'Source{family}', X=integer / 100,
            X_integer=integer.copy(), Y_mean=means, Y_replicates=blocks,
            replicate_variance=variance, mean_observation_variance=variance / 3,
            constraint_sigma=sigma, tau=.2, alpha=.05, origins=tuple(str(i) for i in range(64))))
    return FrozenTransferArchive(tuple(tasks), 80, 'replicated', 'existing-identifier')


def stub_work(monkeypatch):
    original = synthetic_archive()
    log = dict(load=[], draw=[], structural=[], native=[], providers=[], save=[])
    def load(path):
        log['load'].append(str(path))
        return original
    def draw(seed):
        log['draw'].append(seed)
        return np.arange(64)[::-1]
    def provider(name, d, L, sigma, alpha, weights):
        log['providers'].append(dict(name=name, d=d, L=L, sigma=sigma, alpha=alpha, weights=weights))
        return SimpleNamespace(name=name, d=d)
    def structural(archive, providers, *, settings):
        log['structural'].append((archive, deepcopy(settings)))
        assert set(providers) == {'Source0', 'Source1'}
        for problem in providers.values():
            for method in ('simulate', 'true_objective', 'true_constraint_mean', 'true_sigma', 'is_truly_feasible'):
                with pytest.raises(RuntimeError, match='cannot simulate or query scoring truth'):
                    getattr(problem, method)(None)
        return dict(identities=list(range(8)), diagnostics=dict(source_quadrature_nodes=1024,
            source_sites=128, low_importance_ess=True), settings=settings)
    def native(archive, *, heldout_family):
        log['native'].append((archive, heldout_family))
        return SimpleNamespace(n_records=128, diagnostics=lambda: dict(source_sites=128, heldout=heldout_family))
    def save_pickle(path, fitted):
        log['save'].append(str(path))
        path.write_bytes(b'synthetic fitted artifact')
    monkeypatch.setattr(FrozenTransferArchive, 'load', load)
    monkeypatch.setattr(source.archive_ops, 'draw_source_permutation', draw)
    monkeypatch.setattr(source, 'build_scalarized_problem', provider)
    monkeypatch.setattr(source, 'fit_source_models', structural)
    monkeypatch.setattr(source, 'fit_native', native)
    monkeypatch.setattr(source, 'save_pickle', save_pickle)
    return original, log


def test_prepare_is_six_job_metadata_only_and_idempotent_without_archive_model_or_rng(tmp_path, monkeypatch):
    plan = plan_fixture()
    monkeypatch.setattr(FrozenTransferArchive, 'load', forbidden)
    for name in ('fit_source_models', 'fit_native', 'build_scalarized_problem', 'save_pickle'):
        monkeypatch.setattr(source, name, forbidden)
    monkeypatch.setattr(source.archive_ops, 'draw_source_permutation', forbidden)
    monkeypatch.setattr(source.archive_ops.np.random, 'Generator', forbidden)
    result = source.prepare(plan, root=tmp_path)
    assert result['status'] == 'prepared_not_started'
    assert len(result['jobs']) == result['fit_pipeline_count'] == 6
    assert not result['source_fitting_started'] and not result['permutations_materialized']
    assert not result['scientific_execution_started'] and not result['target_execution_allowed']
    assert all(result['cost'][key] == 0 for key in source.ZERO)
    first_ledger = (tmp_path / 'alignment/source_preparation/attempts.jsonl').read_text()
    assert source.prepare(plan, root=tmp_path) == result
    assert (tmp_path / 'alignment/source_preparation/attempts.jsonl').read_text() == first_ledger
    assert not (tmp_path / 'alignment/source_models').exists()
    for job in result['jobs']:
        family = next(row for row in plan['families'] if row['domain'] == job['family'])
        assert job['source_archive_file'] == family['source_archive_file']
        assert job['permutation_seed'] == family['source_permutation_seed']
        assert job['nominal_sigma'] == .04
        assert job['consumers'] == list(source.CONTROL_ARMS)
        assert job['recipe']['structural_settings'] == source.DEFAULTS
        assert job['recipe']['quadrature_nodes'] == 1024 and job['recipe']['identities'] == 8
        assert job['recipe']['native_seed'] == 80


def test_registered_manifest_jobs_preserve_original_nine_seeds_and_terminal_cap():
    root = Path(__file__).resolve().parents[1]
    plan = source.read(root / source.MANIFEST)
    prior = source.read(root / plan['reference_manifest'])
    jobs = source.describe_jobs(plan)
    assert plan['preserved_terminal_arms'] == prior['terminal_arms']
    assert plan['terminal_arms'][:9] == prior['terminal_arms']
    assert len(plan['terminal_arms']) == 11 and plan['budget']['terminal_cells'] == 3168
    assert plan['budget']['verification_cap_per_cell'] == 352
    assert [row['source_permutation_seed'] for row in plan['families']] == [2026190001, 2026190002, 2026190003]
    assert len(jobs) == 6
    for family in plan['families']:
        selected = [row for row in jobs if row['family'] == family['domain']]
        assert [row['kind'] for row in selected] == ['structural', 'native']
        assert [row['output_file'] for row in selected] == [family['source_permuted_assets'][key]
            for key in ('source_models_file', 'native_source_prior_file')]
        assert all(row['recipe']['native_heldout_family'] == family['domain'] for row in selected)


def test_six_fake_fits_share_three_permutations_and_preserve_all_originals(tmp_path, monkeypatch):
    plan = plan_fixture()
    original, log = stub_work(monkeypatch)
    before = deepcopy(original.to_payload())
    source.prepare(plan, root=tmp_path)
    for family in plan['families']:
        for path in family['frozen_assets'].values():
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'original frozen model')
        for kind in source.KINDS:
            result = source.fit_job(plan, family['domain'], kind, root=tmp_path)
            assert result['status'] == 'complete' and not result['target_execution_allowed']
    assert len(log['draw']) == 3 and len(log['structural']) == len(log['native']) == 3
    for (bundle_archive, settings), (native_archive, family) in zip(log['structural'], log['native']):
        assert bundle_archive.to_payload() == native_archive.to_payload()
        assert bundle_archive.fingerprint == ''
        assert settings == source.DEFAULTS and settings['quadrature_power'] == 10
        np.testing.assert_array_equal(bundle_archive.tasks[0].Y_replicates[0], original.tasks[0].Y_replicates[-1])
    assert original.to_payload() == before
    assert all(row['d'] == 50 and row['name'] in original.source_domains for row in log['providers'])
    done = source.collect(plan, root=tmp_path)
    assert done['status'] == 'complete' and done['source_models_ready']
    assert done['completed_fit_pipelines'] == done['source_fit_count'] == 6
    assert done['known_recorded']['current_bundle_fits'] == 3
    assert done['known_recorded']['native_prior_fits'] == 3
    assert done['known_recorded']['source_permutations_generated'] == 3
    assert done['known_recorded']['new_source_calls'] == 0
    assert not done['target_execution_allowed']
    for family in plan['families']:
        for path in family['frozen_assets'].values():
            assert (tmp_path / path).read_bytes() == b'original frozen model'
        reused = source.fit_job(plan, family['domain'], 'structural', root=tmp_path)
        assert reused['cost']['source_archive_calls_reused'] == 384
    assert len(log['draw']) == 3 and len(log['structural']) == 3
    assert source.collect(plan, root=tmp_path)['known_recorded'] == done['known_recorded']


def test_native_failure_preserves_successful_prefix_diagnostics_costs_and_blocks_retry(tmp_path, monkeypatch):
    plan = plan_fixture()
    _, log = stub_work(monkeypatch)
    source.prepare(plan, root=tmp_path)
    family = plan['families'][0]['domain']
    successful = source.fit_job(plan, family, 'structural', root=tmp_path)
    class FailedFit(RuntimeError):
        diagnostics = dict(failure='retained synthetic optimizer failure', low_importance_ess=True)
    def failed_native(archive, *, heldout_family):
        log['native'].append((archive, heldout_family))
        raise FailedFit('native substitute failed')
    monkeypatch.setattr(source, 'fit_native', failed_native)
    with pytest.raises(FailedFit):
        source.fit_job(plan, family, 'native', root=tmp_path)
    result = source.collect(plan, root=tmp_path)
    assert result['status'] == 'failed' and result['completed_fit_pipelines'] == 1
    assert result['source_fit_count'] == 2 and not result['source_models_ready']
    assert result['failures'][0]['fit_diagnostics'] == FailedFit.diagnostics
    assert result['failures'][0]['cost']['native_prior_fits'] == 1
    assert result['failures'][0]['cost']['cpu_seconds'] >= 0
    assert result['known_recorded']['new_source_calls'] == 0
    assert result['historical_source_accounts'] == plan['historical_accounts']
    assert source.fit_job(plan, family, 'structural', root=tmp_path) == successful
    with pytest.raises(RuntimeError, match='no automatic retry'):
        source.fit_job(plan, family, 'native', root=tmp_path)
    with pytest.raises(RuntimeError, match='stop additional source work'):
        source.fit_job(plan, plan['families'][1]['domain'], 'structural', root=tmp_path)
    assert len(log['draw']) == len(log['structural']) == len(log['native']) == 1


def test_native_uses_original_source_only_wrapper_and_actual_heldout_family(tmp_path, monkeypatch):
    from performance import mainline_role_source_holdout as native
    plan = plan_fixture()
    _, log = stub_work(monkeypatch)
    calls = []
    def materialize(family, seed, N):
        calls.append((family, seed, N))
        return dict(heldout=family, seed=seed, args={})
    def provider(name, *args, **kwargs):
        return SimpleNamespace(name=name)
    def train(args, heldout, seed, teacher):
        assert args['source_records_per_domain'] == 64 and not teacher
        for name in ('Source0', 'Source1'):
            problem = native.benchmark.build_scalarized_problem(name)
            with pytest.raises(RuntimeError, match='cannot simulate or query scoring truth'):
                problem.true_objective(None)
        return SimpleNamespace(n_records=128, diagnostics=lambda: dict(heldout=heldout, seed=seed))
    monkeypatch.setattr(native, 'materialize_v69_task', materialize)
    monkeypatch.setattr(native.benchmark, 'build_scalarized_problem', provider)
    monkeypatch.setattr(native.benchmark, 'train_meta_prior', train)
    monkeypatch.setattr(source, 'fit_native', native.fit_native)
    source.prepare(plan, root=tmp_path)
    family = plan['families'][1]['domain']
    source.fit_job(plan, family, 'structural', root=tmp_path)
    source.fit_job(plan, family, 'native', root=tmp_path)
    assert calls == [(family, 80, 13)]
    assert len(log['draw']) == 1


def test_interrupted_prefix_unknown_fees_and_native_dependency_are_not_retried(tmp_path, monkeypatch):
    plan = plan_fixture()
    _, log = stub_work(monkeypatch)
    prepared = source.prepare(plan, root=tmp_path)
    job = prepared['jobs'][0]
    with pytest.raises(ValueError, match='structural fit must finish'):
        source.fit_job(plan, job['family'], 'native', root=tmp_path)
    source.save(tmp_path / job['receipt_directory'] / 'started.json', dict(job=job, status='started'))
    with pytest.raises(RuntimeError, match='interrupted or running'):
        source.fit_job(plan, job['family'], 'structural', root=tmp_path)
    result = source.collect(plan, root=tmp_path)
    assert result['status'] == 'pending' and not result['source_models_ready']
    assert result['cpu_seconds'] is None and result['source_fit_count'] is None
    assert result['known_recorded']['new_source_fits'] == 0
    assert len(result['unknown_attempt_costs']) == 1
    assert not log['draw'] and not log['load']


def test_target_family_in_archive_stops_before_permutation_or_fit(tmp_path, monkeypatch):
    plan = plan_fixture()
    archive, log = stub_work(monkeypatch)
    family = plan['families'][0]['domain']
    archive = replace(archive, tasks=(replace(archive.tasks[0], name=family), archive.tasks[1]))
    monkeypatch.setattr(FrozenTransferArchive, 'load', lambda path: archive)
    source.prepare(plan, root=tmp_path)
    with pytest.raises(ValueError, match='held-out family must be absent'):
        source.fit_job(plan, family, 'structural', root=tmp_path)
    assert not log['draw'] and not log['structural']
    assert source.collect(plan, root=tmp_path)['known_recorded']['new_source_fits'] == 0


def test_provider_construction_failure_is_not_charged_as_a_fit_invocation(tmp_path, monkeypatch):
    plan = plan_fixture()
    _, log = stub_work(monkeypatch)
    source.prepare(plan, root=tmp_path)
    def failed_provider(*args, **kwargs):
        raise RuntimeError('provider construction failed')
    monkeypatch.setattr(source, 'build_scalarized_problem', failed_provider)
    with pytest.raises(RuntimeError, match='provider construction failed'):
        source.fit_job(plan, plan['families'][0]['domain'], 'structural', root=tmp_path)
    result = source.collect(plan, root=tmp_path)
    assert result['status'] == 'failed' and result['source_fit_count'] == 0
    assert result['known_recorded']['current_bundle_fits'] == 0
    assert result['known_recorded']['source_permutations_generated'] == 1
    assert not log['structural']


def test_missing_successful_artifact_cannot_be_ready_or_automatically_refit(tmp_path, monkeypatch):
    plan = plan_fixture()
    _, log = stub_work(monkeypatch)
    source.prepare(plan, root=tmp_path)
    for family in plan['families']:
        for kind in source.KINDS:
            source.fit_job(plan, family['domain'], kind, root=tmp_path)
    family = plan['families'][0]
    (tmp_path / family['source_permuted_assets']['source_models_file']).unlink()
    result = source.collect(plan, root=tmp_path)
    assert result['status'] == 'failed' and not result['source_models_ready']
    assert result['completed_fit_pipelines'] == 5 and result['source_fit_count'] == 6
    assert result['failures'][0]['status'] == 'missing_completed_artifact'
    with pytest.raises(RuntimeError, match='no automatic retry'):
        source.fit_job(plan, family['domain'], 'structural', root=tmp_path)
    assert len(log['draw']) == len(log['structural']) == len(log['native']) == 3


def test_failed_permutation_save_retains_draw_cost_and_does_not_redraw(tmp_path, monkeypatch):
    plan = plan_fixture()
    _, log = stub_work(monkeypatch)
    source.prepare(plan, root=tmp_path)
    real_save = source.save
    def fail_permutation_save(path, value):
        if path.name == 'permutation.json':
            raise OSError('synthetic permutation write failure')
        return real_save(path, value)
    monkeypatch.setattr(source, 'save', fail_permutation_save)
    family = plan['families'][0]['domain']
    with pytest.raises(OSError, match='permutation write failure'):
        source.fit_job(plan, family, 'structural', root=tmp_path)
    result = source.collect(plan, root=tmp_path)
    assert result['status'] == 'failed' and not result['source_models_ready']
    assert result['known_recorded']['source_permutations_generated'] == 1
    assert result['source_fit_count'] == 0
    with pytest.raises(RuntimeError, match='no automatic retry'):
        source.fit_job(plan, family, 'structural', root=tmp_path)
    assert len(log['draw']) == 1 and not log['structural']


def test_output_paths_cannot_overwrite_frozen_models():
    plan = plan_fixture()
    family = plan['families'][0]
    family['planned_permuted_model_directory'] = f"original/{family['domain']}"
    family['source_permuted_assets'] = deepcopy(family['frozen_assets'])
    with pytest.raises(ValueError, match='separate family directory'):
        source.describe_jobs(plan)


def test_copied_other_family_directory_cannot_share_outputs_or_overwrite_its_original():
    plan = plan_fixture()
    first, second = plan['families'][:2]
    first['planned_permuted_model_directory'] = second['planned_permuted_model_directory']
    first['source_permuted_assets'] = deepcopy(second['source_permuted_assets'])
    with pytest.raises(ValueError, match='separate family directory'):
        source.describe_jobs(plan)
    plan = plan_fixture()
    plan['families'][1]['frozen_assets']['source_models_file'] = plan['families'][0]['source_permuted_assets']['source_models_file']
    with pytest.raises(ValueError, match='separate family directory'):
        source.describe_jobs(plan)
