"""Three-family role interface preflight using only the paid seed80 histories."""
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from baselines.transfer_archive import FrozenTransferArchive
from performance import benchmark_lodo_meta_prior as benchmark
from performance.mainline_role_after_init import after_init_point_only_moment_state
from performance.mainline_role_backend import build_fresh_backend
from performance.mainline_role_point_only import point_only_moment_state
from performance.mainline_role_prior import role_inputs, role_moment_state, role_prior
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, load_pickle, read, save
from representation.legacy_function_prior import legacy_source_labels


PARENT = ROOT / 'paper_artifacts/mainline_legacy_backend_bridge_v1'
OUTPUT = ROOT / 'paper_artifacts/mainline_three_family_v1'
FAMILIES = ('InventorySupplyChain', 'QueueResourceControl', 'FactorShockStatePolicyRZDT1')
ARMS = ('legacy_role_full', 'legacy_role_after_init_point_only', 'legacy_role_point_only')


def check(family):
    errors = {}

    def equal(name, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10, err_msg=name)
        errors[name] = float(np.max(np.abs(np.asarray(actual) - expected)))

    with attempt(OUTPUT, 'three-family-preflight', family) as cost:
        cost.update(new_target_initial_calls=0, new_source_fits=0,
                    new_truth_function_calls=0, download_bytes=0)
        source_dir = PARENT / 'source_models' / family
        archive = FrozenTransferArchive.load(
            ROOT / 'paper_artifacts/legacy_joint_bridge_v1/inputs' / family / 'archive.json')
        bundle = load_pickle(source_dir / 'source_models.pkl')
        source_names = [task.name for task in archive.tasks]
        assert set(source_names) == set(FAMILIES) - {family}
        assert bundle['diagnostics']['source_task_names'] == source_names
        assert archive.simulator_calls == bundle['diagnostics']['reused_source_calls'] == 384
        assert bundle['diagnostics']['source_sites'] == 128
        for task, labels in zip(archive.tasks, legacy_source_labels(archive), strict=True):
            raw = np.asarray(task.Y_replicates)
            assert task.X.shape == (64, 50) and raw.shape == (64, 3, 2)
            equal(task.name + '_raw_f', labels['response'][:, 0], raw[:, :, 0].mean(axis=1))
            equal(task.name + '_centered_g', labels['response'][:, 1],
                  raw[:, :, 1].mean(axis=1) - task.tau)
        rows = read(PARENT / family / 'initialization_only/search.json')['history']
        design = read(ROOT / 'paper_artifacts/legacy_joint_bridge_v1/inputs' / family / 'design.json')
        assert design['source_dimension'] == 50 and design['dimension'] == 1000
        assert not design['target_labels_used'] and not design['target_oracle_used']
        assert [row['point'] for row in rows] == design['designs']['80']['points']
        assert len(rows) == 10 and np.asarray([row['raw'] for row in rows]).shape == (10, 2)
        saved = read(PARENT / family / 'legacy_native_v69/search.json')['history']
        assert saved[:10] == rows
        next_row = saved[10]
        bank = list(dict.fromkeys(tuple(row['point']) for row in rows + [next_row]))
        plan = dict(domain=family, frozen_assets=dict(
            source_models_file=str((source_dir / 'source_models.pkl').relative_to(ROOT)),
            native_source_prior_file=str((source_dir / 'native_source_prior.pkl').relative_to(ROOT))),
            geometry_by_seed={'80': {'problem_kwargs': {
                'task_geometry_shift': [0., 0., 0.], 'task_geometry_radius_scale': 1.}}})
        engines = [build_fresh_backend(plan, 80, arm, rows) for arm in ARMS]
        cost['source_model_restorations'] = 7
        assert all(engine.initial_rows == rows and engine.config.n0 == 10
                   and engine.config.N == 13 and engine.problem.d == 1000
                   and engine.problem.L == 100 and engine.problem.tau == 0.
                   and engine.problem.alpha == .05 for engine in engines)
        verifier_kwargs = dict(plan['geometry_by_seed']['80']['problem_kwargs'])
        if family == 'FactorShockStatePolicyRZDT1':
            verifier_kwargs['shared_shock_scale'] = 1.
        verifier = benchmark.build_scalarized_problem(family, 1000, 100, .04, .05,
            (.5, .5), problem_kwargs=verifier_kwargs)
        assert verifier.d == 1000 and verifier.tau == engines[0].problem.tau
        events = []
        with ExitStack() as stack:
            for engine in engines:
                for method in ('simulate', 'true_objective', 'true_constraint_mean', 'true_sigma'):
                    stack.enter_context(patch.object(engine.problem, method,
                        side_effect=AssertionError('Preflight attempted ' + method)))
            prior = role_prior(engines[0])
            source, designs = role_inputs(bundle, engines[0], bank)
            assert len(source['identities']) == 8 and bundle['settings']['quadrature_power'] == 10
            eta_errors = []
            for pair, original, design_pair in zip(source['components'], bundle['components'], designs):
                assert pair[0] is original[0]
                for key in ('eta_mean', 'eta_covariance', 'log_weights'):
                    np.testing.assert_array_equal(pair[1][key], original[1][key])
                    eta_errors.append(0.)
                assert pair[1]['risk'] is original[1]['risk']
                assert all(len(record['log_weights']) == 1024 for record in pair)
                for record, (D, E) in zip(pair, design_pair):
                    assert D.shape == (len(bank), record['mean'].shape[1])
                    assert E.shape == (len(bank), record['eta_mean'].shape[1])
                    assert np.isfinite(D).all() and np.isfinite(E).all()
            zero = role_moment_state(bundle, engines[0], bank)
            D = engines[0].gpr[1].basis_matrix(bank)
            equal('source0_native_g_mean', zero['mean'][:, 2], D @ prior['mean'] - engines[0].problem.tau)
            expected_var = np.einsum('ni,ij,nj->n', D, prior['covariance'], D) + prior['deviation_variance']
            equal('source0_native_g_variance', np.diag(zero['cov'].reshape(4*len(bank), -1))[2::4], expected_var)
            full = role_moment_state(bundle, engines[0], bank, rows)
            cache = {}
            after = after_init_point_only_moment_state(bundle, engines[1], bank, rows, cache=cache)
            point = point_only_moment_state(bundle, engines[2], bank, rows)
            for key in ('mean', 'cov', 'noise', 'weights'):
                equal('initial10_full_after_' + key, after[key], full[key])
            untouched = [i for i, point_ in enumerate(bank) if point_ not in {tuple(r['point']) for r in rows}]
            for key in ('mean', 'noise'):
                if untouched:
                    equal('source0point_unsampled_' + key, point[key][untouched], zero[key][untouched])
            late = after_init_point_only_moment_state(bundle, engines[1], bank, rows + [next_row], cache=cache)
            untouched = [i for i, point_ in enumerate(bank) if point_ != tuple(next_row['point'])]
            for key in ('mean', 'noise', 'policy_weights'):
                equal('after_paid11_other_' + key, late[key][untouched], after[key][untouched])
            for state in (zero, full, after, point, late):
                assert all(np.isfinite(state[key]).all() for key in ('mean', 'cov', 'noise'))
                events.extend(state.get('roundoff_events', []))
            cost['saved_raw_rows_replayed'] = 31
        result = dict(family=family, status='passed', source_names=source_names,
            source_calls_reused=384, source_policy_dimension=50, target_dimension=1000,
            source_rows=128, source_replicates_per_policy=3, raw_head_units=['scalarized_f', 'raw_g'],
            role_g_centering_tau=0., mean_coordinate_dimension=10, risk_coordinate_dimension=10,
            native_prior_dimension=len(prior['mean']), identities=8, nodes_per_head=1024,
            bank_count=len(bank), identical_initialization_count=10, late_saved_raw_count=1,
            built_role_arms=list(ARMS), verifier_family_constructor='passed',
            eta_source_law_max_error=max(eta_errors), errors=errors, maximum_error=max(errors.values()),
            roundoff_events=events)
    result['cost'] = cost
    return result


if __name__ == '__main__':
    results = []
    for family in FAMILIES:
        results.append(check(family))
        save(OUTPUT / 'preflight.json', dict(status='passed' if len(results) == 3 else 'running',
            results=results, new_simulator_calls=0, new_truth_calls=0, new_source_fits=0,
            recorded_cpu_seconds=sum(row['cost']['cpu_seconds'] for row in results)))
        print(family, 'passed', 'cpu', results[-1]['cost']['cpu_seconds'], flush=True)
