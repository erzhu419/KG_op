"""Synthetic whole-block routing; no registered permutation or source file is used."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from baselines import transfer_archive
from baselines.transfer_archive import FrozenTransferArchive, TransferTaskArchive
from performance import mainline_source_alignment_archive as alignment


def forbidden(*args, **kwargs):
    raise AssertionError('No archive loading, new hash or uncontrolled random drawing')


def synthetic_archive():
    tasks = []
    x = np.repeat(np.arange(64)[:, None], 50, axis=1)
    for family in range(2):
        blocks = []
        for index in range(64):
            # Includes a constant block to distinguish the variance floor from
            # native sigma's two nominal-variance degrees of freedom.
            steps = np.array([-1., 0., 1.]) * index / 100
            blocks.append(np.column_stack([
                index + family * 100 + steps,
                family * 10 - index + steps * (family + 2)]))
        values = np.asarray(blocks)
        means = values.mean(axis=1)
        variance = np.maximum(values.var(axis=1, ddof=1), 1e-12)
        sigma = np.array([np.sqrt(max(
            (np.dot(v[:, 1] - m[1], v[:, 1] - m[1]) + 2 * .04 ** 2) / 4,
            1e-12)) for v, m in zip(values, means)])
        tasks.append(TransferTaskArchive(
            name=f'source_{family}', X=x.astype(float) / 100, X_integer=x.copy(),
            Y_mean=means, Y_replicates=tuple(blocks), replicate_variance=variance,
            mean_observation_variance=variance / 3, constraint_sigma=sigma,
            tau=.2 + family, alpha=.05,
            origins=tuple(f'family_{family}_policy_{i}' for i in range(64))))
    return FrozenTransferArchive(tuple(tasks), 80, 'replicated', 'existing-input-identifier')


@pytest.fixture(autouse=True)
def no_real_source_or_hash(monkeypatch):
    monkeypatch.setattr(transfer_archive, '_archive_fingerprint', forbidden)
    monkeypatch.setattr(FrozenTransferArchive, 'load', forbidden)
    monkeypatch.setattr(FrozenTransferArchive, 'save', forbidden)


def test_whole_policy_permutation_preserves_pairing_metadata_and_input(monkeypatch):
    monkeypatch.setattr(alignment.np.random, 'Generator', forbidden)
    original = synthetic_archive()
    before = deepcopy(original.to_payload())
    permutation = np.arange(64)
    permutation[:4] = [2, 0, 3, 1]
    changed = alignment.permute_response_blocks(original, permutation)
    assert original.to_payload() == before
    assert changed.source_seed == original.source_seed
    assert changed.observation_mode == original.observation_mode
    assert changed.fingerprint == '' and original.fingerprint == 'existing-input-identifier'
    assert changed.simulator_calls == original.simulator_calls == 384
    for source, result in zip(original.tasks, changed.tasks):
        assert result.name == source.name and result.origins == source.origins
        assert result.tau == source.tau and result.alpha == source.alpha
        np.testing.assert_array_equal(result.X, source.X)
        np.testing.assert_array_equal(result.X_integer, source.X_integer)
        for index, source_index in enumerate(permutation):
            np.testing.assert_array_equal(result.Y_replicates[index], source.Y_replicates[source_index])
            assert not np.shares_memory(result.Y_replicates[index], source.Y_replicates[source_index])
        for field in ('Y_mean', 'replicate_variance', 'mean_observation_variance', 'constraint_sigma'):
            np.testing.assert_allclose(getattr(result, field), getattr(source, field)[permutation],
                                       rtol=1e-12, atol=1e-12)
            assert not np.shares_memory(getattr(result, field), getattr(source, field))
        assert not np.shares_memory(result.X, source.X)
        assert not np.shares_memory(result.X_integer, source.X_integer)
        # Policy origin stays with X, not with the moved response's former site.
        assert result.origins[0] == source.origins[0] != source.origins[2]
    # Mutating derived fitting arrays must not reach the preserved source input.
    changed.tasks[0].Y_replicates[0][0, 0] = -12345
    changed.tasks[0].X[0, 0] = -12345
    assert original.to_payload() == before


def test_native_sigma_keeps_original_shrinkage_and_sample_variance_floor():
    source = synthetic_archive()
    identity = np.arange(64)
    derived = alignment.permute_response_blocks(source, identity)
    for task in derived.tasks:
        np.testing.assert_array_equal(task.replicate_variance[0], [1e-12, 1e-12])
        np.testing.assert_array_equal(task.mean_observation_variance[0], [1e-12 / 3] * 2)
        assert task.constraint_sigma[0] == pytest.approx(.04 / np.sqrt(2))
        assert task.constraint_sigma[0] != np.sqrt(task.replicate_variance[0, 1])
        centered = task.Y_replicates[17][:, 1] - task.Y_mean[17, 1]
        expected = np.sqrt((centered @ centered + 2 * .04 ** 2) / 4)
        assert task.constraint_sigma[17] == pytest.approx(expected)


@pytest.mark.parametrize('field', [
    'Y_mean', 'replicate_variance', 'mean_observation_variance', 'constraint_sigma',
])
def test_inconsistent_existing_derived_field_stops_before_silent_repair(field):
    archive = synthetic_archive()
    values = getattr(archive.tasks[0], field).copy()
    values.flat[0] += .5
    damaged = replace(archive, tasks=(replace(archive.tasks[0], **{field: values}), archive.tasks[1]))
    with pytest.raises(ValueError, match=field + ' differs'):
        alignment.permute_response_blocks(damaged, np.arange(64))


@pytest.mark.parametrize('damage,match', [
    ('integer_order', 'physical-policy order'),
    ('normalized_order', 'physical-policy order'),
    ('duplicates', 'distinct physical policies'),
    ('repeat_count', 'paired 3 by 2'),
    ('nonfinite', 'nonfinite observations'),
    ('same_family', 'distinct replicated source families'),
    ('wrong_nominal', 'constraint_sigma differs'),
])
def test_source_contract_changes_are_rejected(damage, match):
    archive = synthetic_archive()
    first, second = archive.tasks
    if damage == 'integer_order':
        second = replace(second, X_integer=second.X_integer[::-1].copy())
    elif damage == 'normalized_order':
        second = replace(second, X=second.X[::-1].copy())
    elif damage == 'duplicates':
        x, integer = first.X.copy(), first.X_integer.copy()
        x[1], integer[1] = x[0], integer[0]
        first = replace(first, X=x, X_integer=integer)
        second = replace(second, X=x.copy(), X_integer=integer.copy())
    elif damage == 'repeat_count':
        first = replace(first, Y_replicates=(first.Y_replicates[0][:2], *first.Y_replicates[1:]))
    elif damage == 'nonfinite':
        block = first.Y_replicates[0].copy()
        block[0, 0] = np.nan
        first = replace(first, Y_replicates=(block, *first.Y_replicates[1:]))
    elif damage == 'same_family':
        second = replace(second, name=first.name)
    damaged = replace(archive, tasks=(first, second))
    with pytest.raises(ValueError, match=match):
        alignment.validate_source_archive(damaged, nominal_sigma=.05 if damage == 'wrong_nominal' else .04)


@pytest.mark.parametrize('permutation', [list(range(63)), list(range(63)) + [0],
                                        list(range(64))[::-1] + [64], [float(i) for i in range(64)]])
def test_invalid_permutations_cannot_drop_or_duplicate_source_policies(permutation):
    with pytest.raises(ValueError, match='one permutation'):
        alignment.permute_response_blocks(synthetic_archive(), permutation)


def test_record_keeps_fixed_points_and_attributes_only_the_original_identifier():
    archive = synthetic_archive()
    permutation = np.arange(64)
    permutation[:2] = [1, 0]
    record = alignment.permutation_record(archive, permutation, 123)
    assert record['permutation'] == permutation.tolist()
    assert record['permutation_seed'] == 123 and record['nominal_sigma'] == .04
    assert record['fixed_points'] == list(range(2, 64))
    assert record['same_permutation_across_source_families'] is True
    assert record['source_domains'] == ['source_0', 'source_1']
    assert record['original_archive_fingerprint'] == archive.fingerprint
    assert record['derived_archive_fingerprint'] is None and 'fingerprint' not in record
    assert record['source_only'] and record['new_source_simulations'] == 0


def test_one_future_pcg64_draw_retains_fixed_points_without_screening(monkeypatch):
    calls = []
    explicit = np.arange(64)
    explicit[:2] = [1, 0]

    class Draw:
        def permutation(self, size):
            calls.append(('permutation', size))
            return explicit.copy()

    def pcg(seed):
        calls.append(('PCG64', seed))
        return 'synthetic-generator-state'

    def generator(state):
        calls.append(('Generator', state))
        return Draw()

    monkeypatch.setattr(alignment.np.random, 'PCG64', pcg)
    monkeypatch.setattr(alignment.np.random, 'Generator', generator)
    result = alignment.draw_source_permutation(123)
    np.testing.assert_array_equal(result, explicit)
    assert calls == [('PCG64', 123), ('Generator', 'synthetic-generator-state'), ('permutation', 64)]
