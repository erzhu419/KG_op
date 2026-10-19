"""Public73 index, zero-paid restoration and raw-history contracts.

Index/budget operations need only the standard library. Restoring a real prior
and replaying it use the existing backend; neither operation simulates episodes.
"""


def map_query_indices(candidate_bank_indices, queries, candidate_count=73):
    """Translate old bank-local queries to physical public-bank indices."""
    if any(i < 0 or i >= candidate_count for i in candidate_bank_indices):
        raise ValueError('Source bank contains an invalid public policy index')
    if any(q < 0 or q >= len(candidate_bank_indices) for q in queries):
        raise ValueError('Query must be an index in its original candidate bank')
    return [candidate_bank_indices[q] for q in queries]


def common_bank_mapping(public_bank, frozen_bank):
    """Keep all73 policies and preserve each frozen query's physical point."""
    if len(public_bank) != 73:
        raise ValueError('The common library must contain all73 public policies')
    indices = frozen_bank['candidate_bank_indices']
    mapped = map_query_indices(indices, frozen_bank['queries'])
    if (len(indices) != len(frozen_bank['policies']) or
            any(tuple(public_bank[i]) != tuple(point)
                for i, point in zip(indices, frozen_bank['policies']))):
        raise ValueError('Frozen policies do not match their public-bank indices')
    return {'candidate_bank_indices': list(range(73)), 'queries': mapped}


def remaining_target_calls(paid_episodes, prospective_batch=0, target_cap=362):
    """Search and terminal verification share one target-episode cap."""
    remaining = target_cap-paid_episodes-prospective_batch
    if paid_episodes < 0 or prospective_batch < 0 or remaining < 0:
        raise ValueError('Paid and prospective episodes exceed the target cap')
    return remaining


def _require_zero_paid(model, candidate_count):
    if model.paid_episodes or model.history:
        raise ValueError('Replay must start from a zero-paid prior')
    if any(field.candidate_count != candidate_count or field.paid_episodes
           for pair in model.components for field in pair):
        raise ValueError('Prior fields must cover the full bank with zero paid data')


def restore_zero_paid_prior(spec, pilot, state, connection, old, candidate_count=73):
    """Reuse source-only structural/GP restoration without old target updates."""
    from performance.historical_full_bank_development import restore_full_prior

    model, diagnostics = restore_full_prior(spec, pilot, state, connection, old,
                                            candidate_count)
    _require_zero_paid(model, candidate_count)
    return model, diagnostics


def initialization_batches(record):
    """Preserve the original separate initialization episodes and raw units."""
    indices, observations = record['candidate_bank_indices'], record['paired_observations']
    charged = record['physical_calls'] + record.get('reused_initialization_calls', 0)
    if len(indices) != len(observations) or charged != len(indices):
        raise ValueError('Initialization policy, observation and debit counts differ')
    return [{'public_policy_index': i, 'raw_observations': [pair]}
            for i, pair in zip(indices, observations)]


def legacy_search_batch(record, candidate_bank_indices):
    """Map a saved old decision for state reconstruction, not a new selection.

    Use raw_observation, never subtract tau again from centered_observation.
    """
    index = map_query_indices(candidate_bank_indices, [record['decision']['query']])[0]
    if record['physical_calls'] != 1:
        raise ValueError('Historical search records contain one paid episode')
    return {'public_policy_index': index, 'raw_observations': [record['raw_observation']]}


def replay_raw_batches(model, batches, tau, target_cap=362):
    """Reconstruct paid state once, preserving batch boundaries and order.

    Inputs contain raw [f,g] rows, not centered rows or pseudo variance labels.
    The backend may perform numerical conditioning; this is not a cache reader.
    """
    count = model.components[0][0].candidate_count
    _require_zero_paid(model, count)
    paid = sum(len(batch['raw_observations']) for batch in batches)
    remaining_target_calls(paid, target_cap=target_cap)
    for batch in batches:
        rows = batch['raw_observations']
        if (not rows or batch['public_policy_index'] < 0 or
                batch['public_policy_index'] >= count or any(len(row) != 2 for row in rows)):
            raise ValueError('Each paid batch needs a policy and raw paired observations')
    for batch in batches:
        rows = [[float(f), float(g)-tau] for f, g in batch['raw_observations']]
        index = batch['public_policy_index']
        if len(rows) == 1:
            model.update_raw(index, rows[0])
        else:
            model.update_replicates(index, rows)
    if model.paid_episodes != paid:
        raise ValueError('Raw backend debit differs from replayed episode count')
    return {'paid_episodes': paid,
            'remaining_calls': remaining_target_calls(paid, target_cap=target_cap),
            'replayed_batches': len(batches), 'new_physical_calls': 0}
