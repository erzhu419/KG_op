"""Probability-weighted joint fantasy strata using the historical raw CDF bank."""

import numpy as np


def stratified_record_weights(records, source_weights, uniforms, main_origins, tail_probability,
        tail_count, bulk_prefix_count):
    """Use the original source mass times exact stratum mass per node.

    Both observed labels at a Sobol position stay paired. Stratify the risk
    predictive uniform inside a complete identity,not the posterior identity
    or terminal action. Keep the complementary region and old rare origins.
    """
    low = np.flatnonzero(uniforms[:, 1] < tail_probability)[:tail_count]
    body = np.flatnonzero(uniforms[:bulk_prefix_count, 1] >= tail_probability)
    assert len(low) == tail_count and len(body) > 0
    by_coordinate = {(r['origin_identity'], r['identity_fantasy_position']): r for r in records}
    selected = []; weights = []; regions = []
    for identity, source_mass in enumerate(source_weights):
        if identity in main_origins:
            for name, positions, mass in (('low_constraint', low, tail_probability),
                    ('complement', body, 1.-tail_probability)):
                for position in positions:
                    selected.append(by_coordinate[identity, int(position)])
                    weights.append(source_mass*mass/len(positions)); regions.append(name)
        else:
            retained = sorted((r for r in records if r['origin_identity'] == identity),
                key=lambda r: r['identity_fantasy_position'])
            assert retained
            selected.extend(retained); weights.extend([source_mass/len(retained)]*len(retained))
            regions.extend(['retained_rare_origin']*len(retained))
    weights = np.asarray(weights); regions = np.asarray(regions)
    for identity, source_mass in enumerate(source_weights):
        assert abs(weights[[r['origin_identity'] == identity for r in selected]].sum()-source_mass) < 1e-14
    return selected, weights, regions


def summarize_stratified_futures(records, weights, regions, actions, baseline, raw_powers, rare_origin_mass):
    """Maximize all actions inside each joint fantasy before averaging.

    Rare-origin adaptive utility is in[0,1],so its whole expectation is in
    [0,rare_origin_mass]. Retained samples are reported as a point estimate;
    intervals allow that entire contribution to vary,plus omitted terminal
    identities at computed nodes. Terminal/fantasy quadrature stays separate.
    """
    stop, debited, fixed = baseline; rows = []; index = np.arange(len(records))
    projected = np.asarray([r['projected_values'] for r in records])
    choices = np.asarray([r['projected_action_index'] for r in records])
    rare = regions == 'retained_rare_origin'; main = ~rare
    for power in raw_powers:
        values = np.asarray([r['raw'][str(power)]['values'] for r in records])
        costs = np.asarray([r['raw'][str(power)]['costs'] for r in records])
        selected = np.lexsort((costs, -values), axis=1)[:, 0]
        contrast = values[index, selected]-values[:, fixed]
        achieved = values[index, choices]-values[:, fixed]
        loss = values[index, selected]-values[index, choices]
        tail_mass = np.asarray([r['raw'][str(power)].get('omitted_posterior_utility_mass', 0.) for r in records])
        terminal_bound = float(weights @ tail_mass)
        adaptation = float(weights @ contrast); main_adaptation = float(weights[main] @ contrast[main])
        main_loss = float(weights[main] @ loss[main])
        lower, upper = max(0., main_adaptation-terminal_bound), main_adaptation+terminal_bound+rare_origin_mass
        expected_weights = weights @ np.asarray([r['shared_identity_weights'] for r in records])
        row = {'raw_draw_power': power, 'weighted_joint_futures': len(records), 'terminal_action_count': len(actions),
            'raw_adaptation_value': adaptation, 'raw_net_reward_control_variate': adaptation+debited-stop,
            'raw_adaptation_of_projected_choice': float(weights @ achieved),
            'action_selection_loss': float(weights @ loss),
            'action_selection_loss_identity_interval': [max(0., main_loss-terminal_bound), main_loss+terminal_bound+rare_origin_mass],
            'raw_net_control_variate_identity_interval': [lower+debited-stop, upper+debited-stop],
            'identity_and_rare_origin_error_bound': terminal_bound+rare_origin_mass,
            'weighted_omitted_terminal_identity_mass': terminal_bound, 'rare_predictive_origin_mass': rare_origin_mass,
            'raw_net_reward_uncorrected': float(weights @ values[index, selected])-stop,
            'raw_fixed_action_tower_residual': float(weights @ values[:, fixed])-debited,
            'projected_adaptation_value': float(weights @ (projected[index, choices]-projected[:, fixed])),
            'projected_value_bias_at_selected_action': float(weights @ (projected[index, choices]-values[index, choices])),
            'adaptation_contributions_by_region': {name: float(weights[regions == name] @ contrast[regions == name])
                for name in sorted(set(regions))},
            'predictive_mass_by_region': {name: float(weights[regions == name].sum()) for name in sorted(set(regions))},
            'expected_model_weights': expected_weights.tolist(),
            'raw_selected_action_indices': selected.tolist(),
            'computed_projected_raw_action_disagreement_count': int(np.count_nonzero(selected != choices)),
            'predictive_log_evidence_max_error': max(r['raw'][str(power)]['predictive_log_evidence_max_error'] for r in records),
            'effective_sample_size_min': min(r['raw'][str(power)]['effective_sample_size_min'] for r in records)}
        if rows:
            row['nested_terminal_raw_adaptation_difference'] = abs(adaptation-rows[-1]['raw_adaptation_value'])
        rows.append(row)
    return rows
