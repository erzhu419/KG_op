"""Condition one raw episode without projecting its eta posterior.

Reuse the historical posterior-component/eta importance proposal. Given query
eta and the raw label, the other eta coordinates and all means are Gaussian.
Integrate means with the same guarded terminal functions as current KG. Earlier
paid updates remain the frozen moment fields; only this next update is exact.
"""

import numpy as np
from scipy.special import logsumexp, ndtri, ndtr, roots_hermitenorm
from scipy.stats import qmc

from core.conditional_terminal_kg import (roots, paired_quality_from_moments,
    risk_coordinates_from_moments, guarded_risk_statistics, conditional_covariance_inverse,
    condition_eta_components, guarded_risk_values)
from core.guarded_deployment_kg import _sd_quadrature, _tolerance_factor
from core.legacy_joint_kg import clipped_objective_quality
from core.positive_mixture_lookahead import raw_eta_proposal


def conditional_raw_coordinates(field, updated, indices, nodes, label, head_log_evidence, *, query=None,
        replicates=1, eta_label=None, components=None, log_node_weights=None, query_eta_normals=None):
    """Condition query eta first, including queries outside deployment support."""
    query = indices[0] if query is None else query
    points = list(dict.fromkeys([query]+list(indices))); n = len(points)
    positions = [points.index(i) for i in indices]
    component = (np.minimum(np.searchsorted(np.cumsum(np.exp(updated.log_weights)), nodes[:, 0]),
        len(field.means)-1) if components is None else np.asarray(components))
    selected, position = np.unique(component, return_inverse=True)
    ix = (2*np.asarray(points)[:, None]+np.arange(2)).reshape(-1)
    mean = field.means[:, ix][selected]; C = field.covariances[:, ix][:, :, ix][selected]
    eta_query, logs = raw_eta_proposal(field, updated, query, component,
        ndtri(nodes[:, 1]) if query_eta_normals is None else query_eta_normals,
        label, head_log_evidence, replicates=replicates, eta_label=eta_label)
    rest = np.r_[np.arange(0, 2*n, 2), np.arange(3, 2*n, 2)]
    ev = C[:, 1, 1]; cross = C[:, rest, 1]
    before_covariance = C[:, rest][:, :, rest]-cross[:, :, None]*cross[:, None, :]/ev[:, None, None]
    before_mean = mean[position][:, rest]+cross[position]/ev[position, None]*(eta_query-mean[position, 1])[:, None]
    noise = np.exp(eta_query)/replicates
    total = before_covariance[position, 0, 0]+noise
    if n == 1:
        other_eta = np.empty((len(nodes), 0))
        mean_given_eta, covariance_given_eta = before_mean, before_covariance
    else:
        # Given query eta, the observation downdates the other-eta covariance
        # by one rank. Factor/invert the prior block once per source component,
        # then transport each normal draw with the analytic rank-one square root.
        eta_covariance = before_covariance[:, n:, n:]
        root = roots(eta_covariance); inverse = conditional_covariance_inverse(eta_covariance)
        cross = before_covariance[:, n:, 0]
        whitened = np.einsum('kij,kj->ki', root.transpose(0, 2, 1),
            np.einsum('kij,kj->ki', inverse, cross))
        length = np.sum(whitened**2, axis=1)[position]
        shrink = 1/(total*(1+np.sqrt(np.maximum(1-length/total, 0.))))
        z = ndtri(nodes[:, 2:n+1]); direction = whitened[position]
        transported = z-direction*(np.sum(direction*z, axis=1)*shrink)[:, None]
        eta_mean = before_mean[:, n:]+cross[position]/total[:, None]*(label-before_mean[:, 0])[:, None]
        other_eta = eta_mean+np.einsum('nij,nj->ni', root[position], transported)
        loading = before_covariance[:, :n, n:] @ inverse
        mean_given_eta = before_mean[:, :n]+np.einsum('nij,nj->ni', loading[position], other_eta-before_mean[:, n:])
        covariance_given_eta = before_covariance[:, :n, :n]-loading @ before_covariance[:, n:, :n]
    covariance_given_eta = .5*(covariance_given_eta+covariance_given_eta.transpose(0, 2, 1))
    variance = covariance_given_eta[position, 0, 0]+noise
    gain = covariance_given_eta[position, :, 0]/variance[:, None]
    conditional_mean = mean_given_eta+gain*(label-mean_given_eta[:, 0])[:, None]
    conditional_covariance = covariance_given_eta[position]-gain[:, :, None]*covariance_given_eta[position, None, 0, :]
    conditional_covariance = .5*(conditional_covariance+conditional_covariance.transpose(0, 2, 1))
    if log_node_weights is not None:
        component_errors = [float(logsumexp(logs[component==j]+log_node_weights[component==j])) for j in selected]
        logs = logs+updated.log_weights[component]+log_node_weights
    normalizer = logsumexp(logs); weights = np.exp(logs-normalizer)
    diagnostics = {'predictive_log_evidence_error': float(normalizer-(np.log(len(nodes)) if log_node_weights is None else 0.)),
        'effective_sample_size': float(1/(weights @ weights))}
    if log_node_weights is not None:
        diagnostics.update(component_indices=selected.tolist(),component_log_evidence_errors=component_errors)
    return (np.column_stack([eta_query, other_eta])[:, positions], conditional_mean[:, positions],
        conditional_covariance[:, positions][:, :, positions], weights, diagnostics)


def raw_guarded_head_statistics(field, updated, action, nodes, label, head_log_evidence, head, protocol):
    eta, mean, covariance, weights, diagnostics = conditional_raw_coordinates(field, updated,
        action['shortlist'], nodes, label, head_log_evidence)
    if head == 0:
        quality = clipped_objective_quality(mean, np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.),
            protocol['objective_reference'])
        i = action['incumbent_position']; paired = 0.
        if i:
            pair = [0, i]
            paired = paired_quality_from_moments(eta[:, pair], mean[:, pair], covariance[:, pair][:, :, pair],
                ndtri(nodes[:, 4]), protocol, weights)
        statistics = np.r_[weights @ quality, paired]
    else:
        coordinates = risk_coordinates_from_moments(eta, mean, covariance, np.arange(len(nodes)), nodes)
        statistics = guarded_risk_statistics(coordinates, action['incumbent_position'], action['budgets'],
            protocol, weights)[0]
    return statistics, diagnostics


def conditional_risk_action_values(eta, mean, covariance, position, nodes, action, protocol,
        *, raw, query_points, integrated_position=0):
    """Pointwise branch probabilities and cost, retaining the joint law."""
    if action['kind']=='direct':
        budget=action['budgets'][0];sd,chi_weights=_sd_quadrature(budget)
        sigma=np.exp(.5*eta[:,0]);variance=covariance[:,0,0] if raw else covariance[position,0,0]
        total=np.sqrt(np.maximum(variance,0.)+sigma**2/budget)
        threshold=_tolerance_factor(budget,protocol['chance_alpha'],protocol['familywise_delta'])*sd
        value=ndtr(-(mean[:,0,None]+sigma[:,None]*threshold)/total[:,None])@chi_weights
        return np.column_stack([value,np.zeros((len(nodes),3))]),np.full(len(nodes),float(budget))
    n=len(action['shortlist'])
    coordinates=risk_coordinates_from_moments(eta,mean,covariance,position,nodes,
        mean_normals=ndtri(nodes[:,query_points+1:query_points+n]) if raw else None,
        integrated_position=integrated_position)
    return guarded_risk_values(coordinates,action['incumbent_position'],action['budgets'],protocol,
        integrated_position=integrated_position)


def conditional_risk_action_statistics(eta, mean, covariance, position, nodes, action, protocol,
        weights, *, raw, query_points, integrated_position=0):
    """Reuse the original direct/guarded branches for a weighted joint rule."""
    values,cost=conditional_risk_action_values(eta,mean,covariance,position,nodes,action,protocol,
        raw=raw,query_points=query_points,integrated_position=integrated_position)
    if action['kind']=='direct':expected_cost=float(action['budgets'][0])
    else:
        expected_cost=float(weights @ cost);i=action['incumbent_position']
        if i:expected_cost+=(1-float(np.sum(weights)))*(action['budgets'][0]+action['budgets'][i])
    return weights @ values,expected_cost


def component_risk_statistics(field, updated, action, label, score, protocol, *, query,
        raw, order, power, seed, block_size=16, integrated_position=0):
    """Enumerate all positive components, GH one eta and conditionally QMC the rest.

    Blocks accumulate unnormalized integrals before the single final raw
    normalization. GH normals are passed directly, avoiding a lossy CDF/PPF
    round trip at the actual high-order tail nodes.
    """
    points=list(dict.fromkeys([query]+action['shortlist']));n=len(action['shortlist'])
    dimensions=(len(points)+n if raw else 2*n)
    z,gh_weights=roots_hermitenorm(order);gh_weights/=np.sqrt(2*np.pi)
    count=2**power
    separate=np.ndim(seed)>0
    rest=qmc.Sobol(max(1,dimensions-2),scramble=True,seed=int(seed[0]) if separate else seed).random_base2(power)
    one=np.full((order*count,dimensions),.5)
    if dimensions>2:one[:,2:]=np.tile(rest[:,:dimensions-2],(order,1))
    normal=np.repeat(z,count);log_one=np.repeat(np.log(gh_weights),count)-np.log(count)
    total_statistics=np.zeros(4);total_cost=0.;mass=0.;component_errors=[]
    active=np.flatnonzero(np.isfinite(updated.log_weights))
    for start in range(0,len(active),block_size):
        selected=active[start:start+block_size]
        components=np.repeat(selected,len(one));nodes=np.tile(one,(len(selected),1))
        if separate and dimensions>2:
            for k,j in enumerate(selected):
                rest=qmc.Sobol(dimensions-2,scramble=True,seed=int(seed[j])).random_base2(power)
                nodes[k*len(one):(k+1)*len(one),2:]=np.tile(rest,(order,1))
        normals=np.tile(normal,len(selected));log_nodes=np.tile(log_one,len(selected))
        if raw:
            eta,mean,covariance,weights,diagnostics=conditional_raw_coordinates(field,updated,
                action['shortlist'],nodes,label,score,query=query,components=components,
                log_node_weights=log_nodes,query_eta_normals=normals)
            block_mass=float(np.exp(diagnostics['predictive_log_evidence_error']))
            component_errors.extend(zip(diagnostics['component_indices'],diagnostics['component_log_evidence_errors']))
            position=np.arange(len(nodes))
        else:
            eta_normals=np.column_stack([normals,ndtri(nodes[:,2:n+1])])
            eta,mean,covariance,position=condition_eta_components(updated,action['shortlist'],nodes,
                components=components,eta_normals=eta_normals)
            node_weights=np.exp(updated.log_weights[components]+log_nodes)
            block_mass=float(node_weights.sum());weights=node_weights/block_mass
        statistics,cost=conditional_risk_action_statistics(eta,mean,covariance,position,nodes,action,protocol,
            weights,raw=raw,query_points=len(points),integrated_position=integrated_position)
        total_statistics+=block_mass*statistics;total_cost+=block_mass*cost;mass+=block_mass
    diagnostics=dict(predictive_log_evidence_error=float(np.log(mass)),
        components=len(active),continuous_points_per_component=len(one),
        total_points=len(active)*len(one),all_positive_components_retained=True,
        continuous_scramble_scope='component' if separate else 'shared',
        integrated_mean_position=int(integrated_position),
        distinct_continuous_scrambles=int(len(np.unique(seed[active]))) if separate else 1)
    if raw:
        errors=np.asarray([error for _,error in component_errors]);ids=np.asarray([i for i,_ in component_errors])
        contributions=np.exp(updated.log_weights[ids])*abs(np.expm1(errors))
        largest=np.argsort(contributions)[-3:][::-1]
        diagnostics.update(maximum_component_log_evidence_error=float(np.max(abs(errors))),
            weighted_component_absolute_evidence_error=float(contributions.sum()),
            largest_component_evidence_contributions=[dict(component=int(ids[k]),
                log_evidence_error=float(errors[k]),weighted_absolute_evidence_error=float(contributions[k])) for k in largest])
    return total_statistics/mass,total_cost/mass,diagnostics


def risk_reference_component(updated):
    """Choose an actual component near the posterior mean in its natural scale."""
    active=np.flatnonzero(np.isfinite(updated.log_weights));weights=np.exp(updated.log_weights[active])
    means=updated.means[active];center=weights @ means/weights.sum()
    scale=weights @ (np.diagonal(updated.covariances[active],axis1=1,axis2=2)+(means-center)**2)/weights.sum()
    distance=np.sum((means-center)**2/np.maximum(scale,1e-30),axis=1)
    return int(active[np.argmin(distance)])


def _risk_rule_nodes(dimensions,order,power,seed):
    normal,weights=roots_hermitenorm(order);weights/=np.sqrt(2*np.pi);count=2**power
    nodes=np.full((order*count,dimensions),.5)
    if dimensions>2:
        rest=qmc.Sobol(dimensions-2,scramble=True,seed=int(seed)).random_base2(power)
        nodes[:,2:]=np.tile(rest,(order,1))
    return nodes,np.repeat(normal,count),np.repeat(weights,count)/count


def _risk_component_integrands(field,updated,action,label,score,protocol,query,raw,nodes,normals,components):
    """Unnormalized five-vector, including the original raw importance ratio."""
    points=list(dict.fromkeys([query]+action['shortlist']));n=len(action['shortlist'])
    if raw:
        eta,mean,covariance,_,_=conditional_raw_coordinates(field,updated,action['shortlist'],nodes,
            label,score,query=query,components=components,query_eta_normals=normals)
        position=np.arange(len(nodes))
        _,logs=raw_eta_proposal(field,updated,query,components,normals,label,score)
        density=np.exp(logs)
    else:
        eta_normals=np.column_stack([normals,ndtri(nodes[:,2:n+1])])
        eta,mean,covariance,position=condition_eta_components(updated,action['shortlist'],nodes,
            components=components,eta_normals=eta_normals)
        density=np.ones(len(nodes))
    values,cost=conditional_risk_action_values(eta,mean,covariance,position,nodes,action,protocol,
        raw=raw,query_points=len(points))
    return np.column_stack([values,cost])*density[:,None],density


def shared_risk_reference(field,updated,action,label,score,protocol,*,query,raw,order,power,seed,
        component=None,block_size=32768):
    """Accurately integrate one actual component; this does not replace the mixture."""
    component=risk_reference_component(updated) if component is None else component
    dimensions=len(set([query]+action['shortlist']))+len(action['shortlist']) if raw else 2*len(action['shortlist'])
    nodes,normals,weights=_risk_rule_nodes(dimensions,order,power,seed)
    numerator=np.zeros(5);mass=0.
    for start in range(0,len(nodes),block_size):
        stop=min(start+block_size,len(nodes));w=weights[start:stop]
        values,density=_risk_component_integrands(field,updated,action,label,score,protocol,query,raw,
            nodes[start:stop],normals[start:stop],np.full(stop-start,component))
        numerator+=w @ values;mass+=w @ density
    return dict(component=int(component),numerator=numerator.tolist(),mass=float(mass),
        order=order,power=power,seed=int(seed),points=len(nodes))


def component_control_risk_statistics(field,updated,action,label,score,protocol,reference,*,query,
        raw,order,power,seed,block_size=16):
    """High shared reference plus exact-weight, paired component differences.

    Raw F_j=r_j*g_j includes its non-Gaussian likelihood ratio. The identity is
    sum_j w_j E[F_j]=sum_j w_j E[F_j-F_ref]+sum_j w_j E[F_ref].
    The denominator remains the original GH integral sum_j w_j E[r_j]; no
    block or reference self-normalization occurs before the final division.
    """
    active=np.flatnonzero(np.isfinite(updated.log_weights));component=reference['component']
    dimensions=len(set([query]+action['shortlist']))+len(action['shortlist']) if raw else 2*len(action['shortlist'])
    separate=np.ndim(seed)>0;seed0=int(seed[active[0]]) if separate else seed
    one,normal,node_weights=_risk_rule_nodes(dimensions,order,power,seed0)
    difference=np.zeros(5);ordinary=np.zeros(5);mass=0.;reference_low=np.zeros(5)
    for start in range(0,len(active),block_size):
        selected=active[start:start+block_size];components=np.repeat(selected,len(one))
        if separate:
            nodes=np.concatenate([_risk_rule_nodes(dimensions,order,power,int(seed[j]))[0] for j in selected])
        else:nodes=np.tile(one,(len(selected),1))
        normals=np.tile(normal,len(selected));weights=np.exp(updated.log_weights[components])*np.tile(node_weights,len(selected))
        values,density=_risk_component_integrands(field,updated,action,label,score,protocol,query,raw,
            nodes,normals,components)
        anchor,_=_risk_component_integrands(field,updated,action,label,score,protocol,query,raw,
            nodes,normals,np.full(len(nodes),component))
        difference+=weights @ (values-anchor);ordinary+=weights @ values
        reference_low+=weights @ anchor;mass+=weights @ density
    total_weight=float(np.exp(updated.log_weights[active]).sum())
    reference_high=total_weight*np.asarray(reference['numerator']);corrected=(reference_high+difference)/mass
    diagnostics=dict(predictive_log_evidence_error=float(np.log(mass)),components=len(active),
        all_positive_components_retained=True,reference_component=int(component),reference=reference,
        continuous_points_per_component=len(one),total_component_points=len(active)*len(one),
        reference_pair_points=len(active)*len(one),continuous_scramble_scope='component' if separate else 'shared',
        ordinary_statistics=(ordinary[:4]/mass).tolist(),ordinary_cost=float(ordinary[4]/mass),
        difference_integral=difference.tolist(),low_reference_integral=reference_low.tolist(),
        high_reference_integral=reference_high.tolist())
    return corrected[:4],float(corrected[4]),diagnostics
