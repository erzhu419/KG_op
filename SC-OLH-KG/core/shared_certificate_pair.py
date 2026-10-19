"""Shared incumbent/fallback certification states under the existing verifier."""
import numpy as np
from scipy.special import ndtr,ndtri,roots_hermitenorm
from scipy.stats import qmc

from core.conditional_raw_reference import conditional_raw_coordinates
from core.conditional_terminal_kg import condition_eta_components,risk_coordinates_from_moments
from core.guarded_deployment_kg import conditional_safety_power,_sd_quadrature,_tolerance_factor


def certificate_pair_states(field,updated,pair,allocations,label,score,protocol,*,query,raw,order,power,seed,
        replicates=4,eta_label=None,block_size=16,integrated_position=0):
    """Enumerate positive components; analytically integrate one pair mean.

    State order is00,01,10,11 for the requested pair. Each
    certificate uses the original three-policy guard's delta/3 and SD16.
    The full conditional field covariance is retained.
    """
    points=list(dict.fromkeys([query]+list(pair)));dimensions=len(points)+2 if raw else 4
    z,gh=roots_hermitenorm(order);gh/=np.sqrt(2*np.pi);count=2**power
    one=np.full((order*count,dimensions),.5)
    rest=qmc.Sobol(dimensions-2,scramble=True,seed=seed).random_base2(power)
    one[:,2:]=np.tile(rest,(order,1));normals=np.repeat(z,count)
    log_one=np.repeat(np.log(gh),count)-np.log(count)
    active=np.flatnonzero(np.isfinite(updated.log_weights));total=np.zeros((len(allocations),4));mass=0.
    for start in range(0,len(active),block_size):
        selected=active[start:start+block_size];components=np.repeat(selected,len(one))
        nodes=np.tile(one,(len(selected),1));normal=np.tile(normals,len(selected));log_nodes=np.tile(log_one,len(selected))
        if raw:
            eta,mean,covariance,weights,d=conditional_raw_coordinates(field,updated,pair,nodes,label,score,
                query=query,components=components,log_node_weights=log_nodes,query_eta_normals=normal,
                replicates=replicates,eta_label=eta_label)
            block_mass=float(np.exp(d['predictive_log_evidence_error']))
            covariance=covariance[::count];position=np.repeat(np.arange(len(selected)*order),count)
            mean_normals=ndtri(nodes[:,len(points)+1:len(points)+2])
        else:
            eta,mean,covariance,position=condition_eta_components(updated,pair,nodes,
                components=components,eta_normals=np.column_stack([normal,ndtri(nodes[:,2])]))
            node_weights=np.exp(updated.log_weights[components]+log_nodes);block_mass=float(node_weights.sum())
            weights=node_weights/block_mass;mean_normals=None
        draws,sigma,im,iv=risk_coordinates_from_moments(eta,mean,covariance,position,nodes,
            mean_normals=mean_normals,integrated_position=integrated_position)
        delta=protocol['familywise_delta']/3;cache={}
        for j,(bi,bb) in enumerate(allocations):
            for policy_position,budget in enumerate((bi,bb)):
                if (policy_position,budget) not in cache:
                    if policy_position==integrated_position:
                        sd,w=_sd_quadrature(budget);threshold=_tolerance_factor(budget,protocol['chance_alpha'],delta)*sd
                        denominator=np.sqrt(iv+sigma[:,policy_position]**2/budget)
                        cache[policy_position,budget]=ndtr(-(im[:,None]+sigma[:,policy_position,None]*threshold)/denominator[:,None]) @ w
                    else:cache[policy_position,budget]=conditional_safety_power(draws[:,0],sigma[:,policy_position],budget,protocol['chance_alpha'],delta)
            pi,pb=cache[0,bi],cache[1,bb]
            states=np.column_stack([(1-pi)*(1-pb),(1-pi)*pb,pi*(1-pb),pi*pb])
            total[j]+=block_mass*(weights @ states)
        mass+=block_mass
    return total/mass,dict(components=len(active),all_positive_components_retained=True,
        predictive_log_evidence_error=float(np.log(mass)),order=order,power=power,
        total_points=len(active)*order*count,source_covariance_retained=True,verifier_sd_order=16,
        integrated_mean_position=integrated_position)


def anchor_branch_quality_upper(quality,primary_cap,anchor_upper):
    """Anchor-state LP bound via a three-bin fractional knapsack.

    Before assigning primary passes, reward is qB*pi01+qI*(pi10+pi11).
    Passing primary replaces these branches by qP or qPI; their positive
    improvements share a single primary certification budget. Replacing
    each anchor-state mass by its upper bound is conservative.
    """
    quality=np.asarray(quality);pi=np.asarray(anchor_upper)
    qp,qi,qb,qpi=np.moveaxis(quality,-1,0)
    base=qb*pi[...,1]+qi*(pi[...,2]+pi[...,3])
    gains=np.maximum(np.stack([qp,qp-qb,qpi-qi],axis=-1),0.)
    caps=np.broadcast_to(np.stack([pi[...,0],pi[...,1],pi[...,2]+pi[...,3]],axis=-1),gains.shape)
    order=np.argsort(-gains,axis=-1);gains=np.take_along_axis(gains,order,axis=-1);caps=np.take_along_axis(caps,order,axis=-1)
    available=np.maximum(np.asarray(primary_cap)[...,None]-np.cumsum(caps,axis=-1)+caps,0.)
    return base+np.sum(gains*np.minimum(caps,available),axis=-1)+16*np.finfo(float).eps


def primary_incumbent_quality_upper(quality,pair_upper,fallback_cap):
    """Known P/I states fix three branches; fallback needs both to fail."""
    q=np.asarray(quality);p=np.asarray(pair_upper)
    return q[...,0]*p[...,2]+q[...,1]*p[...,1]+q[...,3]*p[...,3]+q[...,2]*np.minimum(p[...,0],fallback_cap)


def guard_statistics_from_pair_states(states,budgets,protocol):
    """Reconstruct cached numerical guard statistics, omitting fallback reward.

    States are00,01,10,11; budgets are primary,incumbent,fallback. A separate
    fallback omission cap and full mixed-reward admission are required;
    passing pair-state precision alone does not admit an action reference.
    """
    p=np.asarray(states,dtype=float);b=np.asarray(budgets,dtype=float)
    p=np.broadcast_to(p,np.broadcast(p[...,0],b[...,0]).shape+(4,))
    statistics=np.stack([p[...,2],p[...,1],np.zeros_like(p[...,0]),p[...,3]],axis=-1)
    costs=b[...,0]+b[...,1]+2*protocol['comparison_budget']*p[...,3]+b[...,2]*p[...,0]
    return statistics,costs
